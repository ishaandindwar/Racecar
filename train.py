import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)

from racecar_env import make_racecar_env

MODEL_FOLDER = Path("models")
MODEL_PATH = MODEL_FOLDER / "racecar_ppo.zip"
VECNORM_PATH = MODEL_FOLDER / "vecnormalize.pkl"


class ActionRepeat(gym.Wrapper):
    """Repeat each action for a fixed number of physics ticks (see earlier
    explanation: lowers the effective control frequency so each decision
    has a clearer effect on outcome)."""

    def __init__(self, env, repeat=2):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class OffTrackTimeout(gym.Wrapper):
    """CarRacing's own termination only fires when the car is ~50x the road
    width away from the track (see PLAYFIELD vs TRACK_WIDTH). That means a
    car that drifts onto the grass and gives up can burn an entire episode
    doing nothing, wasting rollout budget on a hopeless run instead of
    letting PPO try again from a fresh track.

    We track how many consecutive steps have passed with NO new road-tile
    reward (a raw per-step reward under this threshold means "no tile was
    hit this step" for the un-shaped CarRacing reward). If that streak gets
    too long, we end the episode early ourselves.
    """

    def __init__(self, env, patience=150, no_progress_threshold=-0.05):
        super().__init__(env)
        self.patience = patience
        self.no_progress_threshold = no_progress_threshold
        self._stall_counter = 0

    def reset(self, **kwargs):
        self._stall_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward <= self.no_progress_threshold:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        if self._stall_counter >= self.patience:
            truncated = True
            info["off_track_timeout"] = True

        return obs, reward, terminated, truncated, info


def make_env_fn(render_mode, num_obstacles, obstacle_penalty, action_repeat, seed):
    def _init():
        env = make_racecar_env(
            render_mode=render_mode,
            num_obstacles=num_obstacles,
            obstacle_penalty=obstacle_penalty,
        )
        env = OffTrackTimeout(env, patience=150)
        if action_repeat > 1:
            env = ActionRepeat(env, repeat=action_repeat)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_training_env(n_envs, render, num_obstacles, obstacle_penalty, action_repeat):
    render_mode = "human" if render else "rgb_array"
    env_fns = [
        make_env_fn(render_mode, num_obstacles, obstacle_penalty, action_repeat, seed=i)
        for i in range(n_envs)
    ]
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_env_cls(env_fns)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-penalty", type=float, default=-5.0)
    parser.add_argument("--action-repeat", type=int, default=2)
    args = parser.parse_args()

    MODEL_FOLDER.mkdir(exist_ok=True)

    env = make_training_env(
        n_envs=args.n_envs,
        render=args.render,
        num_obstacles=args.num_obstacles,
        obstacle_penalty=args.obstacle_penalty,
        action_repeat=args.action_repeat,
    )
    eval_env = make_training_env(
        n_envs=1,
        render=False,
        num_obstacles=args.num_obstacles,
        obstacle_penalty=args.obstacle_penalty,
        action_repeat=args.action_repeat,
    )

    if args.resume and MODEL_PATH.exists():
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env)
        if VECNORM_PATH.exists():
            env = VecNormalize.load(str(VECNORM_PATH), env)
            model.set_env(env)
    else:
        print("Creating new model...")
        # Hyperparameters below match Stable-Baselines3's own tuned config
        # for CarRacing (rl-zoo), which is the reference point for what
        # actually works on this specific environment:
        #  - ent_coef=0.0 + use_sde=True: State-Dependent Exploration.
        #    Instead of resampling independent random noise on every action
        #    (which just produces jittery twitching for a continuous
        #    steering task), SDE samples a noise function every
        #    `sde_sample_freq` steps and holds it - so exploration looks
        #    like "steer slightly left for a while" instead of noise, which
        #    is far more likely to stumble onto a working driving policy.
        #  - log_std_init=-2, ortho_init=False: starts the policy's action
        #    distribution narrower and skips orthogonal weight init, both
        #    empirically important for this environment's continuous
        #    action space to converge reliably.
        #  - learning_rate=1e-4, n_steps=512, batch_size=128: smaller,
        #    steadier updates than the generic PPO defaults.
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=dict(log_std_init=-2, ortho_init=False),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=str(MODEL_FOLDER),
        name_prefix="racecar_checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_FOLDER / "best"),
        log_path=str(MODEL_FOLDER / "eval_logs"),
        eval_freq=max(20000 // args.n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )

    model.save(MODEL_PATH)
    env.save(str(VECNORM_PATH))
    env.close()
    eval_env.close()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Best model (by eval score) saved under {MODEL_FOLDER / 'best'}")


if __name__ == "__main__":
    main()