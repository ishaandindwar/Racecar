import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)

from racecar_env import make_racecar_env

VECNORM_PATH = Path("models/vecnormalize.pkl")


def create_env(num_obstacles, obstacle_penalty):
    def _init():
        return make_racecar_env(
            render_mode="human",
            num_obstacles=num_obstacles,
            obstacle_penalty=obstacle_penalty,
        )

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--model", default="models/racecar_ppo.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-penalty", type=float, default=-5.0)
    args = parser.parse_args()

    # Wrap the environment exactly as in training
    env = DummyVecEnv([create_env(args.num_obstacles, args.obstacle_penalty)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    # VecNormalize keeps a running mean/variance of rewards learned during
    # training. Loading it here (instead of creating a fresh one) matters
    # because episode length/finish stats you see below stay comparable to
    # what training reported - a fresh VecNormalize would rescale rewards
    # differently and make the printed totals meaningless.
    if VECNORM_PATH.exists():
        env = VecNormalize.load(str(VECNORM_PATH), env)
        env.training = False  # freeze the running statistics
        env.norm_reward = False  # show raw, human-readable rewards while playing

    print("Opened environment.")

    model = None

    if not args.random:
        model_path = Path(args.model)

        if not model_path.exists():
            raise FileNotFoundError(
                "No trained model found. First run: python train.py --timesteps 100000"
            )

        model = PPO.load(model_path, env=env)

    for episode in range(args.episodes):
        observation = env.reset()
        done = [False]
        total_reward = 0.0

        while not done[0]:
            if args.random:
                action = [env.action_space.sample()]
            else:
                action, _ = model.predict(observation, deterministic=True)

            observation, reward, done, info = env.step(action)
            total_reward += reward[0]

        print(f"Episode {episode + 1}: reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
