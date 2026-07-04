import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from racecar_env import make_racecar_env


MODEL_FOLDER = Path("models")
MODEL_PATH = MODEL_FOLDER / "racecar_ppo.zip"


def make_training_env(render=False):
    render_mode = "human" if render else "rgb_array"

    def create_env():
        env = make_racecar_env(render_mode=render_mode)
        env = Monitor(env)
        return env

    env = DummyVecEnv([create_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    MODEL_FOLDER.mkdir(exist_ok=True)

    env = make_training_env(render=args.render)

    if args.resume and MODEL_PATH.exists():
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=0.00025,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(MODEL_FOLDER),
        name_prefix="racecar_checkpoint",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    env.close()

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()