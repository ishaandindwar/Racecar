
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from racecar_env import make_racecar_env


def create_env():
    return make_racecar_env(render_mode="human")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--model", default="models/racecar_ppo.zip")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # Wrap the environment exactly as in training
    env = DummyVecEnv([create_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

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

