import gymnasium as gym


def make_racecar_env(render_mode=None):
    env_names = ["Racecar-v3", "racecar-v3", "CarRacing-v3"]

    last_error = None

    for env_name in env_names:
        try:
            return gym.make(env_name, render_mode=render_mode)
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not open Racecar-v3 or CarRacing-v3. "
        "Make sure gymnasium[box2d] is installed."
    ) from last_error