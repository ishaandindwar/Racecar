import gymnasium as gym
from gymnasium.envs.registration import register

from obstacle_env import ObstacleCarRacing

_OBSTACLE_ENV_ID = "ObstacleCarRacing-v0"
_registered = False


def _register_obstacle_env():
    """Gymnasium wants every custom Env registered with an id before gym.make()
    can build it. We only need to do this once per process, so we guard it
    with a module-level flag (calling register() twice raises an error)."""
    global _registered
    if _registered:
        return
    register(
        id=_OBSTACLE_ENV_ID,
        entry_point="obstacle_env:ObstacleCarRacing",
        max_episode_steps=2000,
    )
    _registered = True


def make_racecar_env(render_mode=None, num_obstacles=0, obstacle_penalty=-5.0):
    """Build the racing environment.

    Set num_obstacles > 0 to use the obstacle-augmented track instead of the
    stock CarRacing track.
    """
    if num_obstacles > 0:
        _register_obstacle_env()
        return gym.make(
            _OBSTACLE_ENV_ID,
            render_mode=render_mode,
            num_obstacles=num_obstacles,
            obstacle_penalty=obstacle_penalty,
        )

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
