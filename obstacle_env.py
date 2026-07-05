import math

import numpy as np
from Box2D.b2 import fixtureDef, polygonShape

from gymnasium.envs.box2d.car_racing import CarRacing, TRACK_WIDTH


class ObstacleCarRacing(CarRacing):
    """
    CarRacing with static obstacles dropped onto the track.

    Obstacles are real Box2D static bodies, so the physics engine handles
    collision response automatically (the car will physically bump them).
    We additionally apply a reward penalty when contact happens.
    """

    def __init__(self, num_obstacles=8, obstacle_penalty=-5.0,
                 obstacle_half_size=1.2, **kwargs):
        self.num_obstacles = num_obstacles
        self.obstacle_penalty = obstacle_penalty
        self.obstacle_half_size = obstacle_half_size
        self.obstacle_bodies = []
        super().__init__(**kwargs)

    def _destroy(self):
        for body in self.obstacle_bodies:
            self.world.DestroyBody(body)
        self.obstacle_bodies = []
        super()._destroy()

    def _place_obstacles(self):
        self.obstacle_bodies = []
        if self.num_obstacles <= 0 or not self.track:
            return

        track_len = len(self.track)
        safe_start, safe_end = 15, track_len - 15
        if safe_end <= safe_start:
            return

        n = min(self.num_obstacles, safe_end - safe_start)
        indices = self.np_random.choice(
            np.arange(safe_start, safe_end), size=n, replace=False
        )

        s = self.obstacle_half_size
        local_box = [(-s, -s), (s, -s), (s, s), (-s, s)]

        for idx in indices:
            _, beta, x, y = self.track[int(idx)]
            lateral = self.np_random.uniform(-0.55, 0.55) * TRACK_WIDTH
            ox = x - lateral * math.sin(beta)
            oy = y + lateral * math.cos(beta)

            fixture = fixtureDef(
                shape=polygonShape(vertices=local_box),
                density=1.0,
                friction=0.9,
                restitution=0.0,
            )
            body = self.world.CreateStaticBody(position=(ox, oy), fixtures=fixture)
            body.color = (0.65, 0.1, 0.1)
            body.is_obstacle = True
            body.userData = body  # see note in _car_hit_obstacle() below
            self.obstacle_bodies.append(body)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._place_obstacles()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._car_hit_obstacle():
            reward += self.obstacle_penalty
            info["hit_obstacle"] = True
        return obs, reward, terminated, truncated, info

    def _car_hit_obstacle(self):
        if self.car is None:
            return False
        # NOTE: contact.fixtureX.body returns a *new* SWIG proxy each time it's
        # accessed, so a plain attribute we set on our body (e.g. `.is_obstacle`)
        # is lost by the time we read it back here. `.userData`, however, is a
        # real Box2D field that keeps the exact same Python object alive - this
        # is the same trick FrictionDetector uses for road tiles. So we stash a
        # self-reference in userData and read attributes off of *that* instead.
        car_bodies = [self.car.hull] + self.car.wheels
        for contact in self.world.contacts:
            if not contact.touching:
                continue
            ud_a = contact.fixtureA.body.userData
            ud_b = contact.fixtureB.body.userData
            a_is_obstacle = getattr(ud_a, "is_obstacle", False)
            b_is_obstacle = getattr(ud_b, "is_obstacle", False)
            if a_is_obstacle and contact.fixtureB.body in car_bodies:
                return True
            if b_is_obstacle and contact.fixtureA.body in car_bodies:
                return True
        return False

    def _render_road(self, zoom, translation, angle):
        super()._render_road(zoom, translation, angle)
        s = self.obstacle_half_size
        for body in self.obstacle_bodies:
            x, y = body.position
            poly = [(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)]
            self._draw_colored_polygon(
                self.surf, poly, (165, 25, 25), zoom, translation, angle
            )
