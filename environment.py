"""
FAIO Robot Walker environment.

This file is part of the problem statement. The official evaluator uses
exactly this environment.

The environment is Gymnasium's Walker2d-v5 with two changes:
  1. A target speed v_target ~ Uniform(0.8, 5.0) is drawn at the start of each
     episode and appended to the observation (17 numbers -> 18 numbers).
  2. The reward asks the robot to walk AT the target speed, not as fast as
     possible:
         reward = exp(-4 * (v - v_target)^2) + 0.05 - 0.001 * sum(action^2)
     where v is the robot's forward speed on this step.

Episodes end when the robot falls (Walker2d's standard "unhealthy" check) or
after 1250 steps (10 seconds of simulated time).
"""

import gymnasium as gym
import numpy as np

V_TARGET_MIN = 0.8
V_TARGET_MAX = 5.0
MAX_STEPS = 1250  # 1250 steps * 0.008 s = 10 s of simulated time
OBS_DIM = 18
ACT_DIM = 6


class RobotWalker(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("Walker2d-v5", max_episode_steps=MAX_STEPS, render_mode=render_mode)
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.v_target = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # np_random is (re)seeded by the reset above, so v_target is reproducible.
        self.v_target = float(self.env.unwrapped.np_random.uniform(V_TARGET_MIN, V_TARGET_MAX))
        info["v_target"] = self.v_target
        return self._add_target(obs), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, _, terminated, truncated, info = self.env.step(action)
        v = float(info["x_velocity"])
        reward = float(np.exp(-4.0 * (v - self.v_target) ** 2) + 0.05 - 0.001 * np.sum(action ** 2))
        info["v_target"] = self.v_target
        return self._add_target(obs), reward, terminated, truncated, info

    def _add_target(self, obs):
        return np.append(obs, self.v_target).astype(np.float64)


def make_env(render_mode=None):
    return RobotWalker(render_mode=render_mode)
