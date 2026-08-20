"""
Wrapper(s) for extracting the cost signal from safety-gymnasium environments
into a clean (obs, reward, cost, terminated, truncated, info) interface that
the SAC agent's CMDP-aware training loop can consume.
"""

import gymnasium as gym


class CostWrapper(gym.Wrapper):
    """
    Wrapper for safety environments to ensure info['cost'] is populated.
    Handles safety-gymnasium 6-element step output (obs, reward, cost, terminated, truncated, info)
    or 5-element step output with cost in info dict.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        step_returns = self.env.step(action)
        if len(step_returns) == 6:
            obs, reward, cost, terminated, truncated, info = step_returns
            info["cost"] = cost
            return obs, reward, terminated, truncated, info
        elif len(step_returns) == 5:
            obs, reward, terminated, truncated, info = step_returns
            if "cost" not in info:
                info["cost"] = 0.0
            return obs, reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step returns length: {len(step_returns)}")

class VelocityCostWrapper(gym.Wrapper):
    """
    cost = 1.0 if |x_velocity| exceeds velocity_threshold, else 0.0.
    Reward is untouched (task objective unchanged).
    """
    def __init__(self, env: gym.Env, velocity_threshold: float = 2.0):
        super().__init__(env)
        self.velocity_threshold = velocity_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get("x_velocity", 0.0)
        info["cost"] = 1.0 if abs(x_velocity) > self.velocity_threshold else 0.0
        return obs, reward, terminated, truncated, info


def make_safety_ant_velocity(seed: int = 0, velocity_threshold: float = 2.0):
    """Ant-v5 (27-dim obs, matching the rest of this project) + velocity cost."""
    env = gym.make("Ant-v5", include_cfrc_ext_in_observation=False)
    env = VelocityCostWrapper(env, velocity_threshold=velocity_threshold)
    env.reset(seed=seed)
    return env