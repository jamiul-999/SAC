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
