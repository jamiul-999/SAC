"""
Unified environment factory.

Always use the current (non-deprecated) env versions:
- HalfCheetah-v5, Walker2d-v5, Ant-v5 (NOT v4 -- deprecated, will warn/eventually break)
- Ant-v5 is created with include_cfrc_ext_in_observation=False to match the
  27-dim observation used in the original SAC paper and our MDP spec
  (default Ant-v5 is 105-dim: it includes contact forces by default, v4 did not).
- Safety envs: use safety-gymnasium's current IDs (verify exact string when
  the package is installed -- e.g. SafetyAntVelocity-v1).
"""

import gymnasium as gym

STANDARD_ENV_KWARGS = {
    "HalfCheetah-v5": {},
    "Walker2d-v5": {},
    "Ant-v5": {"include_cfrc_ext_in_observation": False},
}


def make_env(env_id: str, seed: int = 0):
    """
    Create and seed a standard (non-safety) Gymnasium environment.
    For safety envs, use make_safety_env below instead.
    """
    if env_id not in STANDARD_ENV_KWARGS:
        raise ValueError(
            f"Unrecognized standard env_id '{env_id}'. "
            f"Known: {list(STANDARD_ENV_KWARGS.keys())}. "
            "If adding a new env, use its current (non-deprecated) version string."
        )
    kwargs = STANDARD_ENV_KWARGS[env_id]
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    return env


from envs.safety_wrappers import CostWrapper


def make_safety_env(env_id: str, seed: int = 0):
    """
    Create and seed a safety-gymnasium environment (e.g. SafetyAntVelocity-v1).
    Wraps it in CostWrapper to expose cost signal in info['cost'].
    """
    try:
        import safety_gymnasium
        env = safety_gymnasium.make(env_id)
        env.reset(seed=seed)
        return CostWrapper(env)
    except ImportError:
        try:
            env = gym.make(env_id)
            env.reset(seed=seed)
            return CostWrapper(env)
        except Exception as e:
            raise RuntimeError(
                f"Could not load safety environment '{env_id}'. "
                f"Please ensure safety-gymnasium is installed (pip install safety-gymnasium). Error: {e}"
            )
