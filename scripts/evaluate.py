"""
Load a trained checkpoint and run evaluation episodes, printing reward stats.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/halfcheetah_base.pt \
        --config configs/halfcheetah_base.yaml --episodes 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sac.agent import SACAgent
from sac.utils import load_config
from envs.make_env import make_env, make_safety_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy mean actions (default)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic sampled actions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = (make_safety_env if cfg.get("is_safety_env") else make_env)(cfg["env_id"], seed=cfg["seed"] + 2000)

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_dim=cfg["hidden_dim"],
        use_twin_q=cfg.get("use_twin_q", True),
        auto_tune_alpha=cfg["auto_tune_alpha"],
    )
    agent.load(args.checkpoint)

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
        print(f"Episode {ep + 1}/{args.episodes}: return={ep_return:.2f}")

    print(f"\nMean return over {args.episodes} episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
