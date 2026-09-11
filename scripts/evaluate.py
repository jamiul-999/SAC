"""
Load trained SAC checkpoint(s) and run evaluation episodes, printing return and cost statistics.

Usage:
    # Single checkpoint evaluation
    python scripts/evaluate.py --checkpoint results/checkpoints/halfcheetah_base.pt --config configs/halfcheetah_base.yaml

    # Compare stochastic vs. deterministic action selection on the same model
    python scripts/evaluate.py --checkpoint results/checkpoints/halfcheetah_base.pt --config configs/halfcheetah_base.yaml --compare-modes

    # Batch evaluate all checkpoints matching a pattern (e.g. from an ablation sweep)
    python scripts/evaluate.py --pattern "results/checkpoints/alpha_*.pt" --config configs/halfcheetah_base.yaml --episodes 10
"""

import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.utils import load_config


def evaluate_agent(env, agent, n_episodes: int = 10, deterministic: bool = True):
    """Runs n_episodes and returns (mean_return, std_return, mean_cost, std_cost)."""
    returns = []
    costs = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_cost = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if isinstance(info, dict) and "cost" in info:
                ep_cost += float(info["cost"])
            done = terminated or truncated
        returns.append(ep_return)
        costs.append(ep_cost)
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(costs)), float(np.std(costs))


def evaluate_single_checkpoint(checkpoint_path: str, cfg: dict, n_episodes: int = 10,
                               deterministic: bool = True, compare_modes: bool = False):
    from sac.agent import SACAgent
    from envs.make_env import make_env, make_safety_env

    is_safety = cfg.get("is_safety_env", False)
    env_fn = make_safety_env if is_safety else make_env
    env = env_fn(cfg["env_id"], seed=cfg.get("seed", 0) + 2000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.get("hidden_dim", 256),
        use_twin_q=cfg.get("use_twin_q", True),
        auto_tune_alpha=cfg.get("auto_tune_alpha", True),
        target_entropy=cfg.get("target_entropy", None),
    )
    agent.load(checkpoint_path)

    results = {}
    if compare_modes:
        ret_det, std_det, c_det, _ = evaluate_agent(env, agent, n_episodes, deterministic=True)
        ret_stoch, std_stoch, c_stoch, _ = evaluate_agent(env, agent, n_episodes, deterministic=False)
        results["deterministic"] = (ret_det, std_det, c_det)
        results["stochastic"] = (ret_stoch, std_stoch, c_stoch)
    else:
        ret, std, c, _ = evaluate_agent(env, agent, n_episodes, deterministic=deterministic)
        results["evaluation"] = (ret, std, c)

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="SAC Evaluation & Ablation Comparison")
    parser.add_argument("--checkpoint", default=None, help="Path to single .pt checkpoint")
    parser.add_argument("--pattern", default=None, help="Glob pattern for batch checkpoint evaluation (e.g. 'results/checkpoints/alpha_*.pt')")
    parser.add_argument("--config", required=True, help="Base config file path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes per checkpoint")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions (default)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic sampled actions")
    parser.add_argument("--compare-modes", action="store_true", help="Compare deterministic vs. stochastic actions head-to-head")
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    elif args.pattern:
        checkpoints = sorted(glob.glob(args.pattern))
        if not checkpoints:
            print(f"No checkpoints found matching pattern: {args.pattern}")
            return
    else:
        raise ValueError("Please provide either --checkpoint or --pattern.")

    print(f"\n=======================================================")
    print(f"SAC Evaluator: Evaluating {len(checkpoints)} checkpoint(s)")
    print(f"Environment: {cfg['env_id']} | Episodes per check: {args.episodes}")
    print(f"=======================================================\n")

    summary = []
    for ckpt in checkpoints:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        if not os.path.isfile(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            continue

        res = evaluate_single_checkpoint(
            ckpt, cfg,
            n_episodes=args.episodes,
            deterministic=args.deterministic,
            compare_modes=args.compare_modes
        )

        if args.compare_modes:
            det_ret, det_std, det_c = res["deterministic"]
            stoch_ret, stoch_std, stoch_c = res["stochastic"]
            diff = det_ret - stoch_ret
            print(f"Model: {name}")
            print(f"  • Deterministic: {det_ret:.2f} ± {det_std:.2f}")
            print(f"  • Stochastic:    {stoch_ret:.2f} ± {stoch_std:.2f}")
            print(f"  • Difference (Det - Stoch): {diff:+.2f}\n")
            summary.append((name, f"{det_ret:.2f} ± {det_std:.2f}", f"{stoch_ret:.2f} ± {stoch_std:.2f}", f"{diff:+.2f}"))
        else:
            ret, std, c = res["evaluation"]
            cost_str = f" | Cost: {c:.1f}" if cfg.get("is_safety_env") else ""
            print(f"Model: {name:<30} -> Return: {ret:.2f} ± {std:.2f}{cost_str}")
            summary.append((name, f"{ret:.2f} ± {std:.2f}", f"{c:.1f}" if cfg.get("is_safety_env") else "N/A"))

    if args.compare_modes and len(summary) > 1:
        print("\n=== SUMMARY: Stochastic vs. Deterministic Evaluation ===")
        print(f"{'Checkpoint':<30} | {'Deterministic':<18} | {'Stochastic':<18} | {'Diff'}")
        print("-" * 75)
        for row in summary:
            print(f"{row[0]:<30} | {row[1]:<18} | {row[2]:<18} | {row[3]}")


if __name__ == "__main__":
    main()
