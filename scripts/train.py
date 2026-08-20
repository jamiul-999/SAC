"""
Train our own SAC implementation on a single environment, driven by a config file.

Usage:
    python scripts/train.py --config configs/halfcheetah_base.yaml
    python scripts/train.py --config configs/halfcheetah_base.yaml --total-timesteps-override 5000  # smoke test

Runs on GPU automatically if available (e.g. on Kaggle) -- no code changes needed
between local CPU testing and a Kaggle GPU run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sac.agent import SACAgent
from sac.buffer import ReplayBuffer
from sac.utils import set_seed, load_config
from envs.make_env import make_env, make_safety_env

RESULTS_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "logs")
RESULTS_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "checkpoints")


def run_eval_episodes(env, agent, n_episodes: int = 5, deterministic: bool = True):
    """Run a few episodes with the current policy (no exploration noise, no learning)."""
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
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(costs))


def run_training(cfg: dict, run_name: str, total_timesteps_override: int = None):
    """Runs one full training job for the given config. Callable directly
    (e.g. from run_ablation.py) or via the CLI entry point below."""
    set_seed(cfg["seed"])
    total_timesteps = total_timesteps_override or cfg["total_timesteps"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] run_name={run_name} env={cfg['env_id']} device={device} total_timesteps={total_timesteps}")

    if cfg.get("is_safety_env"):
        env = make_safety_env(cfg["env_id"], seed=cfg["seed"])
        eval_env = make_safety_env(cfg["env_id"], seed=cfg["seed"] + 1000)
    else:
        env = make_env(cfg["env_id"], seed=cfg["seed"])
        eval_env = make_env(cfg["env_id"], seed=cfg["seed"] + 1000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg["hidden_dim"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        alpha=cfg["alpha"],
        auto_tune_alpha=cfg["auto_tune_alpha"],
        use_twin_q=cfg.get("use_twin_q", True),
        device=device,
    )
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=cfg["buffer_capacity"])

    batch_size = cfg["batch_size"]
    eval_every = cfg.get("eval_every", 10_000)
    start_steps = cfg.get("start_steps", min(10_000, total_timesteps // 10))
    reward_scale = cfg.get("reward_scale", 1.0)  # ablation axis 4: reward scaling

    log_history = {"steps": [], "eval_reward_mean": [], "eval_reward_std": [], "eval_cost_mean": [], "update_metrics": []}

    obs, _ = env.reset(seed=cfg["seed"])
    start_time = time.time()

    for t in range(1, total_timesteps + 1):
        if t <= start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Only bootstrap-zero on true termination, not on time-limit truncation
        # (standard practice -- truncation is an artifact of the episode horizon,
        # not a signal that no future reward was possible).
        buffer.add(obs, action, reward * reward_scale, next_obs, float(terminated))

        obs = next_obs
        if done:
            obs, _ = env.reset()

        if t > start_steps and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size, device=device)
            metrics = agent.update(batch)
        else:
            metrics = None

        if t % eval_every == 0 or t == total_timesteps:
            deterministic_eval = cfg.get("deterministic_eval", True)
            eval_mean, eval_std, eval_cost = run_eval_episodes(
                eval_env, agent, n_episodes=cfg.get("eval_episodes", 5), deterministic=deterministic_eval
            )
            elapsed = time.time() - start_time
            cost_str = f" eval_cost={eval_cost:.1f}" if cfg.get("is_safety_env") else ""
            print(f"[train] step={t}/{total_timesteps} eval_reward={eval_mean:.2f}±{eval_std:.2f}{cost_str} "
                  f"alpha={agent.alpha:.4f} elapsed={elapsed:.1f}s")

            log_history["steps"].append(t)
            log_history["eval_reward_mean"].append(eval_mean)
            log_history["eval_reward_std"].append(eval_std)
            log_history["eval_cost_mean"].append(eval_cost)
            log_history["update_metrics"].append(metrics)

            os.makedirs(RESULTS_LOG_DIR, exist_ok=True)
            os.makedirs(RESULTS_CKPT_DIR, exist_ok=True)
            with open(os.path.join(RESULTS_LOG_DIR, f"{run_name}.json"), "w") as f:
                json.dump({"config": cfg, "history": log_history}, f)
            agent.save(os.path.join(RESULTS_CKPT_DIR, f"{run_name}.pt"))

    env.close()
    eval_env.close()
    print(f"[train] Done. Logs: results/logs/{run_name}.json  Checkpoint: results/checkpoints/{run_name}.pt")
    return log_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None, help="Defaults to the config filename stem.")
    parser.add_argument("--total-timesteps-override", type=int, default=None,
                         help="Use a small value (e.g. 2000) for a smoke test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    run_training(cfg, run_name, args.total_timesteps_override)


if __name__ == "__main__":
    main()
