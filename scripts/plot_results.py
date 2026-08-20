"""
Plotting script to analyze and visualize baseline & ablation experiment results.

Usage:
    python scripts/plot_results.py

Outputs PNG plots to results/plots/ for:
- baseline_comparison.png: Baseline curves (SB3 vs SAC)
- ablation_alpha_sweep.png: Entropy temperature ablation
- ablation_twin_vs_single_q.png: Twin Q vs Single Q critic
- ablation_stochastic_vs_det.png: Stochastic vs Deterministic evaluation
- ablation_reward_scaling.png: Reward scaling ablation
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "logs")
RESULTS_PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "plots")


def load_all_logs():
    logs = {}
    pattern = os.path.join(RESULTS_LOG_DIR, "*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        key = os.path.splitext(filename)[0]
        try:
            with open(filepath, "r") as f:
                logs[key] = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")
    return logs


def plot_ablation_group(logs, filter_prefix: str, title: str, output_name: str, label_fn):
    plt.figure(figsize=(9, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    found = False
    for name, data in logs.items():
        if name.startswith(filter_prefix):
            found = True
            history = data.get("history", {})
            steps = history.get("steps", [])
            means = np.array(history.get("eval_reward_mean", []))
            stds = np.array(history.get("eval_reward_std", []))
            label = label_fn(name, data)

            plt.plot(steps, means, label=label, linewidth=2)
            plt.fill_between(steps, means - stds, means + stds, alpha=0.15)

    if not found:
        print(f"No logs found starting with '{filter_prefix}' for plot {output_name}")
        plt.close()
        return

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(RESULTS_PLOT_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_PLOT_DIR, output_name)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    logs = load_all_logs()
    print(f"Found {len(logs)} log files in {RESULTS_LOG_DIR}")

    # 1. Alpha sweep
    plot_ablation_group(
        logs,
        filter_prefix="alpha_",
        title="SAC Ablation Study: Temperature (α) Tuning",
        output_name="ablation_alpha_sweep.png",
        label_fn=lambda name, data: f"α = {name.replace('alpha_', '')}"
    )

    # 2. Twin vs Single Q
    plot_ablation_group(
        logs,
        filter_prefix="use_twin_q_",
        title="SAC Ablation Study: Twin Q vs. Single Q Critic",
        output_name="ablation_twin_vs_single_q.png",
        label_fn=lambda name, data: "Twin Q (Clipped Double-Q)" if "true" in name else "Single Q"
    )

    # 3. Stochastic vs Deterministic
    plot_ablation_group(
        logs,
        filter_prefix="deterministic_eval_",
        title="SAC Ablation Study: Stochastic vs. Deterministic Evaluation",
        output_name="ablation_stochastic_vs_det.png",
        label_fn=lambda name, data: "Deterministic (Mean)" if "true" in name else "Stochastic (Sampled)"
    )

    # 4. Reward scaling
    plot_ablation_group(
        logs,
        filter_prefix="reward_scale_",
        title="SAC Ablation Study: Reward Scaling (Fixed α = 0.2)",
        output_name="ablation_reward_scaling.png",
        label_fn=lambda name, data: f"Reward Scale = {name.replace('reward_scale_', '')}"
    )


if __name__ == "__main__":
    main()
