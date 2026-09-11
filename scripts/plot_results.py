"""
Plotting and analysis script for SAC baseline and ablation experiment results.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --save-tables --format pdf
    python scripts/plot_results.py --log-dir results/logs --plot-dir results/plots

Generated Plots (in results/plots/):
- baseline_comparison.png: SB3 baseline vs our SAC implementation
- ablation_alpha_sweep.png: Entropy temperature (alpha) sweep
- ablation_alpha_dynamics.png: Evolution of auto-tuned alpha vs fixed values
- ablation_twin_vs_single_q.png: Twin Q vs Single Q critic evaluation returns
- ablation_q_overestimation.png: Q-value estimation curves (overestimation bias analysis)
- ablation_stochastic_vs_det.png: Stochastic vs Deterministic policy evaluation
- ablation_reward_scaling.png: Sensitivity of fixed alpha to reward scale
- ablation_target_entropy.png: Target entropy heuristic sweep
- ablation_safety_cost_penalty.png: Safety cost penalty sweep (SafetyAntVelocity-v1)

Summary Tables (in results/):
- ablation_summary.md: Markdown table of final evaluation rewards and costs
- ablation_summary.tex: LaTeX booktabs table ready for insertion into report/paper
- ablation_summary.csv: Raw CSV export for spreadsheet analysis
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")
DEFAULT_PLOT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "results")


def load_all_logs(log_dir: str = DEFAULT_LOG_DIR) -> dict:
    """Load all experiment logs (*.json) from the given directory."""
    logs = {}
    pattern = os.path.join(log_dir, "*.json")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        key = os.path.splitext(filename)[0]
        try:
            with open(filepath, "r") as f:
                logs[key] = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")
    return logs


def extract_group_key(run_name: str) -> str:
    """Strip seed suffixes like _seed0, _seed1 to group multi-seed runs."""
    return re.sub(r"_seed\d+$", "", run_name)


def aggregate_group_runs(logs: dict, filter_prefix: str, metric_key: str = "eval_reward_mean"):
    """
    Groups matching runs by condition prefix, aggregating curves across seeds.
    Returns: dict of group_key -> {"steps": np.ndarray, "mean": np.ndarray, "std": np.ndarray, "data": dict, "seeds": list}
    """
    grouped = defaultdict(list)
    for name, data in logs.items():
        if name.startswith(filter_prefix):
            group_key = extract_group_key(name)
            history = data.get("history", {})
            steps = history.get("steps", [])
            values = history.get(metric_key, [])
            if steps and values:
                grouped[group_key].append({
                    "name": name,
                    "steps": np.array(steps),
                    "values": np.array(values),
                    "data": data,
                })

    aggregated = {}
    for group_key, runs in grouped.items():
        if not runs:
            continue
        # Use steps from first run
        steps = runs[0]["steps"]
        all_vals = []
        for r in runs:
            # interpolate if steps mismatch slightly
            if len(r["steps"]) == len(steps) and np.all(r["steps"] == steps):
                all_vals.append(r["values"])
            else:
                interp = np.interp(steps, r["steps"], r["values"])
                all_vals.append(interp)

        all_vals = np.array(all_vals)  # shape (n_seeds, n_steps)
        mean_curve = np.mean(all_vals, axis=0)
        # If multiple seeds, std across seeds; if single seed, fall back to within-run eval_reward_std if available
        if len(runs) > 1:
            std_curve = np.std(all_vals, axis=0)
        else:
            std_curve = np.array(runs[0]["data"].get("history", {}).get("eval_reward_std", np.zeros_like(mean_curve)))

        aggregated[group_key] = {
            "steps": steps,
            "mean": mean_curve,
            "std": std_curve,
            "n_seeds": len(runs),
            "sample_data": runs[0]["data"],
        }

    return aggregated


def plot_ablation_group(
    logs: dict,
    filter_prefix: str,
    title: str,
    output_name: str,
    label_fn,
    metric_key: str = "eval_reward_mean",
    ylabel: str = "Evaluation Reward",
    plot_dir: str = DEFAULT_PLOT_DIR,
    fig_format: str = "png",
):
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot {output_name}: matplotlib is not installed.")
        return

    aggregated = aggregate_group_runs(logs, filter_prefix, metric_key=metric_key)
    if not aggregated:
        print(f"No logs found starting with '{filter_prefix}' for plot {output_name}")
        return

    plt.figure(figsize=(8.5, 4.8), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for group_key, agg in sorted(aggregated.items()):
        steps = agg["steps"]
        mean = agg["mean"]
        std = agg["std"]
        label = label_fn(group_key, agg["sample_data"])
        if agg["n_seeds"] > 1:
            label += f" (n={agg['n_seeds']} seeds)"

        line = plt.plot(steps, mean, label=label, linewidth=2)[0]
        plt.fill_between(steps, mean - std, mean + std, alpha=0.16, color=line.get_color())

    plt.xlabel("Environment Timesteps", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    stem, _ = os.path.splitext(output_name)
    out_path = os.path.join(plot_dir, f"{stem}.{fig_format}")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_alpha_trajectory(logs: dict, plot_dir: str = DEFAULT_PLOT_DIR, fig_format: str = "png"):
    """Plot the learned temperature alpha trajectory over timesteps (evolution of entropy tuning)."""
    if not HAS_MATPLOTLIB:
        return

    found = False
    plt.figure(figsize=(8.5, 4.8), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for name, data in sorted(logs.items()):
        if name.startswith("alpha_"):
            history = data.get("history", {})
            steps = history.get("steps", [])
            update_metrics = history.get("update_metrics", [])
            alphas = [m.get("alpha") for m in update_metrics if isinstance(m, dict) and "alpha" in m]
            if steps and alphas and len(steps) == len(alphas):
                found = True
                label = "Auto-tuned α" if "auto" in name.lower() else f"Fixed α = {name.replace('alpha_', '').split('_seed')[0]}"
                plt.plot(steps, alphas, label=label, linewidth=2)

    if not found:
        plt.close()
        return

    plt.xlabel("Environment Timesteps", fontsize=11)
    plt.ylabel("Temperature (α)", fontsize=11)
    plt.title("SAC Ablation Study: Learned Temperature (α) Dynamics Over Training", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"ablation_alpha_dynamics.{fig_format}")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_q_overestimation(logs: dict, plot_dir: str = DEFAULT_PLOT_DIR, fig_format: str = "png"):
    """Plot critic Q-value predictions over timesteps for Single Q vs Twin Q to analyze overestimation bias."""
    if not HAS_MATPLOTLIB:
        return

    found = False
    plt.figure(figsize=(8.5, 4.8), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for name, data in sorted(logs.items()):
        if name.startswith("use_twin_q_"):
            history = data.get("history", {})
            steps = history.get("steps", [])
            update_metrics = history.get("update_metrics", [])
            q1_means = [m.get("q1_mean") for m in update_metrics if isinstance(m, dict) and "q1_mean" in m]
            if steps and q1_means and len(steps) == len(q1_means):
                found = True
                is_twin = "true" in name.lower()
                label = "Twin Q (Clipped Double-Q)" if is_twin else "Single Q Critic"
                plt.plot(steps, q1_means, label=label, linewidth=2)

    if not found:
        plt.close()
        return

    plt.xlabel("Environment Timesteps", fontsize=11)
    plt.ylabel("Mean Predicted Q-Value", fontsize=11)
    plt.title("SAC Ablation Study: Critic Q-Value Estimation Bias (Single Q vs. Twin Q)", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"ablation_q_overestimation.{fig_format}")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_baseline_comparison(logs: dict, plot_dir: str = DEFAULT_PLOT_DIR, fig_format: str = "png"):
    """Plot Stable-Baselines3 baseline curves vs our SAC implementation."""
    if not HAS_MATPLOTLIB:
        return

    sb3_logs = {k: v for k, v in logs.items() if k.startswith("sb3_baseline_")}
    if not sb3_logs:
        return

    plt.figure(figsize=(9, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plotted = False
    for name, data in sb3_logs.items():
        ep_steps = data.get("episode_steps", [])
        ep_rewards = data.get("episode_rewards", [])
        env_id = data.get("env_id", name.replace("sb3_baseline_", ""))
        if ep_steps and ep_rewards:
            plotted = True
            # smooth with rolling window if many points
            if len(ep_rewards) > 10:
                window = max(3, len(ep_rewards) // 20)
                smooth = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
                plt.plot(ep_steps[:len(smooth)], smooth, label=f"SB3 SAC ({env_id})", linestyle="--", linewidth=2)
            else:
                plt.plot(ep_steps, ep_rewards, label=f"SB3 SAC ({env_id})", linestyle="--", linewidth=2)

    # Plot corresponding our SAC runs (e.g. halfcheetah_base, ant_base, or alpha_auto)
    for target in ["halfcheetah_base", "ant_base", "alpha_auto"]:
        if target in logs:
            h = logs[target].get("history", {})
            steps = h.get("steps", [])
            means = h.get("eval_reward_mean", [])
            if steps and means:
                plotted = True
                plt.plot(steps, means, label=f"Our SAC ({target})", linewidth=2.5)

    if not plotted:
        plt.close()
        return

    plt.xlabel("Environment Timesteps", fontsize=11)
    plt.ylabel("Return / Evaluation Reward", fontsize=11)
    plt.title("Baseline Validation: Our SAC vs. Stable-Baselines3", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"baseline_comparison.{fig_format}")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def generate_summary_table(logs: dict, output_dir: str = DEFAULT_REPORT_DIR):
    """Generates comparative summary tables across all ablation runs in Markdown, LaTeX, and CSV."""
    records = []
    for name, data in sorted(logs.items()):
        if name.startswith("sb3_baseline_"):
            continue

        cfg = data.get("config", {})
        history = data.get("history", {})
        means = history.get("eval_reward_mean", [])
        stds = history.get("eval_reward_std", [])
        costs = history.get("eval_cost_mean", [])
        steps = history.get("steps", [])

        if not means:
            continue

        env_id = cfg.get("env_id", "Unknown")
        final_mean = means[-1]
        final_std = stds[-1] if stds else 0.0
        max_mean = max(means)
        total_steps = steps[-1] if steps else 0
        final_cost = f"{costs[-1]:.1f}" if costs and any(c > 0 for c in costs) else "N/A"

        records.append({
            "Experiment": name,
            "Environment": env_id,
            "Timesteps": total_steps,
            "Final Reward": f"{final_mean:.2f} ± {final_std:.2f}",
            "Max Reward": f"{max_mean:.2f}",
            "Final Cost": final_cost,
            "_raw_final": final_mean,
            "_raw_max": max_mean,
        })

    if not records:
        print("No valid evaluation runs to summarize.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Console Output
    col_exp = max(len(r["Experiment"]) for r in records)
    col_env = max(len(r["Environment"]) for r in records)
    print("\n" + "=" * (col_exp + col_env + 48))
    print("SAC EXPERIMENT ABLATION SUMMARY")
    print("=" * (col_exp + col_env + 48))
    header = f"{'Experiment':<{col_exp}} | {'Env':<{col_env}} | {'Steps':<8} | {'Final Return':<18} | {'Max Return':<10} | {'Cost'}"
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['Experiment']:<{col_exp}} | {r['Environment']:<{col_env}} | {r['Timesteps']:<8} | {r['Final Reward']:<18} | {r['Max Reward']:<10} | {r['Final Cost']}")
    print("=" * len(header) + "\n")

    # 2. Markdown File
    md_path = os.path.join(output_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# SAC Ablation Study Summary Table\n\n")
        f.write(f"| Experiment | Environment | Steps | Final Return | Max Return | Safety Cost |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in records:
            f.write(f"| `{r['Experiment']}` | {r['Environment']} | {r['Timesteps']:,} | {r['Final Reward']} | {r['Max Reward']} | {r['Final Cost']} |\n")
    print(f"Saved Markdown summary: {md_path}")

    # 3. LaTeX File (booktabs format for progress report / paper)
    tex_path = os.path.join(output_dir, "ablation_summary.tex")
    with open(tex_path, "w") as f:
        f.write("% SAC Ablation Study Summary Table\n")
        f.write("\\begin{table}[htbp]\n\\centering\\small\n")
        f.write("\\begin{tabular}{l l r c c c}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Experiment} & \\textbf{Environment} & \\textbf{Steps} & \\textbf{Final Return} & \\textbf{Max Return} & \\textbf{Cost} \\\\\n")
        f.write("\\midrule\n")
        for r in records:
            exp_esc = r["Experiment"].replace("_", "\\_")
            f.write(f"{exp_esc} & {r['Environment']} & {r['Timesteps']:,} & {r['Final Reward']} & {r['Max Reward']} & {r['Final Cost']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary of SAC baseline and ablation experiment evaluation returns.}\\label{tab:sac_ablation_summary}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX summary: {tex_path}")

    # 4. CSV File
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Experiment,Environment,Timesteps,FinalReward,MaxReward,FinalCost\n")
        for r in records:
            f.write(f"\"{r['Experiment']}\",\"{r['Environment']}\",{r['Timesteps']},\"{r['Final Reward']}\",\"{r['Max Reward']}\",\"{r['Final Cost']}\"\n")
    print(f"Saved CSV summary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="SAC Result Plotting and Analysis")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory containing JSON logs")
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR, help="Directory to save generated plots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Plot image format")
    parser.add_argument("--no-tables", action="store_true", help="Skip generating summary tables")
    args = parser.parse_args()

    logs = load_all_logs(args.log_dir)
    print(f"Loaded {len(logs)} log files from {args.log_dir}")

    # Baseline comparison (SB3 vs SAC)
    plot_baseline_comparison(logs, plot_dir=args.plot_dir, fig_format=args.format)

    # 1. Alpha sweep (Temperature Tuning)
    plot_ablation_group(
        logs,
        filter_prefix="alpha_",
        title="SAC Ablation Study: Temperature (α) Tuning",
        output_name="ablation_alpha_sweep.png",
        label_fn=lambda name, data: "Auto-tuned α" if "auto" in name.lower() else f"Fixed α = {name.replace('alpha_', '')}",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )
    plot_alpha_trajectory(logs, plot_dir=args.plot_dir, fig_format=args.format)

    # 2. Twin vs Single Q Critic
    plot_ablation_group(
        logs,
        filter_prefix="use_twin_q_",
        title="SAC Ablation Study: Twin Q vs. Single Q Critic",
        output_name="ablation_twin_vs_single_q.png",
        label_fn=lambda name, data: "Twin Q (Clipped Double-Q)" if "true" in name.lower() else "Single Q Critic",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )
    plot_q_overestimation(logs, plot_dir=args.plot_dir, fig_format=args.format)

    # 3. Stochastic vs Deterministic Evaluation
    plot_ablation_group(
        logs,
        filter_prefix="deterministic_eval_",
        title="SAC Ablation Study: Stochastic vs. Deterministic Evaluation",
        output_name="ablation_stochastic_vs_det.png",
        label_fn=lambda name, data: "Deterministic Policy (Mean)" if "true" in name.lower() else "Stochastic Policy (Sampled)",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )

    # 4. Reward Scaling
    plot_ablation_group(
        logs,
        filter_prefix="reward_scale_",
        title="SAC Ablation Study: Reward Scaling Sensitivity (Fixed α = 0.2)",
        output_name="ablation_reward_scaling.png",
        label_fn=lambda name, data: f"Reward Scale = {name.replace('reward_scale_', '')}",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )

    # 5. Target Entropy Heuristic
    plot_ablation_group(
        logs,
        filter_prefix="target_entropy_",
        title="SAC Ablation Study: Target Entropy Heuristic Scaling",
        output_name="ablation_target_entropy.png",
        label_fn=lambda name, data: f"H_target = {name.replace('target_entropy_', '')}",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )

    # 6. Safety Cost Penalty (CMDP)
    plot_ablation_group(
        logs,
        filter_prefix="cost_penalty_",
        title="SAC Ablation Study: Safety Cost Penalty (SafetyAntVelocity-v1)",
        output_name="ablation_safety_cost_penalty.png",
        label_fn=lambda name, data: f"Cost Penalty λ = {name.replace('cost_penalty_', '')}",
        plot_dir=args.plot_dir,
        fig_format=args.format,
    )

    if not args.no_tables:
        generate_summary_table(logs, output_dir=os.path.dirname(args.plot_dir))


if __name__ == "__main__":
    main()
