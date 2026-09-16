"""
Run the four planned SAC ablation axes: reads ablation configs, launches training runs
across parameter values and optional seeds, calling train.run_training() directly.

Usage:
    # Run a single ablation
    python scripts/run_ablation.py --ablation configs/ablation/alpha_sweep.yaml

    # Run all four planned ablations
    python scripts/run_ablation.py --all
    python scripts/run_ablation.py --ablation all

    # Multi-seed sweep
    python scripts/run_ablation.py --ablation configs/ablation/alpha_sweep.yaml --seeds 0 1 2

    # Fast smoke test
    python scripts/run_ablation.py --all --total-timesteps-override 2000

    # Resume interrupted run
    python scripts/run_ablation.py --all --skip-existing

    # Dry-run to inspect scheduled runs
    python scripts/run_ablation.py --all --dry-run
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.utils import load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ABLATION_DIR = os.path.join(PROJECT_ROOT, "configs", "ablation")
RESULTS_LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "results", "ablation_manifest.json")

# These are the four ablation axes specified in the SAC update's remaining-work section.
# Other exploratory configs remain available through explicit --ablation paths.
CANONICAL_ABLATIONS = (
    "alpha_sweep.yaml",
    "stochastic_vs_det.yaml",
    "twin_vs_single_q.yaml",
    "reward_scaling.yaml",
)


def resolve_ablation_files(ablation_inputs, run_all: bool = False):
    """Resolve input paths into a clean list of ablation YAML config paths."""
    if run_all or (ablation_inputs and "all" in [str(x).lower() for x in ablation_inputs]):
        files = [os.path.join(DEFAULT_ABLATION_DIR, name) for name in CANONICAL_ABLATIONS]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing canonical ablation config(s): " + ", ".join(missing)
            )
        return files

    if not ablation_inputs:
        raise ValueError("Please specify --ablation <path> or use --all to run the four planned ablations.")

    resolved = []
    for item in ablation_inputs:
        if os.path.isdir(item):
            resolved.extend(sorted(glob.glob(os.path.join(item, "*.yaml"))))
        elif os.path.isfile(item):
            resolved.append(item)
        else:
            cand = os.path.join(DEFAULT_ABLATION_DIR, item if item.endswith(".yaml") else f"{item}.yaml")
            if os.path.isfile(cand):
                resolved.append(cand)
            else:
                raise FileNotFoundError(f"Could not find ablation config: {item}")

    return sorted(list(dict.fromkeys(resolved)))


def build_sweep_plan(ablation_file: str, base_config_override: str = None,
                     env_override: str = None, seeds: list = None,
                     total_timesteps_override: int = None):
    """
    Constructs a list of runnable job specifications from an ablation configuration.
    Returns list of dicts with keys: run_name, run_cfg, sweep_param, value, seed, ablation_file.
    """
    ablation_cfg = load_config(ablation_file)
    base_cfg_path = base_config_override or ablation_cfg["base_config"]
    if not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(PROJECT_ROOT, base_cfg_path)
    base_cfg = load_config(base_cfg_path)

    for k, v in ablation_cfg.get("override", {}).items():
        base_cfg[k] = v

    if env_override:
        base_cfg["env_id"] = env_override

    sweep_param = ablation_cfg["sweep_param"]
    sweep_values = list(ablation_cfg["sweep_values"])

    if ablation_cfg.get("also_run_auto_tuned"):
        sweep_values.append("auto")

    run_seeds = seeds if seeds is not None and len(seeds) > 0 else [base_cfg.get("seed", 0)]
    multi_seed = len(run_seeds) > 1 or (seeds is not None and len(seeds) == 1 and seeds[0] != base_cfg.get("seed", 0))

    jobs = []
    for seed in run_seeds:
        for value in sweep_values:
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["seed"] = seed

            if value == "auto":
                run_cfg["auto_tune_alpha"] = True
                val_str = "auto"
            else:
                run_cfg[sweep_param] = value
                run_cfg["auto_tune_alpha"] = False if sweep_param == "alpha" else run_cfg.get("auto_tune_alpha", True)
                val_str = str(value)

            if multi_seed:
                run_name = f"{sweep_param}_{val_str}_seed{seed}"
            else:
                run_name = f"{sweep_param}_{val_str}"

            target_steps = total_timesteps_override or run_cfg.get("total_timesteps", 1_000_000)

            jobs.append({
                "run_name": run_name,
                "run_cfg": run_cfg,
                "sweep_param": sweep_param,
                "value": value,
                "seed": seed,
                "target_steps": target_steps,
                "ablation_file": os.path.relpath(ablation_file, PROJECT_ROOT),
            })

    return jobs


def is_run_completed(run_name: str, target_steps: int) -> bool:
    """Checks whether a run has already finished successfully."""
    log_path = os.path.join(RESULTS_LOG_DIR, f"{run_name}.json")
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        steps = data.get("history", {}).get("steps", [])
        return len(steps) > 0 and steps[-1] >= target_steps
    except Exception:
        return False


def load_manifest():
    if os.path.isfile(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def update_manifest(run_name: str, record: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest = load_manifest()
    manifest[run_name] = record
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SAC Ablation Experiment Runner")
    parser.add_argument("--ablation", nargs="*", default=None,
                        help="Path(s) to ablation config(s), or 'all' to run the four planned axes.")
    parser.add_argument("--all", action="store_true",
                        help="Run the four planned ablation configs in configs/ablation/.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of random seeds (e.g. --seeds 0 1 2).")
    parser.add_argument("--env-id", default=None,
                        help="Override environment ID (e.g. HalfCheetah-v5, Ant-v5).")
    parser.add_argument("--base-config", default=None,
                        help="Override base config path.")
    parser.add_argument("--total-timesteps-override", type=int, default=None,
                        help="Override total timesteps for fast smoke tests (e.g. 2000).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that have already completed target steps.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scheduled runs without executing.")
    args = parser.parse_args()

    ablation_files = resolve_ablation_files(args.ablation, run_all=args.all)

    print(f"\n=======================================================")
    print(f"SAC Ablation Runner: {len(ablation_files)} config(s) selected")
    for f in ablation_files:
        print(f"  • {os.path.relpath(f, PROJECT_ROOT)}")
    print(f"=======================================================\n")

    all_jobs = []
    for ab_file in ablation_files:
        jobs = build_sweep_plan(
            ablation_file=ab_file,
            base_config_override=args.base_config,
            env_override=args.env_id,
            seeds=args.seeds,
            total_timesteps_override=args.total_timesteps_override,
        )
        all_jobs.extend(jobs)

    print(f"Total scheduled runs: {len(all_jobs)}")

    if args.dry_run:
        print("\n[DRY RUN PLAN]")
        header = f"{'#':<3} | {'Run Name':<32} | {'Param':<18} | {'Value':<12} | {'Seed':<5} | {'Steps':<8} | {'Status'}"
        print(header)
        print("-" * len(header))
        for idx, job in enumerate(all_jobs, 1):
            already_done = is_run_completed(job["run_name"], job["target_steps"])
            status = "EXISTS (will skip)" if (already_done and args.skip_existing) else ("EXISTS (will overwrite)" if already_done else "READY")
            print(f"{idx:<3} | {job['run_name']:<32} | {job['sweep_param']:<18} | {str(job['value']):<12} | {job['seed']:<5} | {job['target_steps']:<8} | {status}")
        return

    from scripts.train import run_training

    start_total = time.time()
    for idx, job in enumerate(all_jobs, 1):
        run_name = job["run_name"]
        target_steps = job["target_steps"]
        print(f"\n[{idx}/{len(all_jobs)}] Job: {run_name} (param: {job['sweep_param']} = {job['value']}, seed: {job['seed']})")

        if args.skip_existing and is_run_completed(run_name, target_steps):
            print(f"  -> Skipping: {run_name} already completed ({target_steps} steps).")
            update_manifest(run_name, {
                "ablation_file": job["ablation_file"],
                "sweep_param": job["sweep_param"],
                "value": str(job["value"]),
                "seed": job["seed"],
                "status": "skipped_already_completed",
                "target_steps": target_steps,
            })
            continue

        job_start = time.time()
        try:
            run_training(job["run_cfg"], run_name, args.total_timesteps_override)
            duration = time.time() - job_start
            update_manifest(run_name, {
                "ablation_file": job["ablation_file"],
                "sweep_param": job["sweep_param"],
                "value": str(job["value"]),
                "seed": job["seed"],
                "status": "completed",
                "duration_seconds": round(duration, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            print(f"  -> ERROR during run {run_name}: {e}")
            update_manifest(run_name, {
                "ablation_file": job["ablation_file"],
                "sweep_param": job["sweep_param"],
                "value": str(job["value"]),
                "seed": job["seed"],
                "status": f"failed: {e}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            raise e

    print(f"\nAll ablation runs finished in {time.time() - start_total:.1f}s.")
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
