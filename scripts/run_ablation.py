"""
Run an ablation sweep: reads an ablation config (e.g. configs/ablation/alpha_sweep.yaml),
launches one full training run per sweep value, calling train.run_training() directly
(no subprocess overhead across many sweep runs).

Usage:
    python scripts/run_ablation.py --ablation configs/ablation/alpha_sweep.yaml
    python scripts/run_ablation.py --ablation configs/ablation/alpha_sweep.yaml --total-timesteps-override 2000  # smoke test
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.utils import load_config
from scripts.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--total-timesteps-override", type=int, default=None,
                         help="Use a small value for a smoke test, e.g. 2000")
    args = parser.parse_args()

    ablation_cfg = load_config(args.ablation)
    base_cfg = load_config(ablation_cfg["base_config"])

    for k, v in ablation_cfg.get("override", {}).items():
        base_cfg[k] = v

    sweep_param = ablation_cfg["sweep_param"]
    sweep_values = list(ablation_cfg["sweep_values"])

    if ablation_cfg.get("also_run_auto_tuned"):
        # Special-cased for the alpha sweep: run one extra config with
        # auto_tune_alpha=True instead of setting a fixed alpha value.
        sweep_values.append("auto")

    for value in sweep_values:
        run_cfg = copy.deepcopy(base_cfg)
        if value == "auto":
            run_cfg["auto_tune_alpha"] = True
            run_name = f"{sweep_param}_auto"
        else:
            run_cfg[sweep_param] = value
            run_cfg["auto_tune_alpha"] = False if sweep_param == "alpha" else run_cfg.get("auto_tune_alpha", True)
            run_name = f"{sweep_param}_{value}"

        print(f"\n=== Launching run: {run_name} ===")
        run_training(run_cfg, run_name, args.total_timesteps_override)


if __name__ == "__main__":
    main()
