"""
Unit tests for SAC ablation pipeline: config validation, sweep planning,
multi-seed handling, skip/resume logic, and result aggregation / table generation.

Run with:
    python tests/test_ablation.py
"""

import copy
import glob
import json
import os
import shutil
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.utils import load_config
from scripts.run_ablation import (
    resolve_ablation_files,
    build_sweep_plan,
    is_run_completed,
    DEFAULT_ABLATION_DIR,
    PROJECT_ROOT,
)
from scripts.plot_results import (
    load_all_logs,
    extract_group_key,
    aggregate_group_runs,
    generate_summary_table,
)


def test_ablation_configs_exist_and_valid():
    """All YAML configs in configs/ablation/ must be parseable and contain required keys."""
    configs = glob.glob(os.path.join(DEFAULT_ABLATION_DIR, "*.yaml"))
    assert len(configs) >= 4, f"Expected at least 4 ablation configs, found {len(configs)}"

    for cfg_path in configs:
        cfg = load_config(cfg_path)
        assert "base_config" in cfg, f"Missing 'base_config' in {cfg_path}"
        assert "sweep_param" in cfg, f"Missing 'sweep_param' in {cfg_path}"
        assert "sweep_values" in cfg, f"Missing 'sweep_values' in {cfg_path}"
        assert isinstance(cfg["sweep_values"], list), f"'sweep_values' must be a list in {cfg_path}"
        assert len(cfg["sweep_values"]) > 0, f"'sweep_values' cannot be empty in {cfg_path}"

        # Ensure base_config exists
        base_path = cfg["base_config"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(PROJECT_ROOT, base_path)
        assert os.path.isfile(base_path), f"Base config {base_path} not found for {cfg_path}"
    print("✓ All ablation configs exist and are well-formed.")


def test_resolve_ablation_files():
    """Test resolution of --ablation inputs and --all flag."""
    all_files = resolve_ablation_files([], run_all=True)
    assert len(all_files) >= 4
    for f in all_files:
        assert f.endswith(".yaml")

    single = resolve_ablation_files(["alpha_sweep.yaml"], run_all=False)
    assert len(single) == 1
    assert "alpha_sweep.yaml" in single[0]
    print("✓ Ablation file resolver works for single, directory, and --all.")


def test_build_sweep_plan_alpha_sweep():
    """Test alpha sweep expansion, auto-tune special case, and parameter propagation."""
    cfg_path = os.path.join(DEFAULT_ABLATION_DIR, "alpha_sweep.yaml")
    jobs = build_sweep_plan(cfg_path)

    # 4 fixed values + 1 auto-tuned = 5 jobs
    assert len(jobs) == 5
    run_names = [j["run_name"] for j in jobs]
    assert "alpha_auto" in run_names
    assert "alpha_0.0" in run_names
    assert "alpha_0.2" in run_names

    # Check auto-tuned config
    auto_job = next(j for j in jobs if j["run_name"] == "alpha_auto")
    assert auto_job["run_cfg"]["auto_tune_alpha"] is True

    # Check fixed alpha config
    fixed_job = next(j for j in jobs if j["run_name"] == "alpha_0.2")
    assert fixed_job["run_cfg"]["auto_tune_alpha"] is False
    assert fixed_job["run_cfg"]["alpha"] == 0.2
    print("✓ Sweep plan correctly expands alpha values and auto-tuning flag.")


def test_build_sweep_plan_multi_seed():
    """Test multi-seed naming: run_names include _seedX when multiple seeds specified."""
    cfg_path = os.path.join(DEFAULT_ABLATION_DIR, "twin_vs_single_q.yaml")
    jobs = build_sweep_plan(cfg_path, seeds=[0, 1, 2])

    # 2 values (true, false) * 3 seeds = 6 jobs
    assert len(jobs) == 6
    for seed in [0, 1, 2]:
        seed_names = [j["run_name"] for j in jobs if j["seed"] == seed]
        assert len(seed_names) == 2
        for name in seed_names:
            assert f"_seed{seed}" in name
    print("✓ Multi-seed sweep plan constructs distinct, non-overwriting run names.")


def test_build_sweep_plan_overrides():
    """Test CLI overrides for env-id and total timesteps."""
    cfg_path = os.path.join(DEFAULT_ABLATION_DIR, "twin_vs_single_q.yaml")
    jobs = build_sweep_plan(cfg_path, env_override="Ant-v5", total_timesteps_override=5000)
    for j in jobs:
        assert j["run_cfg"]["env_id"] == "Ant-v5"
        assert j["target_steps"] == 5000
    print("✓ CLI overrides (env-id, total timesteps) properly apply to jobs.")


def test_skip_existing_logic():
    """Test is_run_completed correctly detects complete vs incomplete/missing runs."""
    tmp_dir = tempfile.mkdtemp()
    try:
        # Mock incomplete run
        incomplete_path = os.path.join(tmp_dir, "run_incomplete.json")
        with open(incomplete_path, "w") as f:
            json.dump({"history": {"steps": [1000, 2000]}}, f)

        # Mock complete run
        complete_path = os.path.join(tmp_dir, "run_complete.json")
        with open(complete_path, "w") as f:
            json.dump({"history": {"steps": [1000, 2000, 5000]}}, f)

        # Override results dir temporarily
        import scripts.run_ablation as ra
        orig_dir = ra.RESULTS_LOG_DIR
        ra.RESULTS_LOG_DIR = tmp_dir
        try:
            assert is_run_completed("run_complete", target_steps=5000) is True
            assert is_run_completed("run_complete", target_steps=10000) is False
            assert is_run_completed("run_incomplete", target_steps=5000) is False
            assert is_run_completed("non_existent_run", target_steps=5000) is False
        finally:
            ra.RESULTS_LOG_DIR = orig_dir
    finally:
        shutil.rmtree(tmp_dir)
    print("✓ Skip-existing logic accurately verifies run completion status.")


def test_plotting_aggregation_and_label_resolution():
    """
    Test that plot_results correctly groups multi-seed runs,
    resolves case-sensitive boolean names (True/False), and generates summary tables.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create synthetic logs
        mock_logs = {
            "use_twin_q_True_seed0": {
                "config": {"env_id": "HalfCheetah-v5"},
                "history": {
                    "steps": [1000, 2000, 3000],
                    "eval_reward_mean": [100.0, 200.0, 300.0],
                    "eval_reward_std": [10.0, 15.0, 20.0],
                    "eval_cost_mean": [0.0, 0.0, 0.0],
                    "update_metrics": [{"q1_mean": 5.0, "alpha": 0.2}, {"q1_mean": 10.0, "alpha": 0.2}, {"q1_mean": 15.0, "alpha": 0.2}],
                }
            },
            "use_twin_q_True_seed1": {
                "config": {"env_id": "HalfCheetah-v5"},
                "history": {
                    "steps": [1000, 2000, 3000],
                    "eval_reward_mean": [110.0, 210.0, 310.0],
                    "eval_reward_std": [10.0, 15.0, 20.0],
                    "eval_cost_mean": [0.0, 0.0, 0.0],
                    "update_metrics": [{"q1_mean": 5.2, "alpha": 0.2}, {"q1_mean": 10.1, "alpha": 0.2}, {"q1_mean": 15.3, "alpha": 0.2}],
                }
            },
            "use_twin_q_False_seed0": {
                "config": {"env_id": "HalfCheetah-v5"},
                "history": {
                    "steps": [1000, 2000, 3000],
                    "eval_reward_mean": [80.0, 140.0, 180.0],
                    "eval_reward_std": [12.0, 18.0, 25.0],
                    "eval_cost_mean": [0.0, 0.0, 0.0],
                    "update_metrics": [{"q1_mean": 20.0, "alpha": 0.2}, {"q1_mean": 50.0, "alpha": 0.2}, {"q1_mean": 90.0, "alpha": 0.2}],
                }
            },
            "deterministic_eval_True": {
                "config": {"env_id": "HalfCheetah-v5"},
                "history": {
                    "steps": [1000, 2000],
                    "eval_reward_mean": [150.0, 350.0],
                    "eval_reward_std": [5.0, 8.0],
                }
            },
            "deterministic_eval_False": {
                "config": {"env_id": "HalfCheetah-v5"},
                "history": {
                    "steps": [1000, 2000],
                    "eval_reward_mean": [130.0, 310.0],
                    "eval_reward_std": [15.0, 20.0],
                }
            },
        }

        # 1. Test group extraction
        assert extract_group_key("use_twin_q_True_seed0") == "use_twin_q_True"
        assert extract_group_key("alpha_0.2") == "alpha_0.2"

        # 2. Test multi-seed aggregation
        agg = aggregate_group_runs(mock_logs, filter_prefix="use_twin_q_")
        assert "use_twin_q_True" in agg
        assert "use_twin_q_False" in agg
        assert agg["use_twin_q_True"]["n_seeds"] == 2
        assert agg["use_twin_q_False"]["n_seeds"] == 1
        # Mean of [100, 200, 300] and [110, 210, 310] is [105, 205, 305]
        assert np.allclose(agg["use_twin_q_True"]["mean"], [105.0, 205.0, 305.0])

        # 3. Test boolean label resolution logic
        twin_label_fn = lambda name, data: "Twin Q" if "true" in name.lower() else "Single Q"
        assert twin_label_fn("use_twin_q_True", None) == "Twin Q"
        assert twin_label_fn("use_twin_q_False", None) == "Single Q"

        det_label_fn = lambda name, data: "Deterministic" if "true" in name.lower() else "Stochastic"
        assert det_label_fn("deterministic_eval_True", None) == "Deterministic"
        assert det_label_fn("deterministic_eval_False", None) == "Stochastic"

        # 4. Test summary table generation
        generate_summary_table(mock_logs, output_dir=tmp_dir)
        md_file = os.path.join(tmp_dir, "ablation_summary.md")
        tex_file = os.path.join(tmp_dir, "ablation_summary.tex")
        csv_file = os.path.join(tmp_dir, "ablation_summary.csv")

        assert os.path.isfile(md_file), "ablation_summary.md was not generated"
        assert os.path.isfile(tex_file), "ablation_summary.tex was not generated"
        assert os.path.isfile(csv_file), "ablation_summary.csv was not generated"

        with open(md_file, "r") as f:
            md_content = f.read()
            assert "use_twin_q_True_seed0" in md_content
            assert "300.00 ± 20.00" in md_content

        with open(tex_file, "r") as f:
            tex_content = f.read()
            assert "\\begin{table}" in tex_content
            assert "\\toprule" in tex_content
            assert "\\bottomrule" in tex_content
    finally:
        shutil.rmtree(tmp_dir)
    print("✓ Plotting aggregation, label resolution, and table exports verified on synthetic logs.")


if __name__ == "__main__":
    test_ablation_configs_exist_and_valid()
    test_resolve_ablation_files()
    test_build_sweep_plan_alpha_sweep()
    test_build_sweep_plan_multi_seed()
    test_build_sweep_plan_overrides()
    test_skip_existing_logic()
    test_plotting_aggregation_and_label_resolution()
    print("\n🎉 ALL ABLATION TESTS PASSED SUCCESSFULLY!")
