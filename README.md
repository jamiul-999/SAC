# RL Course Project — SAC Reproduction & Entropy-Regularization Ablation Study

## What's in this repo

```
sac/          -> core SAC algorithm (networks, agent update logic, replay buffer)
envs/         -> environment factory + safety-gymnasium wrappers
configs/      -> one config file per experiment (base runs + ablation sweeps)
scripts/      -> runnable entry points (train, validate, run ablation, evaluate)
tests/        -> unit tests for sac/ (run before trusting a training run)
notebooks/    -> Colab/Kaggle exploration notebooks (not source of truth)
results/      -> logs, checkpoints, plots (gitignore large files if needed)
docs/         -> written deliverables (MDP specs, progress report, etc.)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Status

- [x] Environments verified: HalfCheetah-v5, Ant-v5 load correctly, dimensions match `docs/progress_deliverable.tex`.
- [x] Preliminary SB3 SAC baseline run on HalfCheetah-v5 and Ant-v5 (see `scripts/validate_against_sb3.py` and `results/plots/sb3_baseline_curves.png`).
- [x] `sac/networks.py` (GaussianPolicy, QNetwork) — implemented and tested.
- [x] `sac/buffer.py` (ReplayBuffer) — implemented and tested.
- [x] `sac/agent.py` (SACAgent) — implemented and tested: twin/single Q, auto-tuned/fixed alpha, configurable target entropy, Q-value diagnostics, save/load checkpoints, all tests pass.
- [x] `scripts/train.py` — full training loop, GPU-automatic, verified end-to-end (supports standard and CMDP safety tasks).
- [x] `scripts/evaluate.py` — single and batch checkpoint evaluation, head-to-head comparison of deterministic vs. stochastic action selection.
- [x] `scripts/run_ablation.py` — complete ablation runner supporting single configs, `--all`, multi-seed execution (`--seeds 0 1 2`), resume/skip completed runs (`--skip-existing`), CLI overrides (`--env-id`), and `--dry-run`.
- [x] Ablation Configurations (`configs/ablation/`):
  1. `alpha_sweep.yaml`: Fixed temperatures vs. Auto-tuned alpha.
  2. `twin_vs_single_q.yaml`: Twin Q (clipped double-Q) vs. Single Q critic.
  3. `stochastic_vs_det.yaml`: Stochastic sampled policy vs. Deterministic mean policy.
  4. `reward_scaling.yaml`: Reward scale sensitivity under fixed alpha.
  5. `target_entropy_sweep.yaml`: Target entropy heuristic scalings.
  6. `safety_cost_penalty.yaml`: Cost penalty sweep for Constrained MDP on SafetyAntVelocity-v1.
- [x] Plotting & Analysis — `scripts/plot_results.py` and `notebooks/03_ablation_analysis.ipynb` generate curves (returns, alpha dynamics, Q-overestimation bias, baselines) and export comparative tables (Markdown, LaTeX booktabs, CSV).
- [x] Automated Tests — `tests/test_ablation.py` validates configs, sweep planning, seed handling, skip-logic, aggregation, and table generation.

## Ablation Study Quickstart

```bash
# Dry-run all ablations to inspect scheduled runs
python scripts/run_ablation.py --all --dry-run

# Run all ablations with smoke test budget (e.g. on CPU or local dev)
python scripts/run_ablation.py --all --total-timesteps-override 2000

# Run a specific ablation across 3 seeds on GPU
python scripts/run_ablation.py --ablation configs/ablation/twin_vs_single_q.yaml --seeds 0 1 2

# Resume an interrupted sweep without repeating completed runs
python scripts/run_ablation.py --all --skip-existing

# Generate comparison plots and summary tables (Markdown, LaTeX, CSV)
python scripts/plot_results.py

# Evaluate deterministic vs. stochastic policy on a trained checkpoint
python scripts/evaluate.py --checkpoint results/checkpoints/halfcheetah_base.pt --config configs/halfcheetah_base.yaml --compare-modes
```

## Running on Kaggle

See `notebooks/kaggle_cells.txt` for ready-to-paste notebook cells (clone repo, install deps, smoke test, full training run, ablation sweep, pull results back down).

## How to run the SB3 baseline

```bash
python scripts/validate_against_sb3.py HalfCheetah-v5 15000
python scripts/validate_against_sb3.py Ant-v5 15000
```
Saves JSON of per-episode rewards to `results/logs/`. Increase step count (e.g. 100000+) when running on Kaggle GPU for complete curves.

## Next steps

1. Run full (non-smoke-test) training on Kaggle GPU for HalfCheetah-v5, Walker2d-v5, Ant-v5 using the base configs.
2. Run the ablation sweeps (`configs/ablation/*.yaml`) via `python scripts/run_ablation.py --all`.
3. Run `python scripts/plot_results.py` or open `notebooks/03_ablation_analysis.ipynb` to generate final comparison plots in `results/plots/` and `results/ablation_summary.tex` for the report.
