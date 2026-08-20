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
- [x] `sac/agent.py` (SACAgent) — implemented and tested: twin/single Q, auto-tuned/fixed alpha, save/load checkpoints, all 6 tests pass.
- [x] `scripts/train.py` — full training loop, GPU-automatic, verified end-to-end (smoke-tested on real HalfCheetah-v5 runs).
- [x] `scripts/evaluate.py` — loads a checkpoint, runs eval episodes, verified against a real trained checkpoint.
- [x] `scripts/run_ablation.py` — sweep runner, verified end-to-end on both `twin_vs_single_q.yaml` and `alpha_sweep.yaml` (including the auto-tuned special case).
- [x] `safety-gymnasium` integration — `envs/safety_wrappers.py` CostWrapper and `envs/make_env.py` safety environment factory implemented.
- [x] Plotting & Analysis — `scripts/plot_results.py` and `notebooks/03_ablation_analysis.ipynb` created for visualizing all baseline and ablation experiment curves.

**All components are real, tested code — no stubs remain.** All unit tests pass (`python tests/test_agent_update.py`, `python tests/test_buffer.py`, `python tests/test_networks.py` or `pytest tests/`).

## Running on Kaggle

See `notebooks/kaggle_cells.txt` for ready-to-paste notebook cells (clone repo, install deps, smoke test, full training run, ablation sweep, pull results back down).

## How to run the SB3 baseline (already working)

```bash
python scripts/validate_against_sb3.py HalfCheetah-v5 15000
python scripts/validate_against_sb3.py Ant-v5 15000
```
Saves JSON of per-episode rewards to `results/logs/`. Increase step count (e.g. 100000+) when running on Kaggle GPU for complete curves.

## Next steps

1. Run full (non-smoke-test) training on Kaggle GPU for HalfCheetah-v5, Walker2d-v5, Ant-v5 using the base configs.
2. Run the four ablation sweeps (`configs/ablation/*.yaml`) via `scripts/run_ablation.py`.
3. Run `python scripts/plot_results.py` or open `notebooks/03_ablation_analysis.ipynb` to generate final comparison plots in `results/plots/`.
