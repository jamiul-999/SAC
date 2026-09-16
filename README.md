# RL Course Project — SAC Reproduction & Entropy-Regularization Ablation Study

> **For complete architectural documentation, mathematical formulation, feature directory, and an end-to-end Kaggle GPU walkthrough, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).**

## What's in this repo

```
sac/          -> core SAC algorithm (networks, agent update logic, replay buffer)
envs/         -> environment factory + safety-gymnasium wrappers
configs/      -> one config file per experiment (base runs + ablation sweeps)
scripts/      -> runnable entry points (train, validate, run ablation, evaluate)
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
