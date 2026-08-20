# Notebooks

Colab/Kaggle notebooks for exploration and running training on GPU. These are
**not** the source of truth for algorithm logic -- that lives in `sac/`, `envs/`,
`scripts/`. Notebooks should `!git clone` the repo and call into `scripts/*.py`,
not reimplement logic inline.

Planned:
- `01_baseline_sb3.ipynb` -- run scripts/validate_against_sb3.py on Kaggle GPU with a real step budget
- `02_sac_validation.ipynb` -- run scripts/train.py, compare against the SB3 baseline
- `03_ablation_analysis.ipynb` -- load results from all ablation runs, produce final plots
