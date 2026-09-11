"""Small shared utilities: reproducibility seeding and YAML config loading."""

import random
import numpy as np
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
