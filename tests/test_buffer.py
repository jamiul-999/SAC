"""
Run with: pytest tests/ or python tests/test_buffer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sac.buffer import ReplayBuffer

OBS_DIM, ACT_DIM = 17, 6


def test_buffer_add_and_len():
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=100)
    obs = np.random.randn(OBS_DIM)
    next_obs = np.random.randn(OBS_DIM)
    act = np.random.randn(ACT_DIM)
    buf.add(obs, act, 1.0, next_obs, False)
    assert len(buf) == 1


def test_buffer_sample_shapes():
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=100)
    for _ in range(50):
        buf.add(
            np.random.randn(OBS_DIM), np.random.randn(ACT_DIM),
            np.random.rand(), np.random.randn(OBS_DIM), False,
        )
    batch = buf.sample(16)
    assert batch["obs"].shape == (16, OBS_DIM)
    assert batch["action"].shape == (16, ACT_DIM)


def test_buffer_respects_capacity():
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=10)
    for _ in range(25):
        buf.add(
            np.random.randn(OBS_DIM), np.random.randn(ACT_DIM),
            np.random.rand(), np.random.randn(OBS_DIM), False,
        )
    assert len(buf) == 10   # should not exceed capacity, oldest overwritten


if __name__ == "__main__":
    test_buffer_add_and_len()
    test_buffer_sample_shapes()
    test_buffer_respects_capacity()
    print("All buffer tests passed!")
