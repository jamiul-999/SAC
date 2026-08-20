"""
Run with: pytest tests/ or python tests/test_agent_update.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sac.agent import SACAgent
from sac.buffer import ReplayBuffer

OBS_DIM, ACT_DIM, BATCH = 17, 6, 32


def _make_agent(**kwargs):
    defaults = dict(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=32)
    defaults.update(kwargs)
    return SACAgent(**defaults)


def _fill_buffer(n=100):
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=200)
    for _ in range(n):
        buf.add(
            np.random.randn(OBS_DIM), np.random.uniform(-1, 1, ACT_DIM),
            np.random.randn(), np.random.randn(OBS_DIM), False,
        )
    return buf


def test_select_action_shape_and_bounds():
    agent = _make_agent()
    obs = np.random.randn(OBS_DIM)
    action = agent.select_action(obs)
    assert action.shape == (ACT_DIM,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_deterministic_action_is_reproducible():
    agent = _make_agent()
    obs = np.random.randn(OBS_DIM)
    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2), "Deterministic action should be identical across calls"


def test_update_runs_and_returns_finite_losses_twin_q():
    agent = _make_agent(use_twin_q=True, auto_tune_alpha=True)
    buf = _fill_buffer()
    batch = buf.sample(BATCH)
    metrics = agent.update(batch)
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_update_runs_with_single_q_no_autotune():
    agent = _make_agent(use_twin_q=False, auto_tune_alpha=False, alpha=0.2)
    buf = _fill_buffer()
    batch = buf.sample(BATCH)
    metrics = agent.update(batch)
    assert metrics["q2_loss"] == 0.0   # no second critic in this config
    assert np.isfinite(metrics["q1_loss"])


def test_weights_change_after_update():
    agent = _make_agent()
    before = agent.q1.net[0].weight.clone()
    buf = _fill_buffer()
    for _ in range(3):
        agent.update(buf.sample(BATCH))
    after = agent.q1.net[0].weight
    assert not torch.allclose(before, after), "Q1 weights should change after gradient updates"


def test_save_and_load_roundtrip(tmp_path=None):
    agent = _make_agent()
    if tmp_path is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "ckpt.pt")
    else:
        path = str(tmp_path / "ckpt.pt")
    
    agent.save(path)

    agent2 = _make_agent()
    agent2.load(path)

    obs = np.random.randn(OBS_DIM)
    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent2.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2), "Loaded agent should produce identical deterministic actions"


if __name__ == "__main__":
    test_select_action_shape_and_bounds()
    test_deterministic_action_is_reproducible()
    test_update_runs_and_returns_finite_losses_twin_q()
    test_update_runs_with_single_q_no_autotune()
    test_weights_change_after_update()
    test_save_and_load_roundtrip()
    print("All agent update tests passed!")
