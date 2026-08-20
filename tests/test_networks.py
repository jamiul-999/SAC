"""
Run with: pytest tests/ or python tests/test_networks.py

These are shape/sanity tests, not correctness proofs -- but they catch the
most common silent bugs (wrong dims, actions outside [-1, 1], NaNs) before
you burn a full training run on broken code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sac.networks import GaussianPolicy, QNetwork


OBS_DIM, ACT_DIM, BATCH = 17, 6, 32   # HalfCheetah-v5 dims, adjust per env under test


def test_policy_output_shapes():
    policy = GaussianPolicy(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    action, log_prob, mean = policy.sample(obs)
    assert action.shape == (BATCH, ACT_DIM)
    assert log_prob.shape == (BATCH, 1)


def test_policy_action_bounds():
    policy = GaussianPolicy(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    action, _, _ = policy.sample(obs)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0), \
        "Actions must be tanh-squashed into [-1, 1]"


def test_q_network_output_shape():
    q_net = QNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    act = torch.randn(BATCH, ACT_DIM)
    q_val = q_net(obs, act)
    assert q_val.shape == (BATCH, 1)


def test_no_nans_in_forward_pass():
    policy = GaussianPolicy(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    action, log_prob, _ = policy.sample(obs)
    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()


if __name__ == "__main__":
    test_policy_output_shapes()
    test_policy_action_bounds()
    test_q_network_output_shape()
    test_no_nans_in_forward_pass()
    print("All network tests passed!")
