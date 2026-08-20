"""
Fixed-size replay buffer for off-policy training.
Stores (obs, action, reward, next_obs, done) transitions, samples uniformly at random.

Preallocated numpy arrays are used instead of a Python list/deque -- at ~1M
transitions, list-of-tuples overhead (object headers, per-item allocation)
becomes a real bottleneck. This buffer allocates once at capacity and writes
in place, wrapping around (a ring buffer) once full.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity   # wrap around once full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu"):
        """Return a dict of torch tensors: obs, action, reward, next_obs, done."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buf[idxs], device=device),
            "action": torch.as_tensor(self.action_buf[idxs], device=device),
            "reward": torch.as_tensor(self.reward_buf[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs], device=device),
            "done": torch.as_tensor(self.done_buf[idxs], device=device),
        }

    def __len__(self):
        return self.size
