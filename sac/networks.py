"""
Network definitions for SAC.

- GaussianPolicy: the actor. Outputs mean/log_std of a Gaussian over actions,
  squashed through tanh to respect action bounds (Haarnoja et al. 2018, Appendix C).
- QNetwork: a single critic, Q(s, a) -> scalar. Instantiate two independently
  (twin_q1, twin_q2) in the agent for the double-Q trick.
"""

import torch
import torch.nn as nn

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6  # numerical stability for the tanh log-prob correction


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        """Return (mean, log_std) for the pre-squash Gaussian."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """
        Reparameterized sample: apply tanh squashing, return
        (action, log_prob, mean_action) per Appendix C of the SAC paper
        (log_prob needs the tanh Jacobian correction term).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()          # reparameterization trick: mean + std * N(0,1)
        y_t = torch.tanh(x_t)           # squash to (-1, 1)
        action = y_t

        # log_prob of the pre-squash Gaussian sample...
        log_prob = normal.log_prob(x_t)
        # ...minus the tanh Jacobian correction (Appendix C, eq. 21):
        # log_prob -= sum(log(1 - tanh(x_t)^2))
        log_prob -= torch.log(torch.clamp(1 - y_t.pow(2), min=EPS))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)  # deterministic action (ablation axis 2: stochastic vs. deterministic)

        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
