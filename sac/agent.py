"""
SAC agent: ties together GaussianPolicy + twin QNetworks + target networks
+ (optionally) automatic temperature tuning.

Update order per training step (Haarnoja et al. 2018, Algorithm 1):
1. Sample a batch from the replay buffer.
2. Update both Q-networks (critic loss uses the *target* Q-networks and the
   *current* policy's resampled next-action, with the entropy term subtracted).
3. Update the policy (actor loss = entropy-regularized expected Q, using the
   reparameterization trick so gradients flow through the sampled action).
4. (If auto-tuning) update log_alpha via its own loss.
5. Soft-update target Q-networks: target = tau * current + (1 - tau) * target.
"""

import copy

import torch
import torch.nn.functional as F

from sac.networks import GaussianPolicy, QNetwork


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        use_twin_q: bool = True,   # ablation axis: twin vs. single Q-network
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.use_twin_q = use_twin_q
        self.auto_tune_alpha = auto_tune_alpha
        self.device = device
        self.act_dim = act_dim

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.target_q1 = copy.deepcopy(self.q1)
        for p in self.target_q1.parameters():
            p.requires_grad = False

        if use_twin_q:
            self.q2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)
            self.target_q2 = copy.deepcopy(self.q2)
            for p in self.target_q2.parameters():
                p.requires_grad = False
        else:
            self.q2 = None

        if auto_tune_alpha:
            # Target entropy heuristic from the SAC follow-up paper (Haarnoja et al. 2018b):
            # -|A|, i.e. negative of the action dimensionality.
            self.target_entropy = -float(act_dim)
            init_log_alpha = float(np.log(max(alpha, 1e-6)))
            self.log_alpha = torch.tensor([init_log_alpha], requires_grad=True, device=device, dtype=torch.float32)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_actor)
            self._alpha = self.log_alpha.exp().item()
        else:
            self._alpha = alpha

    @property
    def alpha(self) -> float:
        if self.auto_tune_alpha:
            return self.log_alpha.exp().item()
        return self._alpha

    def select_action(self, obs, deterministic: bool = False):
        """
        deterministic=False -> stochastic sample from the policy (training/eval default)
        deterministic=True  -> use the policy mean (ablation axis: stochastic vs. deterministic)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.policy.sample(obs_t)
        chosen = mean_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    def _min_target_q(self, obs, action):
        q1_val = self.target_q1(obs, action)
        if self.use_twin_q:
            q2_val = self.target_q2(obs, action)
            return torch.min(q1_val, q2_val)
        return q1_val

    def _min_current_q(self, obs, action):
        q1_val = self.q1(obs, action)
        if self.use_twin_q:
            q2_val = self.q2(obs, action)
            return torch.min(q1_val, q2_val)
        return q1_val

    def update(self, batch: dict) -> dict:
        """
        Run one gradient step on a sampled batch.
        Returns a dict of scalar losses/metrics for logging.
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        alpha = self.alpha

        # --- 1. Critic update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            target_q_min = self._min_target_q(next_obs, next_action)
            target_value = target_q_min - alpha * next_log_prob
            q_target = reward + self.gamma * (1.0 - done) * target_value

        q1_pred = self.q1(obs, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        q2_loss_val = 0.0
        if self.use_twin_q:
            q2_pred = self.q2(obs, action)
            q2_loss = F.mse_loss(q2_pred, q_target)
            self.q2_optim.zero_grad()
            q2_loss.backward()
            self.q2_optim.step()
            q2_loss_val = q2_loss.item()

        # --- 2. Policy update (reparameterized, gradients flow through sampled action) ---
        new_action, log_prob, _ = self.policy.sample(obs)
        q_new_min = self._min_current_q(obs, new_action)
        policy_loss = (alpha * log_prob - q_new_min).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- 3. Temperature update (if auto-tuning) ---
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_loss_val = alpha_loss.item()

        # --- 4. Soft-update target networks ---
        self._soft_update_targets()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss_val,
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
        }

    def _soft_update_targets(self):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        if self.use_twin_q:
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        ckpt = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "use_twin_q": self.use_twin_q,
            "auto_tune_alpha": self.auto_tune_alpha,
            "alpha": self.alpha,
        }
        if self.use_twin_q:
            ckpt["q2"] = self.q2.state_dict()
            ckpt["target_q2"] = self.target_q2.state_dict()
        if self.auto_tune_alpha:
            ckpt["log_alpha"] = self.log_alpha.detach().cpu()
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.target_q1.load_state_dict(ckpt["target_q1"])
        if self.use_twin_q and "q2" in ckpt:
            self.q2.load_state_dict(ckpt["q2"])
            self.target_q2.load_state_dict(ckpt["target_q2"])
        if self.auto_tune_alpha and "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        elif "alpha" in ckpt:
            self._alpha = ckpt["alpha"]
