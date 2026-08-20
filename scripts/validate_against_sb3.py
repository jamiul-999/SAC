import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json
import sys
import time

env_id = sys.argv[1] if len(sys.argv) > 1 else "HalfCheetah-v5"
total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
seed = 0

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            info = self.locals.get("infos", [{}])[0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_steps.append(self.num_timesteps)
        return True

env = Monitor(gym.make(env_id))
env.reset(seed=seed)

model = SAC("MlpPolicy", env, verbose=0, seed=seed, device="cpu")

logger = RewardLogger()
start = time.time()
model.learn(total_timesteps=total_timesteps, callback=logger, log_interval=None)
elapsed = time.time() - start

result = {
    "env_id": env_id,
    "total_timesteps": total_timesteps,
    "elapsed_seconds": elapsed,
    "episode_steps": logger.episode_steps,
    "episode_rewards": logger.episode_rewards,
}

import os
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "logs")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"sb3_baseline_{env_id}.json")
with open(out_path, "w") as f:
    json.dump(result, f)

print(f"Done: {env_id} | {len(logger.episode_rewards)} episodes | {elapsed:.1f}s")
if logger.episode_rewards:
    print(f"First episode reward: {logger.episode_rewards[0]:.2f}")
    print(f"Last episode reward: {logger.episode_rewards[-1]:.2f}")
    print(f"Mean of last 5: {np.mean(logger.episode_rewards[-5:]):.2f}")
