"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import os

from tqdm import tqdm

# IMPORT: data processing/visualization
import numpy as np


def inference(self, nb_episodes=10):
    print()

    rewards = list()
    mean_reward = list()

    cpt = 0
    for i in tqdm(range(nb_episodes), desc="Inférence en cours, veuillez patienter..."):
        self._env.reset()
        done = False

        while not done:
            obs, reward, done, info = self._env.step(self._env.action_space.sample())

            rewards.append(reward)
            if cpt % 1000 == 0:
                mean_reward.append(np.mean(rewards))

            cpt += 1

    self._env.close()

    # SAVE RESULTS
    self._save_results(file_name=os.path.join(self._results_path, "rewards.png"),
                       values=rewards,
                       title="rewards", x_label="iterations", y_label="rewards")

    self._save_results(file_name=os.path.join(self._results_path, "mean_reward.png"),
                       values=mean_reward,
                       title="mean reward", x_label="iterations", y_label="mean reward")
