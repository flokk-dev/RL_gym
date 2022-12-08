"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import os
import json

import datetime

# IMPORT: deep learning
import torch

# IMPORT: reinforcement learning
import gym
from gym.utils.env_checker import check_env

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

# IMPORT: project
import paths


class Trainer:
    _MODELS = {"DQN": DQN, "A2C": A2C}
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name, game, weights_path=None):
        # Save paths
        creation_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._save_paths = {
            "model_path": os.path.join(paths.MODELS_PATH, creation_time),
            "checks_path": os.path.join(paths.MODELS_PATH, creation_time, "checkpoints"),
        }

        for key, path in self._save_paths.items():
            if not os.path.exists(path):
                os.makedirs(path)

        # Environment
        self._env = gym.make(game)
        check_env(self._env)

        # Model
        if weights_path:
            self._model = self._MODELS[model_name].load(weights_path, env=self._env,
                                                        device=self._DEVICE)
        else:
            with open(paths.CONFIG_PATH) as config_file:
                model_config = json.load(config_file)
            self._model = self._MODELS[model_name](**model_config[model_name], env=self._env,
                                                   device=self._DEVICE)

    def train(self, nb_iter):
        checks = CheckpointCallback(save_freq=nb_iter // 10,
                                    save_path=self._save_paths["checks_path"], name_prefix="model")

        self._model.learn(total_timesteps=nb_iter, progress_bar=True,
                          callback=checks, reset_num_timesteps=False)

        self._model.save(path=os.path.join(self._save_paths["model_path"], "model"))
