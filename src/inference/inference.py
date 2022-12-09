"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import os
from tqdm import tqdm

# IMPORT: image processing
import cv2

# IMPORT: deep learning
import torch

# IMPORT: reinforcement learning
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

# IMPORT: project
import paths
from utils import get_game_id


class Inferencer:
    _MODELS = {"DQN": DQN, "A2C": A2C}
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name, weights_path):
        # Save paths
        weights_path_folder = weights_path.split("/")[-2]
        self._save_paths = {
            "results_path": os.path.join(paths.RESULTS_PATH, weights_path_folder)
        }

        for key, path in self._save_paths.items():
            if not os.path.exists(path):
                os.makedirs(path)

        # Record
        writer = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(os.path.join(self._save_paths["results_path"], "output.mp4"),
                                       writer, 30.0, (160, 210), True)

        # Environment
        game_id = get_game_id(weights_path_folder.split("_")[1])

        self._env = make_atari_env(game_id, n_envs=1)
        # self._env = VecFrameStack(self._env, n_stack=4)

        # Model
        self._model = self._MODELS[model_name].load(weights_path, env=self._env,
                                                    device=self._DEVICE)

    def launch(self):
        obs = self._env.reset()
        done = None

        p_bar = tqdm()
        while not done:
            # Predict an action
            action, _state = self._model.predict(obs, deterministic=True)
            obs, reward, done, info = self._env.step(action)

            # Display and save image
            img = self._env.render(mode="rgb_array")
            self._writer.write(img[:210, :160])

            cv2.imshow("game", img)
            cv2.waitKey(130)

            # Update progress bar
            p_bar.update(1)

        self._env.reset()

        self._writer.release()
        cv2.destroyAllWindows()
