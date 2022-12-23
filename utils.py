"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import os


def get_model_folder(path: str):
    if path.split("/")[-2] == "checkpoints":
        return path.split("/")[-3]
    return path.split("/")[-2]


def get_game_id(path: str):
    return get_model_folder(path).split("_")[1]
