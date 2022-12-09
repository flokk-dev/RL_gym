"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import os


def get_game_name(game_id: str):
    return game_id.split("/")[1].split("-")[0]


def get_game_id(game_name: str):
    return f"ALE/{game_name}-v5"
