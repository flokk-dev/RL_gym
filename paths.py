"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os


"""
ROOT
"""
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


"""
RESOURCES
"""
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources")

CONFIG_PATH = os.path.join(RESOURCES_PATH, "config.json")

MODELS_PATH = os.path.join(RESOURCES_PATH, "models")
RESULTS_PATH = os.path.join(RESOURCES_PATH, "results")
