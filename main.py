"""
Creator: Flokk___
Date: 18/11/2022
Version: V1.0

Purpose:
"""

# IMPORT: utils
import argparse

# IMPORT: projet
from src import Trainer, Inferencer

# WARNINGS SHUT DOWN
import warnings
warnings.filterwarnings("ignore")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reinforcement learning module.")

    parser.add_argument("-p", "--pipe", type=str, nargs="?", default="train",
                        choices=["train", "inference"], help="the pipeline to run.")

    parser.add_argument("-m", "--model", type=str, nargs="?", default="DQN",
                        choices=["DQN", "A2C"], help="the name of the model.")

    parser.add_argument("-g", "--game", type=str, nargs="?", default="ALE/Breakout-v5",
                        choices=["ALE/Breakout-v5"], help="the game to train on.")

    parser.add_argument("-w", "--weights", type=str, nargs="?", default=None,
                        help="the weights to load")

    parser.add_argument("-i", "--iter", type=int, nargs="?", default=10,
                        help="the number of iteration")

    parser.add_argument("-e", "--episode", type=int, nargs="?", default=10000,
                        help="the number of epochs.")

    return parser.parse_args()


if __name__ == "__main__":
    # Parameters
    args = get_args()

    # Train
    if args.pipe == "train":
        trainer = Trainer(model_name=args.model, game=args.game)
        trainer.launch(nb_iter=args.iter)
    elif args.pipe == "inference":
        inferencer = Inferencer(model_name=args.model, game=args.game, weights_path=args.weights)
        inferencer.launch()

