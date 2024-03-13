import os
import argparse
import pandas as pd
from implement_game import run_game

csv_path = ".\Data\ProstateDataset\patient_data_multiple_lesions.csv"


def episode_counter(csv_path):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_path)

    # Count the number of patients
    num_patients = len(df)

    # Calculate the total number of lesions
    num_lesions = df["num_lesions"].sum()

    # Calculate the number of episodes
    num_episode = num_patients * num_lesions

    return num_episode


# no_episodes = episode_counter(csv_path)

parser = argparse.ArgumentParser(
    prog="play", description="Script for playing a simple biopsy game"
)

parser.add_argument(
    "--log_dir",
    "--log",
    metavar="log_dir",
    type=str,
    action="store",
    default="game",
    help="Log dir to save results to",
)

parser.add_argument(
    "--NUM_EPISODES",
    metavar="NUM_EPISODES",
    type=str,
    default="5",
    action="store",
    help="How many times to play the game for",
)

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":

    args = parser.parse_args()
    num_episodes = int(args.NUM_EPISODES)
    log_dir = args.log_dir
    print(num_episodes)

    # print(f"Loading game script. Running for {args.NUM_EPISODES} episodes")
    run_game(NUM_EPISODES=num_episodes, log_dir=log_dir)
