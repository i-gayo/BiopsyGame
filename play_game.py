import os 
import argparse
from implement_game import run_game

parser = argparse.ArgumentParser(prog='play',
                                description="Script for playing a simple biopsy game")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='game',
                    help='Log dir to save results to')

parser.add_argument('--NUM_EPISODES',
                    metavar='NUM_EPISODES',
                    type=str,
                    default='5',
                    action = 'store',
                    help='How many times to play the game for')

# Parse arguments
args = parser.parse_args()

if __name__ == '__main__':
    
    args = parser.parse_args()
    num_episodes = int(args.NUM_EPISODES)
    log_dir = args.log_dir
    print(num_episodes)

    #print(f"Loading game script. Running for {args.NUM_EPISODES} episodes")
    run_game(NUM_EPISODES = num_episodes, log_dir = log_dir)
