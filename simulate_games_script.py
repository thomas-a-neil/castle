import sys
from collections import Counter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from chess_env import ChessEnv
from tictactoe_env import TicTacToeEnv
# from kqk_chess_env import KQKChessEnv
from dual_net import DualNet
from game import self_play_game, play_game, RandomModel


def main(argv):
    # ttt()
    generate_random_dataset()
    # supervised_value_learning()


def generate_random_dataset():
    env = ChessEnv()
    num_iterations = 100
    states_per_iteration = []
    winners_per_iteration = []

    num_turn_per_iteration = []
    for i in range(num_iterations):
        verbose = False
        if i % 10 == 0:
            print(i)
        if i % 100 == 0:
            verbose = True
        states, winner = play_game(RandomModel(env), RandomModel(env), env, verbose=verbose)
        states_per_iteration.append(states)
        winners_per_iteration.append(winner)
        num_turn_per_iteration.append(len(states))

    print(Counter(num_turn_per_iteration))
    np.save('states.npy', np.array(states_per_iteration))
    np.save('winner.npy', np.array(winners_per_iteration))


if __name__ == '__main__':
    main(sys.argv[1:])
