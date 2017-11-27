import sys
from chess_env import ChessEnv
from tictactoe_env import TicTacToeEnv
import numpy as np
import tensorflow as tf
from mcts import backup, select, expand_node, exploration_bonus_for_c_puct, perform_rollouts, get_action_distribution
from tree import Node
from functools import partial
from dual_net import DualNet
# from grandmaster import Game
import pdb
import pickle
import matplotlib.pyplot as plt
from game import self_play_game

def main(argv):
	ttt()

def ttt():
	env = TicTacToeEnv()
	sess = tf.Session()
	network = DualNet(sess, env)
	sess.__enter__()
	tf.global_variables_initializer().run()
	start_state = np.zeros((2, 3, 3))
	n_leaf_expansions = 10
	c_puct = 10
	temperature = 1

	num_games = 500
	num_turns_array = np.zeros(num_games)
	losses = np.zeros(num_games)
	for i in range(num_games):
		print('game number', i)
		states, z, pi = self_play_game(network, env, start_state, n_leaf_expansions, c_puct, temperature, max_num_turns=10, verbose_print_board=True)
		num_turns = len(states)
		num_turns_array[i] = num_turns
		print('result', z)
		losses[i] = network.train(states, pi, z)
		print('curr loss', losses[i])
		print('length of game', num_turns)
	plt.scatter(np.arange(num_games), num_turns_array)
	plt.show()
	plt.scatter(np.arange(num_games), losses)
	plt.show()

def chess():
	state_regime = 'KQK_conv'
	action_regime = 'KQK_pos_pos_piece'
	env = ChessEnv(state_regime, action_regime)
	start_state = np.zeros((8,8,4), dtype=int)
	start_state[2,0,0] = 1
	start_state[3,5,1] = 1
	start_state[0,0,2] = 1
	start_state[:,:,3] = 1
	sess = tf.Session()
	network = DualNet(sess, state_regime, action_regime)
	sess.__enter__()
	tf.global_variables_initializer().run()

	n_leaf_expansions = 20
	c_puct = 100
	temperature = 1
	num_games = 300
	game = Game()
	num_turns_array = np.zeros(num_games)
	for i in range(num_games):
		print('game number', i)
		states, z, pi = game.self_play(network, env, start_state, n_leaf_expansions, c_puct, temperature, verbose=True)
		num_turns = len(states)
		num_turns_array[i] = num_turns
		print('result', z)
		network.train(states, pi, z)
	plt.scatter(np.arange(num_games), num_turns_array)
	plt.show()

if __name__ == '__main__':
	main(sys.argv[1:])