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
from game import self_play_game, random_play_game

def main(argv):
	ttt()
	# generate_random_dataset()
	# supervised_value_learning()

def generate_random_dataset():
	env = TicTacToeEnv()
	start_state = np.zeros((2, 3, 3))
	num_iterations = 10**6
	last_states = np.zeros((num_iterations, 2, 3, 3))
	outcomes = np.zeros(num_iterations)
	for i in range(num_iterations):
		if i % 10000 == 0:
			print(i)
		last_state, outcome = random_play_game(env, start_state)
		# print('last_state', last_state)
		last_states[i,:,:,:] = last_state
		outcomes[i] = outcome
	np.save('last_states_1m.npy', last_states)
	np.save('outcomes_1m.npy', outcomes)

def supervised_value_learning():
	last_states = np.load('last_states.npy')
	outcomes = np.load('outcomes.npy')
	num_test = 1000
	total = 100000
	indices = np.random.choice(total, total, replace=False)
	train_indices = indices[num_test:]
	test_indices = indices[:num_test]
	x_train = last_states[train_indices]
	x_test = last_states[test_indices]
	y_train = outcomes[train_indices]
	y_test = outcomes[test_indices]

	batch_size = 100
	num_train = total - num_test
	runs_per_epoch = num_train / batch_size
	num_epochs = 40

	sess = tf.Session()
	env = TicTacToeEnv()
	network = DualNet(sess, env, learning_rate=0.00001, regularization_mult=0.0, n_residual_layers=0, input_shape=[3, 3, 2], action_size=18, num_convolutional_filters=1)
	sess.__enter__()
	tf.global_variables_initializer().run()

	value_losses = []
	test_losses = np.zeros(num_epochs)

	for i in range(num_epochs):
		print('epoch', i)
		for start_index in range(0, num_train, batch_size):
			batch_indices = np.arange(start_index, start_index + batch_size)
			x_batch = x_train[batch_indices]
			y_batch = y_train[batch_indices]
			value_loss, value_predict = network.train_value(x_batch, y_batch)
			if start_index == 0:
				print('diff', value_predict[:10], y_batch[:10])
			value_losses.append(value_loss / batch_size)
		test_loss = network.test_value(x_test, y_test)[0] / num_test
		print('test_loss', test_loss)
		test_losses[i] = test_loss
	plt.plot(value_losses)
	plt.show()
	plt.plot(test_losses)
	plt.show()

def ttt():
	env = TicTacToeEnv()
	sess = tf.Session()
	network = DualNet(sess, env, learning_rate=0.001, regularization_mult=0.0, n_residual_layers=0, input_shape=[2, 3, 3], action_size=18, num_convolutional_filters=8)
	sess.__enter__()
	tf.global_variables_initializer().run()
	start_state = np.zeros((2, 3, 3))
	n_leaf_expansions = 1
	c_puct = 10
	temperature = 1

	num_games = 2000
	num_turns_array = np.zeros(num_games)
	losses = np.zeros(num_games)
	value_losses = np.zeros(num_games)
	policy_losses = np.zeros(num_games)
	regularization_losses = np.zeros(num_games)
	for i in range(num_games):
		
		states, z, pi, last_state, outcome = self_play_game(network, env, start_state, n_leaf_expansions, c_puct, temperature, max_num_turns=10, verbose_print_board=False)
		num_turns = len(states)
		num_turns_array[i] = num_turns
		# last_state, outcome = random_play_game(env, start_state)


		# value_loss, value_predict = network.train_value(last_state, outcome)
		
		loss, value_loss, policy_loss, regularization_loss, value_predict, policy_predict = network.train(states, pi, z)
		losses[i] = loss 
		value_losses[i] = value_loss
		policy_losses[i] = policy_loss
		regularization_losses[i] = regularization_loss
		if i % 100 == 0:
		# 	# print('states', states)
			print('game number', i)
			print('value_predict', value_predict, z)
			print('policy_predict', policy_predict)

		# print('curr loss', losses[i])
		# print('length of game', num_turns)
	plt.scatter(np.arange(num_games), num_turns_array)
	plt.show()

	plt.scatter(np.arange(num_games), losses)
	plt.title('total loss')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()
	

	plt.scatter(np.arange(num_games), value_losses)
	plt.title('value loss')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()

	plt.scatter(np.arange(num_games), policy_losses)
	plt.title('policy loss')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()

	plt.scatter(np.arange(num_games), regularization_losses)
	plt.title('regularization loss')
	plt.xlabel('iterations')
	plt.ylabel('loss')
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