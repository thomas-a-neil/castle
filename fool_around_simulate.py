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
from game import self_play_game, random_play_game, play_smart_vs_random_game, play_smart1_vs_smart2_game, play_many_vs_random_games, play_many_smart1_vs_smart2

def main(argv):
	# gather_data()
	ttt()
	# test_save()
	test_load()
	

def gather_data():
	env = TicTacToeEnv()
	num_iterations = 10**5
	# last_states = np.zeros((2*num_iterations, 2, 3, 3))
	# last_outcomes = np.zeros(2*num_iterations)
	last_states = []
	last_outcomes = []
	for i in range(num_iterations):
		if i % 10000 == 0:
			print(i)
		states, outcomes, end_state = random_play_game(env)
		# last_states.extend(states)
		# last_outcomes.extend(outcomes)

		# transformed stuff

		for i in range(len(states)):
			transformations, chosen_transform = env.sample_invariant_transformation(states[i])
			for j in range(len(transformations)):
				last_states.append(transformations[j])
				last_outcomes.append(outcomes[i])

			
		# last_states[2*i,:,:,:] = states[-1]
		# last_states[2*i+1,:,:,:] = states[-2]
		# last_outcomes[2*i] = outcomes[-1]
		# last_outcomes[2*i+1] = -outcomes[-1]
	np.save('last_states_fixed_transformed.npy', last_states)
	np.save('outcomes_fixed_transformed.npy', last_outcomes)

def test_save():
	env = TicTacToeEnv()
	sess = tf.Session()
	network = DualNet(sess, env, learning_rate=0.0001, regularization_mult=0.0, n_residual_layers=0, input_shape=[2, 3, 3], action_size=9, num_convolutional_filters=8)
	sess.__enter__()
	tf.global_variables_initializer().run()
	start_state = np.zeros((2, 3, 3))
	n_leaf_expansions = 15
	c_puct = 1
	temperature = [0.4, 0.2]

	

def test_load():
	env = TicTacToeEnv()
	sess = tf.Session()
	sess.__enter__()
	load_graph(sess, 0)


def ttt():
	env = TicTacToeEnv()
	sess = tf.Session()
	network = DualNet(sess, env, learning_rate=0.001, regularization_mult=0.0, n_residual_layers=0, input_shape=[2, 3, 3], action_size=9, num_convolutional_filters=8)
	sess.__enter__()
	tf.global_variables_initializer().run()
	saver = tf.train.Saver()
	start_state = np.zeros((2, 3, 3))
	n_leaf_expansions = 30
	c_puct = 1
	temperature = [0.8, 0.2]


	'''
	Supervised value learning first
	'''
	# verbose = True
	# num_epochs = 400
	# # last_states = np.load('last_states.npy')
	# # outcomes = np.load('outcomes.npy')
	# last_states = np.load('last_states_fixed_transformed.npy')
	# outcomes = np.load('outcomes_fixed_transformed.npy')
	# print('outcomes', outcomes.shape)
	# print('last_states', last_states.shape)
	# num_test = 1000
	# total = 10000
	# indices = np.random.choice(total, total, replace=False)
	# train_indices = indices[num_test:]
	# test_indices = indices[:num_test]
	# x_train = last_states[train_indices]
	# x_test = last_states[test_indices]
	# y_train = outcomes[train_indices]
	# y_test = outcomes[test_indices]

	# batch_size = 100
	# num_train = total - num_test
	# runs_per_epoch = num_train / batch_size

	# value_losses = []
	# test_losses = np.zeros(num_epochs)
	# test_accuracies = np.zeros(num_epochs)

	# for i in range(num_epochs):
	# 	if verbose:
	# 		print('Supervised epoch', i)
	# 	for start_index in range(0, num_train, batch_size):
	# 		batch_indices = np.arange(start_index, start_index + batch_size)
	# 		x_batch = x_train[batch_indices]
	# 		y_batch = y_train[batch_indices]
	# 		value_loss, value_predict = network.train_value(x_batch, y_batch)
	# 		if start_index == 0 and verbose:
	# 			print('diff', value_predict[:20], y_batch[:20])
	# 		value_losses.append(value_loss / batch_size)
	# 	test_loss = network.test_value(x_test, y_test)[0] / num_test
	# 	test_guess, test_accuracy = network.classify_value(x_test, y_test)
	# 	test_accuracies[i] = test_accuracy
	# 	if verbose:
	# 		print('test_accuracy', test_accuracy)
	# 		print('test_loss', test_loss)
	# 	test_losses[i] = test_loss

	'''
	MCTS learning with self play second
	'''

	num_training_games = 100000

	num_games_v_random = 1000

	init_results = play_many_vs_random_games(num_games_v_random, network, env,
		n_leaf_expansions, c_puct=c_puct, temperature=temperature, max_num_turns=10, verbose=True)
	print('init_results', init_results)

	test_vs_random_results = [init_results]

	batch_size = 100
	num_batches = int(num_training_games / batch_size)
	losses = np.zeros(num_batches)
	value_losses = np.zeros(num_batches)
	policy_losses = np.zeros(num_batches)
	for i in range(num_batches):
		batch_states = []
		batch_z = []
		batch_pi = []
		for j in range(batch_size):
			states, z, pi = self_play_game(network, env, n_leaf_expansions, 
				c_puct=c_puct, temperature=temperature, max_num_turns=10, verbose=False)
			batch_states.extend(states)
			batch_z.extend(z)
			batch_pi.extend(pi)
		batch_states = np.array(batch_states)
		batch_z = np.array(batch_z)
		batch_pi = np.array(batch_pi)

		# loss, value_loss, policy_loss = network.train(states, pi, z)
		loss, value_loss, policy_loss, value_predict, policy_predict = network.train(batch_states, batch_pi, batch_z)
		# network.save_graph(saver, sess, 0)
		# with open('filename.pickle', 'wb') as handle:
		# 	pickle.dump(network, handle)
		# return
		losses[i] = loss 
		value_losses[i] = value_loss
		policy_losses[i] = policy_loss
		# regularization_losses[i] = regularization_loss
		
		num_to_print = 10
		print('batch number', i)
		print('policy_loss', policy_loss)
		print('value_loss', value_loss)
		print('loss', loss)
		for j in range(num_to_print):
			env.print_board(batch_states[j])
			print('value_predict', value_predict[j])
			print('policy_predict', policy_predict[j])
			print('pi', batch_pi[j])
			# print('value_predict', value_predict[:num_to_print], batch_z[:num_to_print])
			# env.print_board()
			# print('policy_predict', policy_predict[:num_to_print])
			# print('pi', batch_pi[:num_to_print])

		if i % 20 == 0 and i > 0:
			curr_results = play_many_vs_random_games(num_games_v_random, network, env,
				n_leaf_expansions, c_puct=c_puct, temperature=temperature, max_num_turns=10, verbose=True)
			test_vs_random_results.append(curr_results)
			print('curr_results', curr_results)

		# print('policy_predict', policy_predict)
	final_results = play_many_vs_random_games(num_games_v_random, network, env,
		n_leaf_expansions, c_puct=c_puct, temperature=temperature, max_num_turns=10, verbose=False)
	# print('final_results', final_results)
	print('test_vs_random_results', test_vs_random_results)
	test_vs_random_results.append(final_results)

	plt.scatter(np.arange(len(test_vs_random_results)), np.array([x[0] for x in test_vs_random_results]) / 100.0)
	plt.title('win rate')
	plt.show()

	plt.scatter(np.arange(len(test_vs_random_results)), np.array([x[1] for x in test_vs_random_results]) / 100.0)
	plt.title('tie rate')
	plt.show()

	plt.scatter(np.arange(len(test_vs_random_results)), np.array([x[2] for x in test_vs_random_results]) / 100.0)
	plt.title('loss rate')
	plt.show()

	plt.plot(losses)
	plt.title('total loss')
	plt.show()
	plt.plot(value_losses)
	plt.title('value_losses')
	plt.show()
	plt.plot(policy_losses)
	plt.title('policy_losses')
	plt.show()

def load_graph(sess, index):
	  #Now, save the graph
	  # sess = tf.Session()
	  new_saver = tf.train.import_meta_graph('my_test_model_{0}.meta'.format(index))
	  new_saver.restore(sess, tf.train.latest_checkpoint('./'))


if __name__ == '__main__':
	main(sys.argv[1:])