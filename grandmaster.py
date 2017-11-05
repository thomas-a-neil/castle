from mcts import backup, select, expand_node, exploration_bonus_for_c_puct, perform_rollouts, get_pi, get_action_distribution
from tree import Node
from functools import partial
import numpy as np
import pdb

class Grandmaster(object):
	"""docstring for Grandmaster"""
	def __init__(self, model, env, num_games, n_leaf_expansions):
		self.model = model
		self.num_games = num_games
		self.env = env
		self.start_state = 0
		self.n_leaf_expansions = n_leaf_expansions
		self.c = 100
		self.temperature = 1

	def become_grandmaster():
		for i in range(self.num_games):
			new_game = Game()
			s, v, pi = new_game.self_play(self.model, self.env, self.start_state, self.n_leaf_expansions, self.c)
			model.train(s, pi, z)

class Game(object):
	"""docstring for Game"""
	def self_play(self, model, env, start_state, n_leaf_expansions, c_puct, temperature):
		node = Node(start_state)
		pi_array = []
		num_turns = 0
		states = []
		while not env.game_is_over(node.state):
			state = node.state
			states.append(state)
			exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c_puct)
			perform_rollouts(node, n_leaf_expansions, model, env, exploration_bonus)
			pi, distribution, actions = get_pi(node, temperature, env.action_size)
			pi_array.append(pi)
			chosen_action = np.random.choice(np.arange(actions.size), p=distribution)
			node = node.outgoing_edges[chosen_action].out_node
			num_turns += 1
			if num_turns > 40:
				winner = 0
				v = winner*np.ones(num_turns)
				pi_array = np.array(pi_array)
				return states, v, pi_array
		winner = env.outcome(state)
		v = winner*np.ones(num_turns)
		pi_array = np.array(pi_array)
		return states, v, pi_array
		