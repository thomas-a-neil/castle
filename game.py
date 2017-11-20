from functools import partial

import numpy as np

from mcts import exploration_bonus_for_c_puct, perform_rollouts, get_pi
from tree import Node


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
