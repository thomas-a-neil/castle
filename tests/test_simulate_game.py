import unittest

import numpy as np
import tensorflow as tf

from chess_env import ChessEnv
from dual_net import DualNet
from grandmaster import Game


class TestGame(unittest.TestCase):
    def setUp(self):
        state_regime = 'KQK_conv'
        action_regime = 'KQK_pos_pos_piece'
        self.env = ChessEnv(state_regime, action_regime)
        start_state = np.zeros((8, 8, 4), dtype=int)
        start_state[0, 2, 0] = 1
        start_state[2, 0, 1] = 1
        start_state[3, 3, 2] = 1
        self.start_state = start_state
        sess = tf.Session()
        self.network = DualNet(sess, state_regime, action_regime)
        sess.__enter__()
        tf.global_variables_initializer().run()

    def test_simulate_game(self):
        game = Game()
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = game.self_play(self.network, self.env, self.start_state, n_leaf_expansions, c_puct, temperature)
