import unittest

import numpy as np
import tensorflow as tf

from dual_net import DualNet
from game import self_play_game
from kqk_chess_env import KQKChessEnv, KQK_CHESS_INPUT_SHAPE

from utils import numline_env, mock_model_numline


class TestNumlineGame(unittest.TestCase):
    def test_numline_game(self):
        start_state = 0
        model = mock_model_numline
        env = numline_env

        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(model,
                                       env,
                                       start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=5)


class TestKQKChessGame(unittest.TestCase):
    def setUp(self):
        state_regime = 'KQK_conv'
        action_regime = 'KQK_pos_pos_piece'
        self.env = KQKChessEnv(state_regime, action_regime)
        start_state = np.zeros(KQK_CHESS_INPUT_SHAPE, dtype=int)

        # White King
        start_state[0, 2, 0] = 1
        # White Queen
        start_state[2, 0, 1] = 1
        # Black King
        start_state[3, 3, 2] = 1
        # initial board state:
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . k . . . .
        # K . . . . . . .
        # . . . . . . . .
        # . . Q . . . . .

        self.start_state = start_state
        sess = tf.Session()
        self.network = DualNet(sess, self.env)
        sess.__enter__()
        tf.global_variables_initializer().run()

    def test_simulate_game(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=5)
        self.assertEqual(len(v), 6)
        self.assertEqual(len(states), 6)
        self.assertEqual(len(pi), 6)
