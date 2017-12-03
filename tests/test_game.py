import unittest

import numpy as np
import tensorflow as tf

from dual_net import DualNet
from game import self_play_game, random_play_game, play_smart_vs_random_game, play_smart1_vs_smart2_game
from kqk_chess_env import KQKChessEnv, KQK_CHESS_INPUT_SHAPE
from tictactoe_env import TicTacToeEnv

from utils import numline_env, mock_model_numline
import copy

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


class TestTicTacToeGame(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        sess = tf.Session()
        self.network = DualNet(sess, self.env, input_shape=self.env.input_shape, action_size=self.env.action_size)
        sess.__enter__()
        tf.global_variables_initializer().run()
        self.start_state = self.env.start_state

    def test_ttt_game(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        states, v = random_play_game(self.env, start_state)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)

    def test_ttt_self_play_and_train(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)
        self.network.train(states, pi, v)
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)

    def test_smart1_vs_smart2(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        untrained_model = copy.copy(self.network)
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)
        self.network.train(states, pi, v)
        states, v = play_smart1_vs_smart2_game(untrained_model,
                                       self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)
        self.assertEqual(0, 1)

    def test_smart_vs_random(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)
        self.network.train(states, pi, v)
        states, v = play_smart_vs_random_game(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=9)

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
