import unittest

import numpy as np
import tensorflow as tf

from dual_net import DualNet
from game import self_play_game, random_play_game, play_smart_vs_random_game, play_smart1_vs_smart2_game, play_many_vs_random_games, play_many_smart1_vs_smart2
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
                                       n_leaf_expansions,
                                       start_state=start_state,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=5)


class TestTicTacToeGame(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        sess = tf.Session()
        self.network = DualNet(sess, self.env, input_shape=self.env.input_shape, action_size=self.env.action_size)
        sess.__enter__()
        tf.global_variables_initializer().run()

    def test_ttt_game(self):
        states, v = random_play_game(self.env)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)

    def test_ttt_self_play_and_train(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9,
                                       verbose=True)
        self.network.train(states, pi, v)
        print('states', states, len(states))
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)

    def test_smart1_vs_smart2(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        untrained_model = copy.copy(self.network)
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.network.train(states, pi, v)
        states, v, outcome = play_smart1_vs_smart2_game(untrained_model,
                                       self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)

    def test_smart_vs_random(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)
        self.network.train(states, pi, v)
        states, v, outcome = play_smart_vs_random_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.assertLessEqual(len(states), 9)
        self.assertGreaterEqual(len(states), 5)

    def test_many_smart_vs_random(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        num_games = 100
        outcomes = play_many_vs_random_games(num_games, 
                                       self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)

        # the number of wins for x should be higher than the number of wins for o since x plays first
        self.assertGreater(outcomes[0], outcomes[2])
        self.assertEqual(outcomes[0] + outcomes[1] + outcomes[2], num_games)

    def test_many_smart1_vs_smart2(self):
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        num_games = 100
        untrained_model = copy.copy(self.network)
        states, v, pi = self_play_game(self.network,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)
        self.network.train(states, pi, v)
        outcomes = play_many_smart1_vs_smart2(num_games, 
                                       self.network,
                                       untrained_model,
                                       self.env,
                                       n_leaf_expansions,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=9)

        self.assertEqual(outcomes[0] + outcomes[1] + outcomes[2], num_games)

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
                                       n_leaf_expansions,
                                       start_state=self.start_state,
                                       c_puct=c_puct,
                                       temperature=temperature,
                                       max_num_turns=5)
        self.assertEqual(len(v), 6)
        self.assertEqual(len(states), 6)
        self.assertEqual(len(pi), 6)
