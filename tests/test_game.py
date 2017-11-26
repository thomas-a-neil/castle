import unittest

import numpy as np
import tensorflow as tf

from chess_env import ChessEnv
from kqk_chess_env import KQKChessEnv
from dual_net import DualNet
from kqk_chess_env import KQK_CHESS_INPUT_SHAPE, KQK_POSITION_POSITION_PIECE_ACTION_SIZE
from game import Game

from utils import numline_env, mock_model_numline


class TestNumlineGame(unittest.TestCase):
    def test_numline_game(self):
        start_state = 0
        model = mock_model_numline
        env = numline_env

        game = Game()
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = game.self_play(model,
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
        game = Game()
        n_leaf_expansions = 10
        c_puct = 1000
        temperature = 1
        states, v, pi = game.self_play(self.network,
                                       self.env,
                                       self.start_state,
                                       n_leaf_expansions,
                                       c_puct,
                                       temperature,
                                       max_num_turns=5)
