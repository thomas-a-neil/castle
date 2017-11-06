import unittest

import numpy as np

from chess_env import ChessEnv


class TestChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv('KQK_conv', 'KQK_pos_pos_piece')

    def test_checkmate(self):
        start_state = np.zeros((8, 8, 4), dtype=int)
        start_state[0, 2, 0] = 1
        start_state[2, 0, 1] = 1
        start_state[0, 0, 2] = 1
        start_state[:, :, 3] = np.zeros((8, 8))
        self.assertEqual(self.env.get_legal_actions(start_state).size, 0)
        self.assertTrue(self.env.game_is_over(start_state))
        self.assertEqual(self.env.outcome(start_state), 1)

    def test_2_moves(self):
        start_state = np.zeros((8, 8, 4), dtype=int)
        start_state[0, 2, 0] = 1
        start_state[2, 0, 1] = 1
        start_state[3, 3, 2] = 1
        start_state[:, :, 3] = np.ones((8, 8))
        num_legal_moves = self.env.get_legal_actions(start_state).size
        self.assertTrue(num_legal_moves > 0)
        self.assertFalse(self.env.game_is_over(start_state))

        action = np.zeros((8, 8, 8, 8, 3), dtype=int)
        action[2, 0, 7, 0, 1] = 1
        action = self.env.convert_action_to_int(action)
        next_state = self.env.get_next_state(start_state, action)
        true_next_state = np.zeros((8, 8, 4), dtype=int)
        true_next_state[0, 2, 0] = 1
        true_next_state[7, 0, 1] = 1
        true_next_state[3, 3, 2] = 1
        self.assertTrue(np.array_equal(true_next_state, next_state))

        action_2 = np.zeros((8, 8, 8, 8, 3), dtype=int)
        action_2[3, 3, 4, 4, 2] = 1
        action_2 = self.env.convert_action_to_int(action_2)
        next_state_2 = self.env.get_next_state(next_state, action_2)
        true_next_state_2 = np.zeros((8, 8, 4), dtype=int)
        true_next_state_2[:, :, 3] = np.ones((8, 8))
        true_next_state_2[0, 2, 0] = 1
        true_next_state_2[7, 0, 1] = 1
        true_next_state_2[4, 4, 2] = 1
        self.assertTrue(np.array_equal(true_next_state_2, next_state_2))

    def test_legal_actions(self):
        start_state = np.zeros((8, 8, 4), dtype=int)
        start_state[0, 2, 0] = 1
        start_state[5, 5, 1] = 1
        start_state[0, 0, 2] = 1
        self.assertEqual(self.env.get_legal_actions(start_state).size, 1)
