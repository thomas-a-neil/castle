import unittest

import numpy as np

from tictactoe_env import TicTacToeEnv, InvalidStateException


class TestTicTacToeEnv(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_init(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        self.assertEqual(len(self.env.get_legal_actions(start_state)), 9)

    def test_1_move(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        action_array = np.zeros((2, 3, 3), dtype=int)

        action_array[0, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0, 1, 1] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(len(self.env.get_legal_actions(next_state)), 8)

    def test_2_moves(self):
        start_state = np.zeros((2, 3, 3), dtype=int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 1, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        self.assertEqual(len(self.env.get_legal_actions(next_state)), 8)
        next_state = self.env.get_next_state(next_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0, 1, 1] = 1
        true_next_state[1, 1, 0] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(len(self.env.get_legal_actions(next_state)), 7)

    def test_x_win(self):
        start_state = np.zeros((2, 3, 3), dtype=int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 1, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 2, 2] = 1

        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)

        action_array[1, 2, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        self.assertEqual(self.env.is_game_over(next_state), 0)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 0, 0] = 1

        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)

        true_next_state[0, 1, 1] = 1
        true_next_state[1, 1, 0] = 1
        true_next_state[0, 0, 0] = 1
        true_next_state[0, 2, 2] = 1
        true_next_state[1, 2, 0] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(self.env.is_game_over(next_state), 1)
        self.assertEqual(self.env.outcome(next_state), 1)

    def test_o_win(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        self.env.print_board(start_state)
        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 0, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        self.env.print_board(next_state)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)
        self.env.print_board(next_state)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 2, 2] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 2, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        self.assertEqual(self.env.is_game_over(next_state), 0)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 0, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        self.assertEqual(self.env.is_game_over(next_state), 0)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 0, 2] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0, 0, 1] = 1
        true_next_state[1, 1, 1] = 1
        true_next_state[0, 0, 0] = 1
        true_next_state[0, 2, 2] = 1
        true_next_state[1, 2, 0] = 1
        true_next_state[1, 0, 2] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(self.env.is_game_over(next_state), 1)
        self.assertEqual(self.env.outcome(next_state), -1)

    def test_invalid_state(self):
        invalid_state = np.zeros((2, 3, 3), dtype=int)
        # one o. x must move before o
        invalid_state[1, 0, 1] = 1
        with self.assertRaises(InvalidStateException):
            self.env.is_x_turn(invalid_state)

    def test_convert_action_to_int(self):
        action = np.zeros((2, 3, 3), dtype=int)
        action[1, 0, 2] = 1
        action_int = self.env.convert_action_to_int(action)
        recovered_action = self.env.convert_int_to_action(action_int)
        self.assertEqual(type(action_int), np.int64)
        self.assertEqual(action[1, 0, 2], recovered_action[1, 0, 2])
