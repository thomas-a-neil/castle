import unittest

import numpy as np

from tictactoe_env import TicTacToeEnv


class TestTicTacToeEnv(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_init(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        self.assertEqual(self.env.get_legal_actions(start_state).size, 9)

    def test_1_move(self):
        start_state = np.zeros((2, 3, 3), dtype=int)
        action_array = np.zeros((2, 3, 3), dtype=int)

        action_array[0, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0, 1, 1] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(self.env.get_legal_actions(next_state).size, 8)

    def test_2_moves(self):
        start_state = np.zeros((2, 3, 3), dtype=int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0, 1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1, 1, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        self.assertEqual(self.env.get_legal_actions(next_state).size, 8)
        next_state = self.env.get_next_state(next_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0, 1, 1] = 1
        true_next_state[1, 1, 0] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(self.env.get_legal_actions(next_state).size, 7)

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
        action_array[0,0,1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(start_state, action_int)

        self.env.print_board(next_state)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1,1,1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)
        self.env.print_board(next_state)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0,2,2] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1,2,0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        self.assertEqual(self.env.is_game_over(next_state), 0)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[0,0,0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        self.assertEqual(self.env.is_game_over(next_state), 0)

        action_array = np.zeros((2, 3, 3), dtype=int)
        action_array[1,0,2] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(next_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[0,0,1] = 1
        true_next_state[1,1,1] = 1
        true_next_state[0,0,0] = 1
        true_next_state[0,2,2] = 1
        true_next_state[1,2,0] = 1
        true_next_state[1,0,2] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(self.env.is_game_over(next_state), 1)
        self.assertEqual(self.env.outcome(next_state), -1)





        

