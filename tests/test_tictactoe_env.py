import unittest

import numpy as np

from tictactoe_env import TicTacToeEnv, InvalidStateException

import pdb

class TestTicTacToeEnv(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.start_state = self.env.get_start_state()

    def test_invariant_transformations(self):
        moves = [(1, 1), (0, 1), (2, 2), (2, 0), (0, 0)]
        state = self.start_state
        for i in range(len(moves)):
            action_array = np.zeros(self.env.action_dims, dtype=int)
            action_array[moves[i][0], moves[i][1]] = 1
            action_int = self.env.convert_action_to_int(action_array)
            state = self.env.get_next_state(state, action_int)
        self.env.print_board(state)
        transformations, transformation, transform_index = self.env.sample_invariant_transformation(state)
        for j in range(len(transformations)):
            # transformation_state = self.env.sample_invariant_transformation(state)
            t = transformations[j]
            self.env.print_board(t)
        self.assertEqual(9, 8)

    def test_revert_back(self):
        moves = [(0, 2), (0, 1)]
        state = self.start_state
        for i in range(len(moves)):
            action_array = np.zeros(self.env.action_dims, dtype=int)
            action_array[moves[i][0], moves[i][1]] = 1
            action_int = self.env.convert_action_to_int(action_array)
            state = self.env.get_next_state(state, action_int)
        true_final_state = np.zeros((2, 3, 3), dtype=int)
        true_final_state[0, 0, 2] = 1
        true_final_state[1, 0, 1] = 1

        self.assertTrue(np.array_equal(true_final_state, state))

    def test_init(self):
        self.assertEqual(len(self.env.get_legal_actions(self.start_state)), 9)

    def test_1_move(self):
        action_array = np.zeros(self.env.action_dims, dtype=int)
        action_array[1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        next_state = self.env.get_next_state(self.start_state, action_int)

        true_next_state = np.zeros((2, 3, 3), dtype=int)
        true_next_state[1, 1, 1] = 1

        self.assertTrue(np.array_equal(true_next_state, next_state))
        self.assertEqual(len(self.env.get_legal_actions(next_state)), 8)

    def test_2_moves(self):
        action_array = np.zeros(self.env.action_dims, dtype=int)
        action_array[1, 1] = 1
        action_int = self.env.convert_action_to_int(action_array)
        second_state = self.env.get_next_state(self.start_state, action_int)

        action_array = np.zeros(self.env.action_dims, dtype=int)
        action_array[1, 0] = 1
        action_int = self.env.convert_action_to_int(action_array)
        self.assertEqual(len(self.env.get_legal_actions(second_state)), 8)
        third_state = self.env.get_next_state(second_state, action_int)

        true_third_state = np.zeros((2, 3, 3), dtype=int)
        true_third_state[0, 1, 1] = 1
        true_third_state[1, 1, 0] = 1

        self.assertTrue(np.array_equal(true_third_state, third_state))
        self.assertEqual(len(self.env.get_legal_actions(third_state)), 7)

    def test_x_win(self):
        moves = [(1, 1), (0, 1), (2, 2), (2, 1), (0, 0)]
        state = self.start_state
        for i in range(len(moves)):
            action_array = np.zeros(self.env.action_dims, dtype=int)
            action_array[moves[i][0], moves[i][1]] = 1
            action_int = self.env.convert_action_to_int(action_array)
            state = self.env.get_next_state(state, action_int)
            if i < len(moves) - 1:
                self.assertEqual(self.env.is_game_over(state), 0)
        self.assertEqual(self.env.is_game_over(state), 1)
        self.assertEqual(self.env.outcome(state), 1)

    def test_o_win(self):
        moves = [(1, 1), (0, 1), (2, 2), (0, 0), (1, 2), (0, 2)]
        state = self.start_state
        for i in range(len(moves)):
            action_array = np.zeros(self.env.action_dims, dtype=int)
            action_array[moves[i][0], moves[i][1]] = 1
            action_int = self.env.convert_action_to_int(action_array)
            state = self.env.get_next_state(state, action_int)
            if i < len(moves) - 1:
                self.assertEqual(self.env.is_game_over(state), 0)
        self.assertEqual(self.env.is_game_over(state), 1)
        self.assertEqual(self.env.outcome(state), -1)

    def test_draw(self):
        moves = [(1, 1), (2, 1), (2, 2), (0, 0), (1, 2), (0, 2), (0, 1), (1, 0), (2, 0)]
        state = self.start_state
        for i in range(len(moves)):
            action_array = np.zeros(self.env.action_dims, dtype=int)
            action_array[moves[i][0], moves[i][1]] = 1
            action_int = self.env.convert_action_to_int(action_array)
            state = self.env.get_next_state(state, action_int)
            if i < len(moves) - 1:
                self.assertEqual(self.env.is_game_over(state), 0)
        self.assertEqual(self.env.is_game_over(state), 1)
        self.assertEqual(self.env.outcome(state), 0)

    def test_convert_action_to_int(self):
        action = np.zeros((3, 3), dtype=int)
        action[0, 2] = 1
        action_int = self.env.convert_action_to_int(action)
        recovered_action = self.env.convert_int_to_action(action_int)
        self.assertEqual(type(action_int), np.int64)
        self.assertEqual(action[0, 2], recovered_action[0, 2])

    def test_is_x_turn(self):
        state = np.zeros((2, 3, 3), dtype=int)
        state[0, 0, 2] = 1

        # x must move before o
        with self.assertRaises(InvalidStateException):
            self.env.is_x_turn(state)

        state[1, 0, 1] = 1
        self.assertTrue(self.env.is_x_turn(state))

        state[1, 0, 0] = 1
        self.assertFalse(self.env.is_x_turn(state))
