import unittest

import chess

from chess_env import ChessEnv

INITIAL_BLACK_PAWNS_STRING = ('[[ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 1.  1.  1.  1.  1.  1.  1.  1.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]\n'
                              ' [ 0.  0.  0.  0.  0.  0.  0.  0.]]')


class TestChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()

    def test_move_to_index(self):
        uci_move = chess.Move.from_uci('a1b1')
        index = self.env.move_to_index(uci_move)
        self.assertEqual(index, 1)

        uci_move = chess.Move.from_uci('a2a3')
        index = self.env.move_to_index(uci_move)
        self.assertEqual(index, 528)

    def test_board_to_state(self):
        board = chess.Board()
        state = self.env.map_board_to_state(board)
        # Checks top two rows of the black pawn layer
        self.assertEqual(str(state[:, :, 11]), INITIAL_BLACK_PAWNS_STRING)

    def test_state_to_board(self):
        board = chess.Board()
        state = self.env.map_board_to_state(board)
        new_board = self.env.map_state_to_board(state)
        self.assertEqual(str(new_board), str(board))
