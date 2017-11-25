import numpy as np
import chess

from tests.utils import map_xy_to_square, map_square_to_xy


class ChessEnv(object):
    """
    A simplified chess environment where one king faces off against
    a king and queen on an 8x8 board
    """
    def __init__(self, state_regime, action_regime):
        self.state_regime = state_regime
        self.action_regime = action_regime
        if action_regime == 'KQK_pos_pos_piece':
            self.action_dims = (8, 8, 8, 8, 3)
            self.action_size = np.prod(self.action_dims)
        elif action_regime == 'KQK_pos_pos':
            self.action_dims = (8, 8, 8, 8)
            self.action_size = np.prod(self.action_dims)

    # The following 4 methods are called outside of the environment
    def get_next_state(self, state, action):
        board = self._map_state_to_board(state)
        move = self._map_action_to_move(state, action)
        board.push(move)
        next_state = self._map_board_to_state(board)
        return next_state

    def get_legal_actions(self, state):
        board = self._map_state_to_board(state)
        legal_moves = board.legal_moves
        legal_actions = np.zeros(len(legal_moves), dtype=int)
        i = 0
        for move in legal_moves:
            action = self._map_move_to_action(board, move)
            legal_actions[i] = action
            i += 1
        return legal_actions

    def is_game_over(self, state):
        board = self._map_state_to_board(state)
        return board.is_game_over()

    def outcome(self, state):
        board = self._map_state_to_board(state)
        if board.result() == '1/2-1/2':
            result = 0
        elif board.result() == '1-0':
            result = 1
        elif board.result() == '0-1':
            result = -1
        return result

    def is_legal(self, state, action):
        return action in self.get_legal_actions(state)

    def print_board(self, state):
        board = self._map_state_to_board(state)
        print(board)

    def convert_action_to_int(self, action_array):
        action_array = np.reshape(action_array, -1)
        return np.where(action_array == 1)[0]

    def convert_int_to_action(self, action_int):
        action_array = np.zeros((self.action_dims), dtype=int)
        action_array = np.reshape(action_array, -1)
        action_array[action_int] = 1
        action_array = np.reshape(action_array, (self.action_dims))
        return action_array

    # All methods below are internal and are not called outside of the environment
    def _map_board_to_state(self, board):
        if self.state_regime == 'KQK_conv':
            pieces = board.piece_map()
            state = np.zeros((8, 8, 4), dtype=int)
            for square in pieces:
                piece = pieces[square]
                piece = str(piece)
                x, y = map_square_to_xy(square)
                if piece == 'K':
                    state[x, y, 0] = 1
                elif piece == 'Q':
                    state[x, y, 1] = 1
                elif piece == 'k':
                    state[x, y, 2] = 1
            state[:, :, 3] = board.turn * np.ones((8, 8))
            return state

    def _map_state_to_board(self, state):
        if self.state_regime == 'KQK_conv':
            pieces = {}
            for i in range(3):
                if np.sum(state[:, :, i]) == 1:
                    if i < 2:
                        color = chess.WHITE
                    else:
                        color = chess.BLACK
                    x, y = np.where(state[:, :, i] > 0)
                    square = map_xy_to_square(x, y)
                    if i == 0 or i == 2:
                        piece = chess.KING
                    elif i == 1:
                        piece = chess.QUEEN
                    pieces[square] = chess.Piece(piece, color)
            board = chess.Board()
            board.set_piece_map(pieces)
            board.turn = state[0, 0, 3]
            return board

    def _map_move_to_action(self, board, move):
        if self.state_regime == 'KQK_conv':
            if self.action_regime == 'KQK_pos_pos_piece':
                action = np.zeros((8, 8, 8, 8, 3), dtype=int)
                from_x, from_y = map_square_to_xy(move.from_square)
                to_x, to_y = map_square_to_xy(move.to_square)
                piece = board.piece_at(move.from_square)
                piece_type = piece.piece_type
                piece_color = piece.color
                if piece_type == chess.KING and piece_color == chess.WHITE:
                    action[from_x, from_y, to_x, to_y, 0] = 1
                elif piece_type == chess.QUEEN and piece_color == chess.WHITE:
                    action[from_x, from_y, to_x, to_y, 1] = 1
                elif piece_type == chess.KING and piece_color == chess.BLACK:
                    action[from_x, from_y, to_x, to_y, 2] = 1
                # action = np.reshape(action, 8*8*8*8*3)
                return self.convert_action_to_int(action)
            elif self.action_regime == 'KQK_pos_pos':
                action = np.zeros((8, 8, 8, 8))
                from_x, from_y = map_square_to_xy(move.from_square)
                to_x, to_y = map_square_to_xy(move.to_square)
                piece = board.piece_at(move.from_square)
                piece_type = piece.piece_type
                piece_color = piece.color
                if piece_type == chess.KING and piece_color == chess.WHITE:
                    action[from_x, from_y, to_x, to_y] = 1
                elif piece_type == chess.QUEEN and piece_color == chess.WHITE:
                    action[from_x, from_y, to_x, to_y] = 1
                elif piece_type == chess.KING and piece_color == chess.BLACK:
                    action[from_x, from_y, to_x, to_y] = 1
                # action = np.reshape(action, 8*8*8*8)
                return self.convert_action_to_int(action)

    def _map_action_to_move(self, state, action):
        if self.state_regime == 'KQK_conv':
            if self.action_regime == 'KQK_pos_pos_piece':
                # action = np.reshape(action, (8,8,8,8,3))
                action_array = self.convert_int_to_action(action)
                # dont need to find the piece
                # should have exactly location of value 1 and everything else 0
                from_x, from_y, to_x, to_y, piece = np.where(action_array == 1)
                from_square = map_xy_to_square(from_x, from_y)
                to_square = map_xy_to_square(to_x, to_y)
                return chess.Move(from_square, to_square)
            elif self.action_regime == 'KQK_pos_pos':
                action_array = self.convert_int_to_action(action)
                from_x, from_y, to_x, to_y = np.where(action_array == 1)
                from_square = map_xy_to_square(from_x, from_y)
                to_square = map_xy_to_square(to_x, to_y)
                return chess.Move(from_square, to_square)


'''
Description of different regimes

REGIME 1: 'state_regime = KQK_conv AND action_regime = KQK_pos_pos'

states = {4x8x8}
actions = {8x8x8x8}
0th layer = WK
1st layer = WQ
2nd layer = BK
3rd layer = turn

REGIME 2: 'state_regime = KQK_conv AND action_regime = KQK_pos_pos_piece'

states = {4x8x8}
actions = {8x8x8x8x3 flattened}
0th layer = WK
1st layer = WQ
2nd layer = BK
3rd layer = turn

'''
