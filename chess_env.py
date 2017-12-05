import os

import numpy as np
import chess
import chess.uci

PATH_TO_STOCKFISH_EXE = os.path.expanduser('~/stockfish-8-mac/Mac/stockfish-8-64')

# Map from piece to layer in net input.  0-5 are white.
INDEX_TO_PIECE_MAP = {0: chess.KING, 6: chess.KING,
                      1: chess.QUEEN, 7: chess.QUEEN,
                      2: chess.ROOK, 8: chess.ROOK,
                      3: chess.BISHOP, 9: chess.BISHOP,
                      4: chess.KNIGHT, 10: chess.KNIGHT,
                      5: chess.PAWN, 11: chess.PAWN}

CHAR_TO_INDEX_MAP = {'K': 0, 'k': 6,
                     'Q': 1, 'q': 7,
                     'R': 2, 'r': 8,
                     'B': 3, 'b': 9,
                     'N': 4, 'n': 10,
                     'P': 5, 'p': 11}

FULL_CHESS_INPUT_SHAPE = (8, 8, 13)
PIECE_POSITION_ACTION_SIZE = 32 * 64
POSITION_POSITION_ACTION_SIZE = 64 * 64


def map_xy_to_square(x, y):
    return int(8*y + x)


def map_square_to_xy(square):
    return int(square % 8), int(square // 8)


class ChessEnv(object):
    """
    The full chess environment.
    """
    def __init__(self, input_shape=FULL_CHESS_INPUT_SHAPE, action_size=POSITION_POSITION_ACTION_SIZE):
        self.action_size = action_size
        self.input_shape = input_shape

    def reset(self):
        board = chess.Board()
        state = self.map_board_to_state(board)
        return state

    def get_next_state(self, state, action):
        board = self.map_state_to_board(state)
        move = self.map_action_to_move(action)
        board.push(move)
        next_state = self.map_board_to_state(board)
        return next_state

    def get_legal_actions(self, state):
        board = self.map_state_to_board(state)
        legal_moves = list(board.legal_moves)
        legal_actions = np.zeros(len(legal_moves), dtype=int)
        i = 0
        for move in legal_moves:
            action = self.map_move_to_action(move)
            legal_actions[i] = action
            i += 1
        return legal_actions

    def get_legality_mask(self, state):
        board = self.map_state_to_board(state)
        legal_moves = board.legal_moves
        legal_moves_as_indices = [self.move_to_index(board, move) for move in legal_moves]
        move_legality_mask = np.zeros(self.action_size)
        for index in legal_moves_as_indices:
            move_legality_mask[index] = 1
        return move_legality_mask

    def is_game_over(self, state):
        board = self.map_state_to_board(state)
        return board.is_game_over()

    def outcome(self, state):
        board = self.map_state_to_board(state)
        if board.result() == '1/2-1/2':
            result = 0
        elif board.result() == '1-0':
            result = 1
        elif board.result() == '0-1':
            result = -1
        return result

    def board_str(self, state):
        board = self.map_state_to_board(state)
        return str(board)

    def print_board(self, state):
        board = self.map_state_to_board(state)
        print(board)

    def map_board_to_state(self, board):
        """
        Translates a chess.Board() object to an input state according to the
        full Chess representation.  Should be moved to the chess env
        once that exists.

        Parameters
        ----------
        board: a chess.Board

        Returns the state corresponding do the board
        """
        board_string = str(board)
        rows = board_string.split('\n')
        state = np.zeros(shape=(8, 8, 13))
        for i in range(8):
            row = rows[i]
            pieces = row.split(' ')
            for j in range(8):
                char = pieces[j]
                if char == '.':
                    continue
                state[i][j][CHAR_TO_INDEX_MAP[char]] = 1
        if board.turn:
            state[:, :, 12] = np.ones(shape=(8, 8))
        return state

    def map_state_to_board(self, state):
        """
        Transforms a state representation according to the full Chess board input
        into its chess Board equivalent.
        Parameters
        ----------
        state: a numpy object representing the input board

        Returns a chess.Board object
        """
        pieces = {}
        for i in range(12):
            piece = INDEX_TO_PIECE_MAP[i]
            if i < 6:
                color = chess.WHITE
            else:
                color = chess.BLACK
            indices = np.argwhere(state[:, :, i] == 1)
            squares = []
            for coords in indices:
                x, y = coords
                squares.append(8 * (7 - x) + y)
            for square in squares:
                pieces[square] = chess.Piece(piece, color)
        board = chess.Board()
        board.set_piece_map(pieces)
        board.turn = int(state[0, 0, 12])
        return board

    def map_action_to_move(self, action):
        index = action
        from_pos = int(index / 64)
        to_pos = index % 64
        return chess.Move(from_pos, to_pos)

    def map_move_to_action(self, move):
        index = self.move_to_index(move)
        return index

    def move_to_index(self, move):
        """
        Translates a chess move to the appropriate index in the action space.
        Parameters
        ----------
        move: chess.Move instance

        Returns the index into the action space
        """
        uci = move.uci()
        from_pos = self.position_to_index(uci[:2])
        to_pos = self.position_to_index(uci[2:])
        if self.action_size == POSITION_POSITION_ACTION_SIZE:
            return 64 * from_pos + to_pos
        elif self.action_size == PIECE_POSITION_ACTION_SIZE:
            raise NotImplementedError

    def position_to_index(self, position):
        """
        Translates a position to the appropriate index in the state space.
        Parameters
        ----------
        move: uci string, such as b2 or g1

        Returns the index
        """
        col = position[0]
        row = position[1]
        col_number = ord(col) - ord('a')
        row_number = int(row) - 1
        index = 8 * row_number + col_number
        return index


def score(board, movetime=1000, engine_path=PATH_TO_STOCKFISH_EXE):
    """
    Returns the engine evaluation of board state. board should be a chess.Board

    >>> board = chess.Board()
    >>> score(board)
    0.17
    """
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine(engine_path)

    engine.info_handlers.append(handler)
    engine.position(board)

    engine.go(searchmoves=board.legal_moves, movetime=movetime)

    return handler.info["score"][1].cp / 100.0
