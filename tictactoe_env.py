import numpy as np


class TicTacToeEnv(object):
    def __init__(self):
        """
        state space: 2x3x3
        action space: 18 possible values
            action: make actions a single integer -- in range(0, 18)
        assume x's always go first
        """
        self.action_dims = (2, 3, 3)

    def get_next_state(self, state, action_int):
        action_array = self.convert_int_to_action(action_int)
        next_state = state + action_array
        return next_state

    def get_legal_actions(self, state):
        turn = self.get_turn_tictactoe(state)
        legal_actions = []
        for i in range(3):
            for j in range(3):
                if state[:, i, j].sum() == 0:
                    action_array = np.zeros((2, 3, 3), dtype=int)
                    action_array[turn, i, j] = 1
                    action_int = self.convert_action_to_int(action_array)
                    legal_actions.append(action_int)
        return np.array(legal_actions)

    def is_game_over(self, state):
        """
        Returns True if the state indicates the game is over.
        False otherwise.
        """
        result = self.outcome(state)
        return not result == 2

    def outcome(self, state):
        """
        1 is a win for x's
        -1 is a win for o's
        0 is a tie
        2: the game is not over
        """
        for turn in range(2):
            if all([state[turn, 0, 0] == 1, state[turn, 1, 1] == 1, state[turn, 2, 2] == 1]):
                return self.convert_turn_to_winner(turn)
            if all([state[turn, 2, 0] == 1, state[turn, 1, 1] == 1, state[turn, 0, 2] == 1]):
                return self.convert_turn_to_winner(turn)
            for i in range(3):
                if state[turn, 0, i] == 1 and state[turn, 1, i] == 1 and state[turn, 2, i] == 1:
                    return self.convert_turn_to_winner(turn)
            for j in range(3):
                if state[turn, j, 0] == 1 and state[turn, j, 1] == 1 and state[turn, j, 2] == 1:
                    return self.convert_turn_to_winner(turn)
        if state.sum() >= 9:
            return 0
        # if we reach this point, the game is not over.  return 2
        return 2

    def get_turn_tictactoe(self, state):
        """
        return 0 if x's turn
        return 1 if o's turn
        """
        num_xs = state[0, :, :].sum()
        num_os = state[1, :, :].sum()
        if num_os == num_xs:
            return 0
        elif num_xs > num_os:
            return 1
        else:
            raise InvalidStateException(state)

    def print_board(self, state):
        for i in range(3):
            row = ''
            for j in range(3):
                if state[0, i, j] == 1:
                    row += 'x '
                elif state[1, i, j] == 1:
                    row += 'o '
                else:
                    row += '* '
            print(row)

    def convert_action_to_int(self, action_array):
        """
        action_array is 2x3x3
        """
        action_array = np.reshape(action_array, -1)
        return np.where(action_array == 1)[0]

    def convert_int_to_action(self, action_int):
        action_array = np.zeros((self.action_dims), dtype=int)
        action_array = np.reshape(action_array, -1)
        action_array[action_int] = 1
        action_array = np.reshape(action_array, (self.action_dims))
        return action_array

    def convert_turn_to_winner(self, turn):
        return (1 - turn) * 2 - 1


class InvalidStateException(Exception):
    pass
