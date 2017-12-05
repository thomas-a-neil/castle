import numpy as np


class TicTacToeEnv(object):
    def __init__(self):
        """
        state space: 2x3x3
        action space: 18 possible values
            action: make actions a single integer -- in range(0, 18)
        assume x's always go first
        """
        self.action_dims = (3, 3)
        self.action_size = int(np.prod(self.action_dims))
        self.input_shape = (2, 3, 3)
        self.start_state = self.get_start_state()

    def get_start_state(self):
        return np.zeros(self.input_shape)

    def get_next_state(self, state, action_int):
        action_array = self.convert_int_to_action(action_int)
        next_state = np.zeros(self.input_shape)
        next_state[1,:] +=  action_array
        next_state[0,:] += state[1,:]
        next_state[1,:] += state[0,:]
        return next_state

    def sample_invariant_transformation(self, state):
        '''
        flip rotate
        '''
        rot1 = np.rot90(state, axes=(1,2))
        rot2 = np.rot90(rot1, axes=(1,2))
        rot3 = np.rot90(rot2, axes=(1,2))
        flip_x = np.flip(state, 1)
        flip_y = np.flip(state, 2)

        transformations = [rot1, rot2, rot3, flip_x, flip_y]
        choice = np.random.choice(len(transformations))
        transform_type = choice
        return transformations, transformations[choice], transform_type

    def revert_transform(self, action_array, transform_index):
        # action_array = self.convert_int_to_action(action_int)
        rot1 = np.rot90(action_array, axes=(0,1))
        rot2 = np.rot90(rot1, axes=(0,1))
        rot3 = np.rot90(rot2, axes=(0,1))
        flip_x = np.flip(action_array, 0)
        flip_y = np.flip(action_array, 1)
        transformations = [rot3, rot2, rot1, flip_x, flip_y]
        return transformations[transform_index]

    def get_canonical_actions(self, state):
        if state.sum() == 0:
            return np.array([0, 1, 4])
        elif state.sum() == 1:
            if state[1,0,0] == 1:
                return np.array([1, 2, 4, 5, 8])
            elif state[1,0,1] == 1:
                return np.array([1, 3, 4, 6, 7])
            elif state[1,1,1] == 1:
                return np.array([0, 1])
        else:
            return self.get_legal_actions(state)

    def get_canonical_mask(self, state):
        canon_actions = np.zeros(self.action_size)
        canon_actions_list = self.get_canonical_actions(state)
        for j in canon_actions_list:
            canon_actions[j] = 1
        return np.reshape(canon_actions, -1)

    def get_legality_mask(self, state):
        legal_actions = np.zeros(self.action_dims)
        for i in range(3):
            for j in range(3):
                if state[:, i, j].sum() == 0:
                    legal_actions[i, j] = 1
        return np.reshape(legal_actions, -1)

    def get_legal_actions(self, state):
        turn_index = 0 if self.is_x_turn(state) else 1
        legal_actions = []
        for i in range(3):
            for j in range(3):
                if state[:, i, j].sum() == 0:
                    action_array = np.zeros(self.action_dims, dtype=int)
                    action_array[i, j] = 1
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

        only the current player can lose: check this
        """
        x_turn = 1 if self.is_x_turn(state) else 0
        other_index = 1
        if all([state[other_index, 0, 0] == 1, state[other_index, 1, 1] == 1, state[other_index, 2, 2] == 1]):
            return self.convert_turn_to_winner(x_turn)
        if all([state[other_index, 2, 0] == 1, state[other_index, 1, 1] == 1, state[other_index, 0, 2] == 1]):
            return self.convert_turn_to_winner(x_turn)
        for i in range(3):
            if state[other_index, 0, i] == 1 and state[other_index, 1, i] == 1 and state[other_index, 2, i] == 1:
                return self.convert_turn_to_winner(x_turn)
        for j in range(3):
            if state[other_index, j, 0] == 1 and state[other_index, j, 1] == 1 and state[other_index, j, 2] == 1:
                return self.convert_turn_to_winner(x_turn)
            
        if state.sum() >= 9:
            return 0
        # if we reach this point, the game is not over.  return 2
        return 2

    def is_x_turn(self, state):
        """
        Return True if x's turn. False if o's turn.
        """
        num_curr_stones = state[0, :, :].sum()
        num_other_stones = state[1, :, :].sum()
        if num_curr_stones == num_other_stones - 1:
            return False
        elif num_curr_stones == num_other_stones:
            return True
        else:
            raise InvalidStateException(state)

    def print_board(self, state):
        print('-----')
        if self.is_x_turn(state):
            x_index = 0
        else:
            x_index = 1
        o_index = 1 - x_index
        for i in range(3):
            row = ''
            for j in range(3):
                if state[x_index, i, j] == 1:
                    row += 'x '
                elif state[o_index, i, j] == 1:
                    row += 'o '
                else:
                    row += '* '
            print(row)

    def convert_action_to_int(self, action_array):
        """
        Return an integer representing the index of the chosen action
        in the dimension of space 2x3x3.
        """
        action_array = np.reshape(action_array, -1)
        return np.argmax(action_array)

    def convert_int_to_action(self, action_int):
        """
        Convert the action index (an integer) into the
        array representation of the action
        """
        action_array = np.zeros((self.action_dims), dtype=int)
        action_array = np.reshape(action_array, -1)
        action_array[action_int] = 1
        action_array = np.reshape(action_array, (self.action_dims))
        return action_array

    def convert_turn_to_winner(self, turn):
        '''
            turn   winner
        x.   0.        1   
        o.   1.       -1
        '''
        return (1 - turn) * 2 - 1


class InvalidStateException(Exception):
    pass
