import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import chess

FULL_CHESS_INPUT_SHAPE = (8, 8, 13)
KQK_CHESS_INPUT_SHAPE = (8, 8, 4)

POSITION_POSITION_ACTION_SIZE = 64 * 64
KQK_POSITION_POSITION_PIECE_ACTION_SIZE = 64 * 64 * 3
PIECE_POSITION_ACTION_SIZE = 32 * 64


# A convolutional block as described in AlphaGo Zero
def conv_block(tensor, specs):
    tensor = layers.convolution2d(tensor,
                                  num_outputs=specs['num_outputs'],
                                  kernel_size=specs['kernel_size'],
                                  stride=specs['stride'],
                                  activation_fn=None)
    tensor = tf.layers.batch_normalization(tensor)
    tensor = tf.nn.relu(tensor)
    return tensor


# A residual block as described in AlphaGo Zero
def residual_block(tensor, specs):
    input_tensor = tensor
    tensor = conv_block(tensor, specs)
    tensor = layers.convolution2d(tensor,
                                  num_outputs=specs['num_outputs'],
                                  kernel_size=specs['kernel_size'],
                                  stride=specs['stride'],
                                  activation_fn=None)
    tensor = tf.layers.batch_normalization(tensor)
    tensor += input_tensor
    tensor = tf.nn.relu(tensor)
    return tensor




class DualNet(object):

    def __init__(self,
                 sess,
                 learning_rate=0.01,
                 regularization_mult=0.01,
                 n_residual_layers=2,
                 input_shape=FULL_CHESS_INPUT_SHAPE,
                 action_size=POSITION_POSITION_ACTION_SIZE,
                 num_convolutional_filters=256
                 ):
        """
        sess: tensorflow session
        learning_rate: learning rate for gradient descent
        regularization_mult: multiplier for weight regularization loss
        n_residual_layers: how many residual layers to add, as described in
                           AlphaGo Zero.
        input_shape: a tuple describing the state input shape
        action_size: int describing size of action space
        num_convolutional_filters: how many convolutional filters to have in
                                   each convolutional layer
        """
        self.action_size = action_size
        self.board_placeholder = tf.placeholder(tf.float32, [None] + list(input_shape))

        shared_layers = [{'layer': 'conv', 'num_outputs':
                          num_convolutional_filters, 'stride': 3,
                          'kernel_size': 1, 'activation_fn': tf.nn.relu}]
        # add n_residual_layers to the shared layers
        shared_layers += n_residual_layers*[{'layer': 'residual',
                                             'num_outputs': num_convolutional_filters,
                                             'stride': 1, 'kernel_size': 3,
                                             'activation_fn': tf.nn.relu}]

        policy_layers = [{'layer': 'conv', 'num_outputs': 2, 'stride': 1,
                          'kernel_size': 1, 'activation_fn': tf.nn.relu},
                         {'layer': 'fc', 'num_outputs': action_size,
                          'activation_fn': None}]
        value_layers = [{'layer': 'conv', 'num_outputs': 1, 'stride': 1,
                         'kernel_size': 1, 'activation_fn': tf.nn.relu},
                        {'layer': 'fc', 'num_outputs': num_convolutional_filters,
                         'activation_fn': tf.nn.relu},
                        {'layer': 'fc', 'num_outputs': 1,
                         'activation_fn': tf.nn.tanh}]

        self.boards = None
        self.move_indices_mask = tf.placeholder(tf.int32, [None, None, 1])
        self.move_legality_mask = tf.placeholder(tf.bool, [None, self.action_size])
        self.policy_predict, self.value_predict = self.build_model(self.board_placeholder,
                                                              self.move_legality_mask,
                                                              self.move_indices_mask,
                                                              scope='net',
                                                              shared_layers=shared_layers,
                                                              policy_head=policy_layers,
                                                              value_head=value_layers)
        self.z = tf.placeholder(tf.float32, [None])
        self.pi = tf.placeholder(tf.float32, [None, action_size])
        self.value_loss = tf.reduce_sum(tf.square(self.value_predict - self.z))
        self.policy_loss = tf.reduce_sum(tf.multiply(self.pi, self.policy_predict))
        self.regularization_loss = layers.apply_regularization(layers.l2_regularizer(regularization_mult),
                                                               weights_list=tf.trainable_variables())
        self.loss = self.value_loss - self.policy_loss + tf.reduce_sum(self.regularization_loss)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess = sess


    def build_model(self,
                    board_placeholder,
                    legality_mask_placeholder,
                    legality_indices_placeholder,
                    scope,
                    shared_layers,
                    policy_head,
                    value_head):
        """
        Returns the output tensors for an model based on the layers in shared_layers,
        policy_head, and value_head.
        shared_layers is a list of dicts, each dict representing a layer.
        - Convolutional layers:
           - d['layer'] <- 'conv'
           - d['num_outputs'], d['kernel_size'], d['stride'] should be ints
           - d['activation_fn'] is a function
        - Residual layers:
           - d['layer'] <- 'residual'
           - other keys same as convolutional
        - Fully connected layers:
           - d['layer'] <- 'fc'
           - d['num_outputs'] is an int
           - d['activation_fn'] is a function
        policy_head and value_head have the same structure as above but represent
        the layers for the policy head and value head, respectively.

        returns the policy output and the value output in a tuple
        """
        out = board_placeholder
        for specs in shared_layers:
            if specs['layer'] == 'conv':
                out = conv_block(out, specs)
            elif specs['layer'] == 'residual':
                out = residual_block(out, specs)
            elif specs['layer'] == 'fc':
                out = layers.flatten(out)
                out = layers.fully_connected(out,
                                             num_outputs=specs['num_outputs'],
                                             activation_fn=specs['activation_fn'])
        # Policy head
        policy_out = out
        for specs in policy_head:
            if specs['layer'] == 'conv':
                policy_out = conv_block(policy_out, specs)
            elif specs['layer'] == 'fc':
                policy_out = layers.flatten(policy_out)
                policy_out = layers.fully_connected(policy_out,
                                                    num_outputs=specs['num_outputs'],
                                                    activation_fn=specs['activation_fn'])

        #should only be a function of tf variables
        #[10, 4096, 1] and [10, 20ish, 1]
        policy_out_extra_dim = tf.reshape(policy_out, shape=(tf.shape(policy_out) + (1,)))
        legal_move_logits = tf.gather_nd(policy_out_extra_dim, legality_indices_placeholder) #TODO not correct operation
        normalized_legal_move_logits = tf.nn.log_softmax(legal_move_logits)#TODO only softmax over each row
        zeros = tf.zeros([tf.shape(policy_out)[0], self.action_size])
        policy_out = tf.where(legality_mask_placeholder, normalized_legal_move_logits, zeros) #TODO: need same dim


        # Value head
        value_out = out
        for specs in value_head:
            if specs['layer'] == 'conv':
                value_out = conv_block(value_out, specs)
            elif specs['layer'] == 'fc':
                value_out = layers.flatten(value_out)
                value_out = layers.fully_connected(value_out,
                                                   num_outputs=specs['num_outputs'],
                                                   activation_fn=specs['activation_fn'])
        return policy_out, value_out


    def __call__(self, inp):
        """
        Gets a feed-forward prediction for a batch of input boards of shape set
        during initialization.
        """
        boards = [state_to_board(board) for board in inp]
        all_legal_moves = [board.legal_moves for board in boards]
        legal_moves_as_indices = []
        for legal_moves in all_legal_moves:
            moves = []
            for move in legal_moves:
                moves.append(move_to_index(move))
            legal_moves_as_indices.append(moves)
        move_indices_mask = []
        move_legality_mask = np.zeros(shape=(len(boards), self.action_size))
        for i in range(len(legal_moves_as_indices)):
            move_indices_mask.append(np.array(legal_moves_as_indices[i]))
            for move in legal_moves_as_indices[i]:
                move_legality_mask[i, move] = 1
        move_indices_mask = np.array(move_indices_mask)
        policy, value = self.sess.run([self.policy_predict, self.value_predict],
                                      feed_dict={self.board_placeholder: inp,
                                                 self.move_legality_mask: move_legality_mask,
                                                 self.move_indices_mask: move_indices_mask})
        return policy, value

    def train(self, boards, pi, z):
        """
        Performs one step of gradient descent based on a batch of input boards,
        MCTS policies, and rewards of shape [None, 1].  Shapes of inputs and policies
        should match input_shape and action_size as set during initialization.
        returns the batch loss
        """
        self.boards = [state_to_board(board) for board in boards]
        self.sess.run([self.update_op], feed_dict={self.board_placeholder: boards,
                                                   self.pi: pi,
                                                   self.z: z})
        loss = self.sess.run([self.loss], feed_dict={self.board_placeholder: boards,
                                                     self.pi: pi,
                                                     self.z: z})
        return loss

INDEX_TO_PIECE_MAP = {0: chess.KING, 6: chess.KING,
                      1: chess.QUEEN, 7: chess.QUEEN,
                      2: chess.ROOK, 8: chess.ROOK,
                      3: chess.BISHOP, 9: chess.BISHOP,
                      4: chess.KNIGHT, 10: chess.KNIGHT,
                      5: chess.PAWN, 11: chess.PAWN}

KQK_INDEX_TO_PIECE_MAP = {0: chess.KING,
                          1: chess.QUEEN,
                          2: chess.KING}

CHAR_TO_INDEX_MAP = {'K': 0, 'k': 6,
                     'Q': 1, 'q': 7,
                     'R': 2, 'r': 8,
                     'B': 3, 'b': 9,
                     'N': 4, 'n': 10,
                     'P': 5, 'p': 11}


def state_to_board(state):
    """
    Parameters
    ----------
    state: a numpy object representing the input board

    Returns a chess.Board object
    """
    pieces = {}
    if state.shape[2] == 13:
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
    else:
        for i in range(3):
            piece = KQK_INDEX_TO_PIECE_MAP[i]
            if i < 3:
                color = chess.WHITE
            else:
                color = chess.BLACK
            indices = np.argwhere(state[:, :, i] == 1)
            squares = []
            for coords in indices:
                x, y = coords
                squares.append(8 * y + x)
            for square in squares:
                pieces[square] = chess.Piece(piece, color)
        board = chess.Board()
        board.set_piece_map(pieces)
        board.turn = int(state[0, 0, 3])
        return board


def board_to_state(board):
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
        state[:,:,12] = np.ones(shape=(8,8))
    return state

def move_to_index(move):
    """
    Parameters
    ----------
    move: chess.Move instance

    Returns the index into the action space
    """
    uci = move.uci()
    from_pos = position_to_index(uci[:2])
    to_pos = position_to_index(uci[2:])
    return 64 * from_pos + to_pos

def position_to_index(position):
    col = position[0]
    row = position[1]
    col_number = ord(col) - ord('a')
    row_number = int(row) - 1
    index = 8 * row_number + col_number
    return index
