import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import chess

from chess_env import FULL_CHESS_INPUT_SHAPE, POSITION_POSITION_ACTION_SIZE


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


def build_model(board_placeholder,
                legality_mask_placeholder,
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

    x = tf.exp(policy_out) * legality_mask_placeholder
    # Needs reshape to broadcast properly
    policy_out = x / tf.reshape(tf.reduce_sum(x, axis=1), shape=((tf.shape(x)[0],) + (1,)))

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


class DualNet(object):

    def __init__(self,
                 sess,
                 env,
                 learning_rate=0.01,
                 regularization_mult=0.01,
                 n_residual_layers=2,
                 input_shape=FULL_CHESS_INPUT_SHAPE,
                 action_size=POSITION_POSITION_ACTION_SIZE,
                 num_convolutional_filters=256
                 ):
        """
        sess: tensorflow session
        env: environment to determine move legality with
        learning_rate: learning rate for gradient descent
        regularization_mult: multiplier for weight regularization loss
        n_residual_layers: how many residual layers to add, as described in
                           AlphaGo Zero.
        num_convolutional_filters: how many convolutional filters to have in
                                   each convolutional layer
        """
        self.action_size = env.action_size
        self.board_placeholder = tf.placeholder(tf.float32, [None] + list(env.input_shape))
        self.env = env

        # shared_layers = [{'layer': 'conv', 'num_outputs':
        #                   num_convolutional_filters, 'stride': 3,
        #                   'kernel_size': 1, 'activation_fn': tf.nn.relu}]
        # # add n_residual_layers to the shared layers
        # shared_layers += n_residual_layers*[{'layer': 'residual',
        #                                      'num_outputs': num_convolutional_filters,
        #                                      'stride': 1, 'kernel_size': 3,
        #                                      'activation_fn': tf.nn.relu}]

        # policy_layers = [{'layer': 'conv', 'num_outputs': 2, 'stride': 1,
        #                   'kernel_size': 1, 'activation_fn': tf.nn.relu},
        #                  {'layer': 'fc', 'num_outputs': self.action_size,
        #                   'activation_fn': None}]
        # value_layers = [{'layer': 'conv', 'num_outputs': 1, 'stride': 1,
        #                  'kernel_size': 1, 'activation_fn': tf.nn.relu},
        #                 {'layer': 'fc', 'num_outputs': num_convolutional_filters,
        #                  'activation_fn': tf.nn.relu},
        #                 {'layer': 'fc', 'num_outputs': 1,
        #                  'activation_fn': tf.nn.tanh}]

        # Sunday experiment noon to 3pm
        shared_layers = []
        policy_layers = [{'layer': 'fc', 'num_outputs': 30,
                         'activation_fn': tf.nn.relu},
                         {'layer': 'fc', 'num_outputs': self.action_size,
                          'activation_fn': None}]
        value_layers = [{'layer': 'fc', 'num_outputs': 8,
                         'activation_fn': tf.nn.relu},
                        {'layer': 'fc', 'num_outputs': 1,
                         'activation_fn': tf.nn.tanh}]

        # Sunday experiment noon to 3pm
        # shared_layers = [{'layer': 'conv', 'num_outputs': num_convolutional_filters, 'stride': 1,
        #                   'kernel_size': 2, 'activation_fn': tf.nn.relu},
        #                   {'layer': 'conv', 'num_outputs': num_convolutional_filters, 'stride': 1,
        #                   'kernel_size': 2, 'activation_fn': tf.nn.relu}]
        # policy_layers = [{'layer': 'fc', 'num_outputs': self.action_size,
        #                   'activation_fn': None}]
        # value_layers = [{'layer': 'fc', 'num_outputs': 1,
        #                  'activation_fn': tf.nn.tanh}]

        self.boards = None
        self.move_legality_mask = tf.placeholder(tf.float32, [None, self.action_size])
        self.policy_predict, self.value_predict = build_model(self.board_placeholder,
                                                              self.move_legality_mask,
                                                              scope='net',
                                                              shared_layers=shared_layers,
                                                              policy_head=policy_layers,
                                                              value_head=value_layers)
        self.z = tf.placeholder(tf.float32, [None])
        # Reshape z for proper broadcasting
        reshaped_z = tf.reshape(self.z, [tf.shape(self.z)[0], 1])
        self.pi = tf.placeholder(tf.float32, [None, self.action_size])
        self.value_diff = self.value_predict - reshaped_z
        self.value_loss = tf.reduce_sum(tf.square(self.value_diff))

        # when the 0s become 0.000001s for illegal actions, we are counting on the fact that the are
        # nullified by the corresponding index of self.pi to be 0
        self.policy_loss = -tf.reduce_sum(tf.multiply(self.pi, tf.log(self.policy_predict + 0.0001)))

        self.regularization_loss = layers.apply_regularization(layers.l2_regularizer(regularization_mult),
                                                               weights_list=tf.trainable_variables())
        self.loss = self.value_loss + self.policy_loss + tf.reduce_sum(self.regularization_loss)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.value_update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.value_loss)
        self.sess = sess

    def __call__(self, inp):
        """
        Gets a feed-forward prediction for a batch of input boards of shape set
        during initialization.
        """
        expanded = False
        if inp.ndim == len(self.env.input_shape):
          # our input is not a batch.  it is a single state
          inp = np.expand_dims(inp, axis=0)
          expanded = True
        move_legality_mask = np.zeros(shape=(inp.shape[0], self.action_size))

        for i in range(inp.shape[0]):
            move_legality_mask[i] = self.env.get_legality_mask(inp[i])
        policy, value = self.sess.run([self.policy_predict, self.value_predict],
                                      feed_dict={self.board_placeholder: inp,
                                                 self.move_legality_mask: move_legality_mask})
        if expanded:
          policy = policy[0]
          value = value[0]
        return policy, value

    def train(self, states, pi, z, token_legality_mask=None):
        """
        Performs one step of gradient descent based on a batch of input boards,
        MCTS policies, and rewards of shape [None, 1].  Shapes of inputs and policies
        should match input_shape and action_size as set during initialization.
        returns the batch loss

        The token_legality_mask is just for test purposes so we can input a mask of our choosing
        Otherwise, it gets the legality_mask from the environment
        """
        if token_legality_mask is None:
          move_legality_mask = np.zeros(shape=(len(states), self.action_size))
          for i in range(len(states)):
              move_legality_mask[i] = self.env.get_legality_mask(states[i])
        else:
          move_legality_mask = token_legality_mask
        _, loss, value_loss, policy_loss, value_predict, policy_predict = self.sess.run([self.update_op, self.loss, 
                  self.value_loss, self.policy_loss, self.value_predict, self.policy_predict], 
                  feed_dict={self.board_placeholder: states,
                                                   self.pi: pi,
                                                   self.z: z,
                                                   self.move_legality_mask: move_legality_mask})

        return loss, value_loss, policy_loss, value_predict, policy_predict

    def train_value(self, states, z, token_legality_mask=None):
      if token_legality_mask is None:
        move_legality_mask = np.zeros(shape=(len(states), self.action_size))
        for i in range(len(states)):
            move_legality_mask[i] = self.env.get_legality_mask(states[i])
      else:
        move_legality_mask = token_legality_mask
      _, value_loss, value_predict = self.sess.run([self.value_update_op,
         self.value_loss, self.value_predict], feed_dict={self.board_placeholder: states,
                                                 self.z: z,
                                                 self.move_legality_mask: move_legality_mask})

      return value_loss, value_predict

    def test_value(self, states, z, token_legality_mask=None):
      if token_legality_mask is None:
        move_legality_mask = np.zeros(shape=(len(states), self.action_size))
        for i in range(len(states)):
            move_legality_mask[i] = self.env.get_legality_mask(states[i])
      else:
        move_legality_mask = token_legality_mask
      value_loss = self.sess.run([self.value_loss], 
                                                 feed_dict={self.board_placeholder: states,
                                                 self.z: z,
                                                 self.move_legality_mask: move_legality_mask})

      return value_loss

    def classify_value(self, states, z):
      '''
      This is pseudo-classification
      We simply
      '''
      move_legality_mask = np.zeros(shape=(len(states), self.action_size))
      for i in range(len(states)):
        move_legality_mask[i] = self.env.get_legality_mask(states[i])
      value_loss, value_predict = self.sess.run([self.value_loss, self.value_predict], 
                                                 feed_dict={self.board_placeholder: states,
                                                 self.z: z,
                                                 self.move_legality_mask: move_legality_mask})
      value_guesses = np.around(value_predict)
      total_correct = 0
      total = value_guesses.size
      for i in range(total):
        if z[i] == value_guesses[i]:
          total_correct += 1
      accuracy = total_correct / float(total)
      return value_guesses, accuracy
