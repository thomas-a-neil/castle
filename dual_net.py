import tensorflow as tf
import tensorflow.contrib.layers as layers


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

    policy_out = tf.nn.log_softmax(policy_out)

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

    def __init__(self, sess, learning_rate=0.01, representation=None,
                 regularization_mult=0.01, n_residual_layers=2,
                 input_shape=(8, 8, 13),
                 action_size=64*64,
                 num_convolutional_filters=256
                 ):
        """
        sess: tensorflow session
        representation: 'pos' for (pos1, pos2) action representation,
                        'piece' for (piece, pos) rep. If not None, will override
                        action_size.
        learning_rate: learning rate for gradient descent
        regularization_mult: multiplier for weight regularization loss
        n_residual_layers: how many residual layers to add, as described in
                           AlphaGo Zero.
        input_shape: a tuple describing the state input shape
        action_size: int describing size of action space
        num_convolutional_filters: how many convolutional filters to have in
                                   each convolutional layer
        """
        if representation == 'pos':
            action_size = 64*64
        elif representation == 'piece':
            action_size = 32*64

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

        self.policy_predict, self.value_predict = build_model(self.board_placeholder,
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

    def __call__(self, inp):
        """
        Gets a feed-forward prediction for a batch of input boards of shape [None, 8, 8, 13]
        """
        policy, value = self.sess.run([self.policy_predict, self.value_predict],
                                      feed_dict={self.board_placeholder: inp})
        return policy, value

    def train(self, boards, pi, z):
        """
        Performs one step of gradient descent based on a batch of input boards,
        MCTS policies, and rewards of shape [None, 1].  Shapes of inputs and policies
        should match input_shape and action_size as set during initialization.
        returns the batch loss
        """
        self.sess.run([self.update_op], feed_dict={self.board_placeholder: boards,
                                                   self.pi: pi,
                                                   self.z: z})
        loss = self.sess.run([self.loss], feed_dict={self.board_placeholder: boards,
                                                     self.pi: pi,
                                                     self.z: z})
        return loss
