import unittest
import tensorflow as tf
import numpy as np

from dual_net import (
    DualNet,
    PIECE_POSITION_ACTION_SIZE,
    KQK_POSITION_POSITION_PIECE_ACTION_SIZE,
    KQK_CHESS_INPUT_SHAPE)


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.net = DualNet(self.sess)
        self.piece_net = DualNet(self.sess, action_size=PIECE_POSITION_ACTION_SIZE)

        self.sess.__enter__()
        tf.global_variables_initializer().run()

        self.boards = np.random.random_sample([10, 8, 8, 13])
        self.z = np.random.random_sample([10])

    def test_KQK(self):
        states = np.random.random_sample([10, 8, 8, 4])

        sess = tf.Session()
        net = DualNet(sess, input_shape=KQK_CHESS_INPUT_SHAPE, action_size=KQK_POSITION_POSITION_PIECE_ACTION_SIZE)
        sess.__enter__()
        tf.global_variables_initializer().run()

        policy, value = net(states)
        self.assertEqual(policy.shape, (10, 64*64*3))
        self.assertEqual(value.size, 10)

    def test_predict(self):
        policy, value = self.net(self.boards)
        self.assertEqual(policy.shape, (10, 64*64))

    def test_predict_with_piece_action(self):
        policy, value = self.piece_net(self.boards)
        self.assertEqual(policy.shape, (10, 32*64))

    def test_regularization(self):
        pi = np.random.random_sample([10, 64*64])
        regularization_loss = self.sess.run(self.net.regularization_loss,
                                            feed_dict={self.net.board_placeholder: self.boards,
                                                       self.net.pi: pi,
                                                       self.net.z: self.z})
        self.assertGreater(regularization_loss, 0)

if __name__ == '__main__':
    unittest.main()
