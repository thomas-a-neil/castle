import unittest
import tensorflow as tf
import numpy as np

import dual_net
from dual_net import DualNet


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.net = DualNet(self.sess)
        self.piece_net = DualNet(self.sess, action_size=dual_net.PIECE_POSITION_ACTION_SIZE)

        self.sess.__enter__()
        tf.global_variables_initializer().run()

        self.boards = np.random.random_sample([10, 8, 8, 13])
        self.z = np.random.random_sample([10])

    def test_predict(self):
        sess = tf.Session()
        net = DualNet(sess)
        sess.__enter__()
        tf.global_variables_initializer().run()
        policy, value = net(self.boards)
        self.assertEqual(policy.shape, (10, 64*64))

    def test_predict_with_piece_action(self):
        sess = tf.Session()
        piece_net = DualNet(sess, representation='piece')
        sess.__enter__()
        tf.global_variables_initializer().run()
        policy, value = piece_net(self.boards)
        self.assertEqual(policy.shape, (10, 32*64))

    def test_regularization(self):
        sess = tf.Session()
        net = DualNet(sess)
        sess.__enter__()
        tf.global_variables_initializer().run()
        pi = np.random.random_sample([10, 64*64])
        regularization_loss = sess.run(net.regularization_loss,
                                       feed_dict={net.board_placeholder: self.boards,
                                                  net.pi: pi,
                                                  net.z: self.z})
        self.assertGreater(regularization_loss, 0)

if __name__ == '__main__':
    unittest.main()
