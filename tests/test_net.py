import unittest
import tensorflow as tf
import numpy as np

from dual_net import DualNet


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.net = DualNet(self.sess)
        self.piece_net = DualNet(self.sess, representation='piece')

        self.sess.__enter__()
        tf.global_variables_initializer().run()

        self.boards = np.random.random_sample([10, 8, 8, 13])
        self.z = np.random.random_sample([10])

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
