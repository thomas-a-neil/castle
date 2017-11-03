import unittest
import tensorflow as tf
import numpy as np

from dual_net import DualNet

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.net = DualNet(self.sess)
        self.piece_net = DualNet(self.sess, action_rep='piece')

        self.sess.__enter__()
        tf.global_variables_initializer().run()

        self.boards = np.random.random_sample([10, 8, 8, 13])
        self.z = np.random.random_sample([10])

    def test_predict(self):
        policy, value = self.net.predict(self.boards)
        self.assertEqual(policy.shape, (10, 64*64))

    def test_train(self):
        pi = np.random.random_sample([10, 64*64])
        # violates unit testing protocol, but w/e
        initial_loss = self.sess.run(self.net.loss,
                                     feed_dict={self.net.board_placeholder: self.boards,
                                                self.net.pi: pi,
                                                self.net.z: self.z})
        loss = self.net.train(self.boards, pi, self.z)
        self.assertGreater(initial_loss, loss)

    def test_train_with_piece_action(self):
        pi = np.random.random_sample([10, 32*64])

        initial_loss = self.sess.run([self.piece_net.loss],
                                     feed_dict={self.piece_net.board_placeholder: self.boards,
                                                self.piece_net.pi: pi,
                                                self.piece_net.z: self.z})
        loss = self.piece_net.train(self.boards, pi, self.z)
        self.assertGreater(initial_loss, loss)

    def test_regularization(self):
        pi = np.random.random_sample([10, 64*64])
        regularization_loss = self.sess.run(self.net.regularization_loss,
                                     feed_dict={self.net.board_placeholder: self.boards,
                                                self.net.pi: pi,
                                                self.net.z: self.z})
        self.assertGreater(regularization_loss, 0)

if __name__ == '__main__':
    unittest.main()
