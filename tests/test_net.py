import unittest
import tensorflow as tf
import numpy as np
import chess

from dual_net import (
    DualNet,
    move_to_index,
    board_to_state,
    state_to_board,
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

        #self.boards = np.random.random_sample([10, 8, 8, 13])
        self.boards = [board_to_state(chess.Board()) for _ in range(10)]
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

    def test_board_to_state(self):
        board = chess.Board()
        state = board_to_state(board)
        # Checks top two rows of the black pawn layer
        self.assertEqual(str(state[0:2,:,11]), '[[ 0.  0.  0.  0.  0.  0.  0.  0.]\n [ 1.  1.  1.  1.  1.  1.  1.  1.]]')

    def test_state_to_board(self):
        board = chess.Board()
        state = board_to_state(board)
        new_board = state_to_board(state)
        self.assertEqual(str(new_board), str(board))
    def test_predict(self):
        policy, value = self.net(self.boards)
        self.assertEqual(policy.shape, (10, 64*64))
        # Check that move a1->a1 is illegal
        self.assertEqual(policy[0, 0], 0)
        # Check that move a2->a3 is legal
        self.assertNotEqual(policy[0, 528], 0)

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

    def test_move_to_index(self):
        uci_move = chess.Move.from_uci('a1b1')
        index = move_to_index(uci_move)
        self.assertEqual(index, 1)

        uci_move = chess.Move.from_uci('a2a3')
        index = move_to_index(uci_move)
        self.assertEqual(index, 528)


if __name__ == '__main__':
    unittest.main()
