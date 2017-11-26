import unittest
import tensorflow as tf
import numpy as np
import chess

from dual_net import (
    DualNet,
    )

from chess_env import (
    ChessEnv,
    PIECE_POSITION_ACTION_SIZE,
    POSITION_POSITION_ACTION_SIZE,
    FULL_CHESS_INPUT_SHAPE,
)

from kqk_chess_env import (
    KQKChessEnv,
    KQK_POSITION_POSITION_PIECE_ACTION_SIZE,
    KQK_CHESS_INPUT_SHAPE,
)



class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()
        self.sess = tf.Session()
        self.net = DualNet(self.sess, self.env)
        self.piece_net = DualNet(self.sess, KQKChessEnv('KQK_conv', 'KQK_pos_pos_piece'))

        self.sess.__enter__()
        tf.global_variables_initializer().run()

        self.boards = [self.env.map_board_to_state(chess.Board()) for _ in range(10)]
        self.states = np.zeros(shape=(10,) + FULL_CHESS_INPUT_SHAPE)
        for i in range(10):
            self.states[i] = self.boards[i]
        self.z = np.random.random_sample([10])

    def test_KQK(self):
        self.env = KQKChessEnv('KQK_conv', 'KQK_pos_pos')
        board = chess.Board()
        board.set_piece_map({0: chess.Piece.from_symbol('K'), 8: chess.Piece.from_symbol('Q'), 41: chess.Piece.from_symbol('k')})
        self.boards = [self.env.map_board_to_state(board) for _ in range(10)]

        sess = tf.Session()
        net = DualNet(sess, self.env)
        sess.__enter__()
        tf.global_variables_initializer().run()

        states = np.zeros(shape=(10, 8, 8, 4))
        for i in range(10):
            states[i] = self.boards[i]

        policy, value = net(states)
        self.assertEqual(policy.shape, (10, 64*64))
        self.assertEqual(value.size, 10)


    def test_predict(self):
        policy, value = self.net(self.states)
        self.assertEqual(policy.shape, (10, 64*64))
        # Check that move a1->a1 is illegal
        self.assertEqual(policy[0, 0], 0)
        # Check that move a2->a3 is legal
        self.assertNotEqual(policy[0, 528], 0)

    def test_predict_with_piece_action(self):
        policy, value = self.piece_net(self.states)
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
