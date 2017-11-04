from functools import partial
import unittest

from mcts import backup, select, expand_node, exploration_bonus_for_c_puct, perform_rollouts, get_action_distribution
from tree import Node

from utils import setup_simple_tree, mock_model, mock_env


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.nodes = setup_simple_tree()

    def test_backup(self):
        edge = self.nodes[1].in_edge
        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 0)
        self.assertEqual(edge.total_action_value, 0.0)
        self.assertEqual(edge.mean_action_value, 0.0)

        backup(self.nodes[1], 1)

        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 1)
        self.assertEqual(edge.total_action_value, 1.0)
        self.assertEqual(edge.mean_action_value, 1.0)

    def test_two_level_backup(self):
        edge = self.nodes[1].in_edge
        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 0)
        self.assertEqual(edge.total_action_value, 0.0)
        self.assertEqual(edge.mean_action_value, 0.0)

        backup(self.nodes[4], 1)

        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 1)
        self.assertEqual(edge.total_action_value, 1.0)
        self.assertEqual(edge.mean_action_value, 1.0)

    def test_select_exploration(self):
        self.nodes[1].in_edge.num_visits = 100

        # all things being equal, select unexplored action c=1
        c = 1
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        selected_edge = select(self.nodes[0], exploration_bonus)

        self.assertEqual(selected_edge.action, 1)
        self.assertEqual(selected_edge.num_visits, 0)

    def test_select_no_exploration(self):
        self.nodes[1].in_edge.num_visits = 100
        self.nodes[1].in_edge.total_action_value = 1000

        # select the action with best known reward c=0
        c = 0
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        selected_edge = select(self.nodes[0], exploration_bonus)

        self.assertEqual(selected_edge.action, 0)
        self.assertEqual(selected_edge.num_visits, 100)

    def test_expand_node(self):
        self.assertEqual(len(self.nodes[6].outgoing_edges), 0)
        value = expand_node(self.nodes[6], mock_model, mock_env)
        self.assertEqual(value, 1)
        self.assertEqual(len(self.nodes[6].outgoing_edges), 2)
        next_states = set(edge.out_node.state for edge in self.nodes[6].outgoing_edges)
        self.assertEqual(next_states, {13, 14})


class TestRollouts(unittest.TestCase):
    def test_rollouts(self):
        root_node = Node(0)
        n_leaf_expansions = 2
        c = 100  # to make sure we explore a new path every time
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        perform_rollouts(root_node, n_leaf_expansions, mock_model, mock_env, exploration_bonus)
        edge0, edge1 = root_node.outgoing_edges
        self.assertEquals(edge0.num_visits, 1)
        self.assertEquals(edge1.num_visits, 1)

    def test_get_action_distribution(self):
        start_state = 0
        temperature = 1
        n_leaf_expansions = 2
        c = 100  # to make sure we explore a new path every time
        distribution = get_action_distribution(start_state, temperature, n_leaf_expansions, mock_model, mock_env, c)
        self.assertEquals(tuple(distribution), (0.5, 0.5))
