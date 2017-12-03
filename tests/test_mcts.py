from functools import partial
import unittest

from mcts import backup, select, expand_node, exploration_bonus_for_c_puct, perform_rollouts, get_action_distribution, get_next_state_with_mcts
from tree import Node

from utils import setup_simple_tree, mock_model, mock_env, numline_env, mock_model_numline


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
        self.nodes[6].state = 6
        self.assertEqual(len(self.nodes[6].outgoing_edges), 0)
        value = expand_node(self.nodes[6], mock_model, mock_env)
        self.assertEqual(value, 1)
        self.assertEqual(len(self.nodes[6].outgoing_edges), 2)
        next_states = [edge.out_node.state for edge in self.nodes[6].outgoing_edges]
        self.assertTrue(set(next_states), set([13, 14]))


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

    def test_numline_rollouts(self):
        """
        This is a simple numberline environment with 2 discrete actions: left and right.  The start state is zero.
        The agent gets reward of 10 at locations 2 and 3, reward of -10 from 4 onwards, and 0 on 1 and to the left
        The environment (luckily) pushes actions that are likely to help it achieve its reward based on its state

        The agent should explore the first action being to the right more than being to the left.  The same thing
        applies for the second action since that's when it will really hit its reward.
        But the 5th move should be to the left because it has gone off the cliff
        """

        root_node = Node(0)
        n_leaf_expansions = 100
        c = 100  # to make sure we explore a new path every time
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        perform_rollouts(root_node, n_leaf_expansions, mock_model_numline, numline_env, exploration_bonus)
        edge0, edge1 = root_node.outgoing_edges
        edge10, edge11 = edge1.out_node.outgoing_edges
        edge110, edge111 = edge11.out_node.outgoing_edges
        edge1110, edge1111 = edge111.out_node.outgoing_edges
        edge11110, edge11111 = edge1111.out_node.outgoing_edges

        self.assertTrue(edge0.num_visits < edge1.num_visits)
        self.assertTrue(edge10.num_visits < edge11.num_visits)
        self.assertTrue(edge11110.num_visits > edge11111.num_visits)

    def test_get_action_distribution(self):
        start_state = 0
        root_node = Node(start_state)
        temperature = 1
        n_leaf_expansions = 2
        c = 100  # to make sure we explore a new path every time
        distribution = get_action_distribution(root_node, temperature, n_leaf_expansions, mock_model, mock_env, c)
        self.assertEquals(tuple(distribution), (0.5, 0.5))

    def test_rollouts_on_same_tree(self):
        root_node = Node(0)
        n_leaf_expansions = 1
        c = 100  # to make sure we explore a new path every time
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        perform_rollouts(root_node, n_leaf_expansions, mock_model_numline, numline_env, exploration_bonus)
        self.assertEquals(len(root_node.outgoing_edges), 2)

        # we should only be expanding the root state once.
        perform_rollouts(root_node, n_leaf_expansions, mock_model_numline, numline_env, exploration_bonus)
        self.assertEquals(len(root_node.outgoing_edges), 2)

    def test_nodes_reuse_tree(self):
        '''
        This is a randomized test.  We perform_rollouts for the initial state with 30 leaf expansions
        We test to see that the node returned from get_next_state_with_mcts holds data from those rollouts 
        at its current node (now the second node) in addition to one potential next state.  

        We should fully expand a states that have short depth.
        '''
        n_leaf_expansions = 30
        c = 100
        root_node = Node(0)
        temperature = 1
        exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c)
        perform_rollouts(root_node, n_leaf_expansions, mock_model_numline, numline_env, exploration_bonus)
        
        second_node, action = get_next_state_with_mcts(root_node, temperature, n_leaf_expansions, 
            mock_model_numline, numline_env, c)

        self.assertEqual(len(second_node.outgoing_edges), 2)

        potential_third_node = second_node.outgoing_edges[0].out_node

        self.assertEqual(len(potential_third_node.outgoing_edges), 2)





