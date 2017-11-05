import unittest

from tree import Node, Edge

from utils import setup_simple_tree


class TestInit(unittest.TestCase):

    def test_node_init_defaults(self):
        node = Node('state')
        self.assertEqual(node.state, 'state')
        self.assertEqual(node.outgoing_edges, [])
        self.assertEqual(len(node.outgoing_edges), 0)
        self.assertEqual(node.in_edge, None)

    def test_node_edges(self):
        nodes = [Node(i) for i in range(7)]
        node = nodes[0]
        node.add_outgoing_edge('blah')
        self.assertEqual(node.state, 0)
        self.assertEqual(len(node.outgoing_edges), 1)
        # no other lists were affected
        self.assertEqual(len(nodes[1].outgoing_edges), 0)

    def test_edge_init_defaults(self):
        edge = Edge('in', 'out', 'action', 0.5)
        self.assertEqual(edge.in_node, 'in')
        self.assertEqual(edge.out_node, 'out')
        self.assertEqual(edge.action, 'action')
        self.assertEqual(edge.num_visits, 0)
        self.assertEqual(edge.total_action_value, 0.0)
        self.assertEqual(edge.mean_action_value, 0.0)

    def test_simple_tree(self):
        #      0
        #   1     2
        #  3 4   5 6
        nodes = setup_simple_tree()
        self.assertEqual(len(nodes[0].outgoing_edges), 2)
        self.assertEqual(len(nodes[1].outgoing_edges), 2)
        self.assertEqual(len(nodes[2].outgoing_edges), 2)
        self.assertEqual(len(nodes[3].outgoing_edges), 0)
        # parent of 3 is 1
        self.assertEqual(nodes[3].in_edge.in_node.state, 1)
        self.assertEqual(len(nodes[6].outgoing_edges), 0)
