import unittest

from mcts import Node, Edge, backprop, evaluate, select


def mock_model(state, env):
    return (0.5, 0.5), 1


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


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.nodes = setup_simple_tree()

    def test_backprop(self):
        edge = self.nodes[1].in_edge
        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 0)
        self.assertEqual(edge.total_action_value, 0.0)
        self.assertEqual(edge.mean_action_value, 0.0)

        backprop(self.nodes[1], 1)

        self.assertEqual(edge.action, 0)
        self.assertEqual(edge.num_visits, 1)
        self.assertEqual(edge.total_action_value, 1.0)
        self.assertEqual(edge.mean_action_value, 1.0)

    def test_select_exploration(self):
        self.nodes[1].in_edge.num_visits = 100

        # all things being equal, select unexplored action c=1
        selected_edge = select(self.nodes[0], 1)

        self.assertEqual(selected_edge.action, 1)
        self.assertEqual(selected_edge.num_visits, 0)

    def test_select_no_exploration(self):
        self.nodes[1].in_edge.num_visits = 100
        self.nodes[1].in_edge.total_action_value = 1000

        # select the action with best known reward c=0
        selected_edge = select(self.nodes[0], 0)

        self.assertEqual(selected_edge.action, 0)
        self.assertEqual(selected_edge.num_visits, 100)

    def test_evaluate(self):
        prob_vector, value = evaluate(self.nodes[0], mock_model, 'foo_env')
        self.assertEqual(prob_vector, (0.5, 0.5))
        self.assertEqual(value, 1)


def setup_simple_tree():
    #      0
    #   1     2
    #  3 4   5 6
    nodes = [Node(i) for i in range(7)]
    for i in range(3):
        in_node = nodes[i]
        out_node = nodes[2*(i + 1) - 1]
        edge0 = Edge(in_node, out_node, 0, 0.5)
        in_node.add_outgoing_edge(edge0)
        out_node.add_incoming_edge(edge0)

        out_node = nodes[2*(i + 1)]
        edge1 = Edge(in_node, out_node, 1, 0.5)
        in_node.add_outgoing_edge(edge1)
        out_node.add_incoming_edge(edge1)

    return nodes


if __name__ == '__main__':
    unittest.main()
