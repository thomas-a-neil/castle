import numpy as np


class Node(object):
    def __init__(self, state, outgoing_edges=None, in_edge=None):
        self.state = state
        if outgoing_edges is not None:
            self.outgoing_edges = outgoing_edges
        else:
            self.outgoing_edges = []
        self.in_edge = in_edge

    def add_outgoing_edge(self, edge):
        self.outgoing_edges.append(edge)

    def add_outgoing_edges(self, edges):
        self.outgoing_edges.extend(edges)

    def add_incoming_edge(self, edge):
        self.in_edge = edge


class Edge(object):
    def __init__(self,
                 in_node,
                 out_node,
                 action,
                 prior_probability,
                 num_visits=0,
                 total_action_value=0.0):
        self.in_node = in_node
        self.out_node = out_node  # node
        self.action = action
        self.num_visits = num_visits
        self.total_action_value = total_action_value
        self.prior_probability = prior_probability

    @property
    def mean_action_value(self):
        if self.num_visits == 0:
            return 0.0
        return self.total_action_value / self.num_visits


def exploration_bonus(edge, c_puct):
    """
    Determines a score for an edge that favors exploration
    c_puct is a constant factor that scales how favorable exploration is
    """
    sum_visits = np.sum([edge.num_visits for edge in edge.in_node.outgoing_edges])
    return c_puct * edge.prior_probability * np.sqrt(sum_visits) / (1 + edge.num_visits)


def select(node, c_puct):
    """
    Select the next edge to expand
    c_puct is a constant factor that scales how favorable exploration is
    """
    def score(edge, c_puct):
        return edge.mean_action_value + exploration_bonus(edge, c_puct)
    index = np.argmax([score(edge, c_puct) for edge in node.outgoing_edges])
    return node.outgoing_edges[index]


def evaluate(node, model, env):
    """
    Return the probability vector for actions,
    and the value of the current state
    """
    prob_vector, value = model(node.state, env)
    return prob_vector, value


def backprop(node, value):
    """
    Propagate the value for the current node back
    """
    cur_node = node
    # while not root, move the value up
    while cur_node.in_edge is not None:
        edge = node.in_edge
        edge.num_visits += 1
        edge.total_action_value += value
        cur_node = edge.in_node
