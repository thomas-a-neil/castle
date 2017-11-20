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

    def __repr__(self):
        return 'Node. Children: {} State: {}'.format([edge.out_node.state for edge in self.outgoing_edges], self.state)


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

    def __repr__(self):
        return 'Edge. In: {} Out: {} Action: {}'.format(self.in_node.state, self.out_node.state, self.action)


def create_new_connection(parent_node, child_node, action, prior_probability):
    """
    Returns the edge connecting parent and child
    """
    edge = Edge(parent_node, child_node, action, prior_probability)
    parent_node.add_outgoing_edge(edge)
    child_node.add_incoming_edge(edge)
    return edge
