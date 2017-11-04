from tree import Node, create_new_connection


def mock_model(state):
    return (0.5, 0.5), 1


class MockEnv(object):
    """
    Functionality for the environment
    """
    def get_next_state(self, start_state, action):
        """
        Takes a state, action pair (indexes in the discrete case)
        and returns the state that results from taking that action
        """
        return 2 * (start_state + 1) - 1 * action

    def is_legal(self, state, action):
        """
        Given the current state and submitted action, is it legal?
        """
        return (action == 0 or action == 1)

mock_env = MockEnv()


def setup_simple_tree():
    #      0
    #   1     2
    #  3 4   5 6
    nodes = [Node(i) for i in range(7)]
    for i in range(3):
        in_node = nodes[i]
        out_node = nodes[2*(i + 1) - 1]
        create_new_connection(in_node, out_node, 0, 0.5)

        out_node = nodes[2*(i + 1)]
        create_new_connection(in_node, out_node, 1, 0.5)

    return nodes
