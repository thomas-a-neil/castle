import numpy as np

from tree import Node, create_new_connection


def mock_model(states):
    action_probs = []
    values = []
    for i in range(len(states)):
        action_probs.append(np.array([0.5, 0.5]))
        values.append(np.array([1]))

    return (np.array(action_probs), np.array(values))


def mock_model_numline(states):
    action_probs = []
    values = []
    for state in states:
        if state > 3:
            value = -10
            action_0_prob = 0.9
        elif state > 1:
            value = 10
            action_0_prob = 0.5
        else:
            value = 0
            action_0_prob = 0.1
        action_probs.append(np.array([action_0_prob, 1 - action_0_prob]))
        values.append(np.array([value]))
    return (np.array(action_probs), np.array(values))


class MockEnv(object):
    """
    Functionality for the environment
    """
    @property
    def action_size(self):
        return 2

    def get_next_state(self, start_state, action):
        """
        Takes a state, action pair (indexes in the discrete case)
        and returns the state that results from taking that action
        """
        return 2 * (start_state + 1) - 1 * action

    def get_legal_actions(self, state):
        return np.array([0, 1])

    def is_game_over(self, state):
        return False

mock_env = MockEnv()


class NumlineEnv(object):
    """
    A simple numline environment where an agent can move left or right
    """
    @property
    def action_size(self):
        return 2

    def get_next_state(self, start_state, action):
        """
        Takes a state, action pair (indexes in the discrete case)
        and returns the state that results from taking that action
        """
        return start_state + (2*action - 1)

    def get_legal_actions(self, state):
        return np.array([0, 1])

    def is_game_over(self, state):
        return False

numline_env = NumlineEnv()

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


def setup_uneven_tree():
    #          0
    #     1        2
    #   3   4    5   6
    #  7 8             9
    # 10
    # 11
    nodes = [Node(i) for i in range(12)]
    for i in range(3):
        action = 0
        probability = 0.5
        child_node = nodes[2*(i + 1) - 1]
        nodes[i].add_edge_do_background_work(child_node, action, probability)

        action = 1
        probability = 0.5
        child_node = nodes[2*(i + 1)]
        nodes[i].add_edge_do_background_work(child_node, action, probability)

    # 3->7 and 3->8
    nodes[3].add_edge_do_background_work(nodes[7], 0, 0.2)
    nodes[3].add_edge_do_background_work(nodes[8], 1, 0.8)

    # 6->9
    nodes[6].add_edge_do_background_work(nodes[9], 0, 1.0)

    # 7->10
    nodes[7].add_edge_do_background_work(nodes[10], 1, 1.0)
    return nodes


def map_xy_to_square(x, y):
    return int(8*y + x)


def map_square_to_xy(square):
    return int(square % 8), int(square // 8)
