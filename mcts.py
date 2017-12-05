from functools import partial

import numpy as np

from tree import Node, create_new_connection


def exploration_bonus_for_c_puct(edge, c_puct):
    """
    Determines a score for an edge that favors exploration
    c_puct is a constant factor that scales how favorable exploration is
    """
    sum_visits = np.sum([edge.num_visits for edge in edge.in_node.outgoing_edges])
    return c_puct * edge.prior_probability * np.sqrt(sum_visits) / (1 + edge.num_visits)


def select(node, exploration_bonus):
    """
    Select the next edge to expand
    c_puct is a constant factor that scales how favorable exploration is
    exploration_bonus: function
        Function that takes (edge) as input and returns a score based on
        how good the edge is to explore with our exploration rate.
    """
    def score(edge):
        return edge.mean_action_value + exploration_bonus(edge)
    scores = [score(edge) for edge in node.outgoing_edges]
    index = np.argmax(scores)
    return node.outgoing_edges[index]


def backup(node, value):
    """
    Propagate the value for the current node back up
    the tree
    """
    cur_node = node
    # while not root, move the value up
    count = 1
    while cur_node.in_edge is not None:
        edge = cur_node.in_edge
        edge.num_visits += 1
        edge.total_action_value += (-1)**count * value
        cur_node = edge.in_node
        count += 1


def expand_node(node, model, env):
    """
    For all legal actions possible from a node, create and connect edges
    to subsequent states. Returns the value of the current state as
    calculated by the model.
    """
    if env.is_game_over(node.state):
        node.is_expanded = True
        node.is_terminal = True
        value = -1  # the game is over on my turn, so I have lost
        return value

    vec_action_probs, values = model(np.array([node.state]))
    # need to take [0] index of vector since we're only putting in one state
    action_probs = vec_action_probs[0]
    value = values[0]
    legal_actions = env.get_legal_actions(node.state)
    for action in legal_actions:
        action_prob = action_probs[action]
        next_state = env.get_next_state(node.state, action)
        child_node = Node(next_state)
        create_new_connection(node, child_node, action, action_prob)
    node.is_expanded = True
    # need to take [0] index of value since value is an array of dimension 1
    return value[0]


def perform_rollouts(root_node,
                     n_leaf_expansions,
                     model,
                     env,
                     exploration_bonus):
    """
    Parameters
    ----------
    root_node: Node
        initial node to start MCTS
    n_leaf_expansions: int
        number of leaves to expand in each iteration of MCTS when picking an action
    model: function
        Model to use for computing the value of each state,
        prob_vector, value = model(node.state)
    env:
        game playing environment that can progress game state and give us legal moves
    exploration_bonus: function
        Function that takes (edge) as input and returns a score based on
        how good the edge is to explore with our exploration rate.
    """
    cur_node = root_node
    # add all edges and children for current node
    if not root_node.is_expanded:
        value = expand_node(root_node, model, env)

    while n_leaf_expansions > 0:
        edge = select(root_node, exploration_bonus)
        # find a node you haven't expanded yet, expand it
        # or, if you get to a terminal state, stop expanding
        while edge.num_visits != 0 and not edge.out_node.is_terminal:
            cur_node = edge.out_node
            edge = select(cur_node, exploration_bonus)
        cur_node = edge.out_node
        value = expand_node(cur_node, model, env)
        backup(cur_node, value)

        n_leaf_expansions -= 1


def get_action_distribution(root_node,
                            temperature,
                            n_leaf_expansions,
                            model,
                            env,
                            c_puct):
    """
    Returns the distribution over all actions after exploring the trees.
    This distribution pi(s) should be an improvement over the original p(s)
    output by the model.

    Parameters
    ----------
    root_node: Node
        initial node to start MCTS
    temperature: int
        how much to explore low probability states
    n_leaf_expansions: int
        number of leaves to expand in each iteration of MCTS when picking an action
    model: function
        Model to use for computing the value of each state,
        prob_vector, value = model(node.state, env)
    env:
        game playing environment that can progress game state and give us legal moves
    c_puct: float
        Constant that dictates how much score is assigned to exploring.
    """
    # set up the exploration_bonus function with the constant specified
    exploration_bonus = partial(exploration_bonus_for_c_puct, c_puct=c_puct)

    perform_rollouts(root_node, n_leaf_expansions, model, env, exploration_bonus)
    visit_counts = np.array([edge.num_visits for edge in root_node.outgoing_edges])

    # scale by temperature
    distribution = np.power(visit_counts, 1/temperature)
    # normalize
    distribution = distribution / np.sum(distribution)

    # our distribution is only over legal actions, some subset of the action space
    # all illegal actions have zero probability due to being unexplored
    total_action_distribution = np.zeros(env.action_size)
    action_indexes = [edge.action for edge in root_node.outgoing_edges]
    total_action_distribution[action_indexes] = distribution
    return total_action_distribution


def get_next_state_with_mcts(root_node,
                             temperature,
                             n_leaf_expansions,
                             model,
                             env,
                             c_puct):
    """
    Returns a tuple of (next_node, action_distribution) used to choose the action taken at the
    root node.
    """
    distribution = get_action_distribution(root_node, temperature, n_leaf_expansions, model, env, c_puct)
    action = np.random.choice(env.action_size, p=distribution)
    edge = [edge for edge in root_node.outgoing_edges if edge.action == action]
    next_node = edge[0].out_node
    return next_node, distribution
