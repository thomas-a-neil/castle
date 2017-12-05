import numpy as np

from mcts import get_next_state_with_mcts
from tree import Node


def self_play_game(model,
                   env,
                   start_state=None,
                   n_leaf_expansions=20,
                   c_puct=1.0,
                   temperature=1,
                   max_num_turns=40,
                   verbose=False):
    """
    Plays a game (defined by the env), where a model with MCTS action distribution improvement plays
    itself. Returns a tuple of (states, winner_vector, action_distributions)

    states, winner_vector, action_distributions are vectors of length equal to the number of turns played in the game.
    The winner vector is a vector of either the value 1, -1, or 0 for an eventual win, loss, or tie.
    Last state is not a vector, just a single state.

    Parameters
    ----------
    n_leaf_expansions: int
        number of leaves to expand in each iteration of MCTS when picking an action
    model: function
        Model to use for computing the value of each state,
        [prob_vector], [value] = model([node.state])
    start_state:
        an initial game state (as defined by the environment)
    env:
        game playing environment that can progress game state and give us legal moves
    c_puct: float
        Constant that dictates how much score is assigned to exploring.
    temperature: int
        how much to explore low probability states
    max_num_turns: int
        maximum number of turns to play out before stopping the game
    verbose: boolean
        If set to True, print the board state after each move
    """
    if start_state is None:
        start_state = env.reset()
    state = start_state
    cur_node = Node(state)
    # vector of states
    states = []
    # vector of action distributions for each game state
    action_distributions = []

    num_turns = 0
    while not env.is_game_over(cur_node.state) and num_turns <= max_num_turns:
        states.append(cur_node.state)
        if verbose:
            env.print_board(cur_node.state)

        # we pass nodes in to keep work done in previous mcts rollouts.
        cur_node, distribution = get_next_state_with_mcts(cur_node, temperature, n_leaf_expansions, model, env, c_puct)
        action_distributions.append(distribution)

        num_turns += 1

    if verbose:
        env.print_board(cur_node.state)

    winner = env.outcome(cur_node.state) if num_turns <= max_num_turns else 0
    default_v = [1, -1] * (num_turns // 2) + [1] * (num_turns % 2)
    default_v = np.array(default_v)
    v = winner * default_v
    states = np.array(states)
    action_distributions = np.array(action_distributions)
    return states, v, action_distributions


def play_game(model1,
              model2,
              env,
              start_state=None,
              max_num_turns=40,
              verbose=False):
    """
    Plays a game (defined by the env), where an action is taken each turn by the models specified. model1
    moves first, model2 moves second.
    Returns a tuple of (states, winner_vector)

    states and winner_vector are vectors of length equal to the number of turns played in the game.
    The winner vector is a vector of either the value 1, -1, or 0 for an eventual win, loss, or tie.

    Parameters
    ----------
    model1, model2: function
        Model to use for computing the value of each state,
        [prob_vector], [value] = model([node.state])
        model1 moves first, model2 second
    env:
        game playing environment that can progress game state and give us legal moves
    start_state:
        an initial game state (as defined by the environment)
    max_num_turns: int
        maximum number of turns to play out before stopping the game
    verbose: boolean
        If set to True, print the board state after each move
    """
    if start_state is None:
        start_state = env.reset()
    state = start_state
    # vector of states
    states = []

    num_turns = 0
    while not env.is_game_over(state) and num_turns <= max_num_turns:
        states.append(state)
        if verbose:
            env.print_board(state)

        if num_turns % 2 == 0:
            distribution, value = model1(np.array([state]))
        else:
            distribution, value = model2(np.array([state]))
        action = np.random.choice(env.action_size, p=distribution[0])
        state = env.get_next_state(state, action)

        num_turns += 1

    if verbose:
        env.print_board(state)

    winner = env.outcome(state) if num_turns <= max_num_turns else 0
    default_v = [1, -1] * (num_turns // 2) + [1] * (num_turns % 2)
    default_v = np.array(default_v)
    v = winner * default_v
    states = np.array(states)
    return states, v


class RandomModel(object):
    def __init__(self, env):
        self.env = env

    def __call__(self, states):
        action_probs = []
        values = []
        for state in states:
            legal_actions = self.env.get_legal_actions(state)
            action_index = np.random.choice(legal_actions)

            # one hot encode the chosen, legal action
            action_distribution = np.zeros(self.env.action_size)
            action_distribution[action_index] = 1.0

            action_probs.append(action_distribution)
            values.append(np.array([0]))  # all states have equal value

        return (np.array(action_probs), np.array(values))
