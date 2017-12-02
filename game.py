import numpy as np

from mcts import get_next_state_with_mcts
from tree import Node


def self_play_game(model,
                   env,
                   start_state,
                   n_leaf_expansions,
                   c_puct=1.0,
                   temperature=1,
                   max_num_turns=40,
                   verbose=False):
    """
    Plays a game (defined by the env), where a model with MCTS action distribution improvement plays
    itself. Returns a tuple of (states, winner_vector, action_distributions, last_state)

    tates, winner_vector, action_distributions are vectors of length equal to the number of turns played in the game.
    The winner vector is a vector of either the value 1, -1, or 0 for an eventual win, loss, or tie.
    Last state is not a vector, just a single state.

    Parameters
    ----------
    n_leaf_expansions: int
        number of leaves to expand in each iteration of MCTS when picking an action
    model: function
        Model to use for computing the value of each state,
        prob_vector, value = model(node.state, env)
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


def random_play_game(env,
                     start_state,
                     max_num_turns=40,
                     verbose=False):
    """
    Plays a game (defined by the env), where a random action is taken each turn
    Returns a tuple of (states, winner_vector, last_state)

    states and winner_vector are vectors of length equal to the number of turns played in the game.
    The winner vector is a vector of either the value 1, -1, or 0 for an eventual win, loss, or tie.

    Parameters
    ----------
    start_state:
        an initial game state (as defined by the environment)
    env:
        game playing environment that can progress game state and give us legal moves
    max_num_turns: int
        maximum number of turns to play out before stopping the game
    verbose: boolean
        If set to True, print the board state after each move
    """
    state = start_state
    # vector of states
    states = []

    num_turns = 0
    while not env.is_game_over(state) and num_turns <= max_num_turns:
        states.append(state)
        if verbose:
            env.print_board(state)

        action = np.random.choice(env.get_legal_actions(state))
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
