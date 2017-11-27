import numpy as np

from mcts import get_action_distribution

def self_play_game(model,
                   env,
                   start_state,
                   n_leaf_expansions,
                   c_puct=1.0,
                   temperature=1,
                   max_num_turns=40,
                   verbose_print_board=False):
    """
    Plays a game (defined by the env), where a model with MCTS action distribution improvement plays
    itself. Returns a tuple of (states, winner_vector, action_distributions)

    Where each is a vector of length equal to the number of turns played in the game.
    The winner vector is a vector of either the value 1, -1, or 0 for an eventual win, loss, or tie.

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
    """
    state = start_state
    # vector of states
    states = []
    # vector of action distributions for each game state
    action_distributions = []

    num_turns = 0
    while not env.is_game_over(state) and num_turns <= max_num_turns:
        if verbose_print_board:
            env.print_board(state)
        distribution = get_action_distribution(state, temperature, n_leaf_expansions, model, env, c_puct)

        states.append(state)
        action_distributions.append(distribution)

        chosen_action = np.random.choice(np.arange(env.action_size), p=distribution)
        state = env.get_next_state(state, chosen_action)


        num_turns += 1
    if verbose_print_board:
        env.print_board(state)

    winner = env.outcome(state) if num_turns <= max_num_turns else 0
    default_v = [1, -1] * (num_turns // 2) + [1] * (num_turns % 2)
    default_v = np.array(default_v)
    v = winner * default_v
    action_distributions = np.array(action_distributions)
    return states, v, action_distributions
