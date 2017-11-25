import numpy as np

from mcts import get_action_distribution


class Game(object):
    """
    Module for playing a game with a model, and MCTS action distribution improvement
    """
    def self_play(self,
                  model,
                  env,
                  start_state,
                  n_leaf_expansions,
                  c_puct,
                  temperature,
                  max_num_turns=40):
        """
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
            distribution = get_action_distribution(start_state, temperature, n_leaf_expansions, model, env, c_puct)

            states.append(state)
            action_distributions.append(distribution)

            chosen_action = np.random.choice(np.arange(env.action_size), p=distribution)
            state = env.get_next_state(state, chosen_action)

            num_turns += 1

        winner = env.outcome(state) if num_turns <= max_num_turns else 0
        default_v = [1, -1] * (num_turns // 2) + [1] * (num_turns % 2)
        v = winner * default_v
        action_distributions = np.array(action_distributions)
        return states, v, action_distributions
