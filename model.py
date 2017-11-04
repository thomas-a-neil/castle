import numpy as np


class LegalActionsOnlyModel(object):
    """
    Wrapper for model that only outputs legal action probabilities
    """
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def __call__(self, state):
        """
        Return the probability vector for actions,
        and the value of the current state

        We assume the model will only output probabilities for legal actions
        (TODO wrap actual model in suitable thing that filters)
        """
        action_probs, value = self.model(state)
        legal_action_probs = []
        for i, action_prob in enumerate(action_probs):
            if self.env.is_legal(state, i):
                legal_action_probs.append(action_prob)
            else:
                legal_action_probs.append(0.0)

        legal_action_probs = np.array(legal_action_probs)
        return legal_action_probs / np.sum(legal_action_probs), value
