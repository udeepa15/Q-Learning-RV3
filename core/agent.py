"""
Pure Python Q-Learning Agent for MicroPython EV3.
"""

import random

# MicroPython / standard Python pickle fallback
try:
    import pickle
except ImportError:
    import upickle as pickle

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings


class QLearningAgent:
    """
    Q-Learning Agent implemented in pure Python (no numpy dependency).
    """
    def __init__(self, n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS,
                 alpha=settings.ALPHA, gamma=settings.GAMMA):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Initialize Q-table matrix with 0.0 using pure Python lists
        self.q_table = [[0.0 for _ in range(self.n_actions)] for _ in range(self.n_states)]

    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randrange(self.n_actions)
        else:
            # Exploitation: choose action with maximum Q-value for current state
            q_row = self.q_table[state]
            max_q = q_row[0]
            best_actions = [0]

            for action in range(1, self.n_actions):
                if q_row[action] > max_q:
                    max_q = q_row[action]
                    best_actions = [action]
                elif q_row[action] == max_q:
                    best_actions.append(action)

            # Randomly break ties among actions with equal max Q-value
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Updates Q-value using Bellman Equation:
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        max_next_q = max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        return new_q

    def save(self, filepath):
        """
        Saves the Q-table to a pickle file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print("[QLearningAgent] Q-table successfully saved to {}".format(filepath))

    def load(self, filepath):
        """
        Loads the Q-table from a pickle file.
        """
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print("[QLearningAgent] Q-table successfully loaded from {}".format(filepath))
