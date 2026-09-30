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
                 alpha=settings.PHASE_ALPHA, gamma=settings.PHASE_GAMMA):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.q_table = [[0.0] * n_actions for _ in range(n_states)]

    def snapshot(self):
        return [row[:] for row in self.q_table]

    def restore(self, snapshot):
        self.q_table = [row[:] for row in snapshot]

    def choose_action(self, state, epsilon, allowed_actions=None):
        """
        Epsilon-greedy action selection, optionally restricted to a subset of columns.
        """
        if allowed_actions is None:
            allowed_actions = range(self.n_actions)
        allowed_actions = list(allowed_actions)

        if random.random() < epsilon:
            return random.choice(allowed_actions)

        q_row = self.q_table[state]
        max_q = q_row[allowed_actions[0]]
        best_actions = [allowed_actions[0]]

        for action in allowed_actions[1:]:
            if q_row[action] > max_q:
                max_q = q_row[action]
                best_actions = [action]
            elif q_row[action] == max_q:
                best_actions.append(action)

        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions=None):
        """
        Updates Q-value using Bellman Equation:
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        The max over a' can be restricted to next_actions.
        """
        next_row = self.q_table[next_state]
        if next_actions is None:
            max_next_q = max(next_row)
        else:
            max_next_q = max(next_row[a] for a in next_actions)
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
        Rejects tables whose shape does not match this agent's state/action space
        (e.g. model files from an older state/action layout).
        """
        with open(filepath, 'rb') as f:
            q_table = pickle.load(f)

        if len(q_table) != self.n_states or any(len(row) != self.n_actions for row in q_table):
            raise ValueError("Q-table shape mismatch in {}: expected {}x{}".format(
                filepath, self.n_states, self.n_actions))

        self.q_table = q_table
        print("[QLearningAgent] Q-table successfully loaded from {}".format(filepath))

    def display_q_table(self):
        """
        Prints a dynamic, formatted ASCII snapshot of the Q-table in the terminal.
        Asterisk (*) indicates the current optimal action per state.
        """
        action_names = ["FWD", "M_LFT", "S_LFT", "SH_LFT", "M_RGT", "S_RGT", "SH_RGT", "REV"]
        state_names = [
            "Pure White  ", "Med Drift W ", "Lt Drift W  ",
            "Edge        ", "Drift Black ", "Heavy DriftB", "Pure Black  ", "Lost/IR     "
        ]


        print("\n----------------------- DYNAMIC Q-TABLE SNAPSHOT -----------------------")
        header = "State        | " + " | ".join("{:>7}".format(a) for a in action_names)
        print(header)
        print("-" * len(header))

        for s_idx, q_row in enumerate(self.q_table):
            s_name = state_names[s_idx] if s_idx < len(state_names) else "State {:<5}".format(s_idx)
            max_q = max(q_row)
            formatted_vals = []
            for q_val in q_row:
                if q_val == max_q and q_val != 0.0:
                    formatted_vals.append("{:>6.1f}*".format(q_val))
                else:
                    formatted_vals.append("{:>7.1f}".format(q_val))
            print("{:<12} | ".format(s_name) + " | ".join(formatted_vals))
        print("------------------------------------------------------------------------\n")


