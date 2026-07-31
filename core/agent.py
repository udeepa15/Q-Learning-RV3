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
    Supports 8-State Heuristic Initialization for Clockwise line following.
    """
    def __init__(self, n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS,
                 alpha=settings.ALPHA, gamma=settings.GAMMA):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Initialize Q-table matrix with 5-State heuristic values
        self.q_table = self._initialize_q_table()

    def _initialize_q_table(self):  
        """
        Injects clean initial Q-values (0.0 and 5.0) for 8-State, 8-Action Clockwise line following:
        Actions: [0: FWD, 1: M_LFT, 2: S_LFT, 3: SH_LFT, 4: M_RGT, 5: S_RGT, 6: SH_RGT, 7: REV]
        """
        return [
            [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # Row 0: Pure White   -> Sharp LFT (5.0)
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Row 1: Medium Drift -> Slight LFT (5.0)
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Row 2: Light Drift  -> Micro LFT (5.0)
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Row 3: Micro Drift  -> Micro LFT (5.0)
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Row 4: Perfect Edge -> Drive FWD (5.0)
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],  # Row 5: Drift Black  -> Micro RGT (5.0)
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],  # Row 6: Pure Black   -> Sharp RGT (5.0)
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]   # Row 7: Lost / IR    -> Reverse (5.0)
        ]

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
        Rejects tables whose shape does not match this agent's state/action space
        (e.g. stale 3-State or 5-State model files).
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
            "Pure White  ", "Med Drift W ", "Lt Drift W  ", "Micro DriftW",
            "Edge        ", "Drift Black ", "Pure Black  ", "Lost/IR     "
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


