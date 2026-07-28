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
    Supports Heuristic Initialization for 3-State and 5-State architectures in CW & CCW directions.
    """
    def __init__(self, state_mode=None, direction="CW", n_states=None, n_actions=settings.NUM_ACTIONS,
                 alpha=settings.ALPHA, gamma=settings.GAMMA):
        self.state_mode = state_mode if state_mode is not None else getattr(settings, 'STATE_MODE', 5)
        self.direction = direction.upper() if direction else "CW"
        self.n_states = n_states if n_states is not None else (self.state_mode + 1)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Initialize Q-table matrix with directional heuristic values
        self.q_table = self._initialize_q_table()

    def _initialize_q_table(self):  
        """
        Injects heuristic initial Q-values to guide early exploration for CW and CCW directions.
        """
        if self.state_mode == 3:
            if self.direction == "CCW":
                return [
                    [0.0, 2.0, 2.0, -5.0, -5.0, 0.0],   # Row 0: Too White -> Turn Left
                    [5.0, 0.0, -2.0, 0.0, -2.0, -5.0],  # Row 1: Edge -> FWD
                    [0.0, -5.0, -5.0, 2.0, 2.0, 0.0],   # Row 2: Too Black -> Turn Right
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]      # Row 3: IR / Lost -> REV
                ]
            else:  # CW
                return [
                    [0.0, -5.0, -5.0, 2.0, 2.0, 0.0],   # Row 0: Too White -> Turn Right
                    [5.0, 0.0, -2.0, 0.0, -2.0, -5.0],  # Row 1: Edge -> FWD
                    [0.0, 2.0, 2.0, -5.0, -5.0, 0.0],   # Row 2: Too Black -> Turn Left
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]      # Row 3: IR / Lost -> REV
                ]
        elif self.state_mode == 5:
            if self.direction == "CCW":
                return [
                    [-2.0, 0.0, 5.0, -5.0, -5.0, -1.0],  # Row 0: Pure White -> Sharp LFT
                    [2.0, 3.0, 0.0, -2.0, -5.0, -2.0],   # Row 1: Drift White -> Slight LFT
                    [5.0, 0.0, -2.0, 0.0, -2.0, -5.0],   # Row 2: Edge -> FWD
                    [2.0, -2.0, -5.0, 3.0, 0.0, -2.0],   # Row 3: Drift Black -> Slight RGT
                    [-2.0, -5.0, -5.0, 0.0, 5.0, -1.0],  # Row 4: Pure Black -> Sharp RGT
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]       # Row 5: IR / Lost -> REV
                ]
            else:  # CW
                return [
                    [-2.0, -5.0, -5.0, 0.0, 5.0, -1.0],  # Row 0: Pure White -> Sharp RGT
                    [2.0, -2.0, -5.0, 3.0, 0.0, -2.0],   # Row 1: Drift White -> Slight RGT
                    [5.0, 0.0, -2.0, 0.0, -2.0, -5.0],   # Row 2: Edge -> FWD
                    [2.0, 3.0, 0.0, -2.0, -5.0, -2.0],   # Row 3: Drift Black -> Slight LFT
                    [-2.0, 0.0, 5.0, -5.0, -5.0, -1.0],  # Row 4: Pure Black -> Sharp LFT
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]       # Row 5: IR / Lost -> REV
                ]
        else:
            return [[0.0 for _ in range(self.n_actions)] for _ in range(self.n_states)]

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

    def display_q_table(self):
        """
        Prints a dynamic, formatted ASCII snapshot of the Q-table in the terminal.
        Asterisk (*) indicates the current optimal action per state.
        """
        action_names = ["FWD", "S_LFT", "SH_LFT", "S_RGT", "SH_RGT", "REV"]
        
        if self.state_mode == 3:
            state_names = ["Too White  ", "Edge       ", "Too Black  ", "Lost/IR    "]
        else:
            state_names = ["Pure White ", "Drift White", "Edge       ", "Drift Black", "Pure Black ", "Lost/IR    "]

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

