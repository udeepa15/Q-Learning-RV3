"""
Environment discretizer and reward manager for EV3 Q-Learning line follower.
"""

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings

# 3-State Constants
STATE_TOO_WHITE = 0
STATE_PERFECT_EDGE = 1
STATE_TOO_BLACK = 2
STATE_TOTALLY_LOST = 3

# 5-State Constants
STATE_PURE_WHITE = 0
STATE_DRIFT_WHITE = 1
STATE_PERFECT_EDGE_5 = 2
STATE_DRIFT_BLACK = 3
STATE_PURE_BLACK = 4
STATE_TOTALLY_LOST_5 = 5


class Environment:
    """
    Manages state representation discretizer and Q-learning reward system for 3-State and 5-State modes.
    """
    def __init__(self):
        self.consecutive_lost_count = 0

    def get_state(self, intensity):
        """
        Maps continuous color sensor intensity into discrete states based on settings.STATE_MODE (3 or 5):
        
        3-State Mode:
          0: Too White
          1: Perfect Edge
          2: Too Black
          3: Totally Lost

        5-State Mode:
          0: Pure White
          1: Drift White
          2: Perfect Edge
          3: Drift Black
          4: Pure Black
          5: Totally Lost
        """
        state_mode = getattr(settings, 'STATE_MODE', 5)

        if intensity < settings.TOTALLY_LOST_THRESHOLD:
            self.consecutive_lost_count += 1
        else:
            self.consecutive_lost_count = 0

        # Check for Totally Lost state first based on consecutive low intensity readings
        lost_state = STATE_TOTALLY_LOST if state_mode == 3 else STATE_TOTALLY_LOST_5
        if self.consecutive_lost_count >= settings.TOTALLY_LOST_CONSECUTIVE_STEPS:
            return lost_state

        if state_mode == 3:
            if intensity >= settings.WHITE_THRESHOLD_3:
                return STATE_TOO_WHITE
            elif intensity <= settings.BLACK_THRESHOLD_3:
                return STATE_TOO_BLACK
            else:
                return STATE_PERFECT_EDGE
        else:  # 5-State Mode
            if intensity >= settings.PURE_WHITE_THRESHOLD_5:
                return STATE_PURE_WHITE
            elif intensity >= settings.DRIFT_WHITE_THRESHOLD_5:
                return STATE_DRIFT_WHITE
            elif intensity >= settings.PERFECT_EDGE_LOW_5:
                return STATE_PERFECT_EDGE_5
            elif intensity >= settings.DRIFT_BLACK_THRESHOLD_5:
                return STATE_DRIFT_BLACK
            else:
                return STATE_PURE_BLACK

    def calculate_reward(self, state, action):
        """
        Calculates RL reward for state-action pair based on configured STATE_MODE.
        """
        state_mode = getattr(settings, 'STATE_MODE', 5)

        if state_mode == 3:
            if state == STATE_PERFECT_EDGE:
                return 3.0
            elif state == STATE_TOO_WHITE:
                return -1.0
            elif state == STATE_TOO_BLACK:
                return -2.0
            elif state == STATE_TOTALLY_LOST:
                if action == settings.ACTION_REVERSE:
                    return 5.0
                else:
                    return -5.0
            else:
                return 0.0
        else:  # 5-State Mode
            if state == STATE_PERFECT_EDGE_5:
                return 3.0
            elif state == STATE_DRIFT_WHITE or state == STATE_DRIFT_BLACK:
                return 1.0
            elif state == STATE_PURE_WHITE:
                return -1.0
            elif state == STATE_PURE_BLACK:
                return -2.0
            elif state == STATE_TOTALLY_LOST_5:
                if action == settings.ACTION_REVERSE:
                    return 5.0
                else:
                    return -5.0
            else:
                return 0.0

    def reset(self):
        """
        Resets lost step counters for a new episode.
        """
        self.consecutive_lost_count = 0

