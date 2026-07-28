"""
Environment discretizer and reward manager for EV3 Q-Learning line follower.
"""

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings

# 8-State Constants
STATE_PURE_WHITE         = 0
STATE_MEDIUM_DRIFT_WHITE = 1
STATE_LIGHT_DRIFT_WHITE  = 2
STATE_MICRO_DRIFT_WHITE  = 3
STATE_PERFECT_EDGE       = 4
STATE_DRIFT_BLACK        = 5
STATE_PURE_BLACK         = 6
STATE_TOTALLY_LOST       = 7


class Environment:
    """
    Manages state representation discretizer and Q-learning reward system for 8-State Architecture.
    """
    def __init__(self):
        self.consecutive_lost_count = 0

    def get_state(self, intensity):
        """
        Maps continuous color sensor intensity into 7 discrete color gradient states + 1 Lost state.
        Uses a wide Edge Deadband (PERFECT_EDGE_LOW_8 <= intensity < PERFECT_EDGE_HIGH_8) to prevent waddling.
        """
        if intensity < settings.TOTALLY_LOST_THRESHOLD:
            self.consecutive_lost_count += 1
        else:
            self.consecutive_lost_count = 0

        if self.consecutive_lost_count >= settings.TOTALLY_LOST_CONSECUTIVE_STEPS:
            return STATE_TOTALLY_LOST

        edge_low  = getattr(settings, 'PERFECT_EDGE_LOW_8', 12)
        edge_high = getattr(settings, 'PERFECT_EDGE_HIGH_8', 16)

        if intensity >= getattr(settings, 'PURE_WHITE_THRESHOLD_8', 32):
            return STATE_PURE_WHITE
        elif intensity >= getattr(settings, 'MEDIUM_DRIFT_WHITE_THRESH_8', 28):
            return STATE_MEDIUM_DRIFT_WHITE
        elif intensity >= getattr(settings, 'LIGHT_DRIFT_WHITE_THRESH_8', 25):
            return STATE_LIGHT_DRIFT_WHITE
        elif intensity >= edge_high:
            return STATE_MICRO_DRIFT_WHITE
        elif intensity >= edge_low:
            return STATE_PERFECT_EDGE  # Wide Deadband Range (14 - 22)!
        elif intensity >= getattr(settings, 'DRIFT_BLACK_THRESHOLD_8', 9):
            return STATE_DRIFT_BLACK
        else:
            return STATE_PURE_BLACK


    def calculate_reward(self, state, action):
        """
        Calculates RL reward for state-action pair in 8-State Mode.
        """
        if state == STATE_PERFECT_EDGE:
            if action == settings.ACTION_FORWARD:
                return 5.0
            else:
                return 1.0
        elif state == STATE_MICRO_DRIFT_WHITE:
            if action == settings.ACTION_MICRO_LEFT or action == settings.ACTION_SLIGHT_LEFT or action == settings.ACTION_FORWARD:
                return 4.0
            else:
                return -1.0
        elif state == STATE_LIGHT_DRIFT_WHITE:
            if action == settings.ACTION_MICRO_LEFT or action == settings.ACTION_SLIGHT_LEFT:
                return 3.5
            else:
                return -1.0
        elif state == STATE_MEDIUM_DRIFT_WHITE:
            if action == settings.ACTION_SLIGHT_LEFT or action == settings.ACTION_SHARP_LEFT:
                return 3.0
            else:
                return -1.0
        elif state == STATE_PURE_WHITE:
            if action == settings.ACTION_SHARP_LEFT or action == settings.ACTION_SLIGHT_LEFT:
                return 3.0
            else:
                return -3.0
        elif state == STATE_DRIFT_BLACK:
            if action == settings.ACTION_MICRO_RIGHT or action == settings.ACTION_SLIGHT_RIGHT or action == settings.ACTION_SHARP_RIGHT:
                return 3.5
            else:
                return -1.0

        elif state == STATE_PURE_BLACK:
            if action == settings.ACTION_SHARP_RIGHT or action == settings.ACTION_SLIGHT_RIGHT:
                return 3.0
            else:
                return -3.0
        elif state == STATE_TOTALLY_LOST:
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



