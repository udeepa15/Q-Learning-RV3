"""
Environment discretizer and reward manager for EV3 Q-Learning line follower.
"""

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings

# State Constants
STATE_TOO_WHITE = 0
STATE_PERFECT_EDGE = 1
STATE_TOO_BLACK = 2
STATE_TOTALLY_LOST = 3


class Environment:
    """
    Manages state representation discretizer and Q-learning reward system.
    """
    def __init__(self):
        self.consecutive_lost_count = 0

    def get_state(self, intensity):
        """
        Maps continuous color sensor intensity into 4 discrete states:
        1. Too White
        2. Perfect Edge
        3. Too Black
        4. Totally Lost (pure black for multiple consecutive steps)
        """
        if intensity < settings.TOTALLY_LOST_THRESHOLD:
            self.consecutive_lost_count += 1
        else:
            self.consecutive_lost_count = 0

        # Check for Totally Lost state first based on consecutive low intensity readings
        if self.consecutive_lost_count >= settings.TOTALLY_LOST_CONSECUTIVE_STEPS:
            return STATE_TOTALLY_LOST

        if intensity >= settings.WHITE_THRESHOLD:
            return STATE_TOO_WHITE
        elif intensity <= settings.BLACK_THRESHOLD:
            return STATE_TOO_BLACK
        else:
            return STATE_PERFECT_EDGE

    def calculate_reward(self, state, action):
        """
        Calculates RL reward for state-action pair.
        Implements RULE A: In Totally Lost state, penalizes non-reverse actions (-5.0)
        and heavily rewards the Reverse action (+5.0).
        """
        if state == STATE_PERFECT_EDGE:
            return 3.0
        elif state == STATE_TOO_WHITE:
            return -1.0
        elif state == STATE_TOO_BLACK:
            return -2.0
        elif state == STATE_TOTALLY_LOST:
            # RULE A - The Reverse Trap logic
            if action == settings.ACTION_REVERSE:
                return 5.0  # Heavily reward backing up to refind the track
            else:
                return -5.0 # Penalize moving forward or turning when lost
        else:
            return 0.0

    def reset(self):
        """
        Resets lost step counters for a new episode.
        """
        self.consecutive_lost_count = 0
