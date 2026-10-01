"""
Environment discretizer and reward manager for EV3 Q-Learning line follower.
"""

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings

# 8-State Constants (3 White Drift + Edge + 3 Black Drift + Lost, symmetric on both sides)
STATE_PURE_WHITE         = 0
STATE_MEDIUM_DRIFT_WHITE = 1
STATE_LIGHT_DRIFT_WHITE  = 2
STATE_PERFECT_EDGE       = 3
STATE_DRIFT_BLACK        = 4
STATE_HEAVY_DRIFT_BLACK  = 5
STATE_PURE_BLACK         = 6
STATE_TOTALLY_LOST       = 7

try:
    from pybricks.tools import StopWatch
except ImportError:
    import time

    class StopWatch:
        def __init__(self):
            self._start = time.time()

        def time(self):
            return int((time.time() - self._start) * 1000)


_clock = StopWatch()


_REWARDS = {
    STATE_PERFECT_EDGE: ({settings.ACTION_FORWARD: 5.0}, 1.0),
    STATE_LIGHT_DRIFT_WHITE: ({settings.ACTION_MICRO_LEFT: 3.5, settings.ACTION_SLIGHT_LEFT: 1.5, settings.ACTION_SHARP_LEFT: -1.0}, -1.0),
    STATE_MEDIUM_DRIFT_WHITE: ({settings.ACTION_MICRO_LEFT: 1.0, settings.ACTION_SLIGHT_LEFT: 3.0, settings.ACTION_SHARP_LEFT: 1.5}, -1.0),
    STATE_PURE_WHITE: ({settings.ACTION_SLIGHT_LEFT: 2.0, settings.ACTION_SHARP_LEFT: 3.0}, -3.0),
    STATE_DRIFT_BLACK: ({settings.ACTION_MICRO_RIGHT: 3.5, settings.ACTION_SLIGHT_RIGHT: 1.5, settings.ACTION_SHARP_RIGHT: -1.0}, -1.0),
    STATE_HEAVY_DRIFT_BLACK: ({settings.ACTION_MICRO_RIGHT: 1.0, settings.ACTION_SLIGHT_RIGHT: 3.0, settings.ACTION_SHARP_RIGHT: 1.5}, -1.0),
    STATE_PURE_BLACK: ({settings.ACTION_SLIGHT_RIGHT: 2.0, settings.ACTION_SHARP_RIGHT: 3.0}, -3.0),
    STATE_TOTALLY_LOST: ({settings.ACTION_REVERSE: 5.0}, -5.0),
}


class Environment:
    """
    Manages state representation discretizer and Q-learning reward system for 8-State Architecture.
    """
    def __init__(self):
        self.pure_black_since = None

    def get_state(self, intensity):
        """
        Maps continuous color sensor intensity into 6 discrete color gradient states
        (3 white-side + 3 black-side, symmetric around the edge) + Edge + Lost.
        Lost is reported once the reading has stayed in Pure Black for LOST_TIME_MS.
        Uses a wide Edge Deadband (PERFECT_EDGE_LOW_8 <= intensity < PERFECT_EDGE_HIGH_8) to prevent waddling.
        """
        if intensity < settings.HEAVY_DRIFT_BLACK_THRESHOLD_8:
            now = _clock.time()
            if self.pure_black_since is None:
                self.pure_black_since = now
            if now - self.pure_black_since >= settings.LOST_TIME_MS:
                return STATE_TOTALLY_LOST
        else:
            self.pure_black_since = None

        if intensity >= settings.PURE_WHITE_THRESHOLD_8:
            return STATE_PURE_WHITE
        elif intensity >= settings.MEDIUM_DRIFT_WHITE_THRESH_8:
            return STATE_MEDIUM_DRIFT_WHITE
        elif intensity >= settings.PERFECT_EDGE_HIGH_8:
            return STATE_LIGHT_DRIFT_WHITE
        elif intensity >= settings.PERFECT_EDGE_LOW_8:
            return STATE_PERFECT_EDGE
        elif intensity >= settings.DRIFT_BLACK_THRESHOLD_8:
            return STATE_DRIFT_BLACK
        elif intensity >= settings.HEAVY_DRIFT_BLACK_THRESHOLD_8:
            return STATE_HEAVY_DRIFT_BLACK
        else:
            return STATE_PURE_BLACK


    def calculate_reward(self, state, action):
        """
        Rule-based reward for a state-action pair. The size of the correction
        should match how far off the edge the robot is: micro near the edge,
        slight in the middle band, sharp far away. White-side and black-side
        tiers are mirrored.
        """
        table, default = _REWARDS.get(state, ({}, 0.0))
        return table.get(action, default)

    def progress_reward(self, state, next_state):
        """
        Outcome-based shaping: rewards moving toward the edge and penalises
        drifting away, so Q-values reflect what an action actually did.
        """
        def distance(s):
            return 4 if s == STATE_TOTALLY_LOST else abs(s - STATE_PERFECT_EDGE)

        delta = distance(state) - distance(next_state)
        if delta > 0:
            return settings.PROGRESS_REWARD
        if delta < 0:
            return -settings.PROGRESS_REWARD
        return 0.0

    def reset(self):
        """
        Clears the Pure Black timer (new episode, or after a reflex that moved the robot).
        """
        self.pure_black_since = None



