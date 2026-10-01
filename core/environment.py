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
STATE_LIGHT_DRIFT_BLACK  = 4
STATE_MEDIUM_DRIFT_BLACK = 5
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
    STATE_LIGHT_DRIFT_BLACK: ({settings.ACTION_MICRO_RIGHT: 3.5, settings.ACTION_SLIGHT_RIGHT: 1.5, settings.ACTION_SHARP_RIGHT: -1.0}, -1.0),
    STATE_MEDIUM_DRIFT_BLACK: ({settings.ACTION_MICRO_RIGHT: 1.0, settings.ACTION_SLIGHT_RIGHT: 3.0, settings.ACTION_SHARP_RIGHT: 1.5}, -1.0),
    STATE_PURE_BLACK: ({settings.ACTION_SLIGHT_RIGHT: 2.0, settings.ACTION_SHARP_RIGHT: 3.0}, -3.0),
    STATE_TOTALLY_LOST: ({settings.ACTION_REVERSE: 5.0}, -5.0),
}


class Environment:
    """Turns sensor readings into states and hands out rewards."""
    def __init__(self):
        self.pure_black_since = None

    def get_state(self, intensity):
        """Turn a light reading into one of the 8 states."""
        if intensity < settings.MEDIUM_DRIFT_BLACK_THRESH_8:
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
        elif intensity >= settings.LIGHT_DRIFT_BLACK_THRESH_8:
            return STATE_LIGHT_DRIFT_BLACK
        elif intensity >= settings.MEDIUM_DRIFT_BLACK_THRESH_8:
            return STATE_MEDIUM_DRIFT_BLACK
        else:
            return STATE_PURE_BLACK


    def calculate_reward(self, state, action):
        """Reward for taking this action in this state."""
        table, default = _REWARDS.get(state, ({}, 0.0))
        return table.get(action, default)

    def progress_reward(self, state, next_state):
        """+1 if the robot got closer to the edge, -1 if it got farther."""
        def distance(s):
            return 4 if s == STATE_TOTALLY_LOST else abs(s - STATE_PERFECT_EDGE)

        delta = distance(state) - distance(next_state)
        if delta > 0:
            return settings.PROGRESS_REWARD
        if delta < 0:
            return -settings.PROGRESS_REWARD
        return 0.0

    def reset(self):
        """Clear the pure black timer."""
        self.pure_black_since = None



