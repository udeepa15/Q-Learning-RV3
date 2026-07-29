#!/usr/bin/env pybricks-micropython
"""
Configuration and Hyperparameters for EV3 Pybricks Q-Learning Project.
"""


# Pybricks parameters port imports if available
try:
    from pybricks.parameters import Port
    PORT_LEFT_MOTOR = Port.B
    PORT_RIGHT_MOTOR = Port.C
    PORT_COLOR_SENSOR = Port.S2
    PORT_IR_SENSOR = Port.S3
except ImportError:
    # PC Fallback port representations
    PORT_LEFT_MOTOR = "Port.B"
    PORT_RIGHT_MOTOR = "Port.C"
    PORT_COLOR_SENSOR = "Port.S2"
    PORT_IR_SENSOR = "Port.S3"

# Reinforcement Learning Hyperparameters
ALPHA = 0.2            # Learning rate (smoothed for stable convergence)
GAMMA = 0.7            # Discount factor


EPSILON_START = 0.3    # Initial exploration rate
EPSILON_DECAY = 0.97   # Exploration decay per episode
EPSILON_MIN = 0.01     # Minimum exploration rate

# 8-State Environment Intensity Values and Thresholds (3 sub-gradient white states)
WHITE_INTENSITY = 35
BLACK_INTENSITY = 4
EDGE_INTENSITY = 14

# Intensity Thresholds for 8 States (With Wide Edge Deadband to Stop Penguin Waddling)
PURE_WHITE_THRESHOLD_8       = 32  # State 0: Pure White (>= 32)
MEDIUM_DRIFT_WHITE_THRESH_8  = 28  # State 1: Medium Drift (28 - 32)
LIGHT_DRIFT_WHITE_THRESH_8   = 25  # State 2: Light Drift (25 - 28)
                                   # State 3: Micro Drift (18 - 25)
PERFECT_EDGE_HIGH_8          = 18  # State 4: Edge High Bound
PERFECT_EDGE_LOW_8           = 12  # State 4: Edge Low Bound -> Deadband (12 - 18)
DRIFT_BLACK_THRESHOLD_8      = 9   # State 5: Drift Black (9 - 12)
                                   # State 6: Pure Black (< 9)


TOTALLY_LOST_THRESHOLD = 3          # State 7: Intensity < 3 considered deep off-track black
TOTALLY_LOST_CONSECUTIVE_STEPS = 12 # 12 consecutive steps (1.2s) allowed to steer out of black


# Action Space (8 Actions: Includes Micro Left and Micro Right for non-jerky tracking)
ACTION_FORWARD      = 0
ACTION_MICRO_LEFT   = 1
ACTION_SLIGHT_LEFT  = 2
ACTION_SHARP_LEFT   = 3
ACTION_MICRO_RIGHT  = 4
ACTION_SLIGHT_RIGHT = 5
ACTION_SHARP_RIGHT  = 6
ACTION_REVERSE      = 7

NUM_ACTIONS = 8
NUM_STATES = 8  # 7 Color States + 1 Lost/IR State


# ----------------------------------------------------
# ROBOT SPEED & DIRECTION CONTROL CONFIGURATION
# ----------------------------------------------------
# Base Drive Speed Parameter (deg/s) - Change this number to adjust overall robot speed!
BASE_SPEED = 250

# Turn Direction Mode ("CW" for Clockwise, "CCW" for Counter-Clockwise)
# Runtime-switchable via set_direction() -- used by the 180-degree obstacle turnaround.
TURN_DIRECTION = "CW"

# Which physical edge of the 5cm white strip we follow: "OUTER" or "INNER".
# The two edges are mirror images:
#   OUTER edge, CW : black on the left,  white on the right
#   INNER edge, CW : black on the right, white on the left
LINE_EDGE = "OUTER"

# White is on the robot's right when (direction, edge) is CW+OUTER or CCW+INNER;
# otherwise the left/right speed tuples must be mirrored.
INVERT_TURNS = False

# Dynamic Speed Multipliers derived from BASE_SPEED
FORWARD_SPEED      = BASE_SPEED
MICRO_INNER_SPEED  = int(BASE_SPEED * 0.15)   # e.g., 37 deg/s (gentle micro turn)
SLIGHT_OUTER_SPEED = BASE_SPEED
SLIGHT_INNER_SPEED = int(BASE_SPEED * 0.35)   # e.g., 87 deg/s (slight turn)
SHARP_OUTER_SPEED  = BASE_SPEED
SHARP_INNER_SPEED  = -int(BASE_SPEED * 0.90)  # e.g., -225 deg/s pivot
REVERSE_SPEED      = -int(BASE_SPEED * 0.70)  # e.g., -175 deg/s

# Action Speed Tuples (Left Motor Speed, Right Motor Speed) in deg/s.
# Direction-dependent entries are filled in by set_direction() below.
ACTION_SPEEDS = {
    ACTION_FORWARD: (FORWARD_SPEED, FORWARD_SPEED),
    ACTION_REVERSE: (REVERSE_SPEED, REVERSE_SPEED),
}


def set_direction(direction, edge=None):
    """
    Rebuilds the action -> motor speed mapping for the given travel
    direction ("CW"/"CCW") and followed strip edge ("OUTER"/"INNER").
    The same Q-table drives every combination: whenever white sits on the
    robot's left instead of its right, the speed tuples are mirrored so a
    logical 'left' action physically turns right.
    """
    global TURN_DIRECTION, INVERT_TURNS, LINE_EDGE
    TURN_DIRECTION = direction
    if edge is not None:
        LINE_EDGE = edge
    INVERT_TURNS = (direction == "CCW") != (LINE_EDGE == "INNER")

    if not INVERT_TURNS:
        # White on the right (CW+OUTER / CCW+INNER): White -> Turn Left, Black -> Turn Right
        ACTION_SPEEDS[ACTION_MICRO_LEFT]   = (MICRO_INNER_SPEED, FORWARD_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_LEFT]  = (SLIGHT_INNER_SPEED, SLIGHT_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_LEFT]   = (SHARP_INNER_SPEED, SHARP_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_MICRO_RIGHT]  = (FORWARD_SPEED, MICRO_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_RIGHT] = (SLIGHT_OUTER_SPEED, SLIGHT_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_RIGHT]  = (SHARP_OUTER_SPEED, SHARP_INNER_SPEED)
    else:
        # White on the left (CCW+OUTER / CW+INNER): mirrored left/right tuples
        ACTION_SPEEDS[ACTION_MICRO_LEFT]   = (FORWARD_SPEED, MICRO_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_LEFT]  = (SLIGHT_OUTER_SPEED, SLIGHT_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_LEFT]   = (SHARP_OUTER_SPEED, SHARP_INNER_SPEED)
        ACTION_SPEEDS[ACTION_MICRO_RIGHT]  = (MICRO_INNER_SPEED, FORWARD_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_RIGHT] = (SLIGHT_INNER_SPEED, SLIGHT_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_RIGHT]  = (SHARP_INNER_SPEED, SHARP_OUTER_SPEED)

    print("[Settings] Drive config: direction={}, edge={} (INVERT_TURNS={})".format(
        TURN_DIRECTION, LINE_EDGE, INVERT_TURNS))


set_direction(TURN_DIRECTION)

# 180-degree obstacle turnaround parameters (tune TURN_180_MS for your wheelbase!)
TURN_180_SPEED = 150   # deg/s pivot speed for the turnaround spin
TURN_180_MS    = 1100  # spin duration for ~180 degrees




# Non-RL Reflex / Hardware Parameters
OBSTACLE_DISTANCE_THRESHOLD = 20  # cm / percentage distance threshold for IR sensor
DEFAULT_STEP_TIME_MS = 20         # Action execution duration (20ms = 50Hz fast control loop)
