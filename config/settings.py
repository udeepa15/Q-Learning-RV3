#!/usr/bin/env pybricks-micropython
"""
Configuration and Hyperparameters for EV3 Pybricks Q-Learning Project.
"""


try:
    from pybricks.parameters import Port
    PORT_LEFT_MOTOR = Port.B
    PORT_RIGHT_MOTOR = Port.C
    PORT_COLOR_SENSOR = Port.S1
    PORT_IR_SENSOR = Port.S4
except ImportError:
    PORT_LEFT_MOTOR = "Port.B"
    PORT_RIGHT_MOTOR = "Port.C"
    PORT_COLOR_SENSOR = "Port.S1"
    PORT_IR_SENSOR = "Port.S4"

WHITE_INTENSITY = 24.1  # Raw calibration reading (record only)
BLACK_INTENSITY = 2.5   # Simulator / calibration record only
EDGE_INTENSITY = 11.1

PURE_WHITE_THRESHOLD_8       = 20  # State 0: Pure White (>= 20)
MEDIUM_DRIFT_WHITE_THRESH_8  = 17  # State 1: Medium Drift (17 <= Intensity < 20)
                                   # State 2: Light Drift (14 <= Intensity < 17)
PERFECT_EDGE_HIGH_8          = 14  # State 3: Edge High Bound
PERFECT_EDGE_LOW_8           = 8   # State 3: Edge Low Bound -> Deadband (8 <= Intensity < 14)
DRIFT_BLACK_THRESHOLD_8      = 6   # State 4: Drift Black (6 <= Intensity < 8)
HEAVY_DRIFT_BLACK_THRESHOLD_8 = 4  # State 5: Heavy Drift Black (4 <= Intensity < 6)
                                   # State 6: Pure Black (< 4)


LOST_TIME_MS = 2000                 # State 7: continuous time on Pure Black before the robot counts as Lost


ACTION_FORWARD      = 0
ACTION_MICRO_LEFT   = 1
ACTION_SLIGHT_LEFT  = 2
ACTION_SHARP_LEFT   = 3
ACTION_MICRO_RIGHT  = 4
ACTION_SLIGHT_RIGHT = 5
ACTION_SHARP_RIGHT  = 6
ACTION_REVERSE      = 7

NUM_ACTIONS = 8
NUM_STATES = 8


# ROBOT SPEED & DIRECTION CONTROL CONFIGURATION
# Base Drive Speed Parameter (deg/s) - Change this number to adjust overall robot speed!
BASE_SPEED = 300

# Start configuration. Edit these to change how the robot is set up on the track.
# Turn direction: "CW" for Clockwise, "CCW" for Counter-Clockwise.
START_DIRECTION = "CW"

# Which physical edge of the 5cm white strip we follow: "OUTER" or "INNER".
# The two edges are mirror images:
#   OUTER edge, CW : black on the left,  white on the right
#   INNER edge, CW : black on the right, white on the left
START_EDGE = "OUTER"

# Runtime state, changed by set_direction() (the obstacle turnaround flips the direction).
TURN_DIRECTION = START_DIRECTION
LINE_EDGE = START_EDGE

# White is on the robot's right when (direction, edge) is CW+OUTER or CCW+INNER;
# otherwise the left/right speed tuples must be mirrored.
INVERT_TURNS = False

FORWARD_SPEED      = BASE_SPEED
MICRO_OUTER_SPEED  = BASE_SPEED
MICRO_INNER_SPEED  = int(BASE_SPEED * 0.70)

SLIGHT_OUTER_SPEED = BASE_SPEED
SLIGHT_INNER_SPEED = int(BASE_SPEED * 0.40)

SHARP_OUTER_SPEED  = BASE_SPEED
SHARP_INNER_SPEED  = -int(BASE_SPEED * 0.50)

REVERSE_SPEED      = -int(BASE_SPEED * 0.70)

# Action Speed Tuples (Left Motor Speed, Right Motor Speed) in deg/s.
# Direction-dependent entries are filled in by set_direction() below.
ACTION_SPEEDS = {
    ACTION_FORWARD: (FORWARD_SPEED, FORWARD_SPEED),
    ACTION_REVERSE: (REVERSE_SPEED, REVERSE_SPEED),
}


def set_direction(direction, edge=None):
    """Mirrors the left/right wheel speeds for the given direction and edge."""
    global TURN_DIRECTION, INVERT_TURNS, LINE_EDGE
    TURN_DIRECTION = direction
    if edge is not None:
        LINE_EDGE = edge
    INVERT_TURNS = (direction == "CCW") != (LINE_EDGE == "INNER")

    if not INVERT_TURNS:
        # White on the right (CW+OUTER / CCW+INNER): White -> Turn Left, Black -> Turn Right
        ACTION_SPEEDS[ACTION_MICRO_LEFT]   = (MICRO_INNER_SPEED, MICRO_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_LEFT]  = (SLIGHT_INNER_SPEED, SLIGHT_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_LEFT]   = (SHARP_INNER_SPEED, SHARP_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_MICRO_RIGHT]  = (MICRO_OUTER_SPEED, MICRO_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_RIGHT] = (SLIGHT_OUTER_SPEED, SLIGHT_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_RIGHT]  = (SHARP_OUTER_SPEED, SHARP_INNER_SPEED)
    else:
        # White on the left (CCW+OUTER / CW+INNER): mirrored left/right tuples
        ACTION_SPEEDS[ACTION_MICRO_LEFT]   = (MICRO_OUTER_SPEED, MICRO_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_LEFT]  = (SLIGHT_OUTER_SPEED, SLIGHT_INNER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_LEFT]   = (SHARP_OUTER_SPEED, SHARP_INNER_SPEED)
        ACTION_SPEEDS[ACTION_MICRO_RIGHT]  = (MICRO_INNER_SPEED, MICRO_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SLIGHT_RIGHT] = (SLIGHT_INNER_SPEED, SLIGHT_OUTER_SPEED)
        ACTION_SPEEDS[ACTION_SHARP_RIGHT]  = (SHARP_INNER_SPEED, SHARP_OUTER_SPEED)

    print("[Settings] Drive config: direction={}, edge={} (INVERT_TURNS={})".format(
        TURN_DIRECTION, LINE_EDGE, INVERT_TURNS))


def reset_direction():
    """Puts the direction and edge back to the start setup."""
    set_direction(START_DIRECTION, START_EDGE)


reset_direction()

# 180-degree obstacle turnaround parameters (tune TURN_180_MS for your wheelbase!)
TURN_180_SPEED = 200   # deg/s pivot speed for the turnaround spin
TURN_180_MS    = 5000

# Edge re-acquisition after the turnaround: small steps so the narrow edge band is not skipped.
EDGE_SEARCH_STEP_MS   = 40
EDGE_PROBE_MS         = 600    # max time to probe toward white when already on the edge band
EDGE_SEARCH_TOWARD_MS = 1600   # search time toward the expected white side
EDGE_SEARCH_AWAY_MS   = 3200   # search time back across the strip


# PHASED TRAINING (Straight Line -> Turns)
# Each phase trains its own states (rows) using every action.
STRAIGHT_STATES = (2, 3, 4)     # Light Drift White, Edge, Drift Black
TURN_STATES     = (0, 1, 5, 6, 7)  # Pure White, Medium Drift White, Heavy Drift Black, Pure Black, Lost

STRAIGHT_EPISODE_MS = 4000
TURN_EPISODE_MS     = 5000

PHASE_ALPHA         = 0.3   # Learning rate: few short episodes, near-deterministic rewards
PHASE_GAMMA         = 0.6   # Rewards are mostly immediate; keep the horizon short
PHASE_EPSILON_START = 0.4
PHASE_EPSILON_DECAY = 0.85  # Per episode (expect ~10-20 episodes per phase)
PHASE_EPSILON_MIN   = 0.05
PROGRESS_REWARD     = 1.0   # +/- bonus when the next reading moves toward/away from the edge

STRAIGHT_CHECKPOINT_PATH = "models/straight_q_table_8state.pkl"


# Non-RL Reflex / Hardware Parameters
OBSTACLE_DISTANCE_THRESHOLD = 20  # IR distance reading (0-100 percentage scale) below which the reflex triggers
DEFAULT_STEP_TIME_MS = 20
TRAIN_STEP_TIME_MS   = DEFAULT_STEP_TIME_MS   # Training and evaluation share one step time
