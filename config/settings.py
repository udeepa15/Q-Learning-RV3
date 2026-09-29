#!/usr/bin/env pybricks-micropython
"""
Configuration and Hyperparameters for EV3 Pybricks Q-Learning Project.
"""


# Pybricks parameters port imports if available
try:
    from pybricks.parameters import Port
    PORT_LEFT_MOTOR = Port.B
    PORT_RIGHT_MOTOR = Port.C
    PORT_COLOR_SENSOR = Port.S1
    PORT_IR_SENSOR = Port.S4
except ImportError:
    # PC Fallback port representations
    PORT_LEFT_MOTOR = "Port.B"
    PORT_RIGHT_MOTOR = "Port.C"
    PORT_COLOR_SENSOR = "Port.S1"
    PORT_IR_SENSOR = "Port.S4"

# Reinforcement Learning Hyperparameters
ALPHA = 0.1            # Learning rate (reduced from 0.2 for slower, more stable convergence)
GAMMA = 0.7            # Discount factor


EPSILON_START = 0.3    # Initial exploration rate
EPSILON_DECAY = 0.97   # Exploration decay per episode
EPSILON_MIN = 0.01     # Minimum exploration rate

# Training Loop Duration (per training session, started via train.py)
NUM_EPISODES = 100            # Episodes per training session (increased from 40)
MAX_STEPS_PER_EPISODE = 100   # Max steps per episode (increased from 60)

# 8-State Environment Intensity Values and Thresholds (Updated for calibrated surface readings)
WHITE_INTENSITY = 24.1
BLACK_INTENSITY = 2.5
EDGE_INTENSITY = 11.1

# Intensity Thresholds for 8 States: 3 symmetric drift tiers on each side of the
# edge deadband (Light/Medium/Pure) + Edge + Lost (With Edge Deadband to Stop Penguin Waddling)
PURE_WHITE_THRESHOLD_8       = 23  # State 0: Pure White (>= 23)
MEDIUM_DRIFT_WHITE_THRESH_8  = 20  # State 1: Medium Drift (20 <= Intensity < 23)
                                   # State 2: Light Drift (14 <= Intensity < 20)
PERFECT_EDGE_HIGH_8          = 14  # State 3: Edge High Bound
PERFECT_EDGE_LOW_8           = 8   # State 3: Edge Low Bound -> Deadband (8 <= Intensity < 14)
DRIFT_BLACK_THRESHOLD_8      = 4   # State 4: Drift Black (4 <= Intensity < 8)
HEAVY_DRIFT_BLACK_THRESHOLD_8 = 2  # State 5: Heavy Drift Black (2 <= Intensity < 4)
                                   # State 6: Pure Black (< 2)


TOTALLY_LOST_THRESHOLD = 1          # State 7: Intensity < 1 considered deep off-track black
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
NUM_STATES = 8  # 6 Drift States (3 White + 3 Black) + 1 Edge + 1 Lost/IR State


# ----------------------------------------------------
# ROBOT SPEED & DIRECTION CONTROL CONFIGURATION
# ----------------------------------------------------
# Base Drive Speed Parameter (deg/s) - Change this number to adjust overall robot speed!
BASE_SPEED = 300

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
MICRO_OUTER_SPEED  = BASE_SPEED
MICRO_INNER_SPEED  = int(BASE_SPEED * 0.70)   # Micro turn: gentle curve (diff = 30% BASE_SPEED)

SLIGHT_OUTER_SPEED = BASE_SPEED
SLIGHT_INNER_SPEED = int(BASE_SPEED * 0.40)   # Slight turn: moderate curve (diff = 70% BASE_SPEED)

SHARP_OUTER_SPEED  = BASE_SPEED
SHARP_INNER_SPEED  = -int(BASE_SPEED * 0.50)  # Sharp turn: aggressive pivot spin (diff = 185% BASE_SPEED)

REVERSE_SPEED      = -int(BASE_SPEED * 0.70)  # Reverse speed

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


set_direction(TURN_DIRECTION)

# 180-degree obstacle turnaround parameters (tune TURN_180_MS for your wheelbase!)
TURN_180_SPEED = 200   # deg/s pivot speed for the turnaround spin
TURN_180_MS    = 5000  # spin duration for ~180 degrees




# ----------------------------------------------------
# PHASED TRAINING (Straight Line -> Turns)
# ----------------------------------------------------
# Straight phase: only the near-edge rows and the gentle steering columns are trained.
STRAIGHT_STATES  = (2, 3, 4)   # Light Drift White, Edge, Drift Black
STRAIGHT_ACTIONS = (ACTION_FORWARD, ACTION_SLIGHT_LEFT, ACTION_SLIGHT_RIGHT)

# Turn phase: the remaining rows and columns.
TURN_STATES  = (0, 1, 5, 6, 7)  # Pure White, Medium Drift White, Heavy Drift Black, Pure Black, Lost
TURN_ACTIONS = (ACTION_MICRO_LEFT, ACTION_SHARP_LEFT, ACTION_MICRO_RIGHT,
                ACTION_SHARP_RIGHT, ACTION_REVERSE)

STRAIGHT_EPISODE_MS = 4000
TURN_EPISODE_MS     = 5000
TRAIN_STEP_TIME_MS  = 20   # Longer than DEFAULT_STEP_TIME_MS so each action visibly changes the next reading

# Hyperparameters for the phased trainer (Q-table starts from all zeros)
PHASE_ALPHA         = 0.3   # Higher than ALPHA: few short episodes, near-deterministic rewards
PHASE_GAMMA         = 0.6   # Rewards are mostly immediate; keep the horizon short
PHASE_EPSILON_START = 0.4
PHASE_EPSILON_DECAY = 0.85  # Per episode (expect ~10-20 episodes per phase)
PHASE_EPSILON_MIN   = 0.05
PROGRESS_REWARD     = 1.0   # +/- bonus when the next reading moves toward/away from the edge

STRAIGHT_CHECKPOINT_PATH = "models/straight_q_table_8state.pkl"


def actions_for_state(state):
    """Columns a row is allowed to use under phased training."""
    return STRAIGHT_ACTIONS if state in STRAIGHT_STATES else TURN_ACTIONS


# Non-RL Reflex / Hardware Parameters
OBSTACLE_DISTANCE_THRESHOLD = 20  # cm / percentage distance threshold for IR sensor
DEFAULT_STEP_TIME_MS = 5          # Action execution duration (5ms step delay)
