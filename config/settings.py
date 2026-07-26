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
ALPHA = 0.4            # Learning rate
GAMMA = 0.7            # Discount factor
EPSILON_START = 0.3    # Initial exploration rate
EPSILON_DECAY = 0.97   # Exploration decay per episode
EPSILON_MIN = 0.01     # Minimum exploration rate

# Environment Intensity Values and Thresholds
WHITE_INTENSITY = 35
BLACK_INTENSITY = 3
EDGE_INTENSITY = 19

WHITE_THRESHOLD = 28               # Intensity >= 28 considered White (Too White)
BLACK_THRESHOLD = 11               # Intensity <= 11 considered Black (Too Black)
TOTALLY_LOST_THRESHOLD = 5          # Intensity < 5 considered deep black
TOTALLY_LOST_CONSECUTIVE_STEPS = 3  # Consecutive steps required to trigger TOTALLY_LOST

# Action Space (RULE B - 6 Actions for smooth driving)
ACTION_FORWARD = 0
ACTION_SLIGHT_LEFT = 1
ACTION_SHARP_LEFT = 2
ACTION_SLIGHT_RIGHT = 3
ACTION_SHARP_RIGHT = 4
ACTION_REVERSE = 5

NUM_ACTIONS = 6
NUM_STATES = 4

# Action Speed Tuples (Left Motor Speed, Right Motor Speed) in deg/s
ACTION_SPEEDS = {
    ACTION_FORWARD:      (150, 150),
    ACTION_SLIGHT_LEFT:   (80, 150),
    ACTION_SHARP_LEFT:   (-80, 150),
    ACTION_SLIGHT_RIGHT: (150, 80),
    ACTION_SHARP_RIGHT:  (150, -80),
    ACTION_REVERSE:      (-100, -100),
}

# Non-RL Reflex / Hardware Parameters
OBSTACLE_DISTANCE_THRESHOLD = 20  # cm / percentage distance threshold for IR sensor
DEFAULT_STEP_TIME_MS = 100        # Action execution duration in milliseconds
