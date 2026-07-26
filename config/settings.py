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

# State Architecture Configuration (A/B Testing: 3 or 5)
STATE_MODE = 5

# Environment Intensity Values and Thresholds
WHITE_INTENSITY = 35
BLACK_INTENSITY = 3
EDGE_INTENSITY = 19

# Thresholds for 3-State Mode (Too White, Perfect Edge, Too Black)
WHITE_THRESHOLD_3 = 28               # Intensity >= 28 considered Too White
BLACK_THRESHOLD_3 = 11               # Intensity <= 11 considered Too Black

# Thresholds for 5-State Mode (Pure White, Drift White, Perfect Edge, Drift Black, Pure Black)
PURE_WHITE_THRESHOLD_5 = 30          # Intensity >= 30: Pure White
DRIFT_WHITE_THRESHOLD_5 = 22         # 22 <= Intensity < 30: Drift White
PERFECT_EDGE_LOW_5 = 14              # 14 <= Intensity < 22: Perfect Edge
DRIFT_BLACK_THRESHOLD_5 = 7          # 7 <= Intensity < 14: Drift Black
# Intensity < 7: Pure Black

# Legacy / General Thresholds
WHITE_THRESHOLD = WHITE_THRESHOLD_3
BLACK_THRESHOLD = BLACK_THRESHOLD_3
TOTALLY_LOST_THRESHOLD = 5          # Intensity < 5 considered deep black
TOTALLY_LOST_CONSECUTIVE_STEPS = 3  # Consecutive steps required to trigger TOTALLY_LOST

# Action Space (6 Actions for smooth driving)
ACTION_FORWARD = 0
ACTION_SLIGHT_LEFT = 1
ACTION_SHARP_LEFT = 2
ACTION_SLIGHT_RIGHT = 3
ACTION_SHARP_RIGHT = 4
ACTION_REVERSE = 5

NUM_ACTIONS = 6
NUM_STATES = STATE_MODE + 1

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

