"""
Evaluation / Final Exam Script for EV3 Q-Learning Line Follower Agent.
"""

import sys
import os

# Ensure project root is in sys.path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from pybricks.tools import wait
except ImportError:
    import time
    def wait(ms):
        time.sleep(ms / 1000.0)

from config import settings
from hardware.robot import RobotInterface
from hardware.reflexes import detect_track_direction, hardcoded_obstacle_avoidance
from core.agent import QLearningAgent
from core.environment import Environment


def evaluate_agent(max_iterations=None, use_simulator=False):
    """
    Evaluation loop executing pure Q-table exploitation with track direction detection.
    """
    robot = RobotInterface(use_simulator=use_simulator)
    agent = QLearningAgent(n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS)
    env = Environment()

    print("==================================================")
    print("Starting EV3 Robot Evaluation...")
    print("==================================================")

    # RULE C: Sweep and detect track direction
    direction = detect_track_direction(robot)
    print("[Evaluate] Direction detected:", direction)

    # Determine Q-table file to load based on detected direction
    if direction == "CW":
        model_filename = "models/cw_q_table.pkl"
        fallback_filename = "cw_q_table.pkl"
    else:
        model_filename = "models/ccw_q_table.pkl"
        fallback_filename = "ccw_q_table.pkl"

    # Select existing file path
    if os.path.exists(model_filename):
        load_path = model_filename
    elif os.path.exists(fallback_filename):
        load_path = fallback_filename
    else:
        # If neither exists, attempt to load model_filename (which will print error/raise if missing)
        load_path = model_filename

    try:
        agent.load(load_path)
    except Exception as e:
        print("[Evaluate] Warning: Failed to load {}: {}. Agent will evaluate with zeroed Q-table.".format(load_path, e))

    # Set epsilon = 0.0 for pure exploitation
    epsilon = 0.0
    print("[Evaluate] Epsilon set to 0.0 (Pure Exploitation Mode).")

    iteration = 0
    try:
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                print("[Evaluate] Reached maximum evaluation iterations ({}).".format(max_iterations))
                break

            # RULE D: Non-RL Reflex for Obstacle Avoidance
            if robot.read_ir() < settings.OBSTACLE_DISTANCE_THRESHOLD:
                print("[Evaluate] Obstacle detected by IR sensor! Executing reflex.")
                hardcoded_obstacle_avoidance(robot)
                continue

            # 1. Read current intensity gradient
            intensity = robot.read_intensity()

            # 2. Get state
            state = env.get_state(intensity)

            # 3. Select best action (pure exploitation)
            action = agent.choose_action(state, epsilon=0.0)

            # 4. Execute action
            robot.execute_action(action)
            wait(settings.DEFAULT_STEP_TIME_MS)

            iteration += 1

    except KeyboardInterrupt:
        print("\n[Evaluate] Program interrupted by user.")
    finally:
        robot.stop()
        print("[Evaluate] Robot safely stopped.")


if __name__ == "__main__":
    # If run as main, execute evaluation loop
    evaluate_agent(use_simulator=False)
