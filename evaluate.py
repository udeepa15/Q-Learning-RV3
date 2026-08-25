#!/usr/bin/env pybricks-micropython
"""
Evaluation / Final Exam Script for EV3 Q-Learning Line Follower Agent.
"""

import sys
import os

# Ensure project root is in sys.path (MicroPython os.path fallback)
try:
    import os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
except (ImportError, AttributeError, NameError):
    current_dir = "."

if current_dir and current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from pybricks.tools import wait
except ImportError:
    import time
    def wait(ms):
        time.sleep(ms / 1000.0)

from config import settings
from hardware.robot import RobotInterface
from hardware.reflexes import hardcoded_obstacle_avoidance, calibrate_color_sensor
from core.agent import QLearningAgent
from core.environment import Environment


def file_exists(filename):
    """MicroPython safe file existence check."""
    try:
        os.stat(filename)
        return True
    except Exception:
        return False


def list_saved_q_tables(models_dir="models"):
    """
    Lists saved Q-table .pkl files in models_dir, most recently modified first.
    """
    try:
        filenames = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    except Exception:
        return []

    try:
        filenames.sort(key=lambda f: os.stat(models_dir + "/" + f)[8], reverse=True)  # mtime
    except Exception:
        filenames.sort()

    return [models_dir + "/" + f for f in filenames]


def select_q_table(robot, default_path="models/cw_q_table_8state.pkl"):
    """
    Prompts the user to pick which saved Q-table to load for evaluation.
      - UP/DOWN Buttons : Browse saved Q-tables (most recent first)
      - CENTER Button   : Select the highlighted Q-table
    Supports EV3 brick button menu and terminal input fallback for simulator mode.
    Returns the chosen file path, or default_path if none are found.
    """
    candidates = list_saved_q_tables()
    if not candidates:
        print("[Evaluate] No saved Q-tables found in models/. Falling back to {}.".format(default_path))
        return default_path

    if len(candidates) == 1:
        print("[Evaluate] Found one saved Q-table: {}. Using it.".format(candidates[0]))
        return candidates[0]

    has_buttons = False
    if not robot.is_simulated and hasattr(robot, 'ev3') and robot.ev3 is not None:
        try:
            from pybricks.parameters import Button
            has_buttons = True
        except ImportError:
            has_buttons = False

    if has_buttons:
        from pybricks.parameters import Button
        index = 0

        def show_current():
            print("\n==================================================")
            print("         SELECT Q-TABLE FOR EVALUATION             ")
            print("==================================================")
            print(" [{}/{}] {}".format(index + 1, len(candidates), candidates[index]))
            print(" -> UP/DOWN Button : Browse    CENTER Button : Select")
            print("==================================================\n")

        show_current()
        while True:
            pressed = robot.ev3.buttons.pressed()
            if Button.UP in pressed:
                index = (index - 1) % len(candidates)
                try:
                    robot.ev3.speaker.beep(frequency=900, duration=80)
                except Exception:
                    pass
                show_current()
                wait(300)
            elif Button.DOWN in pressed:
                index = (index + 1) % len(candidates)
                try:
                    robot.ev3.speaker.beep(frequency=900, duration=80)
                except Exception:
                    pass
                show_current()
                wait(300)
            elif Button.CENTER in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=1200, duration=200)
                except Exception:
                    pass
                wait(500)
                print("[Evaluate] Selected Q-table: {}".format(candidates[index]))
                return candidates[index]
            wait(100)
    else:
        print("\n==================================================")
        print("         SELECT Q-TABLE FOR EVALUATION             ")
        print("==================================================")
        for i, path in enumerate(candidates):
            print(" {}. {}".format(i + 1, path))
        print("==================================================")
        try:
            choice = input("Enter number [1-{}] (default: 1 = most recent): ".format(len(candidates))).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1]
            return candidates[0]
        except (EOFError, RuntimeError):
            print("[Evaluate] Non-interactive environment. Using most recent Q-table: {}".format(candidates[0]))
            return candidates[0]


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

    load_path = select_q_table(robot, default_path="models/cw_q_table_8state.pkl")

    try:
        agent.load(load_path)
    except Exception as e:
        print("[Evaluate] Warning: Failed to load {}: {}. Agent will evaluate with the initial heuristic Q-table.".format(load_path, e))

    # Interactive Sensor Calibration (Pure White, Pure Black, Perfect Edge) -- run
    # after the Q-table is picked so evaluation always starts from a fresh reading
    # of the current track surface, not stale thresholds from a past run.
    calibrate_color_sensor(robot)

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
                iteration += 1
                continue

            # 1. Read current intensity gradient
            intensity = robot.read_intensity()

            # 2. Get state
            state = env.get_state(intensity)

            # 3. Select best action (pure exploitation)
            action = agent.choose_action(state, epsilon)

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
