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


def delete_q_table(path):
    """MicroPython safe file delete."""
    try:
        os.remove(path)
        print("[Evaluate] Deleted Q-table: {}".format(path))
        return True
    except Exception as e:
        print("[Evaluate] ERROR: Could not delete {}: {}".format(path, e))
        return False


def select_q_table(robot, default_path="models/cw_q_table_8state.pkl"):
    """
    Prompts the user to pick which saved Q-table to load for evaluation, with
    the option to clean up (delete) old/unwanted tables from the list first.
      - UP/DOWN Buttons : Browse saved Q-tables (most recent first)
      - CENTER Button   : Select the highlighted Q-table
      - LEFT Button     : Delete the highlighted Q-table (press twice to confirm)
    Supports EV3 brick button menu and terminal input fallback for simulator mode.
    Returns the chosen file path, or default_path if none are found/left.
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
        pending_delete = False

        def show_current():
            print("\n==================================================")
            print("         SELECT Q-TABLE FOR EVALUATION             ")
            print("==================================================")
            print(" [{}/{}] {}".format(index + 1, len(candidates), candidates[index]))
            print(" -> UP/DOWN Button : Browse    CENTER Button : Select")
            print(" -> LEFT Button    : Delete (press twice to confirm)")
            print("==================================================\n")

        show_current()
        while True:
            pressed = robot.ev3.buttons.pressed()
            if Button.UP in pressed:
                index = (index - 1) % len(candidates)
                pending_delete = False
                try:
                    robot.ev3.speaker.beep(frequency=900, duration=80)
                except Exception:
                    pass
                show_current()
                wait(300)
            elif Button.DOWN in pressed:
                index = (index + 1) % len(candidates)
                pending_delete = False
                try:
                    robot.ev3.speaker.beep(frequency=900, duration=80)
                except Exception:
                    pass
                show_current()
                wait(300)
            elif Button.LEFT in pressed:
                if not pending_delete:
                    pending_delete = True
                    try:
                        robot.ev3.speaker.beep(frequency=400, duration=150)
                    except Exception:
                        pass
                    print("[Evaluate] Press LEFT again to DELETE '{}', or UP/DOWN to cancel.".format(candidates[index]))
                    wait(400)
                else:
                    try:
                        robot.ev3.speaker.beep(frequency=300, duration=400)
                    except Exception:
                        pass
                    delete_q_table(candidates[index])
                    candidates.pop(index)
                    pending_delete = False
                    if not candidates:
                        print("[Evaluate] No Q-tables remain. Falling back to {}.".format(default_path))
                        return default_path
                    index = index % len(candidates)
                    show_current()
                    wait(400)
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
        while True:
            print("\n==================================================")
            print("         SELECT Q-TABLE FOR EVALUATION             ")
            print("==================================================")
            for i, path in enumerate(candidates):
                print(" {}. {}".format(i + 1, path))
            print("==================================================")
            try:
                choice = input(
                    "Enter number [1-{}] to evaluate, or 'd<number>' to delete (e.g. d2) (default: 1): ".format(len(candidates))
                ).strip()
            except (EOFError, RuntimeError):
                print("[Evaluate] Non-interactive environment. Using most recent Q-table: {}".format(candidates[0]))
                return candidates[0]

            if choice[:1] in ("d", "D") and choice[1:].isdigit() and 1 <= int(choice[1:]) <= len(candidates):
                del_index = int(choice[1:]) - 1
                del_path = candidates[del_index]
                try:
                    confirm = input("Delete '{}'? [y/N]: ".format(del_path)).strip().lower()
                except (EOFError, RuntimeError):
                    confirm = "n"
                if confirm == "y":
                    delete_q_table(del_path)
                    candidates.pop(del_index)
                    if not candidates:
                        print("[Evaluate] No Q-tables remain. Falling back to {}.".format(default_path))
                        return default_path
                continue

            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1]
            return candidates[0]


def is_phased_table(q_table):
    """True if every non-zero cell lies in the columns its row owns under phased training."""
    for state, row in enumerate(q_table):
        allowed = settings.actions_for_state(state)
        for action, q_val in enumerate(row):
            if q_val != 0.0 and action not in allowed:
                return False
    return True


def wait_for_start(robot):
    """Blocks until CENTER is pressed (Enter in simulator mode) so the robot can be placed first."""
    print("\n==================================================")
    print(" Place robot on the line -> Press CENTER to START EVALUATION")
    print("==================================================\n")
    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        return
    try:
        from pybricks.parameters import Button
    except ImportError:
        return
    while Button.CENTER not in robot.ev3.buttons.pressed():
        wait(50)
    try:
        robot.ev3.speaker.beep(frequency=1200, duration=200)
    except Exception:
        pass
    while Button.CENTER in robot.ev3.buttons.pressed():
        wait(50)
    wait(300)


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
    if not calibrate_color_sensor(robot, start_label="EVALUATION"):
        wait_for_start(robot)

    phased = is_phased_table(agent.q_table)
    if phased:
        print("[Evaluate] Phased Q-table detected: each row only uses the columns it was trained on.")

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
            allowed = settings.actions_for_state(state) if phased else None
            action = agent.choose_action(state, epsilon, allowed)

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
