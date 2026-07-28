#!/usr/bin/env pybricks-micropython
"""
Training Script for EV3 Q-Learning Line Follower Agent.
Supports EV3 button selection for CW/CCW track direction and incremental Q-table retraining.
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
from hardware.reflexes import hardcoded_obstacle_avoidance, detect_track_direction, calibrate_color_sensor
from core.agent import QLearningAgent
from core.environment import Environment, STATE_TOTALLY_LOST, STATE_TOTALLY_LOST_5



def file_exists(filename):
    """MicroPython safe file existence check."""
    try:
        os.stat(filename)
        return True
    except Exception:
        return False


def select_training_configuration(robot):
    """
    Allows user to select State Mode (3 or 5) and Direction (CW or CCW) via EV3 buttons:
      Step 1: State Mode
        - Press UP Button    : 3-State Mode
        - Press DOWN Button  : 5-State Mode
      Step 2: Direction
        - Press LEFT Button  : CCW Mode
        - Press RIGHT Button : CW Mode
        - Press CENTER Button: Auto-Detect via Sweep
    """
    default_mode = getattr(settings, 'STATE_MODE', 5)
    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        print("[Train] Simulator mode. Defaulting to State Mode {}, CW direction.".format(default_mode))
        return default_mode, "CW"

    try:
        from pybricks.parameters import Button
    except ImportError:
        return default_mode, "CW"

    # Step 1: State Mode Selection (Blocks until UP or DOWN is pressed)
    print("\n==================================================")
    print("        STEP 1: SELECT STATE ARCHITECTURE         ")
    print(" -> Press UP Button   : 3-State Mode")
    print(" -> Press DOWN Button : 5-State Mode")
    print(" (Waiting for button press...)")
    print("==================================================\n")

    selected_mode = None
    while True:
        pressed = robot.ev3.buttons.pressed()
        if Button.UP in pressed:
            selected_mode = 3
            print("[Train] Button Pressed: UP -> Selected 3-State Architecture.")
            wait(500)
            break
        elif Button.DOWN in pressed:
            selected_mode = 5
            print("[Train] Button Pressed: DOWN -> Selected 5-State Architecture.")
            wait(500)
            break
        wait(100)

    # Step 2: Direction Selection (Blocks until LEFT, RIGHT, or CENTER is pressed)
    print("\n==================================================")
    print("        STEP 2: SELECT TRACK DIRECTION            ")
    print(" -> Press LEFT Button   : Train CCW (Counter-Clockwise)")
    print(" -> Press RIGHT Button  : Train CW  (Clockwise)")
    print(" -> Press CENTER Button : Auto-Detect via Sweep")
    print(" (Waiting for button press...)")
    print("==================================================\n")

    selected_dir = None
    while True:
        pressed = robot.ev3.buttons.pressed()
        if Button.LEFT in pressed:
            selected_dir = "CCW"
            print("[Train] Button Pressed: LEFT -> Selected CCW Mode.")
            wait(500)
            break
        elif Button.RIGHT in pressed:
            selected_dir = "CW"
            print("[Train] Button Pressed: RIGHT -> Selected CW Mode.")
            wait(500)
            break
        elif Button.CENTER in pressed:
            print("[Train] Button Pressed: CENTER -> Running Auto-Detect Sweep...")
            selected_dir = detect_track_direction(robot)
            break
        wait(100)

    # Sync settings.STATE_MODE
    settings.STATE_MODE = selected_mode

    return selected_mode, selected_dir



def train_agent(num_episodes=10, max_steps_per_episode=100, save_path=None, use_simulator=False, force_fresh=False):
    """
    Main RL Training loop for Q-Learning line follower.
    Supports interactive sensor calibration, incremental retraining of existing Q-tables, and logging to CSV.
    """
    robot = RobotInterface(use_simulator=use_simulator)

    # 1. Interactive Sensor Calibration (Pure White, Pure Black, Perfect Edge)
    calibrate_color_sensor(robot)

    # 2. Determine State Mode (3 vs 5) and Direction (CW vs CCW) via EV3 buttons
    state_mode, direction = select_training_configuration(robot)

    
    if save_path is None:
        save_path = "models/{}_q_table_{}state.pkl".format(direction.lower(), state_mode)

    agent = QLearningAgent(state_mode=state_mode, direction=direction, n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS)
    env = Environment()


    # 2. Retraining Logic: Load existing Q-table if available
    if not force_fresh and file_exists(save_path):
        try:
            agent.load(save_path)
            print("[Train] RETRAINING MODE: Loaded existing Q-table from {}. Continuing learning...".format(save_path))
        except Exception as e:
            print("[Train] Could not load existing Q-table ({}). Starting with heuristic table.".format(e))
    else:
        print("[Train] FRESH START MODE: Initializing agent with heuristic Q-values.")

    epsilon = settings.EPSILON_START
    metrics_log = []

    print("==================================================")
    print("Starting Q-Learning Training [Dir: {}, State Mode: {}]...".format(direction, state_mode))
    print("Episodes: {}, Max Steps/Episode: {}".format(num_episodes, max_steps_per_episode))
    print("Target Q-Table File: {}".format(save_path))
    print("==================================================")

    lost_state_id = STATE_TOTALLY_LOST if state_mode == 3 else STATE_TOTALLY_LOST_5

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        hard_corrections = 0
        fatal_off_track = False

        for step in range(1, max_steps_per_episode + 1):
            # RULE D: Non-RL Reflex Interrupt for Obstacle Avoidance
            if robot.read_ir() < settings.OBSTACLE_DISTANCE_THRESHOLD:
                print("[Train] Episode {}, Step {}: IR sensor triggered (<20cm). Skipping Q-update.".format(episode, step))
                hardcoded_obstacle_avoidance(robot)
                continue  # Skip Q-update for this step

            # 1. Observe current state
            intensity = robot.read_intensity()
            state = env.get_state(intensity)

            if state == lost_state_id:
                fatal_off_track = True

            # 2. Select action via Epsilon-Greedy policy
            action = agent.choose_action(state, epsilon)

            # Track hard corrections (Action 2: Sharp LFT, Action 4: Sharp RGT)
            if action == settings.ACTION_SHARP_LEFT or action == settings.ACTION_SHARP_RIGHT:
                hard_corrections += 1

            # 3. Execute action
            robot.execute_action(action)
            wait(settings.DEFAULT_STEP_TIME_MS)

            # 4. Observe next state and calculate reward
            next_intensity = robot.read_intensity()
            next_state = env.get_state(next_intensity)

            if next_state == lost_state_id:
                fatal_off_track = True

            reward = env.calculate_reward(state, action)
            episode_reward += reward

            # 5. Q-table Bellman update
            agent.update(state, action, reward, next_state)

        # Decay exploration rate after each episode
        epsilon = max(settings.EPSILON_MIN, epsilon * settings.EPSILON_DECAY)

        # Lap completed if agent completes max_steps without triggering fatal off-track penalty
        lap_completed = not fatal_off_track

        # Append episode metrics: [episode_number, hard_corrections, lap_completed, total_reward]
        metrics_log.append([episode, hard_corrections, lap_completed, episode_reward])

        print("Episode {:2d}/{} completed | Corrections: {:2d} | Lap Completed: {} | Reward: {:6.1f} | Epsilon: {:.4f}".format(
            episode, num_episodes, hard_corrections, lap_completed, episode_reward, epsilon))

        # Dynamic Q-table snapshot display after each episode
        agent.display_q_table()


    robot.stop()

    # Create target directory if needed and save Q-table (MicroPython compatible)
    if "/" in save_path:
        model_dir = save_path.rsplit("/", 1)[0]
        if model_dir:
            try:
                os.mkdir(model_dir)
            except Exception:
                pass

    agent.save(save_path)
    print("Training finished successfully. Saved updated Q-table to:", save_path)

    # Write metrics to CSV (MicroPython compatible file writer)
    csv_filename = "training_metrics_{}_{}state.csv".format(direction.lower(), state_mode)
    try:
        with open(csv_filename, 'w') as f:
            f.write("episode,hard_corrections,lap_completed,total_reward\n")
            for row in metrics_log:
                f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))
        print("[Train] Metrics logged successfully to:", csv_filename)
    except Exception as e:
        print("[Train] Error writing metrics CSV:", e)

    return agent


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    train_agent(num_episodes=15, max_steps_per_episode=60, save_path=target_file)
