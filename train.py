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
from hardware.reflexes import hardcoded_obstacle_avoidance, calibrate_color_sensor
from core.agent import QLearningAgent
from core.environment import Environment, STATE_TOTALLY_LOST


def file_exists(filename):
    """MicroPython safe file existence check."""
    try:
        os.stat(filename)
        return True
    except Exception:
        return False


def select_model_initialization(robot, save_path, force_fresh=False):
    """
    Prompts the user on EV3 brick before training starts:
      - UP Button   : Continue training previously saved Q-table
      - DOWN Button : Restart fresh from hardcoded heuristic table
    """
    if force_fresh or not file_exists(save_path):
        print("[Train] No previously saved Q-table found at {}. Initializing fresh from hardcoded heuristic table.".format(save_path))
        return False  # False = start fresh from heuristic table

    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        print("[Train] Simulator mode. Defaulting to continuing previous saved model: {}".format(save_path))
        return True

    try:
        from pybricks.parameters import Button
    except ImportError:
        return True

    print("\n==================================================")
    print("        Q-TABLE INITIALIZATION MENU               ")
    print("==================================================")
    print(" Previously saved model detected: {}".format(save_path))
    print(" -> Press UP Button   : CONTINUE Training Previous Saved Model")
    print(" -> Press DOWN Button : RESTART Fresh (Hardcoded Heuristic Table)")
    print(" (Waiting for button press...)")
    print("==================================================\n")

    while True:
        pressed = robot.ev3.buttons.pressed()
        if Button.UP in pressed:
            try:
                robot.ev3.speaker.beep(frequency=1000, duration=150)
            except Exception:
                pass
            print("[Train] Button Pressed: UP -> Continuing training previous saved model.")
            wait(500)
            return True
        elif Button.DOWN in pressed:
            try:
                robot.ev3.speaker.beep(frequency=600, duration=150)
            except Exception:
                pass
            print("[Train] Button Pressed: DOWN -> Restarting fresh from hardcoded heuristic table.")
            wait(500)
            return False
        wait(100)


def train_agent(num_episodes=40, max_steps_per_episode=60, save_path=None, use_simulator=False, force_fresh=False):

    """
    Main RL Training loop for 5-State Q-Learning line follower.
    Supports interactive sensor calibration, model initialization prompt, retraining, and CSV metrics logging.
    """
    robot = RobotInterface(use_simulator=use_simulator)

    # 1. Interactive Sensor Calibration (Pure White, Pure Black, Perfect Edge)
    calibrate_color_sensor(robot)
    
    if save_path is None:
        save_path = "models/cw_q_table_8state.pkl"


    agent = QLearningAgent(n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS)
    env = Environment()

    # 2. Retraining vs Fresh Start Prompt Menu
    use_saved_model = select_model_initialization(robot, save_path, force_fresh=force_fresh)
    if use_saved_model:
        try:
            agent.load(save_path)
            print("[Train] RETRAINING MODE: Successfully loaded existing Q-table from {}.".format(save_path))
        except Exception as e:
            print("[Train] Could not load saved Q-table ({}). Initializing with heuristic table.".format(e))
    else:
        print("[Train] FRESH START MODE: Initialized agent with hardcoded heuristic Q-values.")

    epsilon = settings.EPSILON_START
    metrics_log = []

    print("==================================================")
    print("Starting Q-Learning Training (5-State Clockwise Mode)...")
    print("Episodes: {}, Max Steps/Episode: {}".format(num_episodes, max_steps_per_episode))
    print("Target Q-Table File: {}".format(save_path))
    print("==================================================")

    lost_state_id = STATE_TOTALLY_LOST



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
    csv_filename = "training_metrics_cw_5state.csv"


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
    train_agent(num_episodes=40, max_steps_per_episode=60, save_path=target_file)

