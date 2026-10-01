#!/usr/bin/env pybricks-micropython
"""
Training Script for EV3 Q-Learning Line Follower Agent.
Supports EV3 button selection for CW/CCW track direction, interactive sensor calibration,
retraining, and end-of-training model save/discard options (with date stamping).
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


def generate_dated_filename(base_path):
    """
    Generates a filename with current date and time stamp.
    e.g., models/cw_q_table_8state.pkl -> models/cw_q_table_8state_2026-08-07_09-24.pkl
    """
    import time
    t = time.localtime()
    date_str = "{:04d}-{:02d}-{:02d}_{:02d}-{:02d}".format(t[0], t[1], t[2], t[3], t[4])

    if "." in base_path:
        parts = base_path.rsplit(".", 1)
        return "{}_{}.{}".format(parts[0], date_str, parts[1])
    else:
        return "{}_{}".format(base_path, date_str)


def prompt_save_q_table(agent, save_path, robot):
    """
    Prompts user at the end of training to:
      1. Save Q-table to default path
      2. Save Q-table with current date-time stamp in models directory
      3. Discard Q-table (do not save)
    Supports EV3 brick button menu and terminal input fallback for simulator mode.
    """
    dated_path = generate_dated_filename(save_path)

    has_buttons = False
    if not robot.is_simulated and hasattr(robot, 'ev3') and robot.ev3 is not None:
        try:
            from pybricks.parameters import Button
            has_buttons = True
        except ImportError:
            has_buttons = False

    print("\n==================================================")
    print("           TRAINING FINISHED: SAVE MENU            ")
    print("==================================================")
    print(" Choose how to save the updated Q-table:")
    print(" -> Option 1 (UP Button / '1')    : Default Path ({})".format(save_path))
    print(" -> Option 2 (RIGHT Button / '2') : Date-Stamped ({})".format(dated_path))
    print(" -> Option 3 (DOWN Button / '3')  : DISCARD (Do not save)")
    print("==================================================\n")

    chosen_action = None

    if has_buttons:
        from pybricks.parameters import Button
        print("[Train] Waiting for EV3 button press (UP=Default, RIGHT=Date-Stamped, DOWN=Discard)...")
        while True:
            pressed = robot.ev3.buttons.pressed()
            if Button.UP in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=1000, duration=150)
                except Exception:
                    pass
                chosen_action = 'default'
                wait(500)
                break
            elif Button.RIGHT in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=1200, duration=150)
                except Exception:
                    pass
                chosen_action = 'dated'
                wait(500)
                break
            elif Button.DOWN in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=500, duration=300)
                except Exception:
                    pass
                chosen_action = 'discard'
                wait(500)
                break
            wait(100)
    else:
        try:
            user_choice = input("Enter option [1=Default, 2=Date-Stamped, 3=Discard] (default: 1): ").strip()
            if user_choice == "2":
                chosen_action = 'dated'
            elif user_choice == "3":
                chosen_action = 'discard'
            else:
                chosen_action = 'default'
        except (EOFError, RuntimeError):
            print("[Train] Non-interactive environment. Saving to default path.")
            chosen_action = 'default'

    if chosen_action == 'discard':
        print("\n[Train] DISCARDED: Q-table updates were NOT saved.")
        return None

    target_path = dated_path if chosen_action == 'dated' else save_path

    if "/" in target_path:
        model_dir = target_path.rsplit("/", 1)[0]
        if model_dir:
            try:
                os.mkdir(model_dir)
            except Exception:
                pass

    agent.save(target_path)
    print("\n[Train] SUCCESS: Updated Q-table saved to:", target_path)
    return target_path


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


def train_agent(num_episodes=None, max_steps_per_episode=None, save_path=None,
                 use_simulator=False, force_fresh=False, robot=None, agent=None):
    """
    Main RL Training loop for 8-State Q-Learning line follower.
    Supports interactive sensor calibration, model initialization prompt, retraining, and CSV metrics logging.

    Pass an existing `robot` to reuse hardware across chained training sessions
    (skips re-calibration). Pass an existing `agent` to keep training the same
    in-memory Q-table across sessions (skips the load/fresh-start menu).
    Returns (agent, robot) so callers can chain further sessions.
    """
    if num_episodes is None:
        num_episodes = settings.NUM_EPISODES
    if max_steps_per_episode is None:
        max_steps_per_episode = settings.MAX_STEPS_PER_EPISODE

    if save_path is None:
        save_path = "models/cw_q_table_8state.pkl"

    if robot is None:
        robot = RobotInterface(use_simulator=use_simulator)
        # 1. Interactive Sensor Calibration (Pure White, Pure Black, Perfect Edge)
        calibrate_color_sensor(robot)

    env = Environment()

    if agent is None:
        agent = QLearningAgent(n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS)

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
    else:
        print("[Train] CONTINUING TRAINING: Reusing in-memory Q-table from the previous session.")

    epsilon = settings.EPSILON_START
    metrics_log = []

    print("==================================================")
    print("Starting Q-Learning Training (8-State Clockwise Mode)...")
    print("Episodes: {}, Max Steps/Episode: {}".format(num_episodes, max_steps_per_episode))
    print("Target Q-Table File: {}".format(save_path))
    print("==================================================")

    lost_state_id = STATE_TOTALLY_LOST

    for episode in range(1, num_episodes + 1):
        settings.reset_direction()
        env.reset()
        episode_reward = 0.0
        hard_corrections = 0
        fatal_off_track = False

        for step in range(1, max_steps_per_episode + 1):
            if robot.read_ir() < settings.OBSTACLE_DISTANCE_THRESHOLD:
                print("[Train] Episode {}, Step {}: IR sensor triggered. Skipping Q-update.".format(episode, step))
                hardcoded_obstacle_avoidance(robot)
                env.reset()
                continue

            intensity = robot.read_intensity()
            state = env.get_state(intensity)

            if state == lost_state_id:
                fatal_off_track = True

            action = agent.choose_action(state, epsilon)

            if action == settings.ACTION_SHARP_LEFT or action == settings.ACTION_SHARP_RIGHT:
                hard_corrections += 1

            robot.execute_action(action)
            wait(settings.DEFAULT_STEP_TIME_MS)

            next_intensity = robot.read_intensity()
            next_state = env.get_state(next_intensity)

            if next_state == lost_state_id:
                fatal_off_track = True

            reward = env.calculate_reward(state, action) + env.progress_reward(state, next_state)
            episode_reward += reward

            agent.update(state, action, reward, next_state)

        epsilon = max(settings.EPSILON_MIN, epsilon * settings.EPSILON_DECAY)

        lap_completed = not fatal_off_track

        metrics_log.append([episode, hard_corrections, lap_completed, episode_reward])

        print("Episode {:2d}/{} completed | Corrections: {:2d} | Lap Completed: {} | Reward: {:6.1f} | Epsilon: {:.4f}".format(
            episode, num_episodes, hard_corrections, lap_completed, episode_reward, epsilon))

        agent.display_q_table()

    robot.stop()

    prompt_save_q_table(agent, save_path, robot)

    csv_filename = "training_metrics_cw_8state.csv"

    try:
        with open(csv_filename, 'w') as f:
            f.write("episode,hard_corrections,lap_completed,total_reward\n")
            for row in metrics_log:
                f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))
        print("[Train] Metrics logged successfully to:", csv_filename)
    except Exception as e:
        print("[Train] Error writing metrics CSV:", e)

    return agent, robot


def prompt_continue_or_stop_training(robot):
    """
    Prompts user after a training session completes:
      - UP Button   : Continue Training (start another session)
      - DOWN Button : Stop Training
    Supports EV3 brick button menu and terminal input fallback for simulator mode.
    """
    has_buttons = False
    if not robot.is_simulated and hasattr(robot, 'ev3') and robot.ev3 is not None:
        try:
            from pybricks.parameters import Button
            has_buttons = True
        except ImportError:
            has_buttons = False

    print("\n==================================================")
    print("              CONTINUE TRAINING?                   ")
    print("==================================================")
    print(" -> Option 1 (UP Button / '1')   : CONTINUE Training")
    print(" -> Option 2 (DOWN Button / '2') : STOP Training")
    print("==================================================\n")

    if has_buttons:
        from pybricks.parameters import Button
        print("[Train] Waiting for EV3 button press (UP=Continue, DOWN=Stop)...")
        while True:
            pressed = robot.ev3.buttons.pressed()
            if Button.UP in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=1000, duration=150)
                except Exception:
                    pass
                wait(500)
                return True
            elif Button.DOWN in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=500, duration=300)
                except Exception:
                    pass
                wait(500)
                return False
            wait(100)
    else:
        try:
            user_choice = input("Enter option [1=Continue, 2=Stop] (default: 2): ").strip()
            return user_choice == "1"
        except (EOFError, RuntimeError):
            print("[Train] Non-interactive environment. Stopping training.")
            return False


def prompt_same_or_new_table(robot):
    """
    Prompts user (after choosing to continue) whether to:
      - UP Button   : Keep training the SAME Q-table (continues in-memory)
      - DOWN Button : Train a NEW Q-table (fresh heuristic start, saved separately)
    Returns True to keep the same table, False to start a new one.
    """
    has_buttons = False
    if not robot.is_simulated and hasattr(robot, 'ev3') and robot.ev3 is not None:
        try:
            from pybricks.parameters import Button
            has_buttons = True
        except ImportError:
            has_buttons = False

    print("\n==================================================")
    print("        SAME Q-TABLE OR NEW Q-TABLE?               ")
    print("==================================================")
    print(" -> Option 1 (UP Button / '1')   : SAME Q-table (keep training it)")
    print(" -> Option 2 (DOWN Button / '2') : NEW Q-table (fresh heuristic start)")
    print("==================================================\n")

    if has_buttons:
        from pybricks.parameters import Button
        print("[Train] Waiting for EV3 button press (UP=Same Table, DOWN=New Table)...")
        while True:
            pressed = robot.ev3.buttons.pressed()
            if Button.UP in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=1000, duration=150)
                except Exception:
                    pass
                wait(500)
                return True
            elif Button.DOWN in pressed:
                try:
                    robot.ev3.speaker.beep(frequency=700, duration=150)
                except Exception:
                    pass
                wait(500)
                return False
            wait(100)
    else:
        try:
            user_choice = input("Enter option [1=Same Table, 2=New Table] (default: 1): ").strip()
            return user_choice != "2"
        except (EOFError, RuntimeError):
            print("[Train] Non-interactive environment. Continuing same table.")
            return True


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    save_path = target_file or "models/cw_q_table_8state.pkl"

    session_agent = None
    session_robot = None
    force_fresh = False

    while True:
        session_agent, session_robot = train_agent(
            save_path=save_path,
            force_fresh=force_fresh,
            robot=session_robot,
            agent=session_agent,
        )

        if not prompt_continue_or_stop_training(session_robot):
            print("[Train] Training session ended.")
            break

        if prompt_same_or_new_table(session_robot):
            print("[Train] Continuing training on the SAME Q-table.")
            force_fresh = False
        else:
            print("[Train] Starting a NEW Q-table for the next session.")
            base_path = target_file or "models/cw_q_table_8state.pkl"
            save_path = generate_dated_filename(base_path)
            session_agent = None
            force_fresh = True
