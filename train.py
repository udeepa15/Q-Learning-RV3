#!/usr/bin/env pybricks-micropython
"""Phased training for the Q-learning line follower."""

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
    from pybricks.tools import wait, StopWatch
except ImportError:
    import time

    def wait(ms):
        time.sleep(ms / 1000.0)

    class StopWatch:
        def __init__(self):
            self._start = time.time()

        def time(self):
            return int((time.time() - self._start) * 1000)

from config import settings
from hardware.robot import RobotInterface
from hardware.reflexes import calibrate_color_sensor
from core.agent import QLearningAgent
from core.environment import (Environment, STATE_PERFECT_EDGE, STATE_LIGHT_DRIFT_WHITE,
                              STATE_LIGHT_DRIFT_BLACK, STATE_TOTALLY_LOST)

PHASE_STRAIGHT = "STRAIGHT"
PHASE_TURN = "TURN"

PHASES = {
    PHASE_STRAIGHT: (settings.STRAIGHT_STATES, settings.STRAIGHT_EPISODE_MS),
    PHASE_TURN: (settings.TURN_STATES, settings.TURN_EPISODE_MS),
}


def file_exists(filename):
    """Check if a file exists."""
    try:
        os.stat(filename)
        return True
    except Exception:
        return False


def ensure_parent_dir(path):
    if "/" in path:
        model_dir = path.rsplit("/", 1)[0]
        if model_dir:
            try:
                os.mkdir(model_dir)
            except Exception:
                pass


def generate_dated_filename(base_path):
    """Add the date and time to a file name."""
    import time
    t = time.localtime()
    date_str = "{:04d}-{:02d}-{:02d}_{:02d}-{:02d}".format(t[0], t[1], t[2], t[3], t[4])

    if "." in base_path:
        parts = base_path.rsplit(".", 1)
        return "{}_{}.{}".format(parts[0], date_str, parts[1])
    else:
        return "{}_{}".format(base_path, date_str)


def has_ev3_buttons(robot):
    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        return False
    try:
        from pybricks.parameters import Button  # noqa: F401
        return True
    except ImportError:
        return False


def wait_for_choice(robot, title, options):
    """Show a menu and wait for a button (or a typed answer on a PC)."""
    print("\n==================================================")
    print(" " + title)
    print("==================================================")
    for button_name, key_char, _, label in options:
        print(" -> {} Button / '{}' : {}".format(button_name, key_char, label))
    print("==================================================\n")

    if has_ev3_buttons(robot):
        from pybricks.parameters import Button
        buttons = [(getattr(Button, name), result) for name, _, result, _ in options]
        while True:
            pressed = robot.ev3.buttons.pressed()
            for button, result in buttons:
                if button in pressed:
                    try:
                        robot.ev3.speaker.beep(frequency=1000, duration=120)
                    except Exception:
                        pass
                    while button in robot.ev3.buttons.pressed():
                        wait(50)
                    wait(200)
                    return result
            wait(50)

    try:
        user_choice = input("Enter option (default '{}'): ".format(options[0][1])).strip()
    except (EOFError, RuntimeError):
        user_choice = ""
    for _, key_char, result, _ in options:
        if user_choice == key_char:
            return result
    return options[0][2]


def prompt_save_q_table(agent, save_path, robot):
    """Ask where to save the table, or discard it."""
    dated_path = generate_dated_filename(save_path)

    chosen_action = wait_for_choice(robot, "TRAINING FINISHED: SAVE MENU", [
        ("UP", "1", 'default', "Default Path ({})".format(save_path)),
        ("RIGHT", "2", 'dated', "Date-Stamped ({})".format(dated_path)),
        ("DOWN", "3", 'discard', "DISCARD (Do not save)"),
    ])

    if chosen_action == 'discard':
        print("\n[Train] DISCARDED: Q-table updates were NOT saved.")
        return None

    target_path = dated_path if chosen_action == 'dated' else save_path
    ensure_parent_dir(target_path)
    agent.save(target_path)
    print("\n[Train] SUCCESS: Updated Q-table saved to:", target_path)
    return target_path


def fallback_action(agent, state, phase):
    """Action used in rows this phase does not train."""
    if phase == PHASE_STRAIGHT:
        if state == STATE_TOTALLY_LOST:
            return settings.ACTION_REVERSE
        proxy = STATE_LIGHT_DRIFT_WHITE if state < STATE_PERFECT_EDGE else STATE_LIGHT_DRIFT_BLACK
        return agent.choose_action(proxy, 0.0)
    return agent.choose_action(state, 0.0)


def run_episode(robot, env, agent, phase, epsilon):
    """Run one timed episode, updating only the cells this phase owns."""
    phase_states, duration_ms = PHASES[phase]

    settings.reset_direction()
    env.reset()
    updates = 0
    off_phase_steps = 0
    lost_steps = 0
    total_reward = 0.0

    state = env.get_state(robot.read_intensity())
    watch = StopWatch()

    while watch.time() < duration_ms:
        learning = state in phase_states
        if learning:
            action = agent.choose_action(state, epsilon)
        else:
            action = fallback_action(agent, state, phase)
            off_phase_steps += 1

        robot.execute_action(action)
        wait(settings.TRAIN_STEP_TIME_MS)

        next_state = env.get_state(robot.read_intensity())
        if next_state == STATE_TOTALLY_LOST:
            lost_steps += 1

        if learning:
            reward = env.calculate_reward(state, action) + env.progress_reward(state, next_state)
            agent.update(state, action, reward, next_state)
            total_reward += reward
            updates += 1

        state = next_state

    robot.stop()
    return updates, total_reward, off_phase_steps, lost_steps


def run_phase(robot, env, agent, phase, metrics_log):
    """Run episodes for one phase until the user saves."""
    duration_s = PHASES[phase][1] / 1000.0
    epsilon = settings.PHASE_EPSILON_START
    episode = 0

    print("\n##################################################")
    print(" {} PHASE ({:.0f}s episodes)".format(phase, duration_s))
    print("##################################################")

    while True:
        episode += 1
        wait_for_choice(robot, "{} episode {}: place robot, then START".format(phase, episode), [
            ("CENTER", "", None, "Start episode (epsilon={:.3f})".format(epsilon)),
        ])

        before = agent.snapshot()
        updates, total_reward, off_phase, lost = run_episode(robot, env, agent, phase, epsilon)

        print("[Train] {} episode {} | Updates: {} | Reward: {:.1f} | Off-phase steps: {} | Lost steps: {}".format(
            phase, episode, updates, total_reward, off_phase, lost))

        choice = wait_for_choice(robot, "{} episode {} finished".format(phase, episode), [
            ("UP", "1", 'next', "Keep updates, run NEXT episode"),
            ("LEFT", "2", 'redo', "DISCARD this episode's updates, redo"),
            ("DOWN", "3", 'save', "Keep updates, SAVE table and finish {} phase".format(phase)),
        ])

        if choice == 'redo':
            agent.restore(before)
            episode -= 1
            print("[Train] Episode discarded. Q-table restored.")
            continue

        metrics_log.append([phase, episode, updates, total_reward, off_phase, lost, epsilon])
        epsilon = max(settings.PHASE_EPSILON_MIN, epsilon * settings.PHASE_EPSILON_DECAY)

        if choice == 'save':
            return


def write_metrics(metrics_log, csv_filename="training_metrics_phased.csv"):
    try:
        with open(csv_filename, 'w') as f:
            f.write("phase,episode,updates,total_reward,off_phase_steps,lost_steps,epsilon\n")
            for row in metrics_log:
                f.write("{},{},{},{},{},{},{}\n".format(*row))
        print("[Train] Metrics logged successfully to:", csv_filename)
    except Exception as e:
        print("[Train] Error writing metrics CSV:", e)


def train_agent(save_path="models/cw_q_table_8state.pkl", use_simulator=False):
    robot = RobotInterface(use_simulator=use_simulator)
    calibrate_color_sensor(robot)

    env = Environment()
    agent = QLearningAgent(n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS,
                           alpha=settings.PHASE_ALPHA, gamma=settings.PHASE_GAMMA)
    metrics_log = []
    checkpoint = settings.STRAIGHT_CHECKPOINT_PATH

    start_phase = PHASE_STRAIGHT
    if file_exists(checkpoint):
        start_phase = wait_for_choice(robot, "Straight-phase checkpoint found: {}".format(checkpoint), [
            ("UP", "1", PHASE_STRAIGHT, "Start FRESH (all-zero table, straight phase)"),
            ("RIGHT", "2", PHASE_TURN, "Load checkpoint, go to TURN phase"),
        ])
        if start_phase == PHASE_TURN:
            try:
                agent.load(checkpoint)
            except Exception as e:
                print("[Train] Could not load checkpoint ({}). Starting straight phase fresh.".format(e))
                start_phase = PHASE_STRAIGHT

    print("[Train] alpha={} gamma={} | Q-table start: {}".format(
        settings.PHASE_ALPHA, settings.PHASE_GAMMA,
        "all zeros" if start_phase == PHASE_STRAIGHT else checkpoint))

    try:
        if start_phase == PHASE_STRAIGHT:
            run_phase(robot, env, agent, PHASE_STRAIGHT, metrics_log)
            ensure_parent_dir(checkpoint)
            agent.save(checkpoint)
            print("[Train] Straight phase saved to:", checkpoint)

            go_on = wait_for_choice(robot, "STRAIGHT PHASE COMPLETE", [
                ("UP", "1", True, "Continue to TURN phase"),
                ("DOWN", "2", False, "Stop here (straight table already saved)"),
            ])
            if not go_on:
                return agent, robot

        run_phase(robot, env, agent, PHASE_TURN, metrics_log)
        if not any(agent.q_table[STATE_TOTALLY_LOST]):
            print("\n[Train] WARNING: the Lost row (state 7) was never trained. Run more TURN episodes starting with the robot 15+ cm onto black for {} ms or longer.".format(settings.LOST_TIME_MS))
        prompt_save_q_table(agent, save_path, robot)
    finally:
        robot.stop()
        write_metrics(metrics_log)

    return agent, robot


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "models/cw_q_table_8state.pkl"
    train_agent(save_path=target_file)
