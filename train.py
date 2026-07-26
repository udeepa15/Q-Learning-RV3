"""
Training Script for EV3 Q-Learning Line Follower Agent.
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
from hardware.reflexes import hardcoded_obstacle_avoidance
from core.agent import QLearningAgent
from core.environment import Environment


def train_agent(num_episodes=10, max_steps_per_episode=100, save_path="models/cw_q_table.pkl", use_simulator=False):
    """
    Main RL Training loop for Q-Learning line follower.
    """
    robot = RobotInterface(use_simulator=use_simulator)
    agent = QLearningAgent(n_states=settings.NUM_STATES, n_actions=settings.NUM_ACTIONS)
    env = Environment()

    epsilon = settings.EPSILON_START
    print("==================================================")
    print("Starting Q-Learning Training...")
    print("Episodes: {}, Max Steps/Episode: {}".format(num_episodes, max_steps_per_episode))
    print("Initial Epsilon: {}, Alpha: {}, Gamma: {}".format(epsilon, settings.ALPHA, settings.GAMMA))
    print("==================================================")

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0

        for step in range(1, max_steps_per_episode + 1):
            # RULE D: Non-RL Reflex Interrupt for Obstacle Avoidance
            if robot.read_ir() < settings.OBSTACLE_DISTANCE_THRESHOLD:
                print("[Train] Episode {}, Step {}: IR sensor triggered (<20cm). Skipping Q-update.".format(episode, step))
                hardcoded_obstacle_avoidance(robot)
                continue  # Skip Q-update for this step

            # 1. Observe current state
            intensity = robot.read_intensity()
            state = env.get_state(intensity)

            # 2. Select action via Epsilon-Greedy policy
            action = agent.choose_action(state, epsilon)

            # 3. Execute action
            robot.execute_action(action)
            wait(settings.DEFAULT_STEP_TIME_MS)

            # 4. Observe next state and calculate reward
            next_intensity = robot.read_intensity()
            next_state = env.get_state(next_intensity)
            reward = env.calculate_reward(state, action)
            episode_reward += reward

            # 5. Q-table Bellman update
            agent.update(state, action, reward, next_state)

        # Decay exploration rate after each episode
        epsilon = max(settings.EPSILON_MIN, epsilon * settings.EPSILON_DECAY)

        print("Episode {:2d}/{} completed | Total Reward: {:6.1f} | Epsilon: {:.4f}".format(
            episode, num_episodes, episode_reward, epsilon))

    robot.stop()

    # Create target directory if needed and save Q-table
    model_dir = os.path.dirname(save_path)
    if model_dir and not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
        except Exception:
            pass

    agent.save(save_path)
    print("Training finished successfully. Saved model to:", save_path)
    return agent


if __name__ == "__main__":
    # Check if user specified a save path arg or default
    target_file = sys.argv[1] if len(sys.argv) > 1 else "models/cw_q_table.pkl"
    train_agent(num_episodes=15, max_steps_per_episode=60, save_path=target_file)
