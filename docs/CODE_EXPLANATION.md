# Code Explanation & System Architecture

This document provides a technical walkthrough of the Q-learning implementation, state/action representations, reward dynamics, and compliance with rules A, B, C, and D.

---

## 1. System Architecture & Module Responsibilities

```
ev3_rl_project/
├── config/
│   └── settings.py       # Hyperparameters, thresholds, action speeds & port mappings
├── hardware/
│   ├── robot.py          # Pybricks device driver & PC simulator fallback
│   └── reflexes.py       # Hardcoded reflexes (Obstacle Avoidance, Edge Confirm, & CW/CCW Detection)
├── core/
│   ├── agent.py          # Pure Python Q-Learning agent with Bellman update
│   └── environment.py    # State discretizer & reward function (Reverse Trap logic)
├── models/               # Storage for trained Q-table pickle files (.pkl)
├── train.py              # RL Training pipeline with epsilon decay
└── evaluate.py           # Evaluation pipeline with direction detection & reflex yielding
```

---

## 2. Reinforcement Learning Mathematics

### Bellman Equation Update
The Q-table is represented as a 2D matrix of shape $(N_{\text{states}} \times N_{\text{actions}}) = (8 \times 8)$.

The value of state-action pair $(s, a)$ is updated after each step according to the Bellman Equation:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]
\]

Where:
- $\alpha = 0.2$ (Learning Rate)
- $\gamma = 0.7$ (Discount Factor)
- $r$ is the immediate reward returned by `Environment.calculate_reward(s, a)`
- $s'$ is the resulting next state

---

## 3. Discretized State Space (8 States)

Continuous color sensor reflection values ($0 - 100\%$) are mapped into 8 discrete states using calibrated thresholds:

| State Index | State Name | Intensity Criteria | Description |
|---|---|---|---|
| `0` | `STATE_PURE_WHITE` | $\text{intensity} \ge 23$ | Sensor is fully over the pure white surface ($\text{White} = 24.1$). |
| `1` | `STATE_MEDIUM_DRIFT_WHITE` | $20 \le \text{intensity} < 23$ | Sensor is drifting white. |
| `2` | `STATE_LIGHT_DRIFT_WHITE` | $17 \le \text{intensity} < 20$ | Sensor is lightly drifting white. |
| `3` | `STATE_MICRO_DRIFT_WHITE` | $14 \le \text{intensity} < 17$ | Sensor is micro drifting white. |
| `4` | `STATE_PERFECT_EDGE` | $8 \le \text{intensity} < 14$ | Sensor is in the optimal Forward Deadband ($\text{Edge} = 11.1$). |
| `5` | `STATE_DRIFT_BLACK` | $4 \le \text{intensity} < 8$ | Sensor is drifting into black territory. |
| `6` | `STATE_PURE_BLACK` | $\text{intensity} < 4$ | Sensor is over pure black ($\text{Black} = 2.5$). |
| `7` | `STATE_TOTALLY_LOST` | $\text{intensity} < 1$ for $\ge 12$ steps | Robot has driven completely off the line. |

---

## 4. Action Space (8 Actions - RULE B)

To guarantee smooth curvature tracking and sharp turns, the action space contains 8 motor speed actions (Base Speed = $250\text{ deg/s}$):

| Action ID | Name | Left Speed (deg/s) | Right Speed (deg/s) | Motion Profile |
|---|---|---|---|---|
| `0` | `ACTION_FORWARD` | $250$ | $250$ | Straight line acceleration |
| `1` | `ACTION_MICRO_LEFT` | $37$ | $250$ | Gentle micro-turn left |
| `2` | `ACTION_SLIGHT_LEFT` | $87$ | $250$ | Soft curve left |
| `3` | `ACTION_SHARP_LEFT` | $-250$ | $250$ | Equal-speed pivot spin left |
| `4` | `ACTION_MICRO_RIGHT` | $250$ | $37$ | Gentle micro-turn right |
| `5` | `ACTION_SLIGHT_RIGHT` | $250$ | $87$ | Soft curve right |
| `6` | `ACTION_SHARP_RIGHT` | $250$ | $-250$ | Equal-speed pivot spin right |
| `7` | `ACTION_REVERSE` | $-175$ | $-175$ | Backward reversal |

---

## 5. Critical Rule Implementations

### RULE A: The Reverse Trap
- **Requirement**: The robot must learn to reverse natively using Q-learning when lost; reversing cannot be hardcoded into the RL decision loop.
- **Mechanism**:
  - In `Environment.get_state()`, consecutive steps below intensity $1$ are counted.
  - When $\ge 12$ steps ($1.2\text{s}$) are recorded, state transitions to `STATE_TOTALLY_LOST`.
  - In `Environment.calculate_reward()`:
    - If action is `ACTION_REVERSE` (7): **$+5.0$ reward**.
    - If action is forward/turning: **$-5.0$ penalty**.

### RULE B: Action Smoothness
- 8-action space includes `MICRO_LEFT` / `MICRO_RIGHT` for deadband tracking without penguin waddling.

### RULE C: Clockwise vs Anti-Clockwise (CCW) Support
- `detect_track_direction(robot)` sweeps left at startup and determines CW vs CCW based on intensity.

### RULE D: Non-RL Obstacle Reflex
- `hardcoded_obstacle_avoidance(robot)` handles IR obstacle avoidance with a 180° turnaround reflex.
