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
│   └── reflexes.py       # Hardcoded reflexes (Obstacle Avoidance & CW/CCW Detection)
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
The Q-table is represented as a 2D matrix of shape $(N_{\text{states}} \times N_{\text{actions}}) = (4 \times 6)$.

The value of state-action pair $(s, a)$ is updated after each step according to the Bellman Equation:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]
\]

Where:
- $\alpha = 0.4$ (Learning Rate)
- $\gamma = 0.7$ (Discount Factor)
- $r$ is the immediate reward returned by `Environment.calculate_reward(s, a)`
- $s'$ is the resulting next state

---

## 3. Discretized State Space (4 States)

Continuous color sensor reflection values ($0 - 100\%$) are mapped into 4 discrete states:

| State Index | State Name | Intensity Criteria | Description |
|---|---|---|---|
| `0` | `STATE_TOO_WHITE` | $\text{intensity} \ge 28$ | Sensor is fully over the white track background. |
| `1` | `STATE_PERFECT_EDGE` | $11 < \text{intensity} < 28$ | Sensor is tracking the boundary gradient between white and black. |
| `2` | `STATE_TOO_BLACK` | $5 \le \text{intensity} \le 11$ | Sensor is drifting too far into black territory. |
| `3` | `STATE_TOTALLY_LOST` | $\text{intensity} < 5$ for $\ge 3$ steps | Robot has driven completely off the line into pure black. |

---

## 4. Action Space (6 Actions - RULE B)

To guarantee smooth curvature tracking, the action space contains 6 distinct motor speed pairs:

| Action ID | Name | Left Speed (deg/s) | Right Speed (deg/s) | Motion Profile |
|---|---|---|---|---|
| `0` | `ACTION_FORWARD` | $150$ | $150$ | Straight line acceleration |
| `1` | `ACTION_SLIGHT_LEFT` | $80$ | $150$ | Soft left curve |
| `2` | `ACTION_SHARP_LEFT` | $-80$ | $150$ | In-place / sharp left pivot |
| `3` | `ACTION_SLIGHT_RIGHT` | $150$ | $80$ | Soft right curve |
| `4` | `ACTION_SHARP_RIGHT` | $150$ | $-80$ | In-place / sharp right pivot |
| `5` | `ACTION_REVERSE` | $-100$ | $-100$ | Backward reversal |

---

## 5. Critical Rule Implementations

### RULE A: The Reverse Trap
- **Requirement**: The robot must learn to reverse natively using Q-learning when lost; reversing cannot be hardcoded into the RL decision loop.
- **Mechanism**:
  - In `Environment.get_state()`, consecutive steps below intensity $5$ are counted.
  - When $\ge 3$ steps are recorded, state transitions to `STATE_TOTALLY_LOST`.
  - In `Environment.calculate_reward()`:
    - If action is `ACTION_REVERSE` (5): **$+5.0$ reward**.
    - If action is forward/turning (0-4): **$-5.0$ penalty**.
  - As a result, the Bellman equation naturally updates $Q(\text{TOTALLY\_LOST}, \text{REVERSE})$ to be the dominant maximum value.

### RULE B: Action Smoothness
- 6-action space allows gentle steering corrections (`SLIGHT_LEFT`, `SLIGHT_RIGHT`) rather than abrupt binary left/right oscillation.

### RULE C: Clockwise vs Anti-Clockwise (CCW) Support
- **Mechanism**:
  - `detect_track_direction(robot)` sweeps left at startup.
  - If intensity $\ge 28$ (White), returns `"CCW"`.
  - If intensity $\le 11$ (Black), returns `"CW"`.
  - `evaluate.py` dynamically loads `models/cw_q_table.pkl` or `models/ccw_q_table.pkl`.

### RULE D: Non-RL Obstacle Reflex
- **Mechanism**:
  - `hardcoded_obstacle_avoidance(robot)` is a deterministic non-RL safety routine.
  - Triggered whenever `read_ir() < 20cm`.
  - Reverses, then sweeps left and right until `read_intensity()` detects an edge intensity ($11 < \text{intensity} < 28$).
  - `train.py` and `evaluate.py` yield control immediately and skip Q-table updates while the obstacle reflex is active.
