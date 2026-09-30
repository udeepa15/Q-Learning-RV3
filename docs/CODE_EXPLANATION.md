# Code Explanation & System Architecture

## 1. Project Layout

```
├── config/settings.py      # Hyperparameters, thresholds, speeds, ports, phased-training config
├── core/
│   ├── agent.py            # Tabular Q-learning agent (pure Python lists, no numpy)
│   └── environment.py      # State discretizer, reward function, progress reward
├── hardware/
│   ├── robot.py            # Pybricks driver + PC simulator fallback
│   └── reflexes.py         # Obstacle-avoidance reflex, edge search, sensor calibration
├── models/                 # Q-tables (.pkl) and calibration.json
├── train.py                # Phased trainer (straight -> turns)
├── evaluate.py             # Greedy evaluation
└── main.py                 # Brick entry point (runs evaluate_agent)
```

## 2. Q-Learning

Q-table: 8 states x 8 actions, initialised to **all zeros**. After each learning step:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

The max over `a'` only considers the columns allowed for `s'` (see phased training). Training uses `PHASE_ALPHA = 0.3` and `PHASE_GAMMA = 0.6`. `ALPHA` / `GAMMA` in settings are the agent's constructor defaults (used by `evaluate.py`, where no updates happen).

`QLearningAgent` provides `choose_action(state, epsilon, allowed_actions=None)`, `update(...)`, `snapshot()` / `restore()` (used to discard an episode), `save()`, `load()` and `display_q_table()`.

## 3. States (8)

Thresholds are **overwritten by calibration** at the start of every training/evaluation run (see RUNNING_GUIDE). Values below are the `settings.py` defaults.

| State | Name | Intensity |
|---|---|---|
| 0 | Pure White | >= 23 |
| 1 | Medium Drift White | 20 - 23 |
| 2 | Light Drift White | 14 - 20 |
| 3 | Perfect Edge (deadband) | 8 - 14 |
| 4 | Drift Black | 4 - 8 |
| 5 | Heavy Drift Black | 2 - 4 |
| 6 | Pure Black | < 2 |
| 7 | Totally Lost | < `TOTALLY_LOST_THRESHOLD` for 12 consecutive steps |

## 4. Actions (8)

Speeds are (left, right) in deg/s with `BASE_SPEED = 300`, for `CW` direction on the `OUTER` edge. They are mirrored automatically by `set_direction()` for other direction/edge combinations.

| ID | Action | Speeds |
|---|---|---|
| 0 | Forward | 300, 300 |
| 1 | Micro left | 210, 300 |
| 2 | Slight left | 120, 300 |
| 3 | Sharp left | -150, 300 |
| 4 | Micro right | 300, 210 |
| 5 | Slight right | 300, 120 |
| 6 | Sharp right | 300, -150 |
| 7 | Reverse | -210, -210 |

## 5. Reward Function

Per learning step: `reward = calculate_reward(s, a) + progress_reward(s, s')`.

**Rule-based** (`Environment.calculate_reward`):

| State | Rewarded actions | Reward | Otherwise |
|---|---|---|---|
| 3 Edge | Forward | +5 | +1 |
| 2 Light Drift White | any left | +3.5 | -1 |
| 1 Medium Drift White | slight / sharp left | +3 | -1 |
| 0 Pure White | slight / sharp left | +3 | -3 |
| 4 Drift Black | any right | +3.5 | -1 |
| 5 Heavy Drift Black | slight / sharp right | +3 | -1 |
| 6 Pure Black | slight / sharp right | +3 | -3 |
| 7 Lost | Reverse | +5 | -5 |

**Progress shaping** (`Environment.progress_reward`): distance from the edge is `|state - 3|` (Lost counts as 4). `+PROGRESS_REWARD` (1.0) if the next state is closer, `-1.0` if farther, `0` if unchanged.

## 6. Phased Training (`train.py`)

Training is split into two phases of short, time-limited episodes; only the (row, column) cells owned by the current phase are updated.

| Phase | Episode length | States updated | Actions used |
|---|---|---|---|
| Straight | 4 s | 2, 3, 4 | forward, slight left, slight right |
| Turn | 5 s | 0, 1, 5, 6, 7 | micro/sharp left, micro/sharp right, reverse |

Per step inside an episode:
1. If the IR sensor sees an obstacle, run the obstacle reflex (no Q-update) and continue.
2. If the state belongs to the phase: epsilon-greedy over the phase's actions. Otherwise use `fallback_action` (no update, counted as an off-phase step).
3. Execute the action, wait `TRAIN_STEP_TIME_MS` (20 ms), read the next state.
4. For learning steps, compute the reward and update the Q-table.

Epsilon starts at `PHASE_EPSILON_START` (0.4) and is multiplied by `PHASE_EPSILON_DECAY` (0.85) after each kept episode, down to `PHASE_EPSILON_MIN` (0.05). After each episode you choose: next episode, redo (the snapshot is restored), or save and finish the phase.

The straight phase is saved to `models/straight_q_table_8state.pkl` so you can resume at the turn phase later. At the end you choose where to save the final table. Per-episode metrics are written to `training_metrics_phased.csv`.

## 7. Evaluation (`evaluate.py`)

1. Pick a Q-table from `models/` (browse, select, or delete).
2. Calibrate the sensor (or skip and load `models/calibration.json`).
3. Loop with epsilon = 0: obstacle check, read intensity, pick the best action, execute, wait `DEFAULT_STEP_TIME_MS` (3 ms).
4. If the table is phased (every non-zero cell lies in its row's trained columns), each state only uses its trained columns. Loading failure aborts the run.

## 8. Obstacle Reflex (non-RL)

`hardcoded_obstacle_avoidance(robot)` runs when the IR distance is below `OBSTACLE_DISTANCE_THRESHOLD`:
1. Back away, then pivot ~180 degrees (`TURN_180_SPEED`, `TURN_180_MS`).
2. Flip the travel direction (CW <-> CCW); the followed edge (OUTER/INNER) stays the same.
3. Sweep to re-acquire the edge; fall back to an expanding spiral search if needed.
4. Confirm it grabbed the correct edge of the strip, not its mirror, then return control to the agent.

## 9. Calibration

`calibrate_color_sensor()` measures white, black and edge (10-sample averages), derives all state thresholds from them (edge deadband = max(3, 12% of the white-black range); white and black sides each split into thirds; lost threshold = black + 0.8), writes them into `settings` in memory, and saves them to `models/calibration.json`. Skipping, abnormal readings, or simulator mode load the saved JSON instead.
