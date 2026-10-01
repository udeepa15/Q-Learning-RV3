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

The max over `a'` is taken over all 8 actions of the next state. Hyperparameters are `PHASE_ALPHA = 0.3` and `PHASE_GAMMA = 0.6`, which are also the agent's constructor defaults.

`QLearningAgent` provides `choose_action(state, epsilon)`, `update(...)`, `snapshot()` / `restore()` (used to discard an episode), `save()`, `load()` and `display_q_table()`.

## 3. States (8)

Thresholds are **overwritten by calibration** at the start of every training/evaluation run (see RUNNING_GUIDE). Values below are the `settings.py` defaults.

| State | Name | Intensity |
|---|---|---|
| 0 | Pure White | >= 20 |
| 1 | Medium Drift White | 17 - 20 |
| 2 | Light Drift White | 14 - 17 |
| 3 | Perfect Edge (deadband) | 8 - 14 |
| 4 | Light Drift Black | 6 - 8 |
| 5 | Medium Drift Black | 4 - 6 |
| 6 | Pure Black | < 4 |
| 7 | Totally Lost | Pure Black continuously for `LOST_TIME_MS` (2000 ms) |

## 4. Actions (8)

Speeds are (left, right) in deg/s with `BASE_SPEED = 300`, for `CW` direction on the `OUTER` edge. They are mirrored automatically by `set_direction()` for other direction/edge combinations. The start configuration is `START_DIRECTION` / `START_EDGE` in settings; `reset_direction()` restores it at the start of every training episode and every evaluation run, because the obstacle reflex flips the direction at runtime.

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

**Rule-based** (`Environment.calculate_reward`, table `_REWARDS` in `core/environment.py`). The size of the correction matches how far off the edge the robot is: micro near the edge, slight in the middle band, sharp far away. Black-side rows mirror the white-side rows with right actions.

| State | Micro left | Slight left | Sharp left | Other actions |
|---|---|---|---|---|
| 3 Edge | +1 | +1 | +1 | Forward +5, rest +1 |
| 2 Light Drift White | +3.5 | +1.5 | -1 | -1 |
| 1 Medium Drift White | +1 | +3 | +1.5 | -1 |
| 0 Pure White | -3 | +2 | +3 | -3 |
| 7 Lost | | | | Reverse +5, rest -5 |

States 4, 5 and 6 use the same values with micro, slight and sharp **right**.

**Progress shaping** (`Environment.progress_reward`): distance from the edge is `|state - 3|` (Lost counts as 4). `+PROGRESS_REWARD` (1.0) if the next state is closer, `-1.0` if farther, `0` if unchanged.

## 6. Phased Training (`train.py`)

Training is split into two phases of short, time-limited episodes. Each phase owns some states (rows) and trains them with **every action**; only the rows owned by the current phase are updated.

| Phase | Episode length | States updated (rows) | Actions |
|---|---|---|---|
| Straight | 4 s | 2, 3, 4 (near the edge) | all 8 |
| Turn | 5 s | 0, 1, 5, 6, 7 (far off or lost) | all 8 |

Per step inside an episode:
1. If the state belongs to the phase: epsilon-greedy over all actions. Otherwise use `fallback_action` (no update, counted as an off-phase step).
2. Execute the action, wait `TRAIN_STEP_TIME_MS` (20 ms, same as evaluation), read the next state.
3. For learning steps, compute the reward and update the Q-table.

The obstacle reflex is not active during training, so an episode is never interrupted. It still runs during evaluation.

Epsilon starts at `PHASE_EPSILON_START` (0.4) and is multiplied by `PHASE_EPSILON_DECAY` (0.85) after each kept episode, down to `PHASE_EPSILON_MIN` (0.05). After each episode you choose: next episode, redo (the snapshot is restored), or save and finish the phase.

The straight phase is saved to `models/straight_q_table_8state.pkl` so you can resume at the turn phase later. At the end you choose where to save the final table. Per-episode metrics are written to `training_metrics_phased.csv`.

## 7. Evaluation (`evaluate.py`)

1. Pick a Q-table from `models/` (browse, select, or delete).
2. Calibrate the sensor (or skip and load `models/calibration.json`).
3. Loop with epsilon = 0: obstacle check, read intensity, pick the best action, execute, wait `DEFAULT_STEP_TIME_MS` (20 ms).
4. Loading failure aborts the run.

## 8. Obstacle Reflex (non-RL)

`hardcoded_obstacle_avoidance(robot)` runs when the IR distance is below `OBSTACLE_DISTANCE_THRESHOLD`:
1. Back away, then pivot ~180 degrees (`TURN_180_SPEED`, `TURN_180_MS`).
2. Flip the travel direction (CW <-> CCW); the followed edge (OUTER/INNER) stays the same.
3. `reacquire_edge()` finds OUR edge of the strip. It reads the intensity before moving, then moves in small steps (`EDGE_SEARCH_STEP_MS`, 40 ms) and reads again after each one. The two edges are mirror images, so the direction of the change identifies them: moving toward the expected white side the reading rises at the correct edge, and moving away from it the reading falls. The robot stops the moment it sees the expected change.
   - Already on the edge band: probe toward white for up to `EDGE_PROBE_MS`. A reading that reaches the white threshold confirms the correct edge. A reading that falls means the opposite edge, so it searches across the strip.
   - Otherwise: search toward white (`EDGE_SEARCH_TOWARD_MS`), then back the other way (`EDGE_SEARCH_AWAY_MS`).
4. If nothing is found, an expanding spiral search runs, then `reacquire_edge()` runs again to confirm the identity.
5. Control returns to the agent with the mirrored left/right mapping in effect and a cleared Lost timer.

## 9. Calibration

`calibrate_color_sensor()` measures white, black and edge (10-sample averages), derives all state thresholds from them (edge deadband = max(3, 12% of the white-black range); white and black sides each split into thirds; , writes them into `settings` in memory, and saves them to `models/calibration.json`. Skipping, abnormal readings, or simulator mode load the saved JSON instead.
