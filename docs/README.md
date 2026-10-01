# EV3 Q-Learning Line Follower

Pybricks MicroPython project: an EV3 robot learns to follow the edge of a white line with tabular Q-learning (8 states x 8 actions), with a hardcoded reflex for obstacle avoidance. Runs on the EV3 brick or on a PC simulator.

## Documents

1. [RUNNING_GUIDE.md](RUNNING_GUIDE.md) - setup, calibration, training, evaluation, and the menus/buttons.
2. [CODE_EXPLANATION.md](CODE_EXPLANATION.md) - architecture, states, actions, reward function, training logic, and the obstacle reflex.
3. [PRECAUTIONS_AND_PITFALLS.md](PRECAUTIONS_AND_PITFALLS.md) - calibration, battery, MicroPython limits, edge selection, IR units.
4. [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) - possible extensions.

## Quick Start

```bash
python train.py                      # training from the heuristic table (simulator on PC)
python evaluate.py                   # greedy evaluation
```

On the brick, run `main.py` (evaluation) or `train.py` from the EV3 VS Code extension.
