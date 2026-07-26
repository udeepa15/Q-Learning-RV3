# EV3 Reinforcement Learning (Q-Learning) Path-Following Documentation

Welcome to the comprehensive documentation suite for the EV3 Pybricks Q-Learning Path-Following Robot.

## Document Directory

1. [**RUNNING_GUIDE.md**](file:///d:/UDEEPA/UCSC/4th%20year/1st%20Sem/SCS4202%20Cognitive%20Robotics/micropython/Qlearning/ev3_rl_project/docs/RUNNING_GUIDE.md)
   - Quickstart guide for PC simulation mode.
   - Deploying and running on physical LEGO Mindstorms EV3 hardware via Pybricks MicroPython.
   - Model training, execution, and troubleshooting commands.

2. [**CODE_EXPLANATION.md**](file:///d:/UDEEPA/UCSC/4th%20year/1st%20Sem/SCS4202%20Cognitive%20Robotics/micropython/Qlearning/ev3_rl_project/docs/CODE_EXPLANATION.md)
   - In-depth system architecture and modular component design.
   - Explanation of the 4 discrete states, 6-action motor space, and Bellman equation.
   - Deep dive into critical rules: **Rule A** (Reverse Trap), **Rule B** (Action Smoothness), **Rule C** (CW/CCW Detection), and **Rule D** (Non-RL IR Reflex).

3. [**PRECAUTIONS_AND_PITFALLS.md**](file:///d:/UDEEPA/UCSC/4th%20year/1st%20Sem/SCS4202%20Cognitive%20Robotics/micropython/Qlearning/ev3_rl_project/docs/PRECAUTIONS_AND_PITFALLS.md)
   - Light sensor reflection threshold calibration.
   - Motor power drop and battery voltage compensation.
   - MicroPython vs Standard Python compatibility caveats.

4. [**FUTURE_IMPROVEMENTS.md**](file:///d:/UDEEPA/UCSC/4th%20year/1st%20Sem/SCS4202%20Cognitive%20Robotics/micropython/Qlearning/ev3_rl_project/docs/FUTURE_IMPROVEMENTS.md)
   - Advanced enhancements: SARSA, Q-learning with eligibility traces ($Q(\lambda)$), dual color sensors, continuous state spaces, and dynamic hyperparameter schedules.
