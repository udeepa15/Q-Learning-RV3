# Future Improvements & Advanced Expansion Areas

While the current Q-learning codebase completely satisfies all requirements and rules, this document outlines potential future enhancements for advanced research and competitions.

---

## 1. Algorithmic Enhancements

### A. On-Policy SARSA Learning
- **Current Approach**: Q-learning (Off-policy) updates using $\max_{a'} Q(s', a')$.
- **Enhancement**: Implement SARSA (On-policy):
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]
  \]
- **Benefit**: SARSA takes exploration risks into account, producing safer control trajectories when driving near sharp turns or obstacles.

### B. Eligibility Traces ($Q(\lambda)$)
- **Enhancement**: Maintain an eligibility trace matrix $E(s, a)$ to propagate rewards backward across multiple recent steps rather than just single-step updates:
  \[
  E(s, a) \leftarrow \gamma \lambda E(s, a) + 1
  \]
- **Benefit**: Dramatically speeds up learning convergence during early training episodes.

### C. Dynamic Learning Rate & Epsilon Schedules
- Decay $\alpha$ over time alongside $\epsilon$:
  \[
  \alpha_t = \max(\alpha_{\min}, \alpha_{\text{start}} \cdot \text{decay}^t)
  \]
- **Benefit**: Prevents late-stage Q-value oscillations once optimal policies are established.

---

## 2. Sensor & Hardware Upgrades

### A. Dual Color Sensor Differential Tracking
- **Enhancement**: Mount a second Color Sensor on Port S2.
- **State Representation**: Compute intensity differential $\Delta I = I_{\text{left}} - I_{\text{right}}$.
- **Benefit**: Allows the robot to center directly over the line rather than following a single edge gradient.

### B. Continuous State Space Discretization (Tile Coding / Linear Function Approximation)
- Expand state representation from 4 discrete buckets to 10 fine-grained intensity bins or tile codings ($0-10, 10-20, \dots, 90-100$).
- **Benefit**: Smoother velocity transitions and finer steering control on complex tracks.

---

## 3. Automated Model Selection & Online Fine-Tuning

### A. Hybrid Online Evaluation
- In `evaluate.py`, allow the agent to continue performing low-rate ($\epsilon = 0.02$) Q-updates during evaluation runs.
- **Benefit**: Enables real-time adaptation to changing track surface reflectivity or fading battery voltage.

### B. Model Checkpointing
- Periodically save intermediate Q-tables during long training runs to prevent data loss in the event of battery disconnects.
