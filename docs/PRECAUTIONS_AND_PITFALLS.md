# Precautions, Hardware Calibration & Common Pitfalls

This document outlines key physical robot calibration steps, hardware considerations, MicroPython runtime limitations, and practical troubleshooting strategies.

---

## 1. Light Sensor Calibration & Ambient Lighting

> [!IMPORTANT]
> Reflection values returned by `ColorSensor.reflection()` are sensitive to ambient lighting and physical sensor height above the track surface.

- **Sensor Height**: The color sensor should be mounted vertically **$8\text{mm} - 12\text{mm}$** above the surface. If mounted too high ($>20\text{mm}$), reflected light intensity drops dramatically and ambient light introduces noise.
- **Thresholds are calibrated at every run**: `calibrate_color_sensor()` measures white, black and the edge, derives all state thresholds from them and saves them to `models/calibration.json`. Re-calibrate whenever lighting or the track surface changes. The defaults in `config/settings.py` are only used when no calibration file can be loaded.
- **Skipping calibration** loads the saved file, so make sure it came from the same lighting and sensor height.
- Calibration is skipped in the simulator.

---

## 2. Battery Voltage & Motor Power Drop

> [!WARNING]
> LEGO EV3 AA batteries or rechargeable battery packs experience voltage drops as they discharge.

- Lower battery voltage reduces actual motor angular velocity even if motor target speeds (e.g. 150 deg/s) are specified.
- **Symptom**: Robot turns too slowly or fails to complete sweep reflexes.
- **Solution**: Keep EV3 battery level above **7.5V**. If motor response degrades during training, replace or recharge batteries before saving `.pkl` models.

---

## 3. MicroPython Runtime Limitations

1. **No `numpy` Library**:
   - EV3 MicroPython v2.0 is based on MicroPython 1.11, which does not support heavy C-extension packages like `numpy` or `scipy`.
   - **Enforcement**: All matrices in `core/agent.py` are built using pure Python nested lists (`[[0.0] * n_actions for _ in range(n_states)]`). Do not introduce `numpy` imports into core modules.

2. **`pickle` vs `upickle` Module**:
   - On full Python desktop environments, the module is `pickle`. On MicroPython, it is often `upickle`.
   - **Enforcement**: Imports use `try: import pickle except ImportError: import upickle as pickle` to maintain cross-platform compatibility.

3. **Sensor Sampling Delays**:
   - Sensor read commands (`color_sensor.reflection()`, `ir_sensor.distance()`) take $\sim 5 - 10\text{ms}$ on I2C/analog EV3 buses.
   - Avoid zero-delay loops (`while True: pass`). Always include a small wait interval (`DEFAULT_STEP_TIME_MS` = 3 ms in evaluation, `TRAIN_STEP_TIME_MS` = 20 ms in training) between RL iterations.

---

## 4. Track Edge Selection (Left vs Right Edge)

- The robot tracks the **gradient edge** between white and black.
- On a clockwise (CW) track:
  - If tracking the **outer edge**, white is on the left, black is on the right.
  - If tracking the **inner edge**, black is on the left, white is on the right.
- The direction and edge are set in `config/settings.py` (`TURN_DIRECTION`, `LINE_EDGE`); the left/right action speeds are mirrored automatically.
- The obstacle reflex flips the direction (CW <-> CCW) after its 180-degree turn but keeps the same physical edge.
- Ensure the trained model (trained with the same `TURN_DIRECTION` and `LINE_EDGE` in `config/settings.py`) matches the specific edge gradient your robot is placed on during evaluation.

---

## 5. Infrared Sensor Distance Units

- Pybricks `InfraredSensor.distance()` returns distance as an estimated percentage ($0 - 100\%$).
- $100\%$ corresponds to approximately $70\text{cm}$.
- An IR reading of $20$ corresponds to approximately $14 - 20\text{cm}$.
- If using an **Ultrasonic Sensor** (`UltrasonicSensor`) instead of an Infrared Sensor, `distance()` returns distance in millimeters ($20\text{cm} = 200\text{mm}$). Update `settings.OBSTACLE_DISTANCE_THRESHOLD` accordingly if swapping sensor hardware.

---

## 6. Lost-State Timing

`TOTALLY_LOST_CONSECUTIVE_STEPS` (12) counts steps, not time. A step is 20 ms in training and 3 ms in evaluation, so the robot has less real time to recover before being declared lost during evaluation.
