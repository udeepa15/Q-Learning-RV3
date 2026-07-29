"""
Hardcoded Reflex Behaviors: Obstacle Avoidance and Track Direction Detection.
"""

try:
    from pybricks.tools import wait
except ImportError:
    import time
    def wait(ms):
        time.sleep(ms / 1000.0)

try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings


def is_on_edge(intensity):
    """
    Helper function to check if reflection intensity is within the perfect edge gradient range.
    """
    return settings.PERFECT_EDGE_LOW_8 <= intensity < settings.PERFECT_EDGE_HIGH_8



def hardcoded_obstacle_avoidance(robot):
    """
    RULE D: Hardcoded non-RL obstacle avoidance reflex.
    Called when IR sensor reads distance < 20cm.
    Reverses and sweeps left/right until read_intensity() detects the edge again.
    """
    print("[Reflex] Obstacle detected! Interrupting RL agent control loop.")
    robot.stop()
    wait(100)

    # 1. Reverse away from the obstacle
    robot.turn_direct(-120, -120, 600)
    robot.stop()
    wait(100)

    # 2. Sweep left to refind the track edge
    edge_found = False
    max_sweep_steps = 15

    for _ in range(max_sweep_steps):
        robot.turn_direct(-100, 100, 100)  # Spin left slightly
        intensity = robot.read_intensity()
        if is_on_edge(intensity):
            edge_found = True
            print("[Reflex] Edge refound during left sweep (intensity={}).".format(intensity))
            break

    # 3. If left sweep fails, sweep right past center to refind edge
    if not edge_found:
        print("[Reflex] Edge not found on left sweep. Sweeping right...")
        for _ in range(max_sweep_steps * 2):
            robot.turn_direct(100, -100, 100)  # Spin right slightly
            intensity = robot.read_intensity()
            if is_on_edge(intensity):
                edge_found = True
                print("[Reflex] Edge refound during right sweep (intensity={}).".format(intensity))
                break

    robot.stop()
    wait(100)
    print("[Reflex] Obstacle cleared & edge refound. Returning control to RL agent.")


def detect_track_direction(robot):
    """
    RULE C: Sweep-and-detect reflex to determine track direction.
    Sweeps left at startup:
      - If it sees white (intensity >= WHITE_THRESHOLD), returns 'CCW'
      - If it sees black (intensity <= BLACK_THRESHOLD), returns 'CW'
    """
    print("[Reflex] Detecting track direction via initial sweep...")

    # Sweep left
    robot.turn_direct(-100, 100, 400)
    robot.stop()
    wait(100)

    intensity = robot.read_intensity()
    print("[Reflex] Post-sweep intensity reading:", intensity)

    # Determine CW vs CCW
    if intensity >= settings.PURE_WHITE_THRESHOLD_8:
        direction = "CCW"
    else:
        direction = "CW"

    # Return robot to initial orientation by sweeping back right
    robot.turn_direct(100, -100, 400)
    robot.stop()
    wait(100)

    print("[Reflex] Track direction detected: {}".format(direction))
    return direction


def calibrate_color_sensor(robot):
    """
    Interactive color sensor calibration routine on EV3 brick:
      1. Pure White surface
      2. Pure Black surface
      3. Perfect Edge boundary
    Calculates dynamic drift white & drift black intensity thresholds for 3-State and 5-State modes.
    """
    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        print("[Calibration] Simulator mode detected. Skipping interactive calibration.")
        return

    try:
        from pybricks.parameters import Button
    except ImportError:
        return

    def wait_for_center_button(prompt_text):
        print("\n==================================================")
        print(" CALIBRATION: " + prompt_text)
        print(" -> Place sensor, then press CENTER button")
        print("==================================================")
        
        while Button.CENTER not in robot.ev3.buttons.pressed():
            wait(100)
        
        try:
            robot.ev3.speaker.beep(frequency=1000, duration=150)
        except Exception:
            pass

        while Button.CENTER in robot.ev3.buttons.pressed():
            wait(100)

        # Average 10 readings for accuracy
        total = 0
        for _ in range(10):
            total += robot.read_intensity()
            wait(30)
        avg_val = total / 10.0
        print("[Calibration] Recorded Intensity: {:.1f}".format(avg_val))
        return avg_val

    print("\n==================================================")
    print("      SENSOR INTENSITY CALIBRATION MENU           ")
    print(" -> Press CENTER Button : Start Sensor Calibration")
    print(" -> Press DOWN Button   : Skip Calibration (Defaults)")
    print(" (Waiting for button press...)")
    print("==================================================\n")

    # Block execution until user explicitly presses a button
    while True:
        pressed = robot.ev3.buttons.pressed()
        if Button.CENTER in pressed:
            try:
                robot.ev3.speaker.beep(frequency=800, duration=150)
            except Exception:
                pass
            wait(500)
            break
        elif Button.DOWN in pressed:
            print("[Calibration] Skipped calibration. Using default thresholds.")
            wait(500)
            return
        wait(100)


    # 1. Pure White
    white_val = wait_for_center_button("1/3 PURE WHITE SURFACE")

    # 2. Pure Black
    black_val = wait_for_center_button("2/3 PURE BLACK SURFACE")

    # 3. Perfect Edge
    edge_val = wait_for_center_button("3/3 PERFECT EDGE BOUNDARY")

    # Sanity check: ensure white > edge > black
    if not (white_val > edge_val > black_val):
        print("[Calibration] WARNING: Readings abnormal (White={:.1f}, Edge={:.1f}, Black={:.1f}). Using defaults.".format(
            white_val, edge_val, black_val))
        return

    # Update base intensities in settings
    settings.WHITE_INTENSITY = int(white_val)
    settings.BLACK_INTENSITY = int(black_val)
    settings.EDGE_INTENSITY = int(edge_val)

    # Proportionally lower TOTALLY_LOST_THRESHOLD so normal black doesn't trigger lost state
    settings.TOTALLY_LOST_THRESHOLD = max(1, int(black_val * 0.4))

    # 8-State Thresholds (With Edge Deadband zone to eliminate penguin waddling)
    deadband_offset = max(3, int((white_val - black_val) * 0.12))
    settings.PERFECT_EDGE_HIGH_8 = int(edge_val + deadband_offset)
    settings.PERFECT_EDGE_LOW_8  = int(edge_val - deadband_offset)

    upper_span = white_val - settings.PERFECT_EDGE_HIGH_8
    lower_span = settings.PERFECT_EDGE_LOW_8 - black_val

    step_w = upper_span / 3.0
    settings.LIGHT_DRIFT_WHITE_THRESH_8  = int(settings.PERFECT_EDGE_HIGH_8 + step_w * 1)
    settings.MEDIUM_DRIFT_WHITE_THRESH_8 = int(settings.PERFECT_EDGE_HIGH_8 + step_w * 2)
    settings.PURE_WHITE_THRESHOLD_8      = int(white_val - step_w * 0.3)

    settings.DRIFT_BLACK_THRESHOLD_8   = int(black_val + lower_span * 0.4)

    print("\n==================================================")
    print("      CALIBRATION COMPLETE & THRESHOLDS UPDATED   ")
    print("==================================================")
    print(" Raw Surface Intensity Readings:")
    print("   -> Pure White Surface : {:.1f}".format(white_val))
    print("   -> Perfect Edge Line  : {:.1f}".format(edge_val))
    print("   -> Pure Black Surface : {:.1f}".format(black_val))
    print("--------------------------------------------------")
    print(" Computed 8-State Intensity Thresholds:")
    print("   -> State 0 (Pure White)   : Intensity >= {}".format(settings.PURE_WHITE_THRESHOLD_8))
    print("   -> State 1 (Medium Drift) : {} <= Intensity < {}".format(settings.MEDIUM_DRIFT_WHITE_THRESH_8, settings.PURE_WHITE_THRESHOLD_8))
    print("   -> State 2 (Light Drift)  : {} <= Intensity < {}".format(settings.LIGHT_DRIFT_WHITE_THRESH_8, settings.MEDIUM_DRIFT_WHITE_THRESH_8))
    print("   -> State 3 (Micro Drift)  : {} <= Intensity < {}".format(settings.PERFECT_EDGE_HIGH_8, settings.LIGHT_DRIFT_WHITE_THRESH_8))
    print("   -> State 4 (PERFECT EDGE) : {} <= Intensity < {} [FORWARD DEADBAND]".format(settings.PERFECT_EDGE_LOW_8, settings.PERFECT_EDGE_HIGH_8))
    print("   -> State 5 (Drift Black)  : {} <= Intensity < {}".format(settings.DRIFT_BLACK_THRESHOLD_8, settings.PERFECT_EDGE_LOW_8))
    print("   -> State 6 (Pure Black)   : Intensity < {}".format(settings.DRIFT_BLACK_THRESHOLD_8))
    print("   -> State 7 (Totally Lost) : Intensity < {} (for {} steps)".format(settings.TOTALLY_LOST_THRESHOLD, settings.TOTALLY_LOST_CONSECUTIVE_STEPS))
    print("==================================================")
    print(" -> PRESS CENTER BUTTON TO CONFIRM & START TRAINING")
    print("==================================================\n")



    # Hold execution until user presses CENTER button
    while True:
        pressed = robot.ev3.buttons.pressed()
        if Button.CENTER in pressed:
            try:
                robot.ev3.speaker.beep(frequency=1200, duration=200)
            except Exception:
                pass
            wait(500)
            break
        wait(100)



