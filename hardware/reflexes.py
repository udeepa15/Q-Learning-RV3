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



def _spin_toward_white():
    """
    Spin tuple (left_speed, right_speed) that pans the sensor toward the
    side where WHITE is expected for the current (direction, edge) config:
    white on the right when INVERT_TURNS is False, on the left when True.
    """
    if not settings.INVERT_TURNS:
        return (100, -100)   # spin right
    return (-100, 100)       # spin left


def confirm_correct_edge(robot):
    """
    The 5cm white strip has two edges that read identically at a single
    point, but are mirror images (OUTER edge CW: black left / white right,
    INNER edge CW: black right / white left). After re-acquiring an edge
    band reading, nudge the sensor toward the expected WHITE side:
      - Reading gets whiter (or stays in band) -> correct edge, nudge back.
      - Reading gets blacker -> we grabbed the opposite edge of the strip;
        sweep back across the white strip until the far edge is reached.
    """
    spin = _spin_toward_white()
    back = (-spin[0], -spin[1])

    robot.turn_direct(spin[0], spin[1], 150)
    robot.stop()
    intensity = robot.read_intensity()

    if intensity >= settings.PERFECT_EDGE_LOW_8:
        robot.turn_direct(back[0], back[1], 150)
        robot.stop()
        print("[Reflex] Edge identity confirmed: {} edge (probe intensity={}).".format(settings.LINE_EDGE, intensity))
        return True

    # Mirror image detected: white was on the unexpected side -> wrong edge.
    print("[Reflex] WRONG edge acquired (probe intensity={}). Crossing strip back to the {} edge...".format(
        intensity, settings.LINE_EDGE))
    crossed_white = False
    for _ in range(30):
        robot.turn_direct(back[0], back[1], 100)
        intensity = robot.read_intensity()
        if not crossed_white:
            if intensity >= settings.LIGHT_DRIFT_WHITE_THRESH_8:
                crossed_white = True
        elif is_on_edge(intensity):
            robot.stop()
            print("[Reflex] Correct {} edge re-acquired (intensity={}).".format(settings.LINE_EDGE, intensity))
            return True

    robot.stop()
    print("[Reflex] WARNING: Could not cross back to the {} edge.".format(settings.LINE_EDGE))
    return False


def spiral_search_for_edge(robot, max_steps=45):
    """
    Expanding Archimedean spiral search routine to re-acquire the track edge
    when initial left/right sweeps fail to locate the white strip.
    Gradually increases inner wheel speed relative to outer wheel speed
    so the robot drives in an expanding spiral arc, checking intensity at each step.
    """
    print("[Reflex] Initial sweeps failed. Starting expanding spiral search recovery...")
    
    outer_speed = 200
    base_inner = 30
    
    # Determine curve direction towards the side where white is expected
    if not settings.INVERT_TURNS:
        # White on right -> curve rightward in an expanding arc
        left_is_outer = True
    else:
        # White on left -> curve leftward in an expanding arc
        left_is_outer = False

    for step in range(max_steps):
        # Increase inner wheel speed progressively (caps at 160 to maintain expanding curve)
        inner_speed = min(160, base_inner + int(step * 3))
        
        if left_is_outer:
            ls, rs = outer_speed, inner_speed
        else:
            ls, rs = inner_speed, outer_speed
            
        robot.turn_direct(ls, rs, 80)
        intensity = robot.read_intensity()
        
        if is_on_edge(intensity) or intensity >= settings.LIGHT_DRIFT_WHITE_THRESH_8:
            robot.stop()
            print("[Reflex] Track edge refound during spiral search at step {} (intensity={:.1f})!".format(step + 1, intensity))
            return True
            
    robot.stop()
    print("[Reflex] Spiral search completed without detecting track edge.")
    return False


def hardcoded_obstacle_avoidance(robot):
    """
    RULE D: Hardcoded non-RL obstacle avoidance reflex.
    Called when IR sensor reads distance below threshold.
    Backs away, pivots 180 degrees, flips the travel direction mapping
    (LINE_EDGE stays the same -- we return along the SAME physical edge),
    re-acquires the track edge, verifies it is not the strip's opposite
    edge, then returns control to the RL agent travelling the other way.
    """
    print("[Reflex] Obstacle detected! Turning 180 degrees to go back.")
    robot.stop()
    wait(100)

    # 1. Back away from the obstacle
    robot.turn_direct(-120, -120, 500)
    robot.stop()
    wait(100)

    # 2. Pivot ~180 degrees in place
    robot.turn_direct(settings.TURN_180_SPEED, -settings.TURN_180_SPEED, settings.TURN_180_MS)
    robot.stop()
    wait(100)

    # 3. Flip travel direction; the followed edge (OUTER/INNER) is unchanged,
    #    so set_direction recomputes which side white is on after the U-turn
    new_direction = "CCW" if settings.TURN_DIRECTION == "CW" else "CW"
    settings.set_direction(new_direction)

    # 4. Re-acquire the track edge, sweeping toward the white/strip side first
    first_spin = _spin_toward_white()
    second_spin = (-first_spin[0], -first_spin[1])

    edge_found = False
    max_sweep_steps = 15

    for _ in range(max_sweep_steps):
        robot.turn_direct(first_spin[0], first_spin[1], 100)
        intensity = robot.read_intensity()
        if is_on_edge(intensity):
            edge_found = True
            print("[Reflex] Edge refound during first sweep (intensity={}).".format(intensity))
            break

    if not edge_found:
        print("[Reflex] Edge not found on first sweep. Sweeping back the other way...")
        for _ in range(max_sweep_steps * 2):
            robot.turn_direct(second_spin[0], second_spin[1], 100)
            intensity = robot.read_intensity()
            if is_on_edge(intensity):
                edge_found = True
                print("[Reflex] Edge refound during second sweep (intensity={}).".format(intensity))
                break

    # 4.5. If edge is STILL not found after both sweeps -> Perform Expanding Spiral Search!
    if not edge_found:
        edge_found = spiral_search_for_edge(robot)

    # 5. Make sure we grabbed OUR edge of the 5cm strip, not its mirror twin
    if edge_found:
        confirm_correct_edge(robot)

    robot.stop()
    wait(100)
    if edge_found:
        print("[Reflex] Turnaround complete. Now driving {} on the {} edge. Returning control to RL agent.".format(
            settings.TURN_DIRECTION, settings.LINE_EDGE))
    else:
        print("[Reflex] WARNING: Edge not refound after turnaround. RL agent resumes anyway ({}, {} edge).".format(
            settings.TURN_DIRECTION, settings.LINE_EDGE))
 

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


def save_calibration(filepath="models/calibration.json"):
    """
    Saves current color sensor intensity thresholds to JSON file.
    """
    import os
    try:
        import json
    except ImportError:
        import ujson as json

    if "/" in filepath:
        d = filepath.rsplit("/", 1)[0]
        if d:
            try:
                os.mkdir(d)
            except Exception:
                pass

    cal_data = {
        "WHITE_INTENSITY": settings.WHITE_INTENSITY,
        "BLACK_INTENSITY": settings.BLACK_INTENSITY,
        "EDGE_INTENSITY": settings.EDGE_INTENSITY,
        "TOTALLY_LOST_THRESHOLD": settings.TOTALLY_LOST_THRESHOLD,
        "PERFECT_EDGE_HIGH_8": settings.PERFECT_EDGE_HIGH_8,
        "PERFECT_EDGE_LOW_8": settings.PERFECT_EDGE_LOW_8,
        "LIGHT_DRIFT_WHITE_THRESH_8": settings.LIGHT_DRIFT_WHITE_THRESH_8,
        "MEDIUM_DRIFT_WHITE_THRESH_8": settings.MEDIUM_DRIFT_WHITE_THRESH_8,
        "PURE_WHITE_THRESHOLD_8": settings.PURE_WHITE_THRESHOLD_8,
        "DRIFT_BLACK_THRESHOLD_8": settings.DRIFT_BLACK_THRESHOLD_8
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(cal_data, f)
        print("[Calibration] Saved calibration thresholds to:", filepath)
        return True
    except Exception as e:
        print("[Calibration] ERROR saving calibration ({}):".format(e))
        return False


def load_calibration(filepath="models/calibration.json"):
    """
    Loads saved color sensor intensity thresholds from JSON file into settings.
    """
    try:
        import json
    except ImportError:
        import ujson as json

    try:
        with open(filepath, 'r') as f:
            cal_data = json.load(f)

        for key, val in cal_data.items():
            if hasattr(settings, key):
                setattr(settings, key, val)

        print("[Calibration] Successfully loaded saved intensity thresholds from:", filepath)
        return True
    except Exception as e:
        print("[Calibration] Note: Could not load saved calibration ({}). Using settings defaults.".format(e))
        return False


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
        load_calibration()
        return

    try:
        from pybricks.parameters import Button
    except ImportError:
        load_calibration()
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
    print(" -> Press DOWN Button   : Skip Calibration (Load Saved/Defaults)")
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
            print("[Calibration] Skipped calibration. Loading saved thresholds or defaults.")
            load_calibration()
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
        load_calibration()
        return

    # Update base intensities in settings
    settings.WHITE_INTENSITY = int(white_val)
    settings.BLACK_INTENSITY = int(black_val)
    settings.EDGE_INTENSITY = int(edge_val)

    # Set TOTALLY_LOST_THRESHOLD slightly above pure black reading so off-track black triggers lost state
    settings.TOTALLY_LOST_THRESHOLD = int(black_val + 0.8)

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

    # Save calibrated thresholds to models/calibration.json
    save_calibration()

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



