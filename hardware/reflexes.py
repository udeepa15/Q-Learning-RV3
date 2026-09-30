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
        return (100, -100)
    return (-100, 100)


def _avg_intensity(robot, samples=3):
    total = 0.0
    for _ in range(samples):
        total += robot.read_intensity()
    return total / samples


def reacquire_edge(robot):
    """
    Finds OUR edge of the 5cm strip after a turnaround. The strip's two edges
    are mirror images, so the direction of the intensity change identifies them:
    moving toward the expected WHITE side the reading rises at the correct edge;
    moving away from it the reading falls at the correct edge.
    The intensity is read before moving and after every small step, and the robot
    stops as soon as the expected change is seen.
    """
    step = settings.EDGE_SEARCH_STEP_MS
    mid = (settings.PERFECT_EDGE_LOW_8 + settings.PERFECT_EDGE_HIGH_8) / 2.0
    toward = _spin_toward_white()
    away = (-toward[0], -toward[1])

    if is_on_edge(_avg_intensity(robot)):
        for _ in range(settings.EDGE_PROBE_MS // step):
            robot.turn_direct(toward[0], toward[1], step)
            robot.stop()
            intensity = _avg_intensity(robot)
            if intensity >= settings.PERFECT_EDGE_HIGH_8:
                for _ in range(10):
                    robot.turn_direct(away[0], away[1], step)
                    robot.stop()
                    if is_on_edge(_avg_intensity(robot)):
                        break
                print("[Reflex] Correct {} edge confirmed (reading rose toward white).".format(settings.LINE_EDGE))
                return True
            if intensity < settings.PERFECT_EDGE_LOW_8:
                print("[Reflex] Reading fell toward white: opposite edge. Searching across the strip...")
                break

    previous = _avg_intensity(robot)
    for _ in range(settings.EDGE_SEARCH_TOWARD_MS // step):
        robot.turn_direct(toward[0], toward[1], step)
        intensity = robot.read_intensity()
        if previous < mid <= intensity:
            robot.stop()
            print("[Reflex] Correct {} edge found (rising toward white, intensity={}).".format(settings.LINE_EDGE, intensity))
            return True
        previous = intensity

    previous = _avg_intensity(robot)
    for _ in range(settings.EDGE_SEARCH_AWAY_MS // step):
        robot.turn_direct(away[0], away[1], step)
        intensity = robot.read_intensity()
        if previous >= mid > intensity:
            robot.stop()
            print("[Reflex] Correct {} edge found (falling away from white, intensity={}).".format(settings.LINE_EDGE, intensity))
            return True
        previous = intensity

    robot.stop()
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
    
    if not settings.INVERT_TURNS:
        left_is_outer = True
    else:
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
        
        if is_on_edge(intensity) or intensity >= settings.PERFECT_EDGE_HIGH_8:
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

    robot.turn_direct(-120, -120, 500)
    robot.stop()
    wait(100)

    robot.turn_direct(settings.TURN_180_SPEED, -settings.TURN_180_SPEED, settings.TURN_180_MS)
    robot.stop()
    wait(100)

    # 3. Flip travel direction; the followed edge (OUTER/INNER) is unchanged,
    #    so set_direction recomputes which side white is on after the U-turn
    new_direction = "CCW" if settings.TURN_DIRECTION == "CW" else "CW"
    settings.set_direction(new_direction)

    # 4. Re-acquire OUR edge (identity checked by the direction of the intensity change)
    edge_found = reacquire_edge(robot)

    if not edge_found:
        print("[Reflex] Edge not found. Starting spiral search...")
        if spiral_search_for_edge(robot):
            edge_found = reacquire_edge(robot)

    robot.stop()
    wait(100)
    if edge_found:
        print("[Reflex] Turnaround complete. Now driving {} on the {} edge. Returning control to RL agent.".format(
            settings.TURN_DIRECTION, settings.LINE_EDGE))
    else:
        print("[Reflex] WARNING: Edge not refound after turnaround. RL agent resumes anyway ({}, {} edge).".format(
            settings.TURN_DIRECTION, settings.LINE_EDGE))
 

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
        "PERFECT_EDGE_HIGH_8": settings.PERFECT_EDGE_HIGH_8,
        "PERFECT_EDGE_LOW_8": settings.PERFECT_EDGE_LOW_8,
        "MEDIUM_DRIFT_WHITE_THRESH_8": settings.MEDIUM_DRIFT_WHITE_THRESH_8,
        "PURE_WHITE_THRESHOLD_8": settings.PURE_WHITE_THRESHOLD_8,
        "DRIFT_BLACK_THRESHOLD_8": settings.DRIFT_BLACK_THRESHOLD_8,
        "HEAVY_DRIFT_BLACK_THRESHOLD_8": settings.HEAVY_DRIFT_BLACK_THRESHOLD_8
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


def calibrate_color_sensor(robot, start_label="TRAINING"):
    """
    Interactive color sensor calibration routine on EV3 brick:
      1. Pure White surface
      2. Pure Black surface
      3. Perfect Edge boundary
    Derives the 8-state intensity thresholds from the measured values and saves them.
    """
    if robot.is_simulated or not hasattr(robot, 'ev3') or robot.ev3 is None:
        print("[Calibration] Simulator mode detected. Skipping interactive calibration.")
        load_calibration()
        return False

    try:
        from pybricks.parameters import Button
    except ImportError:
        load_calibration()
        return False

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
            return False
        wait(100)

    white_val = wait_for_center_button("1/3 PURE WHITE SURFACE")

    black_val = wait_for_center_button("2/3 PURE BLACK SURFACE")

    edge_val = wait_for_center_button("3/3 PERFECT EDGE BOUNDARY")

    if not (white_val > edge_val > black_val):
        print("[Calibration] WARNING: Readings abnormal (White={:.1f}, Edge={:.1f}, Black={:.1f}). Using defaults.".format(
            white_val, edge_val, black_val))
        load_calibration()
        return False

    settings.WHITE_INTENSITY = int(white_val)
    settings.BLACK_INTENSITY = int(black_val)
    settings.EDGE_INTENSITY = int(edge_val)

    # 8-State Thresholds (With Edge Deadband zone to eliminate penguin waddling)
    deadband_offset = max(3, int((white_val - black_val) * 0.12))
    settings.PERFECT_EDGE_HIGH_8 = int(edge_val + deadband_offset)
    settings.PERFECT_EDGE_LOW_8  = int(edge_val - deadband_offset)

    upper_span = white_val - settings.PERFECT_EDGE_HIGH_8
    lower_span = settings.PERFECT_EDGE_LOW_8 - black_val

    # White side and black side are each split into thirds (Light/Medium/Pure and
    # Drift/Heavy Drift/Pure), symmetric around the edge, so a deeper drift on
    # either side earns a proportionally stronger turn.
    step_w = upper_span / 3.0
    settings.MEDIUM_DRIFT_WHITE_THRESH_8 = int(settings.PERFECT_EDGE_HIGH_8 + step_w * 1)
    settings.PURE_WHITE_THRESHOLD_8      = int(settings.PERFECT_EDGE_HIGH_8 + step_w * 2)

    step_b = lower_span / 3.0
    settings.DRIFT_BLACK_THRESHOLD_8       = int(settings.PERFECT_EDGE_LOW_8 - step_b * 1)
    settings.HEAVY_DRIFT_BLACK_THRESHOLD_8 = int(settings.PERFECT_EDGE_LOW_8 - step_b * 2)

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
    print("   -> State 0 (Pure White)      : Intensity >= {}".format(settings.PURE_WHITE_THRESHOLD_8))
    print("   -> State 1 (Medium Drift)    : {} <= Intensity < {}".format(settings.MEDIUM_DRIFT_WHITE_THRESH_8, settings.PURE_WHITE_THRESHOLD_8))
    print("   -> State 2 (Light Drift)     : {} <= Intensity < {}".format(settings.PERFECT_EDGE_HIGH_8, settings.MEDIUM_DRIFT_WHITE_THRESH_8))
    print("   -> State 3 (PERFECT EDGE)    : {} <= Intensity < {} [FORWARD DEADBAND]".format(settings.PERFECT_EDGE_LOW_8, settings.PERFECT_EDGE_HIGH_8))
    print("   -> State 4 (Drift Black)     : {} <= Intensity < {}".format(settings.DRIFT_BLACK_THRESHOLD_8, settings.PERFECT_EDGE_LOW_8))
    print("   -> State 5 (Heavy Drift Blk) : {} <= Intensity < {}".format(settings.HEAVY_DRIFT_BLACK_THRESHOLD_8, settings.DRIFT_BLACK_THRESHOLD_8))
    print("   -> State 6 (Pure Black)      : Intensity < {}".format(settings.HEAVY_DRIFT_BLACK_THRESHOLD_8))
    print("   -> State 7 (Totally Lost)    : Pure Black for {} ms".format(settings.LOST_TIME_MS))
    print("==================================================")
    print(" -> PRESS CENTER BUTTON TO CONFIRM & START {}".format(start_label))
    print("==================================================\n")

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

    return True
