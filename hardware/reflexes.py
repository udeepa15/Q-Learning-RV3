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
    Helper function to check if reflection intensity is within the edge gradient range.
    """
    return settings.BLACK_THRESHOLD < intensity < settings.WHITE_THRESHOLD


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
    if intensity >= settings.WHITE_THRESHOLD:
        direction = "CCW"
    else:
        direction = "CW"

    # Return robot to initial orientation by sweeping back right
    robot.turn_direct(100, -100, 400)
    robot.stop()
    wait(100)

    print("[Reflex] Track direction detected: {}".format(direction))
    return direction
