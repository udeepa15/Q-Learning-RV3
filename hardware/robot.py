"""
Hardware Abstraction Layer for EV3 Pybricks Robot with PC Simulator Fallback.
"""

import sys

# Try importing EV3 Pybricks modules
IS_HARDWARE = True
try:
    from pybricks.hubs import EV3Brick
    from pybricks.ev3devices import Motor, ColorSensor, InfraredSensor
    from pybricks.tools import wait
except ImportError:
    IS_HARDWARE = False
    # Mock wait function for PC environment
    import time
    def wait(ms):
        time.sleep(ms / 1000.0)

# Relative or package imports of settings
try:
    from config import settings
except ImportError:
    from ev3_rl_project.config import settings


class RobotInterface:
    """
    Interface wrapper for Pybricks EV3 motors and sensors with PC simulator fallback.
    """
    def __init__(self, use_simulator=False):
        self.is_simulated = use_simulator or not IS_HARDWARE

        # Simulated state values for PC testing
        self.sim_intensity = settings.EDGE_INTENSITY
        self.sim_ir_distance = 100

        if not self.is_simulated:
            try:
                self.ev3 = EV3Brick()
                self.left_motor = Motor(settings.PORT_LEFT_MOTOR)
                self.right_motor = Motor(settings.PORT_RIGHT_MOTOR)
                self.color_sensor = ColorSensor(settings.PORT_COLOR_SENSOR)
                self.ir_sensor = InfraredSensor(settings.PORT_IR_SENSOR)
            except Exception as e:
                print("[RobotInterface] Pybricks Hardware Init Failed. Falling back to Simulator:", e)
                self.is_simulated = True
        else:
            print("[RobotInterface] Running in PC Simulator Mode.")

    def read_intensity(self):
        """
        Reads reflection intensity from the color sensor (0 - 100).
        """
        if self.is_simulated:
            return self.sim_intensity
        return self.color_sensor.reflection()

    def read_ir(self):
        """
        Reads distance percentage / cm from the IR sensor.
        """
        if self.is_simulated:
            return self.sim_ir_distance
        return self.ir_sensor.distance()

    def execute_action(self, action_id):
        """
        Executes a 6-action motor command.
        """
        if action_id not in settings.ACTION_SPEEDS:
            raise ValueError("Invalid action_id: {}".format(action_id))

        left_speed, right_speed = settings.ACTION_SPEEDS[action_id]

        if not self.is_simulated:
            self.left_motor.run(left_speed)
            self.right_motor.run(right_speed)
        else:
            # Update simulated intensity based on action for realistic mock runs
            if action_id in (settings.ACTION_SLIGHT_LEFT, settings.ACTION_SHARP_LEFT):
                self.sim_intensity = max(0, self.sim_intensity - 5)
            elif action_id in (settings.ACTION_SLIGHT_RIGHT, settings.ACTION_SHARP_RIGHT):
                self.sim_intensity = min(100, self.sim_intensity + 5)
            elif action_id == settings.ACTION_REVERSE:
                # Reversing shifts intensity back towards edge in lost state
                if self.sim_intensity < settings.TOTALLY_LOST_THRESHOLD:
                    self.sim_intensity = settings.BLACK_INTENSITY + 2

    def stop(self):
        """
        Stops both drive motors.
        """
        if not self.is_simulated:
            self.left_motor.stop()
            self.right_motor.stop()

    def turn_direct(self, left_speed, right_speed, duration_ms):
        """
        Direct motor control helper for reflex behaviors.
        """
        if not self.is_simulated:
            self.left_motor.run(left_speed)
            self.right_motor.run(right_speed)
        wait(duration_ms)
