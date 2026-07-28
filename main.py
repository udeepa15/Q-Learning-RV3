#!/usr/bin/env pybricks-micropython
"""
Main Entrypoint for Pybricks EV3 MicroPython Execution.
"""

import sys
import os

# Ensure local package path is accessible (MicroPython os.path fallback)
try:
    import os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
except (ImportError, AttributeError, NameError):
    current_dir = "."

if current_dir and current_dir not in sys.path:
    sys.path.append(current_dir)


from evaluate import evaluate_agent

if __name__ == "__main__":
    print("Launching EV3 Q-Learning Line Follower Controller...")
    evaluate_agent()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            