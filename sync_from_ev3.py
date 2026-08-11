#!/usr/bin/env python3
"""
Sync Script: Downloads Q-tables (.pkl) and metric logs (.csv) from EV3 Brick to PC.

Usage:
  python sync_from_ev3.py [EV3_IP_OR_HOSTNAME]

Default Hostname: ev3dev.local
Default User: robot
"""

import sys
import os
import subprocess

EV3_HOST = sys.argv[1] if len(sys.argv) > 1 else "ev3dev.local"
EV3_USER = "robot"
REMOTE_PROJECT_DIR = "/home/robot/ev3_rl_project"

LOCAL_MODELS_DIR = os.path.join(".", "models")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

def run_scp(remote_src, local_dst):
    scp_cmd = ["scp", "-r", f"{EV3_USER}@{EV3_HOST}:{remote_src}", local_dst]
    print(f"[Sync] Running command: {' '.join(scp_cmd)}")
    try:
        subprocess.run(scp_cmd, check=True)
        print(f"[Sync] Successfully copied {remote_src} -> {local_dst}")
    except subprocess.CalledProcessError as e:
        print(f"[Sync] ERROR copying from EV3: {e}")
    except FileNotFoundError:
        print("[Sync] ERROR: 'scp' tool not found in system PATH. Ensure SSH/OpenSSH is installed on PC.")

if __name__ == "__main__":
    print("==================================================")
    print(f"       SYNCING Q-TABLES & METRICS FROM EV3        ")
    print(f" Target Host: {EV3_HOST} ({EV3_USER})")
    print("==================================================")

    # 1. Sync models directory
    print("\n1. Downloading Q-table model files (.pkl)...")
    run_scp(f"{REMOTE_PROJECT_DIR}/models/*", LOCAL_MODELS_DIR)

    # 2. Sync metrics CSV files
    print("\n2. Downloading training metrics (.csv)...")
    run_scp(f"{REMOTE_PROJECT_DIR}/*.csv", "./")

    print("\n==================================================")
    print(" Sync complete!")
    print("==================================================")
