#!/usr/bin/env python3
"""Test quote handling in the scheduler"""
import subprocess
import time
import threading

# Start the scheduler
proc = subprocess.Popen(
    ["python", "gpu_scheduler.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for it to start
time.sleep(2)

# Queue a job with quotes
test_command = 'queue python -c "print(\'Hello from scheduler\'); import time; time.sleep(5); print(\'Job complete\')"'
print(f"Sending command: {test_command}")

proc.stdin.write(test_command + "\n")
proc.stdin.flush()

# Wait for response
time.sleep(1)

# Check job status
proc.stdin.write("jobs\n")
proc.stdin.flush()

# Wait for job to complete
print("Waiting for job to complete...")
time.sleep(8)

# Check logs
proc.stdin.write("jobs\n")
proc.stdin.flush()
time.sleep(1)

# Exit
proc.stdin.write("exit\n")
proc.stdin.flush()

# Get the latest log file
import os
import glob

time.sleep(1)
log_files = glob.glob("gpu_scheduler_logs/job_0_*.log")
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"\nLog file: {latest_log}")
    print("Contents:")
    with open(latest_log, 'r') as f:
        print(f.read())
else:
    print("No log files found")

proc.terminate()