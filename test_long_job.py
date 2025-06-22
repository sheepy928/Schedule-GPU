#!/usr/bin/env python3
"""Test long-running job with quotes"""
import subprocess
import time
import os

# Start scheduler
proc = subprocess.Popen(
    ["python", "gpu_scheduler.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(2)

# Queue the 60-second job
print("Queueing a 60-second job...")
proc.stdin.write('queue python -c "import time; print(\'Starting 60 second job\'); time.sleep(60); print(\'Job complete\')"\n')
proc.stdin.flush()

time.sleep(1)

# Check status
print("\nChecking initial status...")
proc.stdin.write("status\n")
proc.stdin.flush()
proc.stdin.write("jobs\n") 
proc.stdin.flush()

time.sleep(2)

# Kill the job before it completes
print("\nGetting job ID and killing it...")
proc.stdin.write("kill #0\n")
proc.stdin.flush()

time.sleep(1)

# Check final status
print("\nChecking final status...")
proc.stdin.write("jobs\n")
proc.stdin.flush()

time.sleep(1)

# Exit
proc.stdin.write("exit\n")
proc.stdin.flush()

# Check log
time.sleep(1)
import glob
log_files = glob.glob("gpu_scheduler_logs/job_0_*.log")
if log_files:
    latest = max(log_files, key=os.path.getmtime)
    print(f"\nLog file {latest}:")
    with open(latest, 'r') as f:
        print(f.read()[:200] + "..." if len(f.read()) > 200 else f.read())

proc.terminate()