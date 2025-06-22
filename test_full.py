#!/usr/bin/env python3
"""Test the GPU scheduler functionality"""
import subprocess
import time

def run_command(cmd):
    """Run a scheduler command and print result"""
    result = subprocess.run(
        ["python", "gpu_scheduler.py", "--command", cmd],
        capture_output=True,
        text=True
    )
    print(f"Command: {cmd}")
    print(f"Output: {result.stdout.strip()}")
    print("-" * 50)
    return result.stdout

# Test 1: Queue a simple job
print("=== Test 1: Queue a simple job ===")
run_command("queue echo Hello GPU World")

# Test 2: Check status
print("\n=== Test 2: Check status ===")
time.sleep(1)
run_command("status")

# Test 3: Queue multiple jobs with priorities
print("\n=== Test 3: Queue multiple jobs with priorities ===")
run_command("queue python test_job.py --job 1 --priority 1")
run_command("queue python test_job.py --job 2 --priority 5")
run_command("queue python test_job.py --job 3 --priority 3")

# Test 4: List jobs
print("\n=== Test 4: List jobs ===")
run_command("jobs")

# Test 5: Check GPU info
print("\n=== Test 5: GPU info ===")
run_command("gpus")

# Wait a bit for jobs to process
print("\n=== Waiting for jobs to complete... ===")
time.sleep(15)

# Test 6: Final status
print("\n=== Test 6: Final status ===")
run_command("status")
run_command("jobs")