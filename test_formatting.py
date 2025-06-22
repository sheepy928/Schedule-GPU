#!/usr/bin/env python3
"""Test the improved formatting of GPU scheduler"""
import subprocess
import time

# Start the scheduler
proc = subprocess.Popen(
    ["python", "gpu_scheduler.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for startup
time.sleep(2)

# Test commands to show formatting
commands = [
    ("help", "help"),
    ("status", "status"),
    ("queue job", 'queue python -c "print(\'Test job\')"'),
    ("jobs", "jobs"),
    ("gpus", "gpus"),
    ("info", "info #0"),
]

for desc, cmd in commands:
    print(f"\n{'='*60}")
    print(f"Testing: {desc}")
    print('='*60)
    
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()
    time.sleep(0.5)

# Exit
proc.stdin.write("exit\n")
proc.stdin.flush()

# Capture and print output
time.sleep(1)
proc.terminate()

# Show a portion of the output
output, _ = proc.communicate()
lines = output.split('\n')

# Print key sections
print("\n\nSample output sections:")
print("="*60)

in_command = False
line_count = 0
max_lines = 150

for line in lines:
    if "scheduler>" in line:
        in_command = True
        if line_count > max_lines:
            print("\n... (truncated)")
            break
    
    if in_command and line_count < max_lines:
        print(line)
        line_count += 1
        if line.strip() == "" and lines[lines.index(line)-1].strip() == "":
            in_command = False