#!/usr/bin/env python3
"""Demo of the GPU scheduler functionality"""
import subprocess
import threading
import time
import sys

def run_scheduler_with_commands():
    """Run scheduler and send commands to it"""
    
    # Start the scheduler process
    proc = subprocess.Popen(
        ["python", "gpu_scheduler.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Create a thread to read output
    output_lines = []
    def read_output():
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end='')
    
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    
    # Wait for scheduler to start
    time.sleep(2)
    
    # Send commands
    commands = [
        ("Queue a quick job", "queue echo 'Hello from GPU scheduler!'"),
        ("Queue job with priority 1", "queue python test_job.py --name low-priority --priority 1"),
        ("Queue job with priority 5", "queue python test_job.py --name high-priority --priority 5"),
        ("Queue job with priority 3", "queue python test_job.py --name medium-priority --priority 3"),
        ("Check status", "status"),
        ("List jobs", "jobs"),
    ]
    
    for desc, cmd in commands:
        print(f"\n>>> {desc}: {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        time.sleep(1)
    
    # Wait for jobs to complete
    print("\n>>> Waiting for jobs to complete...")
    for i in range(25):
        time.sleep(1)
        if i % 5 == 4:
            print(f"\n>>> Checking jobs after {i+1} seconds")
            proc.stdin.write("jobs\n")
            proc.stdin.flush()
            time.sleep(0.5)
    
    # Final status
    print("\n>>> Final status")
    proc.stdin.write("status\n")
    proc.stdin.flush()
    time.sleep(1)
    
    # Exit
    print("\n>>> Exiting scheduler")
    proc.stdin.write("exit\n")
    proc.stdin.flush()
    
    # Wait for process to end
    proc.wait()
    reader.join(timeout=2)

if __name__ == "__main__":
    print("=== GPU Scheduler Demo ===")
    print("This demo will show jobs being scheduled and executed on available GPUs")
    print("Jobs with higher priority will run first when a GPU becomes available")
    print("-" * 60)
    
    run_scheduler_with_commands()