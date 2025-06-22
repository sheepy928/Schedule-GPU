#!/usr/bin/env python3
import time
import subprocess
import threading

def run_scheduler():
    """Run the scheduler in interactive mode"""
    proc = subprocess.Popen(
        ["python", "gpu_scheduler.py", "--log-level", "DEBUG"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for scheduler to start
    time.sleep(2)
    
    # Queue a job
    proc.stdin.write("queue python test_job.py --test 1\n")
    proc.stdin.flush()
    
    # Check status
    time.sleep(1)
    proc.stdin.write("status\n")
    proc.stdin.flush()
    
    # Wait for job to complete
    time.sleep(8)
    
    # Check jobs
    proc.stdin.write("jobs\n")
    proc.stdin.flush()
    
    # Exit
    time.sleep(1)
    proc.stdin.write("exit\n")
    proc.stdin.flush()
    
    # Read output
    output = []
    reader = threading.Thread(target=lambda: output.extend(proc.stdout.readlines()))
    reader.start()
    
    proc.wait()
    reader.join()
    
    # Print relevant lines
    for line in output:
        if any(keyword in line for keyword in ["GPU", "Job", "Allocating", "Freeing", "queued", "Status", "Queued:", "Running:", "Completed:"]):
            print(line.strip())

if __name__ == "__main__":
    run_scheduler()