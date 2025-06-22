#!/usr/bin/env python3
import time
import os
import sys

print(f"Job started on GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
print(f"Arguments: {sys.argv[1:]}")

# Simulate some work
for i in range(5):
    print(f"Working... {i+1}/5")
    time.sleep(1)

print("Job completed successfully!")