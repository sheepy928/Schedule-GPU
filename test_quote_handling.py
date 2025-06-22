#!/usr/bin/env python3
import shlex

# Test different command formats
test_commands = [
    'python -c "import time; time.sleep(60)"',
    "python -c 'import time; time.sleep(60)'",
    'python -c "print(\'Hello World\')"',
    'echo "Hello World"',
]

print("Testing shlex parsing:")
for cmd in test_commands:
    print(f"\nOriginal: {cmd}")
    try:
        parts = shlex.split(cmd)
        print(f"Parsed: {parts}")
        print(f"Reconstructed: {' '.join(parts)}")
    except Exception as e:
        print(f"Error: {e}")

# Test what happens when we pass through argparse
import sys
print("\n\nCommand line args received:")
print(f"sys.argv: {sys.argv}")

if len(sys.argv) > 1:
    print(f"\nParsing argv[1]: {sys.argv[1]}")
    try:
        parts = shlex.split(sys.argv[1])
        print(f"Parsed: {parts}")
    except Exception as e:
        print(f"Error: {e}")