#!/usr/bin/env python3
"""Test client for GPU scheduler server"""

import socket
import json
import sys
import time

def send_command(host, port, command):
    """Send a command to the server and print the response"""
    try:
        # Create socket and connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print(f"Connected to {host}:{port}")
        
        # Send command with newline
        print(f"Sending: '{command}'")
        command_with_newline = command + '\n' if not command.endswith('\n') else command
        sock.send(command_with_newline.encode('utf-8'))
        
        # Wait a bit for processing
        time.sleep(0.1)
        
        # Receive response (may need multiple reads for full response)
        response = ""
        sock.settimeout(2.0)  # 2 second timeout
        try:
            while True:
                chunk = sock.recv(4096).decode('utf-8')
                if not chunk:
                    break
                response += chunk
                if '\n' in response:  # Got complete response
                    break
        except socket.timeout:
            print("Timeout waiting for response")
        
        print(f"Received: {response.strip()}")
        
        # Parse and display result
        try:
            result = json.loads(response.strip())
            print(f"Result: {result['result']}")
        except json.JSONDecodeError:
            print(f"Could not parse JSON from response")
            
        sock.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Handle both "host port" and just "port" formats
    if len(sys.argv) > 2:
        host = sys.argv[1]
        port = int(sys.argv[2])
    elif len(sys.argv) > 1:
        host = "localhost"
        port = int(sys.argv[1])
    else:
        host = "localhost"
        port = 8000
    
    # Test various commands
    print("=== Testing GPU Scheduler Server ===\n")
    
    commands = [
        "status",
        "gpus",
        "jobs",
        "queue echo 'Hello from test job'",
        "invalid_command",
        "",  # Empty command
        "  \n  ",  # Whitespace only
    ]
    
    for i, cmd in enumerate(commands):
        print(f"\nTest {i+1}:")
        send_command(host, port, cmd)
        time.sleep(0.5)