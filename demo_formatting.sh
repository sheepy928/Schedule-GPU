#!/bin/bash

echo "GPU Scheduler - Improved Visual Formatting Demo"
echo "=============================================="
echo ""

# Test help command
echo "1. Testing 'help' command:"
python gpu_scheduler.py --command "help" | head -30
echo ""

# Test status command  
echo -e "\n2. Testing 'status' command:"
python gpu_scheduler.py --command "status"
echo ""

# Test queue command
echo -e "\n3. Testing 'queue' command:"
python gpu_scheduler.py --command 'queue python -c "print(\"Hello GPU\")" --priority 5'
echo ""

# Test gpus command
echo -e "\n4. Testing 'gpus' command:"
python gpu_scheduler.py --command "gpus"
echo ""

echo "Demo complete!"