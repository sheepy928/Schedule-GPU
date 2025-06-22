#!/bin/bash

echo "=== GPU Scheduler Test ==="
echo "This test will queue a job and show it running"
echo ""

# Queue a job and check status
echo "1. Queueing a test job..."
python gpu_scheduler.py --command "queue python test_job.py --demo"

echo ""
echo "2. The job will run in the background. Check the logs:"
sleep 2
ls -la gpu_scheduler_logs/

echo ""
echo "3. Latest log content:"
cat gpu_scheduler_logs/$(ls -t gpu_scheduler_logs/ | head -1)

echo ""
echo "Test complete!"