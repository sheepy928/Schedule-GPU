#!/bin/bash
echo "Starting GPU Scheduler Demo..."
echo "The scheduler will run in the background and process jobs"
echo ""

# Start the scheduler in the background
python gpu_scheduler.py &
SCHEDULER_PID=$!

# Give it time to start
sleep 2

echo "=== Queueing jobs ==="
echo "queue echo 'Quick job'" | nc localhost 5555
echo "queue python test_job.py --job 1 --priority 1" | nc localhost 5555
echo "queue python test_job.py --job 2 --priority 5" | nc localhost 5555
echo "queue python test_job.py --job 3 --priority 3" | nc localhost 5555

echo ""
echo "=== Checking status ==="
echo "status" | nc localhost 5555

echo ""
echo "=== Jobs will process in priority order (5, 3, 1) ==="
echo "Waiting for jobs to complete..."

# Monitor for a bit
for i in {1..20}; do
    sleep 1
    if [ $((i % 5)) -eq 0 ]; then
        echo ""
        echo "=== Status at $i seconds ==="
        echo "jobs" | nc localhost 5555
    fi
done

echo ""
echo "=== Final status ==="
echo "status" | nc localhost 5555

# Clean up
kill $SCHEDULER_PID
echo ""
echo "Demo complete!"