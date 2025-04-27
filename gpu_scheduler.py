#!/usr/bin/env python3
import os
import sys
import time
import json
import queue
import socket
import argparse
import subprocess
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter

# Install rich traceback handler
install()

# Create rich console
console = Console()

def get_available_gpus() -> List[int]:
    """
    Detect all GPUs using nvidia-smi command.

    Returns:
        List of all GPU IDs detected by nvidia-smi
    """
    try:
        # Run nvidia-smi to get the list of GPUs
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output to get GPU indices
        gpu_ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]

        if not gpu_ids:
            console.print("[yellow]Warning:[/yellow] No GPUs detected by nvidia-smi.")
            return [0]  # Default to GPU 0 if none detected

        return gpu_ids
    except subprocess.CalledProcessError:
        console.print("[bold red]Error[/bold red] running nvidia-smi. Make sure NVIDIA drivers are installed.")
        return [0]  # Default to GPU 0 if nvidia-smi fails
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] nvidia-smi command not found. Make sure NVIDIA drivers are installed.")
        return [0]  # Default to GPU 0 if nvidia-smi not found
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return [0]  # Default to GPU 0 for any other exception

def check_gpu_status(gpu_id: int) -> str:
    """
    Check if a GPU is free based on memory usage and utilization.

    A GPU is considered free if:
    - Its VRAM usage is <= a threshold (50 MB)
    - Its utilization is <= a threshold (10%)

    Args:
        gpu_id: GPU ID to check

    Returns:
        "free" if GPU meets criteria, "occupied by other" otherwise
    """
    try:
        # Query this specific GPU
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output
        output = result.stdout.strip()
        if not output:
            return "occupied by other"  # If we can't get data, assume it's occupied

        parts = output.split(',')
        if len(parts) != 2:
            return "occupied by other"

        memory_used = float(parts[0].strip())
        utilization = float(parts[1].strip())

        # Check against thresholds
        if memory_used <= 50 and utilization <= 10:
            return "free"
        else:
            return "occupied by other"

    except Exception as e:
        console.print(f"[bold red]Error checking GPU {gpu_id} status:[/bold red] {str(e)}")
        return "occupied by other"  # Assume occupied if there's an error

class GPUJobScheduler:
    def __init__(self, gpu_ids: List[int], log_dir: str = "gpu_scheduler_logs", status_check_interval: int = 10):
        """
        Initialize the GPU job scheduler.

        Args:
            gpu_ids: List of GPU IDs to use
            log_dir: Directory to store log files
            status_check_interval: Interval in seconds to check GPU status
        """
        self.gpu_ids = gpu_ids
        self.log_dir = log_dir
        self.status_check_interval = status_check_interval

        # Priority queue for jobs (priority, job_id, command)
        # Lower priority value = higher priority
        self.queue = queue.PriorityQueue()

        # Counter for local job IDs (0-based)
        self.next_local_id = 0

        # Track local_id to job_id mapping
        self.local_to_uuid = {}
        self.uuid_to_local = {}

        # Initial GPU status check
        self.gpu_status = {}
        for gpu_id in gpu_ids:
            status = check_gpu_status(gpu_id)
            self.gpu_status[gpu_id] = status
            status_color = "green" if status == "free" else "yellow"
            console.print(f"GPU {gpu_id} initial status: [{status_color}]{status}[/{status_color}]")

        self.gpu_locks = {gpu_id: threading.Lock() for gpu_id in gpu_ids}
        self.job_status = {}  # job_id -> status
        self.job_commands = {}  # job_id -> command
        self.job_results = {}  # job_id -> (return_code, stdout, stderr)
        self.job_priorities = {}  # job_id -> priority
        self.job_gpus = {}  # job_id -> requested GPU ID (or actual running GPU ID)
        self.job_processes = {}  # job_id -> process object for running jobs
        self.stop_event = threading.Event()

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Start worker threads for each GPU
        self.workers = []
        for gpu_id in gpu_ids:
            worker = threading.Thread(
                target=self._gpu_worker,
                args=(gpu_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # Start GPU status monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_gpu_status,
            daemon=True
        )
        self.monitor_thread.start()

        console.print(f"[bold green]GPU scheduler initialized[/bold green] with GPUs: {gpu_ids}")

    def _monitor_gpu_status(self):
        """Thread that periodically checks GPU status and updates availability"""
        while not self.stop_event.is_set():
            for gpu_id in self.gpu_ids:
                # Only check GPUs that are marked as occupied but not actively running a job
                if self.gpu_status.get(gpu_id) == "occupied by other":
                    # Check if GPU is now available
                    new_status = check_gpu_status(gpu_id)
                    if new_status == "free":
                        with self.gpu_locks[gpu_id]:
                            # Only update if it was previously occupied by other
                            if self.gpu_status.get(gpu_id) == "occupied by other":
                                self.gpu_status[gpu_id] = "free"
                                console.print(f"GPU {gpu_id} is now [green]free[/green]")

            # Sleep before next check
            time.sleep(self.status_check_interval)


    def enqueue(self, command: str, priority: int = 100, gpu_id: Optional[int] = None) -> str:
        """
        Add a command to the queue.

        Args:
            command: The command to run
            priority: Priority level (lower values = higher priority, default: 100)
            gpu_id: Specific GPU ID to run this job on (default: None, meaning any available GPU)

        Returns:
            job_id: Unique ID for the job
        """
        if gpu_id is not None and gpu_id not in self.gpu_ids:
            raise ValueError(f"GPU ID {gpu_id} is not available. Available GPUs: {self.gpu_ids}")

        job_id = str(uuid.uuid4())
        local_id = self.next_local_id
        self.next_local_id += 1

        # Store both mappings
        self.local_to_uuid[local_id] = job_id
        self.uuid_to_local[job_id] = local_id

        self.job_status[job_id] = "queued"
        self.job_commands[job_id] = command  # Store the command
        self.job_priorities[job_id] = priority
        self.job_gpus[job_id] = gpu_id  # Store preferred GPU (can be None)

        # Add to priority queue: (priority, local_id, job_id, command)
        # Local ID added to tiebreak equal priorities based on FIFO order
        self.queue.put((priority, local_id, job_id, command))

        gpu_msg = f" on GPU {gpu_id}" if gpu_id is not None else ""
        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] [bold]queued[/bold]{gpu_msg} (priority: {priority}): [dim]{command[:100]}{'...' if len(command) > 100 else ''}[/dim]")
        return job_id

    def prioritize(self, job_identifier: str, new_priority: int = 10) -> Dict[str, Any]:
        """
        Prioritize a queued job by changing its priority.

        Args:
            job_identifier: Local ID (as string with # prefix) or UUID of the job
            new_priority: New priority value (lower = higher priority)

        Returns:
            Result dictionary with status information
        """
        # Determine if using local ID or UUID
        job_id = None
        local_id = None

        if job_identifier.startswith('#'):
            try:
                local_id = int(job_identifier[1:])
                if local_id in self.local_to_uuid:
                    job_id = self.local_to_uuid[local_id]
                else:
                    return {"error": f"Local job ID {local_id} not found"}
            except ValueError:
                return {"error": f"Invalid local job ID format: {job_identifier}"}
        else:
            job_id = job_identifier
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found"}
            local_id = self.uuid_to_local.get(job_id)

        # Can only prioritize queued jobs
        if self.job_status[job_id] != "queued":
            return {"error": f"Can only prioritize queued jobs. Job #{local_id} is '{self.job_status[job_id]}'"}

        # Update the job's priority
        old_priority = self.job_priorities[job_id]
        self.job_priorities[job_id] = new_priority

        # We need to remove and re-add the job to the queue with new priority
        # This is tricky since PriorityQueue doesn't support removing items
        # We'll use a new queue and transfer all items with modified priority for our target job

        new_queue = queue.PriorityQueue()
        while not self.queue.empty():
            try:
                priority, item_local_id, item_job_id, item_command = self.queue.get(block=False)

                # If this is our target job, use the new priority
                if item_job_id == job_id:
                    new_queue.put((new_priority, item_local_id, item_job_id, item_command))
                else:
                    new_queue.put((priority, item_local_id, item_job_id, item_command))
            except queue.Empty:
                break

        # Replace the old queue with our new one
        self.queue = new_queue

        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] priority changed: {old_priority} → [bold]{new_priority}[/bold]")
        return {
            "job_id": job_id,
            "local_id": local_id,
            "status": "prioritized",
            "old_priority": old_priority,
            "new_priority": new_priority
        }

    def cancel_job(self, job_identifier: str) -> Dict[str, Any]:
        """
        Cancel a queued job.

        Args:
            job_identifier: Local ID (as string with # prefix) or UUID of the job

        Returns:
            Result dictionary with status information
        """
        # Determine if using local ID or UUID
        job_id = None
        local_id = None

        if job_identifier.startswith('#'):
            try:
                local_id = int(job_identifier[1:])
                if local_id in self.local_to_uuid:
                    job_id = self.local_to_uuid[local_id]
                else:
                    return {"error": f"Local job ID {local_id} not found"}
            except ValueError:
                return {"error": f"Invalid local job ID format: {job_identifier}"}
        else:
            job_id = job_identifier
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found"}
            local_id = self.uuid_to_local.get(job_id)

        # Check if the job is in the queue (can only cancel queued jobs)
        if self.job_status[job_id] != "queued":
            return {"error": f"Can only cancel queued jobs. Job #{local_id} is '{self.job_status[job_id]}'"}

        # Update job status BEFORE modifying the queue to avoid race conditions with worker threads
        self.job_status[job_id] = "cancelled"

        # Create a new queue without the job to cancel
        new_queue = queue.PriorityQueue()
        job_found = False

        while not self.queue.empty():
            try:
                priority, item_local_id, item_job_id, item_command = self.queue.get(block=False)

                # If this is not our target job, keep it in the queue
                if item_job_id != job_id:
                    new_queue.put((priority, item_local_id, item_job_id, item_command))
                else:
                    job_found = True

                # Always mark the task as done regardless of whether we're keeping it
                # self.queue.task_done() # Removed: task_done should only be called by worker after processing
            except queue.Empty:
                break

        # Replace the old queue with our new one
        self.queue = new_queue

        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] has been [bold red]cancelled[/bold red]")
        return {
            "job_id": job_id,
            "local_id": local_id,
            "status": "cancelled"
        }

    def kill_job(self, job_identifier: str) -> Dict[str, Any]:
        """
        Kill a running job.

        Args:
            job_identifier: Local ID (as string with # prefix) or UUID of the job

        Returns:
            Result dictionary with status information
        """
        # Determine if using local ID or UUID
        job_id = None
        local_id = None

        if job_identifier.startswith('#'):
            try:
                local_id = int(job_identifier[1:])
                if local_id in self.local_to_uuid:
                    job_id = self.local_to_uuid[local_id]
                else:
                    return {"error": f"Local job ID {local_id} not found"}
            except ValueError:
                return {"error": f"Invalid local job ID format: {job_identifier}"}
        else:
            job_id = job_identifier
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found"}
            local_id = self.uuid_to_local.get(job_id)

        # Check if the job is running
        if self.job_status[job_id] != "running":
            return {"error": f"Can only kill running jobs. Job #{local_id} is '{self.job_status[job_id]}'"}

        # Check if we have a process object for this job
        if job_id not in self.job_processes or self.job_processes[job_id] is None:
            return {"error": f"No process found for job #{local_id}"}

        # Find which GPU this job is running on (stored when job starts)
        job_gpu = self.job_gpus.get(job_id) # This should now hold the actual running GPU

        # Attempt to kill the process
        try:
            process = self.job_processes[job_id]
            process.terminate()

            # Give it a moment to terminate gracefully, then kill if needed
            time.sleep(2)
            if process.poll() is None:
                # Process didn't terminate, force kill
                process.kill()

            # Update job status
            self.job_status[job_id] = "killed"

            # Clean up process reference
            self.job_processes.pop(job_id, None)

            # Update logs with termination information
            stdout_log = os.path.join(self.log_dir, f"{job_id}_stdout.log")
            stderr_log = os.path.join(self.log_dir, f"{job_id}_stderr.log")

            try:
                with open(stderr_log, 'a') as f:
                    f.write("\n\n=== JOB KILLED BY USER ===\n")
            except:
                pass

            console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] has been [bold red]killed[/bold red]")

            # Free up GPU status if we know which GPU this job was running on
            if job_gpu is not None:
                with self.gpu_locks[job_gpu]:
                    # Check the actual status again
                    current_status = check_gpu_status(job_gpu)
                    # Only update if we still think it's busy (avoid overwriting reserved/occupied)
                    if self.gpu_status.get(job_gpu) == "busy":
                        self.gpu_status[job_gpu] = current_status
                        status_color = "green" if current_status == "free" else "yellow"
                        console.print(f"GPU {job_gpu} status after kill: [{status_color}]{current_status}[/{status_color}]")
            else:
                # If for some reason we don't know the GPU, check all busy ones
                gpus_to_check = [gid for gid, status in self.gpu_status.items() if status == "busy"]
                for gpu_id_to_check in gpus_to_check:
                    with self.gpu_locks[gpu_id_to_check]:
                        current_status = check_gpu_status(gpu_id_to_check)
                        if self.gpu_status.get(gpu_id_to_check) == "busy":
                            self.gpu_status[gpu_id_to_check] = current_status
                            status_color = "green" if current_status == "free" else "yellow"
                            console.print(f"GPU {gpu_id_to_check} status after kill (re-checked): [{status_color}]{current_status}[/{status_color}]")


            return {
                "job_id": job_id,
                "local_id": local_id,
                "status": "killed"
            }
        except Exception as e:
            return {"error": f"Failed to kill job #{local_id}: {str(e)}"}

    def requeue_job(self, job_identifier: str) -> Dict[str, Any]:
        """
        Kill a running job and put it back into the queue.

        Args:
            job_identifier: Local ID (as string with # prefix) or UUID of the job

        Returns:
            Result dictionary with status information
        """
        # Determine if using local ID or UUID
        job_id = None
        local_id = None

        if job_identifier.startswith('#'):
            try:
                local_id = int(job_identifier[1:])
                if local_id in self.local_to_uuid:
                    job_id = self.local_to_uuid[local_id]
                else:
                    return {"error": f"Local job ID {local_id} not found"}
            except ValueError:
                return {"error": f"Invalid local job ID format: {job_identifier}"}
        else:
            job_id = job_identifier
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found"}
            local_id = self.uuid_to_local.get(job_id)

        # Check if the job is running
        if self.job_status[job_id] != "running":
            return {"error": f"Can only requeue running jobs. Job #{local_id} is '{self.job_status[job_id]}'"}

        # Get original job details before killing
        original_command = self.job_commands.get(job_id)
        original_priority = self.job_priorities.get(job_id, 100) # Default priority if somehow missing
        original_gpu_preference = self.job_gpus.get(job_id) # Get original preference (might be None or specific GPU)

        if not original_command:
             return {"error": f"Could not retrieve original command for job #{local_id}"}

        console.print(f"Attempting to kill and requeue job #{local_id} [cyan]{job_id}[/cyan]...")

        # Kill the job
        kill_result = self.kill_job(job_identifier)

        if "error" in kill_result:
            console.print(f"[bold red]Failed to kill job #{local_id} before requeueing:[/bold red] {kill_result['error']}")
            return {"error": f"Failed to kill job before requeueing: {kill_result['error']}"}

        # If kill was successful, enqueue the job again
        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] killed successfully. Requeueing command...")
        try:
            # Requeue with original command, priority, and GPU preference
            new_job_id = self.enqueue(original_command, original_priority, original_gpu_preference)
            new_local_id = self.uuid_to_local.get(new_job_id)
            console.print(f"Original job #{local_id} [cyan]{job_id}[/cyan] requeued as job #{new_local_id} [cyan]{new_job_id}[/cyan]")
            return {
                "success": True,
                "killed_job_id": job_id,
                "killed_local_id": local_id,
                "new_job_id": new_job_id,
                "new_local_id": new_local_id,
                "status": "requeued"
            }
        except Exception as e:
             console.print(f"[bold red]Error requeueing job #{local_id}:[/bold red] {str(e)}")
             return {"error": f"Job was killed, but failed to requeue: {str(e)}"}


    def reserve_gpus(self, count: int) -> Dict[str, Any]:
        """
        Reserve a specified number of free GPUs.

        Args:
            count: Number of GPUs to reserve

        Returns:
            Result dictionary with status information and list of reserved GPU IDs
        """
        if count <= 0:
            return {"error": "Must reserve at least one GPU"}

        # Find free GPUs
        free_gpus = []
        for gpu_id in self.gpu_ids:
            with self.gpu_locks[gpu_id]:
                if self.gpu_status.get(gpu_id) == "free":
                    free_gpus.append(gpu_id)

        # Check if we have enough free GPUs
        if len(free_gpus) < count:
            return {"error": f"Not enough free GPUs. Requested: {count}, Available: {len(free_gpus)}"}

        # Reserve the GPUs (limited to the requested count)
        reserved_gpus = []
        for gpu_id in free_gpus[:count]:
            with self.gpu_locks[gpu_id]:
                self.gpu_status[gpu_id] = "reserved"
                reserved_gpus.append(gpu_id)
            console.print(f"GPU {gpu_id} is now [magenta]reserved[/magenta]")

        return {
            "success": True,
            "message": f"Reserved {len(reserved_gpus)} GPUs",
            "reserved_gpus": reserved_gpus
        }

    def release_gpu(self, gpu_id: int) -> Dict[str, Any]:
        """
        Release a reserved GPU back to free state.

        Args:
            gpu_id: GPU ID to release

        Returns:
            Result dictionary with status information
        """
        if gpu_id not in self.gpu_ids:
            return {"error": f"GPU ID {gpu_id} is not valid. Available GPUs: {self.gpu_ids}"}

        with self.gpu_locks[gpu_id]:
            current_status = self.gpu_status.get(gpu_id)

            if current_status != "reserved":
                return {"error": f"GPU {gpu_id} is not reserved. Current status: {current_status}"}

            # Check actual GPU status before marking as free
            actual_status = check_gpu_status(gpu_id)
            self.gpu_status[gpu_id] = actual_status
            status_color = "green" if actual_status == "free" else "yellow"
            console.print(f"GPU {gpu_id} released from reservation, current status: [{status_color}]{actual_status}[/{status_color}]")

        return {
            "success": True,
            "gpu_id": gpu_id,
            "status": actual_status,
            "message": f"GPU {gpu_id} released from reservation"
        }

    def _gpu_worker(self, gpu_id: int):
        """
        Worker thread that executes jobs on a specific GPU.

        Args:
            gpu_id: The GPU ID to use
        """
        current_job_id = None # Track the job this worker is running
        while not self.stop_event.is_set():
            job_acquired = False
            job_id = None # Reset job_id for each attempt
            try:
                # Check if GPU is available for processing jobs
                with self.gpu_locks[gpu_id]: # Lock needed to read/write gpu_status safely
                    current_gpu_status = self.gpu_status.get(gpu_id)

                # Skip if GPU is marked as occupied by other processes, reserved, or already busy with our job
                if current_gpu_status in ["occupied by other", "busy", "reserved"]:
                    # Double-check actual GPU status to detect cases where a job was killed externally
                    if current_gpu_status == "busy":
                        actual_status = check_gpu_status(gpu_id)
                        if actual_status == "free":
                            # GPU shows as free but we think it's busy - update our status
                            with self.gpu_locks[gpu_id]:
                                # Check again inside lock to avoid race condition
                                if self.gpu_status.get(gpu_id) == "busy":
                                    self.gpu_status[gpu_id] = "free"
                                    console.print(f"GPU {gpu_id} was marked as busy but is actually free. Status updated.")
                                    current_job_id = None # Clear the job ID if the GPU became free unexpectedly
                            # Continue to next iteration to try and get a job
                            continue

                    time.sleep(1)
                    continue

                # Try to get a job from the queue with a timeout
                try:
                    priority, local_id, job_id, command = self.queue.get(timeout=1)
                    job_acquired = True  # Flag that we got a job from the queue

                    # Check if this job has been cancelled already
                    if self.job_status.get(job_id) == "cancelled":
                        # Job was cancelled while in queue, don't process it
                        self.queue.task_done()  # Mark it as done in the queue
                        job_acquired = False  # Reset flag since we've handled it
                        job_id = None # Clear job_id
                        continue

                    # Check if this job has a GPU preference
                    preferred_gpu = self.job_gpus.get(job_id)
                    if preferred_gpu is not None and preferred_gpu != gpu_id:
                        # Job has a specific GPU preference and it's not this GPU
                        # Put job back in queue and skip
                        self.queue.put((priority, local_id, job_id, command))
                        self.queue.task_done()  # Mark it as done for this attempt
                        job_acquired = False  # Reset flag since we've handled it
                        job_id = None # Clear job_id
                        continue

                except queue.Empty:
                    job_id = None # Clear job_id
                    continue

                # Lock this GPU to claim it
                with self.gpu_locks[gpu_id]:
                    # Double check GPU is still free (or became free)
                    current_status = check_gpu_status(gpu_id)
                    if current_status != "free":
                        # GPU became occupied/reserved while we were waiting
                        # Put job back in queue and skip
                        self.queue.put((priority, local_id, job_id, command))
                        self.queue.task_done()  # Mark it as done for this attempt
                        job_acquired = False  # Reset flag since we've handled it
                        job_id = None # Clear job_id
                        time.sleep(0.5) # Short sleep before retrying
                        continue

                    # Check again if job has been cancelled (last check before running)
                    if self.job_status.get(job_id) == "cancelled":
                        # Job was cancelled while we were checking GPU status
                        self.queue.task_done()  # Mark it as done in the queue
                        job_acquired = False  # Reset flag since we've handled it
                        job_id = None # Clear job_id
                        continue

                    # Mark GPU as busy for our job and update job status
                    self.gpu_status[gpu_id] = "busy"
                    self.job_status[job_id] = "running"
                    current_job_id = job_id # Track the job this worker is running
                    # Store the actual GPU this job is running on (overwrites preference if None)
                    self.job_gpus[job_id] = gpu_id

                # --- Job Execution (GPU lock released after this point) ---
                try:
                    # Set up environment variables to make this GPU visible
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    # Log files
                    stdout_log = os.path.join(self.log_dir, f"{job_id}_stdout.log")
                    stderr_log = os.path.join(self.log_dir, f"{job_id}_stderr.log")

                    console.print(f"Running job #{local_id} [cyan]{job_id}[/cyan] on GPU [bold green]{gpu_id}[/bold green]: [dim]{command[:100]}{'...' if len(command) > 100 else ''}[/dim]")

                    # Execute the command
                    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
                        process = subprocess.Popen(
                            command,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            shell=True,
                            env=env
                        )

                        # Store the process object for potential killing later
                        self.job_processes[job_id] = process

                        # Wait for the process to complete
                        return_code = process.wait()

                        # Remove process from tracking dict only if it finished naturally
                        # If killed, kill_job will remove it.
                        if self.job_status.get(job_id) != "killed":
                             self.job_processes.pop(job_id, None)


                    # Read the logs (only if job wasn't killed, results might be incomplete if killed)
                    stdout = ""
                    stderr = ""
                    if self.job_status.get(job_id) != "killed":
                        try:
                            with open(stdout_log, 'r') as f:
                                stdout = f.read()
                        except FileNotFoundError: pass # Log might not exist if command failed instantly
                        try:
                            with open(stderr_log, 'r') as f:
                                stderr = f.read()
                        except FileNotFoundError: pass

                    # Update job status (check if it was killed during execution)
                    final_status = self.job_status.get(job_id)
                    if final_status == "killed":
                        status_color = "red"
                        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] was [bold red]killed[/bold red] while running.")
                        # Logs might be incomplete, store what we have
                        self.job_results[job_id] = (-1, stdout, stderr + "\n=== JOB KILLED ===")
                    elif final_status == "running": # If not killed, determine completed/failed
                        self.job_status[job_id] = "completed" if return_code == 0 else "failed"
                        self.job_results[job_id] = (return_code, stdout, stderr)
                        status_color = "green" if return_code == 0 else "red"
                        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] [bold {status_color}]{self.job_status[job_id]}[/bold {status_color}] with return code {return_code}")
                    # else: job status might have been changed by another operation (e.g., requeue starting) - leave it

                finally:
                    # --- Cleanup after job execution ---
                    current_job_id = None # No longer running this job

                    # Always update GPU status after job attempt, acquire lock first
                    with self.gpu_locks[gpu_id]:
                        # Check the actual status again
                        current_status = check_gpu_status(gpu_id)
                        # Only update if we still think it's busy (avoid overwriting reserved/occupied)
                        if self.gpu_status.get(gpu_id) == "busy":
                            self.gpu_status[gpu_id] = current_status
                            status_color = "green" if current_status == "free" else "yellow"
                            console.print(f"GPU {gpu_id} status after job: [{status_color}]{current_status}[/{status_color}]")

                    # Job was successfully processed or handled (even if failed/killed), mark it as done in the queue
                    if job_acquired:
                        self.queue.task_done()
                        job_acquired = False # Reset flag
                        job_id = None # Clear job_id

            except Exception as e:
                console.print(f"[bold red]Error[/bold red] in GPU worker {gpu_id}: {e}")
                # Attempt to update GPU status in case of unexpected error
                try:
                    with self.gpu_locks[gpu_id]:
                        current_status = check_gpu_status(gpu_id)
                        # Only update if we thought it was busy
                        if self.gpu_status.get(gpu_id) == "busy":
                             self.gpu_status[gpu_id] = current_status
                except Exception as e_inner:
                     console.print(f"[bold red]Error updating GPU {gpu_id} status after worker error:[/bold red] {e_inner}")

            finally:
                # If we got a job from the queue but didn't process it due to an exception,
                # make sure we mark it as done to avoid queue getting stuck
                if job_acquired and job_id is not None: # Check job_id is not None here
                    try:
                        self.queue.task_done()
                    except ValueError: # Can happen if task_done() called too many times
                        pass
                    job_acquired = False # Reset flag
                    job_id = None # Clear job_id


    def get_status(self, job_id: Optional[str] = None, include_cancelled: bool = False) -> Dict:
        """
        Get the status of a specific job or all jobs.

        Args:
            job_id: Optional job ID to get status for
            include_cancelled: Whether to include cancelled jobs in the status output

        Returns:
            Dictionary with job status information
        """
        # First refresh GPU statuses (only for non-busy GPUs)
        for gpu_id in self.gpu_ids:
             with self.gpu_locks[gpu_id]: # Lock needed to safely read status
                 current_internal_status = self.gpu_status.get(gpu_id)
             # Only check externally if not busy AND not reserved (avoids unnecessary nvidia-smi calls)
             if current_internal_status not in ["busy", "reserved"]:
                     external_status = check_gpu_status(gpu_id)
                     with self.gpu_locks[gpu_id]: # Lock needed to safely write status
                         # Only update if the status actually changed
                         if self.gpu_status.get(gpu_id) != external_status:
                             self.gpu_status[gpu_id] = external_status


        if job_id:
            # Check if job_id is actually a local ID with # prefix
            original_identifier = job_id
            if job_id.startswith('#'):
                try:
                    local_id = int(job_id[1:])
                    if local_id in self.local_to_uuid:
                        job_id = self.local_to_uuid[local_id]
                    else:
                        return {"error": f"Local job ID {local_id} not found"}
                except ValueError:
                    return {"error": f"Invalid local job ID format: {original_identifier}"}

            # Now job_id should be a UUID
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found (from identifier '{original_identifier}')"}

            local_id = self.uuid_to_local.get(job_id)
            status = self.job_status[job_id]

            # Skip cancelled jobs if requested
            if status == "cancelled" and not include_cancelled:
                 return {"error": f"Job ID {job_id} (#{local_id}) was cancelled."}


            result = {
                "job_id": job_id,
                "local_id": local_id,
                "status": status,
                "priority": self.job_priorities.get(job_id, "N/A"),
                "command": self.job_commands.get(job_id, "Unknown command"),
                "preferred_gpu": self.job_gpus.get(job_id) # Show preferred/running GPU
            }

            if status in ["completed", "failed", "killed"] and job_id in self.job_results:
                return_code, stdout, stderr = self.job_results[job_id]
                result["return_code"] = return_code
                # Include truncated stdout/stderr for readability
                result["stdout_preview"] = stdout[:500] + "..." if len(stdout) > 500 else stdout
                result["stderr_preview"] = stderr[:500] + "..." if len(stderr) > 500 else stderr

            return result
        else:
            # Filter jobs based on include_cancelled flag
            filtered_jobs = {}
            filtered_priorities = {}
            filtered_commands = {}
            filtered_local_ids = {}
            filtered_gpus = {}

            for jid, status in self.job_status.items():
                 if status == "cancelled" and not include_cancelled:
                     continue
                 filtered_jobs[jid] = status
                 filtered_priorities[jid] = self.job_priorities.get(jid)
                 filtered_commands[jid] = self.job_commands.get(jid)
                 filtered_local_ids[jid] = self.uuid_to_local.get(jid)
                 filtered_gpus[jid] = self.job_gpus.get(jid)


            # Get a count of jobs by status
            status_counts = {}
            for status in filtered_jobs.values():
                status_counts[status] = status_counts.get(status, 0) + 1

            # Get current queue contents (need to drain and refill to inspect)
            queued_items = []
            temp_queue = queue.PriorityQueue()
            while not self.queue.empty():
                 try:
                     item = self.queue.get(block=False)
                     queued_items.append({
                         "priority": item[0],
                         "local_id": item[1],
                         "job_id": item[2],
                         "command": item[3]
                     })
                     temp_queue.put(item)
                 except queue.Empty:
                     break
            self.queue = temp_queue # Restore queue


            return {
                "queue_size": len(queued_items),
                "queued_jobs": queued_items, # List of jobs currently in the priority queue
                "gpu_status": self.gpu_status.copy(), # Return a copy to avoid modification issues
                "all_jobs_status": filtered_jobs, # Status of all tracked jobs (UUID -> status)
                "job_priorities": filtered_priorities, # UUID -> priority
                "job_commands": filtered_commands, # UUID -> command
                "job_local_ids": filtered_local_ids, # UUID -> local_id
                "job_gpus": filtered_gpus, # UUID -> preferred/running GPU
                "status_counts": status_counts, # Counts of jobs by status
            }

    def get_log(self, job_identifier: str, output_type: str = "both", last_lines: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the logs (stdout and stderr) for a specific job.

        Args:
            job_identifier: Job identifier (either UUID or local ID with # prefix)
            output_type: Type of output to return: "both", "stdout", or "stderr"
            last_lines: If specified, returns only the last N lines of the logs

        Returns:
            Dictionary with job logs and metadata
        """
        # Check if job_id is actually a local ID with # prefix
        job_id = job_identifier
        if job_identifier.startswith('#'):
            try:
                local_id = int(job_identifier[1:])
                if local_id in self.local_to_uuid:
                    job_id = self.local_to_uuid[local_id]
                else:
                    return {"error": f"Local job ID {local_id} not found"}
            except ValueError:
                return {"error": f"Invalid local job ID format: {job_identifier}"}

        # Now job_id should be a UUID
        if job_id not in self.job_status:
            return {"error": f"Job ID {job_id} not found"}

        local_id = self.uuid_to_local.get(job_id)
        status = self.job_status[job_id]

        stdout_log_path = os.path.join(self.log_dir, f"{job_id}_stdout.log")
        stderr_log_path = os.path.join(self.log_dir, f"{job_id}_stderr.log")
        stdout_content = ""
        stderr_content = ""
        message = ""

        # Try reading logs from files first, works for running, completed, failed, killed
        if os.path.exists(stdout_log_path) and (output_type == "both" or output_type == "stdout"):
             try:
                 with open(stdout_log_path, 'r') as f:
                     if last_lines is None:
                         stdout_content = f.read()
                     else:
                         # Read only the last N lines (using a simpler approach)
                         lines = f.readlines()
                         stdout_content = "".join(lines[-last_lines:])
             except Exception as e:
                 stdout_content = f"Error reading stdout log: {str(e)}"

        if os.path.exists(stderr_log_path) and (output_type == "both" or output_type == "stderr"):
             try:
                 with open(stderr_log_path, 'r') as f:
                     if last_lines is None:
                         stderr_content = f.read()
                     else:
                         lines = f.readlines()
                         stderr_content = "".join(lines[-last_lines:])
             except Exception as e:
                 stderr_content = f"Error reading stderr log: {str(e)}"


        # Handle specific statuses
        if status == "queued":
            message = "Job is queued. No logs available yet."
        elif status == "running":
             message = "Job is running. Logs may be incomplete."
        elif status in ["completed", "failed", "killed"]:
             message = f"Job finished with status: {status}"
             # If logs were read from file, they are already populated.
             # If results are stored in memory (e.g., from older version or direct run), use them.
             if not stdout_content and not stderr_content and job_id in self.job_results:
                  _, stdout_mem, stderr_mem = self.job_results[job_id]
                  if output_type == "both" or output_type == "stdout":
                       stdout_content = "\n".join(stdout_mem.splitlines()[-last_lines:]) if last_lines else stdout_mem
                  if output_type == "both" or output_type == "stderr":
                       stderr_content = "\n".join(stderr_mem.splitlines()[-last_lines:]) if last_lines else stderr_mem
        elif status == "cancelled":
             message = "Job was cancelled before running. No logs available."


        result = {
            "job_id": job_id,
            "local_id": local_id,
            "status": status,
            "message": message,
            "command": self.job_commands.get(job_id, "Unknown command"),
            "return_code": self.job_results.get(job_id, (None, None, None))[0] # Get return code if available
        }

        # Add stdout/stderr based on output_type
        if output_type == "both" or output_type == "stdout":
            result["stdout"] = stdout_content
        if output_type == "both" or output_type == "stderr":
            result["stderr"] = stderr_content

        return result

    def _get_last_n_lines(self, file_obj, n: int):
        """Helper method to efficiently get the last N lines of a file"""
        # This implementation was complex and potentially buggy, using simpler readlines approach now
        # Kept here for reference if needed later
        raise NotImplementedError("Using simpler readlines approach in get_log")


    def shutdown(self):
        """Shutdown the scheduler and wait for all workers to complete"""
        console.print("[yellow]Shutting down GPU scheduler...[/yellow]")
        self.stop_event.set()
        # Wait for monitor thread first
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2) # Short timeout for monitor

        # Wait for worker threads
        for worker in self.workers:
             if worker.is_alive():
                 worker.join(timeout=5) # Give workers some time to finish current step

        # Check if any workers are still alive (forceful shutdown)
        alive_workers = [w for w in self.workers if w.is_alive()]
        if alive_workers:
             console.print(f"[yellow]Warning: {len(alive_workers)} worker threads did not exit gracefully.[/yellow]")

        console.print("[bold green]GPU scheduler shutdown complete[/bold green]")

    def kill_all_jobs(self) -> Dict[str, Any]:
        """
        Kill all running jobs.

        Returns:
            Result dictionary with status information about killed jobs
        """
        killed_jobs = []
        failed_jobs = []

        # Find all running jobs based on status
        running_job_ids = [job_id for job_id, status in self.job_status.items() if status == "running"]

        if not running_job_ids:
            return {
                "success": True,
                "message": "No running jobs to kill",
                "killed_jobs": [],
                "failed_jobs": []
            }

        console.print(f"Attempting to kill {len(running_job_ids)} running jobs...")
        # Attempt to kill each running job
        for job_id in running_job_ids:
            local_id = self.uuid_to_local.get(job_id, "?")
            kill_result = self.kill_job(job_id) # Use kill_job logic for each
            if "error" in kill_result:
                 failed_jobs.append((job_id, local_id, kill_result["error"]))
                 console.print(f"[red]Failed to kill job #{local_id} ({job_id}): {kill_result['error']}[/red]")
            else:
                 killed_jobs.append((job_id, local_id))
                 # kill_job already prints success message

        # Free up GPU status after killing jobs (redundant if kill_job works, but safe)
        for gpu_id, status in self.gpu_status.items():
            if status == "busy":
                current_status = check_gpu_status(gpu_id)
                with self.gpu_locks[gpu_id]:
                    self.gpu_status[gpu_id] = current_status

        return {
            "success": True,
            "message": f"Attempted to kill {len(running_job_ids)} jobs. Killed: {len(killed_jobs)}, Failed: {len(failed_jobs)}",
            "killed_jobs": killed_jobs,
            "failed_jobs": failed_jobs
        }


class SchedulerServer:
    def __init__(self, scheduler: GPUJobScheduler, host: str = 'localhost', port: int = 8000):
        """
        Initialize the scheduler server.

        Args:
            scheduler: The GPU job scheduler
            host: Host to bind to
            port: Port to bind to
        """
        self.scheduler = scheduler
        self.host = host
        self.port = port
        self.server_socket = None
        self.stop_event = threading.Event()
        self.server_thread = None # Initialize server_thread attribute

    def start(self):
        """Start the scheduler server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Allow port reuse
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind and listen
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1)  # 1 second timeout for accept

            console.print(f"[bold green]Scheduler server started[/bold green] on [blue]{self.host}:{self.port}[/blue]")

            # Start server loop in a thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
        except OSError as e:
             console.print(f"[bold red]Error starting server on {self.host}:{self.port}: {e}[/bold red]")
             console.print("[yellow]Server could not start. Running in CLI-only mode.[/yellow]")
             self.server_socket = None # Ensure socket is None if start failed
        except Exception as e:
             console.print(f"[bold red]Unexpected error starting server: {e}[/bold red]")
             self.server_socket = None


    def _server_loop(self):
        """Server loop to accept connections and handle requests"""
        if not self.server_socket: # Check if socket was initialized
             console.print("[red]Server socket not available. Server loop cannot run.[/red]")
             return

        while not self.stop_event.is_set():
            try:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except OSError as e: # Handle cases where socket might be closed during shutdown
                 if not self.stop_event.is_set():
                      console.print(f"[bold red]Socket error in server loop:[/bold red] {e}")
                 break # Exit loop on socket error
            except Exception as e:
                if not self.stop_event.is_set():
                    console.print(f"[bold red]Error accepting connection:[/bold red] {e}")

    def _handle_client(self, client_socket: socket.socket, addr: Tuple[str, int]):
        """
        Handle a client connection.

        Args:
            client_socket: Client socket
            addr: Client address
        """
        response = "" # Initialize response
        try:
            # Set a timeout for client operations
            client_socket.settimeout(10) # Increased timeout

            # Receive data (handle potential larger commands)
            buffer = ""
            while True:
                 chunk = client_socket.recv(4096).decode('utf-8')
                 if not chunk:
                      break # Connection closed
                 buffer += chunk
                 # Basic check for end of command (newline) - adjust if needed
                 if '\n' in buffer:
                      break

            data = buffer.strip()
            if not data: # Handle empty messages
                 response = "Error: Received empty command.\n"
                 client_socket.sendall(response.encode('utf-8'))
                 return


            console.print(f"Received command from [blue]{addr[0]}:{addr[1]}[/blue]: [dim]{data[:100]}{'...' if len(data)>100 else ''}[/dim]")

            # Process command
            if data.startswith("> queue "):
                # Check for GPU specification with "on gpu X" or similar pattern
                command_part = data[8:].strip()
                command = command_part
                gpu_id = None

                # Check for GPU specification at the end: "command on gpu X"
                # More robust parsing allowing spaces in command itself
                gpu_spec_found = False
                for pattern in [" on gpu ", " on GPU ", " --gpu ", " --GPU "]:
                    if pattern in command_part:
                         parts = command_part.rsplit(pattern, 1) # Split only on the last occurrence
                         if len(parts) == 2 and parts[1].strip().isdigit():
                              command = parts[0].strip() # The actual command
                              try:
                                   gpu_id = int(parts[1].strip())
                                   gpu_spec_found = True
                                   break
                              except ValueError:
                                   pass # Not a valid integer after the pattern

                try:
                    job_id = self.scheduler.enqueue(command, gpu_id=gpu_id)
                    local_id = self.scheduler.uuid_to_local.get(job_id)
                    gpu_msg = f" on GPU {gpu_id}" if gpu_id is not None else ""
                    response = f"Job submitted with ID: {job_id} (#{local_id}){gpu_msg}\n"
                except ValueError as e: # Catch specific error from enqueue
                     response = f"Error: {str(e)}\n"
                except Exception as e: # Catch other potential errors
                     response = f"Error submitting job: {str(e)}\n"


            elif data.startswith("> prioritize "):
                parts = data.split(maxsplit=3)
                if len(parts) < 3:
                    response = "Error: Usage: > prioritize <job_id|#local_id> <priority>\n"
                else:
                    job_identifier = parts[1] # Correct index for job identifier
                    try:
                        priority = int(parts[2]) # Correct index for priority
                        result = self.scheduler.prioritize(job_identifier, priority)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            response = f"Job prioritized: #{result['local_id']} {result['job_id']} (priority: {result['new_priority']})\n"
                    except ValueError:
                        response = "Error: Priority must be an integer\n"
                    except Exception as e:
                         response = f"Error prioritizing job: {str(e)}\n"

            elif data.startswith("> reserve "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > reserve <count>\n"
                else:
                    try:
                        count = int(parts[1])
                        result = self.scheduler.reserve_gpus(count)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            response = f"Successfully reserved GPUs: {result['reserved_gpus']}\n"
                    except ValueError:
                        response = "Error: Count must be an integer\n"
                    except Exception as e:
                         response = f"Error reserving GPUs: {str(e)}\n"

            elif data.startswith("> release "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > release <gpu_id>\n"
                else:
                    try:
                        gpu_id = int(parts[1])
                        result = self.scheduler.release_gpu(gpu_id)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            status_color = "green" if result['status'] == "free" else "yellow"
                            response = f"Successfully released GPU {gpu_id}. Current status: [{status_color}]{result['status']}[/{status_color}]\n"
                    except ValueError:
                        response = "Error: GPU ID must be an integer\n"
                    except Exception as e:
                         response = f"Error releasing GPU: {str(e)}\n"

            elif data.startswith("> cancel "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > cancel <job_id|#local_id>\n"
                else:
                    job_identifier = parts[1]
                    try:
                        result = self.scheduler.cancel_job(job_identifier)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            response = f"Job cancelled: #{result['local_id']} {result['job_id']}\n"
                    except Exception as e:
                         response = f"Error cancelling job: {str(e)}\n"

            elif data.startswith("> kill "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > kill <job_id|#local_id>\n"
                else:
                    job_identifier = parts[1]
                    try:
                        result = self.scheduler.kill_job(job_identifier)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            response = f"Job killed: #{result['local_id']} {result['job_id']}\n"
                    except Exception as e:
                         response = f"Error killing job: {str(e)}\n"

            # --- NEW REQUEUE COMMAND ---
            elif data.startswith("> requeue "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > requeue <job_id|#local_id>\n"
                else:
                    job_identifier = parts[1]
                    try:
                        result = self.scheduler.requeue_job(job_identifier)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        elif result.get("success"):
                             response = (f"Job #{result['killed_local_id']} ({result['killed_job_id']}) killed "
                                         f"and requeued as Job #{result['new_local_id']} ({result['new_job_id']}).\n")
                        else: # Should not happen if error/success not present, but handle defensively
                             response = "Error: Unknown error during requeue operation.\n"
                    except Exception as e:
                         response = f"Error requeueing job: {str(e)}\n"
            # --- END REQUEUE COMMAND ---

            # --- NEW SHOW_COMMAND COMMAND ---
            elif data.startswith("> show_command "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > show_command <job_id|#local_id>\n"
                else:
                    job_identifier = parts[1]
                    job_id = None
                    local_id = None
                    try:
                        # Resolve job identifier
                        if job_identifier.startswith('#'):
                            try:
                                local_id = int(job_identifier[1:])
                                if local_id in self.scheduler.local_to_uuid:
                                    job_id = self.scheduler.local_to_uuid[local_id]
                                else:
                                    response = f"Error: Local job ID {local_id} not found\n"
                            except ValueError:
                                response = f"Error: Invalid local job ID format: {job_identifier}\n"
                        else:
                            job_id = job_identifier
                            if job_id not in self.scheduler.job_status:
                                response = f"Error: Job ID {job_id} not found\n"
                            else:
                                local_id = self.scheduler.uuid_to_local.get(job_id)

                        # If job_id was found and no error response yet
                        if job_id and not response.startswith("Error"):
                            command = self.scheduler.job_commands.get(job_id)
                            if command is not None:
                                response = f"Command for Job #{local_id} ({job_id}):\n{command}\n"
                            else:
                                # Should not happen if job_id exists in job_status, but check anyway
                                response = f"Error: Command not found for job #{local_id} ({job_id})\n"
                    except Exception as e:
                         response = f"Error showing command: {str(e)}\n"
            # --- END SHOW_COMMAND COMMAND ---

            elif data.startswith("> shutdown"):
                # Check if there's a force flag
                force = False
                if len(data.split()) > 1 and data.split()[1].lower() == "force":
                    force = True

                # Get running jobs - include cancelled jobs here to maintain full job history
                try:
                    status_data = self.scheduler.get_status(include_cancelled=True)
                    running_jobs = [(jid, status_data["job_local_ids"].get(jid))
                                    for jid, status in status_data["all_jobs_status"].items()
                                    if status == "running"]

                    if running_jobs and not force:
                        response = "WARNING: There are running jobs that will be killed.\n"
                        for job_id, local_id in running_jobs:
                            response += f"  #{local_id} {job_id}\n"
                        response += "To confirm shutdown and kill all jobs, use: > shutdown force\n"
                    else:
                        if running_jobs:
                            # Kill all running jobs
                            kill_result = self.scheduler.kill_all_jobs()
                            response = f"Killed {len(kill_result['killed_jobs'])} running jobs.\n"
                            if kill_result['failed_jobs']:
                                response += f"Failed to kill {len(kill_result['failed_jobs'])} jobs.\n"
                        else:
                            response = "No running jobs to kill.\n"

                        response += "Shutting down the scheduler...\n"
                        # Set a timer to shutdown after sending response
                        shutdown_timer = threading.Timer(1.0, self.stop) # Call server stop method
                        shutdown_timer.daemon = True
                        shutdown_timer.start()
                        # Also trigger scheduler shutdown
                        scheduler_shutdown_timer = threading.Timer(1.5, self.scheduler.shutdown)
                        scheduler_shutdown_timer.daemon = True
                        scheduler_shutdown_timer.start()

                except Exception as e:
                     response = f"Error during shutdown process: {str(e)}\n"


            elif data.startswith("> log "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > log <job_id|#local_id>\n"
                else:
                    job_identifier = parts[1]
                    try:
                        log_data = self.scheduler.get_log(job_identifier)

                        if "error" in log_data:
                            response = f"Error: {log_data['error']}\n"
                        else:
                            response = f"=== Job Details ===\n"
                            response += f"Job ID: {log_data['job_id']} (#{log_data['local_id']})\n"
                            response += f"Status: {log_data['status']}\n"

                            if "message" in log_data:
                                response += f"{log_data['message']}\n"

                            if "command" in log_data:
                                response += f"Command: {log_data['command']}\n"

                            if "return_code" in log_data and log_data['return_code'] is not None:
                                response += f"Return Code: {log_data['return_code']}\n"

                            if "stdout" in log_data:
                                response += f"\n=== STDOUT ===\n{log_data['stdout']}\n"

                            if "stderr" in log_data:
                                response += f"\n=== STDERR ===\n{log_data['stderr']}\n"
                    except Exception as e:
                         response = f"Error retrieving logs: {str(e)}\n"

            elif data.startswith("> status"):
                parts = data.split(maxsplit=2)
                job_identifier = parts[1] if len(parts) > 1 else None
                try:
                    # Use include_cancelled=False to filter out cancelled jobs by default
                    status_data = self.scheduler.get_status(job_identifier, include_cancelled=False)
                    # Pretty print JSON for status response
                    response = json.dumps(status_data, indent=2) + "\n"
                except Exception as e:
                     response = f"Error getting status: {str(e)}\n"

            elif data.startswith("> help"):
                response = "Available commands:\n"
                response += "  > queue <command> [on gpu <gpu_id>]: Queue a command (optionally on a specific GPU)\n"
                response += "  > prioritize <job_id|#local_id> <priority>: Change job priority (lower = higher priority)\n"
                response += "  > reserve <count>: Reserve a number of free GPUs\n"
                response += "  > release <gpu_id>: Release a reserved GPU back to free state\n"
                response += "  > cancel <job_id|#local_id>: Cancel a queued job\n"
                response += "  > kill <job_id|#local_id>: Kill a running job\n"
                response += "  > requeue <job_id|#local_id>: Kill a running job and add it back to the queue\n"
                response += "  > status [job_id|#local_id]: Get status of a job or all jobs\n"
                response += "  > log <job_id|#local_id>: Display full stdout and stderr logs for a job\n"
                response += "  > show_command <job_id|#local_id>: Display the full command for a job\n" # Added show_command
                response += "  > shutdown [force]: Shutdown the scheduler (warns if jobs are running unless force is used)\n"
                response += "  > help: Show this help message\n"

            else:
                response = "Unknown command. Type '> help' for available commands.\n"

            # Send response
            client_socket.sendall(response.encode('utf-8'))

        except socket.timeout:
             console.print(f"[yellow]Client {addr} timed out.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error handling client {addr}:[/bold red] {e}")
            # Try sending an error message back to the client if possible
            try:
                 error_response = f"Server error handling command: {str(e)}\n"
                 client_socket.sendall(error_response.encode('utf-8'))
            except:
                 pass # Ignore errors during error reporting

        finally:
            client_socket.close()

    def stop(self):
        """Stop the scheduler server"""
        self.stop_event.set()
        if self.server_socket:
            try:
                 # Unblock the accept call by connecting to the server socket itself
                 # This is a common pattern to interrupt a blocking accept()
                 with socket.create_connection((self.host, self.port), timeout=1):
                      pass
            except: # Ignore errors during this unblocking attempt
                 pass
            self.server_socket.close() # Close the server socket
            self.server_socket = None # Set to None after closing

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2) # Wait briefly for the server thread

        if self.server_thread and self.server_thread.is_alive():
             console.print("[yellow]Warning: Server thread did not stop gracefully.[/yellow]")

        console.print("[bold green]Scheduler server stopped[/bold green]")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU Job Scheduler")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: auto-detect using nvidia-smi)")
    parser.add_argument("--log-dir", type=str, default="gpu_scheduler_logs",
                        help="Directory to store logs (default: gpu_scheduler_logs)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host to bind server to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the server (default: 8000)")
    parser.add_argument("--command", type=str, help="Single command to run and exit")
    parser.add_argument("--server-only", action="store_true",
                        help="Run only in server mode without interactive CLI")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect GPUs if not specified
    if args.gpus is None:
        args.gpus = get_available_gpus()
        console.print(f"Auto-detected GPUs: [bold cyan]{args.gpus}[/bold cyan]")

    # Initialize scheduler
    scheduler = GPUJobScheduler(gpu_ids=args.gpus, log_dir=args.log_dir)

    # If a single command is provided, run it and exit
    if args.command:
        try:
            job_id = scheduler.enqueue(args.command)
            console.print(f"Job submitted with ID: [cyan]{job_id}[/cyan]")
            # Wait for all jobs to complete
            console.print("Waiting for job to complete...")
            scheduler.queue.join() # Wait for the queue to be empty
            # Need to wait for the specific job to finish, not just the queue
            while scheduler.job_status.get(job_id) in ["queued", "running"]:
                 time.sleep(1)

            console.print(f"Job {job_id} finished.")
            final_status = scheduler.get_status(job_id)
            console.print(Panel(json.dumps(final_status, indent=2), title=f"Final Status Job {job_id}"))

        except Exception as e:
             console.print(f"[bold red]Error running single command: {e}[/bold red]")
        finally:
             scheduler.shutdown()
        return

    # Start scheduler server if not in command mode
    server = SchedulerServer(scheduler, host=args.host, port=args.port)
    server.start()

    # Determine if running interactive or server-only
    is_interactive = not args.server_only

    # --- Main Loop (Interactive or Server-Only) ---
    try:
        if is_interactive:
            panel_content = (
                "GPU Scheduler running in interactive mode. Enter commands to queue them.\n"
                f"The scheduler is also available via socket server at [blue]{args.host}:{args.port}[/blue]\n\n"
                "Available commands:\n"
                "  [cyan]queue <command> \[on gpu <gpu_id>][/cyan]: Queue a command (optionally on a specific GPU)\n"
                "  [cyan]prioritize <job_id|#local_id> <priority>[/cyan]: Change job priority (lower = higher priority)\n"
                "  [cyan]reserve <count>[/cyan]: Reserve a number of free GPUs\n"
                "  [cyan]release <gpu_id>[/cyan]: Release a reserved GPU back to free state\n"
                "  [cyan]cancel <job_id|#local_id>[/cyan]: Cancel a queued job\n"
                "  [cyan]kill <job_id|#local_id>[/cyan]: Kill a running job\n"
                "  [cyan]requeue <job_id|#local_id>[/cyan]: Kill a running job and add it back to the queue\n"
                "  [cyan]status \[job_id|#local_id][/cyan]: Get status of a job or all jobs\n"
                "  [cyan]log <job_id|#local_id>[/cyan]: Display full stdout and stderr logs for a job\n"
                "  [cyan]show_command <job_id|#local_id>[/cyan]: Display the full command for a job\n" # Added show_command
                "  [cyan]shutdown [force][/cyan]: Shutdown the scheduler (warns if jobs are running unless force is used)\n"
                "  [cyan]help[/cyan]: Show this help message\n"
                "  [cyan]exit[/cyan]: Exit the scheduler\n\n"
                "Keyboard shortcuts:\n"
                "  [magenta]↑/↓[/magenta]: Navigate command history\n"
                "  [magenta]←/→[/magenta]: Move cursor within command\n"
                "  [magenta]Tab[/magenta]: Complete commands\n"
                "  [magenta]Ctrl+C[/magenta]: Cancel current input\n"
                "  [magenta]Ctrl+D[/magenta]: Exit the scheduler"
            )
            console.print(Panel(panel_content, title="GPU Scheduler", border_style="green"))

            # Initialize prompt_toolkit session with history
            command_history = InMemoryHistory()
            prompt_style = Style.from_dict({
                'prompt': 'bold green',
            })

            # Set up command completion
            command_completer = WordCompleter([
                'help', 'exit', 'quit',
                'queue', 'status', 'log', 'prioritize', 'cancel', 'kill', 'requeue', 'show_command', # Added show_command
                'reserve', 'release', 'on', 'gpu', 'shutdown', 'force' # Added force for shutdown
            ], ignore_case=True)

            session = PromptSession(
                history=command_history,
                auto_suggest=AutoSuggestFromHistory(),
                style=prompt_style,
                completer=command_completer,
                complete_while_typing=True # Enable completion while typing
            )
            use_fallback = False # Flag for fallback input

        else: # Server-only mode
             console.print(Panel.fit(
                  f"Server running at [blue]{args.host}:{args.port}[/blue]. Press Ctrl+C to stop.",
                  title="GPU Scheduler Server",
                  border_style="green"
             ))
             console.print("[yellow]Server-only mode. Use Ctrl+C to exit.[/yellow]")
             # Keep main thread alive while server runs in background
             while not server.stop_event.is_set():
                  time.sleep(1)
             # Skip the interactive loop below if in server-only mode
             return


        # --- Interactive Loop ---
        while True:
            try:
                # Get user input (with or without prompt_toolkit based on fallback flag)
                if use_fallback:
                    # Fallback to standard input if prompt_toolkit doesn't work
                    console.print("[bold green]>[/bold green] ", end="")
                    user_input = input().strip()
                else:
                    # Try to use prompt_toolkit
                    try:
                        user_input = session.prompt("\n> ").strip()
                    except Exception as e:
                        console.print(f"[yellow]Warning: prompt_toolkit error ({str(e)}). Falling back to standard input.[/yellow]")
                        use_fallback = True
                        console.print("[bold green]>[/bold green] ", end="")
                        user_input = input().strip()

                if not user_input: # Skip empty input
                     continue

                # --- Command Processing ---
                if user_input.lower() == "exit" or user_input.lower() == "quit":
                    break
                elif user_input.lower() == "help":
                    help_table = Table(title="Available Commands")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description")
                    help_table.add_row("queue <command> \[on gpu <gpu_id>]", "Queue a command (optionally on a specific GPU)")
                    help_table.add_row("prioritize <job_id|#local_id> <priority>", "Change job priority (lower = higher priority)")
                    help_table.add_row("reserve <count>", "Reserve a number of free GPUs")
                    help_table.add_row("release <gpu_id>", "Release a reserved GPU back to free state")
                    help_table.add_row("cancel <job_id|#local_id>", "Cancel a queued job")
                    help_table.add_row("kill <job_id|#local_id>", "Kill a running job")
                    help_table.add_row("requeue <job_id|#local_id>", "Kill a running job and add it back to the queue")
                    help_table.add_row("status \[job_id|#local_id]", "Get status of a job or all jobs")
                    help_table.add_row("log <job_id|#local_id>", "Display full stdout and stderr logs for a job")
                    help_table.add_row("show_command <job_id|#local_id>", "Display the full command for a job") # Added show_command
                    help_table.add_row("shutdown [force]", "Shutdown the scheduler (warns if jobs are running unless force is used)")
                    help_table.add_row("help", "Show this help message")
                    help_table.add_row("exit", "Exit the scheduler")
                    console.print(help_table)

                    # Also show keyboard shortcuts
                    shortcuts_table = Table(title="Keyboard Shortcuts")
                    shortcuts_table.add_column("Key", style="magenta")
                    shortcuts_table.add_column("Action")
                    shortcuts_table.add_row("↑/↓", "Navigate command history")
                    shortcuts_table.add_row("←/→", "Move cursor within command")
                    shortcuts_table.add_row("Tab", "Complete commands")
                    shortcuts_table.add_row("Ctrl+C", "Cancel current input")
                    shortcuts_table.add_row("Ctrl+D", "Exit the scheduler")
                    console.print(shortcuts_table)

                elif user_input.lower().startswith("status"):
                    parts = user_input.split(maxsplit=1)
                    job_identifier = parts[1] if len(parts) > 1 else None

                    try:
                        status_data = scheduler.get_status(job_identifier)

                        if job_identifier:
                            # Single job status
                            if "error" in status_data:
                                console.print(f"[bold red]{status_data['error']}[/bold red]")
                            else:
                                status_color = "green" if status_data.get("status") == "completed" else (
                                    "red" if status_data.get("status") in ["failed", "killed"] else "yellow")

                                details = [
                                     f"Job ID: [cyan]{status_data['job_id']}[/cyan] (#{status_data['local_id']})",
                                     f"Priority: {status_data.get('priority', 'N/A')}",
                                     f"Command: [dim]{status_data.get('command', 'N/A')[:100]}{'...' if len(status_data.get('command', 'N/A')) > 100 else ''}[/dim]", # Truncate command in status
                                     f"Preferred/Running GPU: {status_data.get('preferred_gpu', 'Any')}",
                                     f"Status: [bold {status_color}]{status_data['status']}[/bold {status_color}]"
                                ]
                                if "return_code" in status_data and status_data['return_code'] is not None:
                                     details.append(f"Return Code: {status_data['return_code']}")
                                if "stdout_preview" in status_data:
                                     details.append(f"Stdout Preview: {status_data['stdout_preview']}")
                                if "stderr_preview" in status_data:
                                     details.append(f"Stderr Preview: {status_data['stderr_preview']}")


                                status_panel = Panel("\n".join(details),
                                                     title="Job Status",
                                                     border_style=status_color)
                                console.print(status_panel)
                        else:
                            # All jobs status
                            gpu_table = Table(title="GPU Status")
                            gpu_table.add_column("GPU ID", style="cyan")
                            gpu_table.add_column("Status")

                            for gpu_id, status in status_data["gpu_status"].items():
                                status_color = "green" if status == "free" else ("magenta" if status == "reserved" else "yellow")
                                gpu_table.add_row(str(gpu_id), f"[{status_color}]{status}[/{status_color}]")
                            console.print(gpu_table)

                            console.print(f"Jobs in queue: [yellow]{status_data['queue_size']}[/yellow]")
                            # Display queued jobs if any
                            if status_data['queued_jobs']:
                                 queue_table = Table(title="Queued Jobs (Top 5)")
                                 queue_table.add_column("Priority", style="magenta")
                                 queue_table.add_column("Local ID", style="cyan")
                                 queue_table.add_column("Job ID", style="dim")
                                 queue_table.add_column("Command", style="dim")
                                 for item in sorted(status_data['queued_jobs'], key=lambda x: (x['priority'], x['local_id']))[:5]:
                                      cmd = item['command']
                                      if len(cmd) > 40: cmd = cmd[:37] + "..."
                                      queue_table.add_row(str(item['priority']), f"#{item['local_id']}", item['job_id'], cmd)
                                 console.print(queue_table)


                            if status_data["all_jobs_status"]:
                                jobs_table = Table(title="Tracked Jobs (Excluding Cancelled)")
                                jobs_table.add_column("Local ID", style="cyan")
                                jobs_table.add_column("Job ID", style="dim")
                                jobs_table.add_column("Status")
                                jobs_table.add_column("Priority")
                                jobs_table.add_column("GPU")
                                jobs_table.add_column("Command", style="dim")

                                # Sort jobs by local ID for consistent display
                                sorted_job_ids = sorted(status_data["all_jobs_status"].keys(),
                                                        key=lambda jid: status_data["job_local_ids"].get(jid, -1))

                                for job_id in sorted_job_ids:
                                    status = status_data["all_jobs_status"][job_id]
                                    status_color = "green" if status == "completed" else (
                                        "red" if status in ["failed", "killed"] else "yellow")
                                    local_id = status_data["job_local_ids"].get(job_id, "?")
                                    priority = status_data["job_priorities"].get(job_id, "N/A")
                                    command = status_data["job_commands"].get(job_id, "Unknown command")
                                    gpu_pref = status_data["job_gpus"].get(job_id)
                                    gpu_display = str(gpu_pref) if gpu_pref is not None else "Any"

                                    # Truncate command if too long
                                    if len(command) > 40:
                                        command = command[:37] + "..."
                                    jobs_table.add_row(
                                        f"#{local_id}",
                                        job_id,
                                        f"[{status_color}]{status}[/{status_color}]",
                                        str(priority),
                                        gpu_display,
                                        command
                                    )
                                console.print(jobs_table)
                            else:
                                 console.print("No active or completed jobs found.")
                    except Exception as e:
                         console.print(f"[bold red]Error getting status: {str(e)}[/bold red]")


                elif user_input.lower().startswith("log "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: log <job_id|#local_id>")
                    else:
                        job_identifier = parts[1]
                        try:
                            log_data = scheduler.get_log(job_identifier)

                            if "error" in log_data:
                                console.print(f"[bold red]Error:[/bold red] {log_data['error']}")
                            else:
                                status_color = "green" if log_data.get("status") == "completed" else (
                                    "red" if log_data.get("status") in ["failed", "killed"] else "yellow")

                                # Create a panel for job details
                                details_content = [
                                     f"Job ID: [cyan]{log_data['job_id']}[/cyan] (#{log_data['local_id']})",
                                     f"Status: [bold {status_color}]{log_data['status']}[/bold {status_color}]"
                                ]

                                if "message" in log_data:
                                    details_content.append(f"{log_data['message']}")

                                if "command" in log_data:
                                    details_content.append(f"Command: [dim]{log_data['command']}[/dim]")

                                if "return_code" in log_data and log_data['return_code'] is not None:
                                    details_content.append(f"Return Code: {log_data['return_code']}")

                                details_panel = Panel("\n".join(details_content),
                                                      title="Job Details",
                                                      border_style=status_color)
                                console.print(details_panel)

                                # Show stdout in a separate panel if available
                                if "stdout" in log_data and log_data["stdout"].strip():
                                    stdout_panel = Panel(log_data["stdout"],
                                                         title="STDOUT",
                                                         border_style="green",
                                                         expand=False) # Prevent excessive expansion
                                    console.print(stdout_panel)
                                elif "stdout" in log_data:
                                     console.print("[dim]No output in stdout.[/dim]")

                                # Show stderr in a separate panel if available and not empty
                                if "stderr" in log_data and log_data["stderr"].strip():
                                    stderr_panel = Panel(log_data["stderr"],
                                                         title="STDERR",
                                                         border_style="red",
                                                         expand=False)
                                    console.print(stderr_panel)
                                elif "stderr" in log_data:
                                     console.print("[dim]No output in stderr.[/dim]")
                        except Exception as e:
                             console.print(f"[bold red]Error getting logs: {str(e)}[/bold red]")

                # --- NEW SHOW_COMMAND (CLI) ---
                elif user_input.lower().startswith("show_command "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: show_command <job_id|#local_id>")
                    else:
                        job_identifier = parts[1]
                        job_id = None
                        local_id = None
                        error_msg = None
                        try:
                            # Resolve job identifier
                            if job_identifier.startswith('#'):
                                try:
                                    local_id = int(job_identifier[1:])
                                    if local_id in scheduler.local_to_uuid:
                                        job_id = scheduler.local_to_uuid[local_id]
                                    else:
                                        error_msg = f"Local job ID {local_id} not found"
                                except ValueError:
                                    error_msg = f"Invalid local job ID format: {job_identifier}"
                            else:
                                job_id = job_identifier
                                if job_id not in scheduler.job_status:
                                    error_msg = f"Job ID {job_id} not found"
                                else:
                                    local_id = scheduler.uuid_to_local.get(job_id)

                            # If job_id was found and no error yet
                            if job_id and not error_msg:
                                command = scheduler.job_commands.get(job_id)
                                if command is not None:
                                    cmd_panel = Panel(command,
                                                      title=f"Full Command for Job #{local_id} ({job_id})",
                                                      border_style="blue",
                                                      expand=False)
                                    console.print(cmd_panel)
                                else:
                                    # Should not happen if job_id exists in job_status
                                    error_msg = f"Command not found for job #{local_id} ({job_id})"
                        except Exception as e:
                             error_msg = f"Error showing command: {str(e)}"

                        if error_msg:
                            console.print(f"[bold red]Error:[/bold red] {error_msg}")
                # --- END SHOW_COMMAND (CLI) ---

                elif user_input.lower().startswith("queue "):
                    command_part = user_input[6:].strip()
                    command = command_part
                    gpu_id = None

                    # Check for GPU specification at the end: "command on gpu X"
                    gpu_spec_found = False
                    for pattern in [" on gpu ", " on GPU ", " --gpu ", " --GPU "]:
                        if pattern in command_part:
                             parts = command_part.rsplit(pattern, 1)
                             if len(parts) == 2 and parts[1].strip().isdigit():
                                  command = parts[0].strip()
                                  try:
                                       gpu_id = int(parts[1].strip())
                                       gpu_spec_found = True
                                       break
                                  except ValueError:
                                       pass

                    if not command:
                         console.print("[bold red]Error:[/bold red] No command provided to queue.")
                         continue

                    try:
                        job_id = scheduler.enqueue(command, gpu_id=gpu_id)
                        # Message is printed by enqueue method
                    except ValueError as e:
                        console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    except Exception as e:
                         console.print(f"[bold red]Error queueing job: {str(e)}[/bold red]")


                elif user_input.lower().startswith("prioritize "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: prioritize <job_id|#local_id> [priority]")
                    else:
                        job_identifier = parts[1]
                        try:
                            priority = int(parts[2]) if len(parts) > 2 else 10 # Default priority 10
                            result = scheduler.prioritize(job_identifier, priority)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            # Success message printed by prioritize method
                        except ValueError:
                            console.print("[bold red]Error:[/bold red] Priority must be an integer")
                        except Exception as e:
                             console.print(f"[bold red]Error prioritizing job: {str(e)}[/bold red]")


                elif user_input.lower().startswith("reserve "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: reserve <count>")
                    else:
                        try:
                            count = int(parts[1])
                            result = scheduler.reserve_gpus(count)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            else:
                                console.print(f"[bold green]Success:[/bold green] Reserved GPUs: [magenta]{result['reserved_gpus']}[/magenta]")
                        except ValueError:
                            console.print("[bold red]Error:[/bold red] Count must be an integer")
                        except Exception as e:
                             console.print(f"[bold red]Error reserving GPUs: {str(e)}[/bold red]")


                elif user_input.lower().startswith("release "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: release <gpu_id>")
                    else:
                        try:
                            gpu_id = int(parts[1])
                            result = scheduler.release_gpu(gpu_id)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            # Success message printed by release_gpu method
                        except ValueError:
                            console.print("[bold red]Error:[/bold red] GPU ID must be an integer")
                        except Exception as e:
                             console.print(f"[bold red]Error releasing GPU: {str(e)}[/bold red]")


                elif user_input.lower().startswith("cancel "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: cancel <job_id|#local_id>")
                    else:
                        job_identifier = parts[1]
                        try:
                            result = scheduler.cancel_job(job_identifier)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            # Success message printed by cancel_job method
                        except Exception as e:
                             console.print(f"[bold red]Error cancelling job: {str(e)}[/bold red]")


                elif user_input.lower().startswith("kill "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: kill <job_id|#local_id>")
                    else:
                        job_identifier = parts[1]
                        try:
                            result = scheduler.kill_job(job_identifier)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            # Success message printed by kill_job method
                        except Exception as e:
                             console.print(f"[bold red]Error killing job: {str(e)}[/bold red]")

                # --- NEW REQUEUE COMMAND (CLI) ---
                elif user_input.lower().startswith("requeue "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: requeue <job_id|#local_id>")
                    else:
                        job_identifier = parts[1]
                        try:
                            result = scheduler.requeue_job(job_identifier)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            # Success messages printed by requeue_job method
                        except Exception as e:
                             console.print(f"[bold red]Error requeueing job: {str(e)}[/bold red]")
                # --- END REQUEUE COMMAND (CLI) ---

                elif user_input.lower().startswith("shutdown"):
                    # Check if there's a force flag
                    force = False
                    if len(user_input.split()) > 1 and user_input.split()[1].lower() == "force":
                        force = True

                    try:
                        # Get running jobs
                        status_data = scheduler.get_status(include_cancelled=True) # Include all for check
                        running_jobs = [(jid, status_data["job_local_ids"].get(jid))
                                        for jid, status in status_data["all_jobs_status"].items()
                                        if status == "running"]

                        if running_jobs and not force:
                            console.print("[bold yellow]WARNING:[/bold yellow] There are running jobs that will be killed:")
                            for job_id, local_id in running_jobs:
                                console.print(f"  #{local_id} [cyan]{job_id}[/cyan]")
                            console.print("To confirm shutdown and kill all jobs, use: [bold]shutdown force[/bold]")
                        else:
                            if running_jobs:
                                # Kill all running jobs
                                with Progress(
                                    SpinnerColumn(),
                                    TextColumn("[bold green]Killing running jobs...[/bold green]")
                                ) as progress:
                                    task = progress.add_task("", total=None)
                                    kill_result = scheduler.kill_all_jobs()
                                    progress.update(task, completed=True)

                                console.print(f"Killed [bold red]{len(kill_result['killed_jobs'])}[/bold red] running jobs.")
                                if kill_result['killed_jobs']:
                                     for job_id, local_id in kill_result['killed_jobs']:
                                          console.print(f"  Killed job #{local_id} [cyan]{job_id}[/cyan]")

                                if kill_result['failed_jobs']:
                                    console.print(f"Failed to kill {len(kill_result['failed_jobs'])} jobs:")
                                    for job_id, local_id, error in kill_result['failed_jobs']:
                                         console.print(f"  Job #{local_id} [cyan]{job_id}[/cyan]: {error}")

                            console.print("\n[yellow]Shutting down...[/yellow]")
                            break # Exit the interactive loop
                    except Exception as e:
                         console.print(f"[bold red]Error during shutdown: {str(e)}[/bold red]")

                else: # Only show error for non-empty, unrecognized input
                     console.print("[bold red]Unknown command.[/bold red] Type [cyan]help[/cyan] for available commands.")

            except KeyboardInterrupt:
                # Handle Ctrl+C during input (allow canceling current input)
                console.print("\nInput cancelled. Enter command or 'exit'.")
                continue
            except EOFError:
                # Handle Ctrl+D (exit)
                console.print("\nExiting...")
                break

    except KeyboardInterrupt: # Handle Ctrl+C outside the prompt loop (e.g., while command is running)
        console.print("\n[yellow]Interrupt received. Shutting down...[/yellow]")

    finally:
        # Shutdown the server and scheduler
        console.print("Final shutdown sequence initiated...")
        server.stop()
        scheduler.shutdown()
        console.print("Scheduler finished.")


if __name__ == "__main__":
    main()