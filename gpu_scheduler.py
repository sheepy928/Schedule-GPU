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
                if self.gpu_status[gpu_id] == "occupied by other":
                    # Check if GPU is now available
                    new_status = check_gpu_status(gpu_id)
                    if new_status == "free":
                        with self.gpu_locks[gpu_id]:
                            # Only update if it was previously occupied by other
                            if self.gpu_status[gpu_id] == "occupied by other":
                                self.gpu_status[gpu_id] = "free"
                                console.print(f"GPU {gpu_id} is now [green]free[/green]")
            
            # Sleep before next check
            time.sleep(self.status_check_interval)
                    
    def enqueue(self, command: str, priority: int = 100) -> str:
        """
        Add a command to the queue.
        
        Args:
            command: The command to run
            priority: Priority level (lower values = higher priority, default: 100)
            
        Returns:
            job_id: Unique ID for the job
        """
        job_id = str(uuid.uuid4())
        local_id = self.next_local_id
        self.next_local_id += 1
        
        # Store both mappings
        self.local_to_uuid[local_id] = job_id
        self.uuid_to_local[job_id] = local_id
        
        self.job_status[job_id] = "queued"
        self.job_commands[job_id] = command  # Store the command
        self.job_priorities[job_id] = priority
        
        # Add to priority queue: (priority, local_id, job_id, command)
        # Local ID added to tiebreak equal priorities based on FIFO order
        self.queue.put((priority, local_id, job_id, command))
        
        console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] [bold]queued[/bold] (priority: {priority}): [dim]{command}[/dim]")
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
    
    def _gpu_worker(self, gpu_id: int):
        """
        Worker thread that executes jobs on a specific GPU.
        
        Args:
            gpu_id: The GPU ID to use
        """
        while not self.stop_event.is_set():
            job_acquired = False
            try:
                # Skip if GPU is marked as occupied by other processes or already busy with our job
                if self.gpu_status[gpu_id] in ["occupied by other", "busy"]:
                    time.sleep(1)
                    continue
                
                # Try to get a job from the queue with a timeout
                try:
                    priority, local_id, job_id, command = self.queue.get(timeout=1)
                    job_acquired = True  # Flag that we got a job from the queue
                except queue.Empty:
                    continue
                
                # Lock this GPU
                with self.gpu_locks[gpu_id]:
                    # Double check GPU is still free
                    current_status = check_gpu_status(gpu_id)
                    if current_status != "free":
                        # Put job back in queue and skip
                        self.queue.put((priority, local_id, job_id, command))
                        time.sleep(1)
                        # Set job_acquired to False since we're putting it back
                        job_acquired = False
                        continue
                    
                    # Mark GPU as busy for our job
                    self.gpu_status[gpu_id] = "busy"
                    self.job_status[job_id] = "running"
                    
                    # Set up environment variables to make this GPU visible
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    
                    # Log files
                    stdout_log = os.path.join(self.log_dir, f"{job_id}_stdout.log")
                    stderr_log = os.path.join(self.log_dir, f"{job_id}_stderr.log")
                    
                    console.print(f"Running job #{local_id} [cyan]{job_id}[/cyan] on GPU [bold green]{gpu_id}[/bold green]: [dim]{command}[/dim]")
                    
                    # Execute the command
                    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
                        process = subprocess.Popen(
                            command,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            shell=True,
                            env=env
                        )
                        
                        # Wait for the process to complete
                        return_code = process.wait()
                    
                    # Read the logs
                    with open(stdout_log, 'r') as f:
                        stdout = f.read()
                    with open(stderr_log, 'r') as f:
                        stderr = f.read()
                    
                    # Update job status
                    self.job_status[job_id] = "completed" if return_code == 0 else "failed"
                    self.job_results[job_id] = (return_code, stdout, stderr)
                    
                    status_color = "green" if return_code == 0 else "red"
                    console.print(f"Job #{local_id} [cyan]{job_id}[/cyan] [bold {status_color}]{self.job_status[job_id]}[/bold {status_color}] with return code {return_code}")
                    
                    # Check GPU status after job completes
                    current_status = check_gpu_status(gpu_id)
                    self.gpu_status[gpu_id] = current_status
                    status_color = "green" if current_status == "free" else "yellow"
                    console.print(f"GPU {gpu_id} status after job: [{status_color}]{current_status}[/{status_color}]")
                
                    # Job was successfully processed, mark it as done in the queue
                    if job_acquired:
                        self.queue.task_done()
                        job_acquired = False
                
            except Exception as e:
                console.print(f"[bold red]Error[/bold red] in GPU worker {gpu_id}: {e}")
                # Update GPU status in case of error
                with self.gpu_locks[gpu_id]:
                    current_status = check_gpu_status(gpu_id)
                    self.gpu_status[gpu_id] = current_status
                    
            finally:
                # If we got a job from the queue but didn't process it due to an exception,
                # make sure we mark it as done to avoid queue getting stuck
                if job_acquired:
                    self.queue.task_done()
                    job_acquired = False
    
    def get_status(self, job_id: Optional[str] = None) -> Dict:
        """
        Get the status of a specific job or all jobs.
        
        Args:
            job_id: Optional job ID to get status for
            
        Returns:
            Dictionary with job status information
        """
        # First refresh GPU statuses
        for gpu_id in self.gpu_ids:
            if self.gpu_status[gpu_id] != "busy":  # Don't check GPUs that are running jobs
                current_status = check_gpu_status(gpu_id)
                self.gpu_status[gpu_id] = current_status
        
        if job_id:
            # Check if job_id is actually a local ID with # prefix
            if job_id.startswith('#'):
                try:
                    local_id = int(job_id[1:])
                    if local_id in self.local_to_uuid:
                        job_id = self.local_to_uuid[local_id]
                    else:
                        return {"error": f"Local job ID {local_id} not found"}
                except ValueError:
                    return {"error": f"Invalid local job ID format: {job_id}"}
            
            # Now job_id should be a UUID
            if job_id not in self.job_status:
                return {"error": f"Job ID {job_id} not found"}
            
            local_id = self.uuid_to_local.get(job_id)
            result = {
                "job_id": job_id,
                "local_id": local_id,
                "status": self.job_status[job_id],
                "priority": self.job_priorities.get(job_id, "N/A"),
                "command": self.job_commands.get(job_id, "Unknown command")
            }
            
            if self.job_status[job_id] in ["completed", "failed"] and job_id in self.job_results:
                return_code, stdout, stderr = self.job_results[job_id]
                result["return_code"] = return_code
                # Include truncated stdout/stderr for readability
                result["stdout"] = stdout[:500] + "..." if len(stdout) > 500 else stdout
                result["stderr"] = stderr[:500] + "..." if len(stderr) > 500 else stderr
            
            return result
        else:
            # Get a count of jobs by status
            status_counts = {}
            for status in self.job_status.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "queue_size": self.queue.qsize(),
                "gpu_status": self.gpu_status,
                "jobs": self.job_status,
                "job_priorities": self.job_priorities,
                "status_counts": status_counts,
                "commands": self.job_commands,
                "local_ids": self.uuid_to_local
            }
    
    def get_log(self, job_identifier: str) -> Dict[str, Any]:
        """
        Get the full logs (stdout and stderr) for a specific job.
        
        Args:
            job_identifier: Job identifier (either UUID or local ID with # prefix)
            
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
        
        # Check if we have results for this job
        if job_id not in self.job_results:
            if self.job_status[job_id] == "queued":
                return {
                    "job_id": job_id,
                    "local_id": local_id,
                    "status": "queued",
                    "message": "Job is queued. No logs available yet."
                }
            elif self.job_status[job_id] == "running":
                # Try to read in-progress logs from file
                stdout_log = os.path.join(self.log_dir, f"{job_id}_stdout.log")
                stderr_log = os.path.join(self.log_dir, f"{job_id}_stderr.log")
                
                stdout = ""
                stderr = ""
                
                if os.path.exists(stdout_log):
                    try:
                        with open(stdout_log, 'r') as f:
                            stdout = f.read()
                    except Exception as e:
                        stdout = f"Error reading stdout: {str(e)}"
                
                if os.path.exists(stderr_log):
                    try:
                        with open(stderr_log, 'r') as f:
                            stderr = f.read()
                    except Exception as e:
                        stderr = f"Error reading stderr: {str(e)}"
                
                return {
                    "job_id": job_id,
                    "local_id": local_id,
                    "status": "running",
                    "message": "Job is running. Partial logs follow:",
                    "stdout": stdout,
                    "stderr": stderr
                }
            else:
                return {
                    "job_id": job_id,
                    "local_id": local_id,
                    "status": self.job_status[job_id],
                    "message": "Job completed but no logs found."
                }
        
        # Get full logs
        return_code, stdout, stderr = self.job_results[job_id]
        
        return {
            "job_id": job_id,
            "local_id": local_id,
            "status": self.job_status[job_id],
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "command": self.job_commands.get(job_id, "Unknown command")
        }
    
    def shutdown(self):
        """Shutdown the scheduler and wait for all workers to complete"""
        console.print("[yellow]Shutting down GPU scheduler...[/yellow]")
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        self.monitor_thread.join()
        console.print("[bold green]GPU scheduler shutdown complete[/bold green]")


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
    
    def start(self):
        """Start the scheduler server"""
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
    
    def _server_loop(self):
        """Server loop to accept connections and handle requests"""
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
        try:
            # Set a timeout for client operations
            client_socket.settimeout(5)
            
            # Receive data
            data = client_socket.recv(4096).decode('utf-8').strip()
            console.print(f"Received command from [blue]{addr[0]}:{addr[1]}[/blue]: [dim]{data}[/dim]")
            
            # Process command
            if data.startswith("> queue "):
                command = data[8:].strip()
                job_id = self.scheduler.enqueue(command)
                local_id = self.scheduler.uuid_to_local.get(job_id)
                response = f"Job submitted with ID: {job_id} (#{local_id})\n"
            
            elif data.startswith("> prioritize "):
                parts = data.split(maxsplit=3)
                if len(parts) < 3:
                    response = "Error: Usage: > prioritize <job_id> <priority>\n"
                else:
                    job_id = parts[2]
                    try:
                        priority = int(parts[3]) if len(parts) > 3 else 10
                        result = self.scheduler.prioritize(job_id, priority)
                        if "error" in result:
                            response = f"Error: {result['error']}\n"
                        else:
                            response = f"Job prioritized: #{result['local_id']} {result['job_id']} (priority: {result['new_priority']})\n"
                    except ValueError:
                        response = "Error: Priority must be an integer\n"
            
            elif data.startswith("> log "):
                parts = data.split(maxsplit=2)
                if len(parts) < 2:
                    response = "Error: Usage: > log <job_id|#local_id>\n"
                else:
                    job_id = parts[1]
                    log_data = self.scheduler.get_log(job_id)
                    
                    if "error" in log_data:
                        response = f"Error: {log_data['error']}\n"
                    else:
                        response = f"=== Job Details ===\n"
                        response += f"Job ID: {log_data['job_id']} (#{log_data['local_id']})\n"
                        response += f"Status: {log_data['status']}\n"
                        
                        if "message" in log_data:
                            response += f"{log_data['message']}\n\n"
                        
                        if "command" in log_data:
                            response += f"Command: {log_data['command']}\n"
                        
                        if "return_code" in log_data:
                            response += f"Return Code: {log_data['return_code']}\n"
                        
                        if "stdout" in log_data:
                            response += f"\n=== STDOUT ===\n{log_data['stdout']}\n"
                        
                        if "stderr" in log_data:
                            response += f"\n=== STDERR ===\n{log_data['stderr']}\n"
            
            elif data.startswith("> status"):
                parts = data.split(maxsplit=2)
                job_id = parts[2] if len(parts) > 2 else None
                status = self.scheduler.get_status(job_id)
                response = json.dumps(status, indent=2) + "\n"
            
            elif data.startswith("> help"):
                response = "Available commands:\n"
                response += "  > queue <command>: Queue a command\n"
                response += "  > prioritize <job_id|#local_id> [priority]: Change job priority (lower = higher priority)\n"
                response += "  > status [job_id|#local_id]: Get status of a job or all jobs\n"
                response += "  > log <job_id|#local_id>: Display full stdout and stderr logs for a job\n"
                response += "  > help: Show this help message\n"
            
            else:
                response = "Unknown command. Type '> help' for available commands.\n"
            
            # Send response
            client_socket.sendall(response.encode('utf-8'))
        
        except Exception as e:
            console.print(f"[bold red]Error handling client {addr}:[/bold red] {e}")
        
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop the scheduler server"""
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()
        if self.server_thread:
            self.server_thread.join()
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
        job_id = scheduler.enqueue(args.command)
        console.print(f"Job submitted with ID: [cyan]{job_id}[/cyan]")
        # Wait for all jobs to complete
        scheduler.queue.join()
        console.print(f"Final status: {scheduler.get_status(job_id)}")
        scheduler.shutdown()
        return
    
    # Start scheduler server if not in command mode
    server = SchedulerServer(scheduler, host=args.host, port=args.port)
    server.start()
    
    # If in server-only mode, just wait for termination
    if args.server_only:
        try:
            console.print(Panel.fit(
                f"Server running at [blue]{args.host}:{args.port}[/blue]. Press Ctrl+C to stop.",
                title="GPU Scheduler Server",
                border_style="green"
            ))
            
            # Even in server-only mode, provide a basic interactive CLI
            console.print("[yellow]Server-only mode is active, but you can still enter commands.[/yellow]")
            console.print("[yellow]Type 'exit' to quit, or 'help' for available commands.[/yellow]")
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    
                    if user_input.lower() == "exit" or user_input.lower() == "quit":
                        break
                    elif user_input.lower() == "help":
                        console.print("Available commands in server-only mode:")
                        console.print("  [cyan]status[/cyan]: Show GPU and job status")
                        console.print("  [cyan]exit[/cyan]: Exit the scheduler")
                    elif user_input.lower() == "status":
                        status_data = scheduler.get_status()
                        
                        # Display GPU status
                        status_table = Table(title="GPU Status")
                        status_table.add_column("GPU ID", style="cyan")
                        status_table.add_column("Status")
                        
                        for gpu_id, status in status_data["gpu_status"].items():
                            status_color = "green" if status == "free" else "yellow"
                            status_table.add_row(str(gpu_id), f"[{status_color}]{status}[/{status_color}]")
                        
                        console.print(status_table)
                        
                        # Display queue and job counts
                        if status_data["queue_size"] > 0:
                            console.print(f"Jobs in queue: [yellow]{status_data['queue_size']}[/yellow]")
                            
                        status_counts = status_data.get("status_counts", {})
                        if status_counts:
                            console.print("Job status counts:")
                            for status, count in status_counts.items():
                                status_color = "green" if status == "completed" else ("red" if status == "failed" else "yellow")
                                console.print(f"  [{status_color}]{status}[/{status_color}]: {count}")
                    elif user_input:
                        console.print("[bold red]Unknown command.[/bold red] Type [cyan]help[/cyan] for available commands.")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    
            console.print("\n[yellow]Shutting down...[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            server.stop()
            scheduler.shutdown()
        return
    
    # Interactive mode
    try:
        panel_content = (
            "GPU Scheduler running in interactive mode. Enter commands to queue them.\n"
            f"The scheduler is also available via socket server at [blue]{args.host}:{args.port}[/blue]\n\n"
            "Available commands:\n"
            "  [cyan]queue <command>[/cyan]: Queue a command\n"
            "  [cyan]prioritize <job_id|#local_id> [priority][/cyan]: Change job priority (lower = higher priority)\n"
            "  [cyan]status [job_id|#local_id][/cyan]: Get status of a job or all jobs\n"
            "  [cyan]log <job_id|#local_id>[/cyan]: Display full stdout and stderr logs for a job\n"
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
            'queue', 'status', 'log', 'prioritize',
        ], ignore_case=True)
        
        session = PromptSession(
            history=command_history,
            auto_suggest=AutoSuggestFromHistory(),
            style=prompt_style,
            completer=command_completer
        )
        
        # Store command history for fallback mode
        fallback_history = []
        use_fallback = False
        
        while True:
            try:
                # Get user input (with or without prompt_toolkit based on fallback flag)
                if use_fallback:
                    # Fallback to standard input if prompt_toolkit doesn't work
                    console.print("[bold green]>[/bold green] ", end="")
                    user_input = input().strip()
                    
                    # Simple history navigation with standard input
                    if user_input == "!!" and fallback_history:
                        user_input = fallback_history[-1]
                        console.print(f"[dim]Executing: {user_input}[/dim]")
                else:
                    # Try to use prompt_toolkit with a simple prompt and message
                    try:
                        # Using a simple ">" prompt that works reliably
                        user_input = session.prompt("\n> ", complete_while_typing=True).strip()
                    except Exception as e:
                        console.print(f"[yellow]Warning: Falling back to standard input due to error: {str(e)}[/yellow]")
                        use_fallback = True
                        console.print("[bold green]>[/bold green] ", end="")
                        user_input = input().strip()
                
                # Store command in history (if not empty)
                if user_input and user_input not in fallback_history:
                    fallback_history.append(user_input)
                
                if user_input.lower() == "exit" or user_input.lower() == "quit":
                    break
                elif user_input.lower() == "help":
                    help_table = Table(title="Available Commands")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description")
                    help_table.add_row("queue <command>", "Queue a command")
                    help_table.add_row("prioritize <job_id|#local_id> [priority]", "Change job priority (lower = higher priority)")
                    help_table.add_row("status [job_id|#local_id]", "Get status of a job or all jobs")
                    help_table.add_row("log <job_id|#local_id>", "Display full stdout and stderr logs for a job")
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
                    job_id = parts[1] if len(parts) > 1 else None
                    
                    status_data = scheduler.get_status(job_id)
                    
                    if job_id:
                        # Single job status
                        if "error" in status_data:
                            console.print(f"[bold red]{status_data['error']}[/bold red]")
                        else:
                            status_color = "green" if status_data.get("status") == "completed" else (
                                "red" if status_data.get("status") == "failed" else "yellow")
                            
                            status_panel = Panel(
                                f"Job ID: [cyan]{status_data['job_id']}[/cyan] (#{status_data['local_id']})\n"
                                f"Priority: {status_data.get('priority', 'N/A')}\n"
                                f"Command: [dim]{status_data.get('command', 'N/A')}[/dim]\n"
                                f"Status: [bold {status_color}]{status_data['status']}[/bold {status_color}]\n"
                                + (f"Return Code: {status_data.get('return_code', 'N/A')}\n" if "return_code" in status_data else "")
                                + (f"Stdout: {status_data.get('stdout', 'N/A')}\n" if "stdout" in status_data else "")
                                + (f"Stderr: {status_data.get('stderr', 'N/A')}" if "stderr" in status_data else ""),
                                title="Job Status",
                                border_style=status_color
                            )
                            console.print(status_panel)
                    else:
                        # All jobs status
                        status_table = Table(title="GPU Status")
                        status_table.add_column("GPU ID", style="cyan")
                        status_table.add_column("Status")
                        
                        for gpu_id, status in status_data["gpu_status"].items():
                            status_color = "green" if status == "free" else "yellow"
                            status_table.add_row(str(gpu_id), f"[{status_color}]{status}[/{status_color}]")
                        
                        console.print(status_table)
                        
                        if status_data["queue_size"] > 0:
                            console.print(f"Jobs in queue: [yellow]{status_data['queue_size']}[/yellow]")
                        
                        if status_data["jobs"]:
                            jobs_table = Table(title="Jobs")
                            jobs_table.add_column("Local ID", style="cyan")
                            jobs_table.add_column("Job ID", style="dim")
                            jobs_table.add_column("Status")
                            jobs_table.add_column("Priority")
                            jobs_table.add_column("Command", style="dim")
                            
                            for job_id, status in status_data["jobs"].items():
                                status_color = "green" if status == "completed" else (
                                    "red" if status == "failed" else "yellow")
                                local_id = status_data["local_ids"].get(job_id, "?")
                                priority = status_data["job_priorities"].get(job_id, "N/A")
                                command = status_data["commands"].get(job_id, "Unknown command")
                                # Truncate command if too long
                                if len(command) > 40:
                                    command = command[:37] + "..."
                                jobs_table.add_row(
                                    f"#{local_id}", 
                                    job_id, 
                                    f"[{status_color}]{status}[/{status_color}]",
                                    str(priority),
                                    command
                                )
                            
                            console.print(jobs_table)
                elif user_input.lower().startswith("log "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: log <job_id|#local_id>")
                    else:
                        job_id = parts[1]
                        log_data = scheduler.get_log(job_id)
                        
                        if "error" in log_data:
                            console.print(f"[bold red]Error:[/bold red] {log_data['error']}")
                        else:
                            status_color = "green" if log_data.get("status") == "completed" else (
                                "red" if log_data.get("status") == "failed" else "yellow")
                            
                            # Create a panel for job details
                            details_content = [
                                f"Job ID: [cyan]{log_data['job_id']}[/cyan] (#{log_data['local_id']})",
                                f"Status: [bold {status_color}]{log_data['status']}[/bold {status_color}]"
                            ]
                            
                            if "message" in log_data:
                                details_content.append(f"{log_data['message']}")
                            
                            if "command" in log_data:
                                details_content.append(f"Command: [dim]{log_data['command']}[/dim]")
                            
                            if "return_code" in log_data:
                                details_content.append(f"Return Code: {log_data['return_code']}")
                            
                            details_panel = Panel("\n".join(details_content), 
                                                title="Job Details", 
                                                border_style=status_color)
                            console.print(details_panel)
                            
                            # Show stdout in a separate panel if available
                            if "stdout" in log_data and log_data["stdout"].strip():
                                stdout_panel = Panel(log_data["stdout"], 
                                                    title="STDOUT", 
                                                    border_style="green")
                                console.print(stdout_panel)
                            elif "stdout" in log_data:
                                console.print("[dim]No output in stdout.[/dim]")
                            
                            # Show stderr in a separate panel if available and not empty
                            if "stderr" in log_data and log_data["stderr"].strip():
                                stderr_panel = Panel(log_data["stderr"], 
                                                    title="STDERR", 
                                                    border_style="red")
                                console.print(stderr_panel)
                            elif "stderr" in log_data:
                                console.print("[dim]No output in stderr.[/dim]")
                elif user_input.lower().startswith("queue "):
                    command = user_input[6:].strip()
                    job_id = scheduler.enqueue(command)
                    local_id = scheduler.uuid_to_local.get(job_id)
                    console.print(f"Job submitted with ID: [cyan]{job_id}[/cyan] (#{local_id})")
                elif user_input.lower().startswith("prioritize "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        console.print("[bold red]Error:[/bold red] Usage: prioritize <job_id|#local_id> [priority]")
                    else:
                        job_id = parts[1]
                        try:
                            priority = int(parts[2]) if len(parts) > 2 else 10
                            result = scheduler.prioritize(job_id, priority)
                            if "error" in result:
                                console.print(f"[bold red]Error:[/bold red] {result['error']}")
                            else:
                                console.print(f"Job prioritized: #{result['local_id']} [cyan]{result['job_id']}[/cyan] (priority: {result['new_priority']})")
                        except ValueError:
                            console.print("[bold red]Error:[/bold red] Priority must be an integer")
                elif user_input:  # Only show error for non-empty input
                    console.print("[bold red]Unknown command.[/bold red] Type [cyan]help[/cyan] for available commands.")
            except KeyboardInterrupt:
                # Handle Ctrl+C during input (allow canceling current input)
                continue
            except EOFError:
                # Handle Ctrl+D (exit)
                break
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    
    finally:
        # Shutdown the server and scheduler
        server.stop()
        scheduler.shutdown()


if __name__ == "__main__":
    main() 