#!/usr/bin/env python3
"""
GPU Job Scheduler - A clean implementation for managing GPU compute jobs
"""

import os
import sys
import time
import json
import queue
import socket
import signal
import logging
import argparse
import datetime
import threading
import subprocess
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from prompt_toolkit import prompt
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.completion import WordCompleter
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install rich prompt_toolkit")
    sys.exit(1)


# GPU States
class GPUState(Enum):
    FREE = "free"
    BUSY = "busy"
    RESERVED = "reserved"
    OCCUPIED_BY_OTHER = "occupied_by_other"
    UNAVAILABLE = "unavailable"


# Job States
class JobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    CANCELLED = "cancelled"


@dataclass
class GPU:
    """Represents a single GPU"""
    id: int
    name: str
    state: GPUState = GPUState.FREE
    reserved_by: Optional[str] = None
    current_job_id: Optional[str] = None
    
    def is_available(self) -> bool:
        """Check if GPU is available for new jobs"""
        return self.state == GPUState.FREE and not self.reserved_by


@dataclass
class Job:
    """Represents a compute job"""
    id: str
    command: List[str]
    priority: int = 0
    state: JobState = JobState.QUEUED
    gpu_id: Optional[int] = None
    preferred_gpus: Optional[List[int]] = None
    process: Optional[subprocess.Popen] = None
    submitted_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    exit_code: Optional[int] = None
    log_file: Optional[Path] = None
    
    def __lt__(self, other):
        """For priority queue comparison (higher priority first)"""
        return self.priority > other.priority


class GPUManager:
    """Manages GPU detection and status tracking"""
    
    def __init__(self):
        self.gpus: Dict[int, GPU] = {}
        self.logger = logging.getLogger(__name__)
        self._detect_gpus()
    
    def _detect_gpus(self):
        """Detect available GPUs using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith("GPU "):
                    parts = line.split(": ", 1)
                    gpu_id = int(parts[0].split()[-1])
                    gpu_name = parts[1].split(" (UUID")[0] if " (UUID" in parts[1] else parts[1]
                    self.gpus[gpu_id] = GPU(id=gpu_id, name=gpu_name)
                    
            self.logger.info(f"Detected {len(self.gpus)} GPUs")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Failed to detect GPUs. Is nvidia-smi available?")
            
    def update_gpu_states(self):
        """Update GPU states based on nvidia-smi output"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    memory_used = int(parts[1])
                    gpu_util = int(parts[2])
                    
                    if gpu_id in self.gpus:
                        gpu = self.gpus[gpu_id]
                        # Only update if not reserved or running our job
                        if gpu.state not in [GPUState.RESERVED, GPUState.BUSY]:
                            # Consider GPU occupied if significant memory used or high utilization
                            # Note: Some memory is always used by the driver/display
                            if memory_used > 5000 or gpu_util > 80:
                                gpu.state = GPUState.OCCUPIED_BY_OTHER
                            else:
                                gpu.state = GPUState.FREE
                                
        except (subprocess.CalledProcessError, ValueError):
            self.logger.warning("Failed to update GPU states")
    
    def get_available_gpu(self, preferred_gpus: Optional[List[int]] = None) -> Optional[GPU]:
        """Get an available GPU, considering preferences"""
        self.update_gpu_states()
        
        # First try preferred GPUs
        if preferred_gpus:
            for gpu_id in preferred_gpus:
                if gpu_id in self.gpus and self.gpus[gpu_id].is_available():
                    return self.gpus[gpu_id]
        
        # Then try any available GPU
        for gpu in self.gpus.values():
            if gpu.is_available():
                return gpu
                
        return None
    
    def reserve_gpu(self, gpu_id: int, reserved_by: str) -> bool:
        """Reserve a GPU"""
        if gpu_id in self.gpus and self.gpus[gpu_id].state == GPUState.FREE:
            self.gpus[gpu_id].state = GPUState.RESERVED
            self.gpus[gpu_id].reserved_by = reserved_by
            return True
        return False
    
    def unreserve_gpu(self, gpu_id: int) -> bool:
        """Unreserve a GPU"""
        if gpu_id in self.gpus and self.gpus[gpu_id].state == GPUState.RESERVED:
            self.gpus[gpu_id].state = GPUState.FREE
            self.gpus[gpu_id].reserved_by = None
            return True
        return False
    
    def mark_gpu_busy(self, gpu_id: int, job_id: str):
        """Mark GPU as busy with a job"""
        if gpu_id in self.gpus:
            self.gpus[gpu_id].state = GPUState.BUSY
            self.gpus[gpu_id].current_job_id = job_id
    
    def mark_gpu_free(self, gpu_id: int):
        """Mark GPU as free after job completion"""
        if gpu_id in self.gpus:
            self.gpus[gpu_id].state = GPUState.FREE
            self.gpus[gpu_id].current_job_id = None


class JobQueue:
    """Priority queue for jobs"""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.all_jobs: Dict[str, Job] = {}
        self.job_counter = 0
        self.lock = threading.Lock()
        
    def add_job(self, command: List[str], priority: int = 0, preferred_gpus: Optional[List[int]] = None) -> Job:
        """Add a new job to the queue"""
        with self.lock:
            job_id = f"#{self.job_counter}"
            self.job_counter += 1
            
            job = Job(
                id=job_id,
                command=command,
                priority=priority,
                preferred_gpus=preferred_gpus
            )
            
            self.all_jobs[job_id] = job
            self.queue.put(job)
            
            return job
    
    def get_job(self) -> Optional[Job]:
        """Get the next job from queue"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        with self.lock:
            if job_id in self.all_jobs:
                job = self.all_jobs[job_id]
                if job.state == JobState.QUEUED:
                    job.state = JobState.CANCELLED
                    # Remove from queue by rebuilding it
                    new_queue = queue.PriorityQueue()
                    while not self.queue.empty():
                        j = self.queue.get()
                        if j.id != job_id:
                            new_queue.put(j)
                    self.queue = new_queue
                    return True
            return False
    
    def get_queued_jobs(self) -> List[Job]:
        """Get all queued jobs sorted by priority"""
        with self.lock:
            return sorted(
                [j for j in self.all_jobs.values() if j.state == JobState.QUEUED],
                reverse=True
            )


class GPUScheduler:
    """Main scheduler coordinating jobs and GPUs"""
    
    def __init__(self, log_dir: str = "gpu_scheduler_logs"):
        self.gpu_manager = GPUManager()
        self.job_queue = JobQueue()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []
        self.executor = ThreadPoolExecutor(max_workers=len(self.gpu_manager.gpus) + 2)
        
        self.logger = logging.getLogger(__name__)
        self.should_stop = threading.Event()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        pending_job = None
        
        while not self.should_stop.is_set():
            # Update GPU states first
            self.gpu_manager.update_gpu_states()
            
            # Try to schedule pending job first
            if pending_job:
                gpu = self.gpu_manager.get_available_gpu(pending_job.preferred_gpus)
                if gpu:
                    # Allocate GPU and start job
                    self.logger.debug(f"Allocating GPU {gpu.id} to pending job {pending_job.id}")
                    self.gpu_manager.mark_gpu_busy(gpu.id, pending_job.id)
                    pending_job.gpu_id = gpu.id
                    pending_job.state = JobState.RUNNING
                    pending_job.started_at = datetime.datetime.now()
                    
                    self.running_jobs[pending_job.id] = pending_job
                    self.executor.submit(self._run_job, pending_job)
                    pending_job = None  # Clear pending job
            
            # If no pending job, try to get a new one
            if not pending_job:
                job = self.job_queue.get_job()
                if job:
                    gpu = self.gpu_manager.get_available_gpu(job.preferred_gpus)
                    if gpu:
                        # Allocate GPU and start job
                        self.logger.debug(f"Allocating GPU {gpu.id} to job {job.id}")
                        self.gpu_manager.mark_gpu_busy(gpu.id, job.id)
                        job.gpu_id = gpu.id
                        job.state = JobState.RUNNING
                        job.started_at = datetime.datetime.now()
                        
                        self.running_jobs[job.id] = job
                        self.executor.submit(self._run_job, job)
                    else:
                        # Keep job as pending
                        self.logger.debug(f"No GPU available for job {job.id}, keeping as pending")
                        pending_job = job
            
            # Small sleep to prevent busy waiting
            time.sleep(0.5)
    
    def _run_job(self, job: Job):
        """Run a job on its allocated GPU"""
        try:
            # Create log file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"job_{job.id.replace('#', '')}_{timestamp}.log"
            job.log_file = self.log_dir / log_filename
            
            # Prepare environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
            
            # Run the job
            with open(job.log_file, 'w') as log_file:
                # Log the command being executed for debugging
                log_file.write(f"Command: {job.command}\n")
                log_file.write(f"Command as string: {' '.join(job.command)}\n")
                log_file.write("-" * 50 + "\n")
                log_file.flush()
                
                job.process = subprocess.Popen(
                    job.command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid if sys.platform != 'win32' else None
                )
                
                # Wait for completion
                job.exit_code = job.process.wait()
                
            # Update job state
            job.finished_at = datetime.datetime.now()
            if job.exit_code == 0:
                job.state = JobState.COMPLETED
            else:
                job.state = JobState.FAILED
                
        except Exception as e:
            self.logger.error(f"Error running job {job.id}: {e}")
            if job.state != JobState.KILLED:  # Don't override KILLED state
                job.state = JobState.FAILED
            if not job.finished_at:
                job.finished_at = datetime.datetime.now()
            
        finally:
            # Clean up
            job.process = None
            del self.running_jobs[job.id]
            self.completed_jobs.append(job)
            
            # Free the GPU
            if job.gpu_id is not None:
                self.logger.debug(f"Freeing GPU {job.gpu_id} after job {job.id} completed")
                self.gpu_manager.mark_gpu_free(job.gpu_id)
    
    def queue_job(self, command: List[str], priority: int = 0, preferred_gpus: Optional[List[int]] = None) -> Job:
        """Queue a new job"""
        return self.job_queue.add_job(command, priority, preferred_gpus)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        return self.job_queue.cancel_job(job_id)
    
    def kill_job(self, job_id: str) -> bool:
        """Kill a running job"""
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            if job.process:
                try:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
                    else:
                        job.process.terminate()
                    job.state = JobState.KILLED
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to kill job {job_id}: {e}")
                    return False
        return False
    
    def get_job_info(self, job_id: str) -> Optional[Job]:
        """Get information about a job"""
        return self.job_queue.all_jobs.get(job_id)
    
    def reserve_gpu(self, gpu_id: int, user: str = "user") -> bool:
        """Reserve a GPU"""
        return self.gpu_manager.reserve_gpu(gpu_id, user)
    
    def unreserve_gpu(self, gpu_id: int) -> bool:
        """Unreserve a GPU"""
        return self.gpu_manager.unreserve_gpu(gpu_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall scheduler status"""
        return {
            "gpus": {
                gpu_id: {
                    "name": gpu.name,
                    "state": gpu.state.value,
                    "current_job": gpu.current_job_id,
                    "reserved_by": gpu.reserved_by
                }
                for gpu_id, gpu in self.gpu_manager.gpus.items()
            },
            "queued_jobs": len(self.job_queue.get_queued_jobs()),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs)
        }
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.should_stop.set()
        self.executor.shutdown(wait=True)


class CommandProcessor:
    """Process text commands for the scheduler"""
    
    def __init__(self, scheduler: GPUScheduler):
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__)
        
        self.commands = {
            "queue": self._handle_queue,
            "cancel": self._handle_cancel,
            "kill": self._handle_kill,
            "status": self._handle_status,
            "jobs": self._handle_jobs,
            "gpus": self._handle_gpus,
            "reserve": self._handle_reserve,
            "unreserve": self._handle_unreserve,
            "info": self._handle_info,
            "help": self._handle_help,
            "exit": self._handle_exit,
            "quit": self._handle_exit,
        }
    
    def process_command(self, command_str: str) -> str:
        """Process a command string and return response"""
        parts = command_str.strip().split(None, 1)
        if not parts:
            return "Empty command"
        
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in self.commands:
            try:
                return self.commands[cmd](args)
            except Exception as e:
                return f"Error executing command: {e}"
        else:
            return f"Unknown command: {cmd}. Type 'help' for available commands."
    
    def _handle_queue(self, args: str) -> str:
        """Handle queue command"""
        if not args:
            return "Usage: queue <command> [--priority N] [--gpus GPU1,GPU2,...]"
        
        # Use shlex to properly parse command line with quotes
        import shlex
        try:
            parts = shlex.split(args)
        except ValueError as e:
            return f"Error parsing command: {e}"
        
        priority = 0
        preferred_gpus = None
        command_parts = []
        
        i = 0
        while i < len(parts):
            if parts[i] == "--priority" and i + 1 < len(parts):
                try:
                    priority = int(parts[i + 1])
                    i += 2
                except ValueError:
                    return "Invalid priority value"
            elif parts[i] == "--gpus" and i + 1 < len(parts):
                try:
                    preferred_gpus = [int(g) for g in parts[i + 1].split(",")]
                    i += 2
                except ValueError:
                    return "Invalid GPU list"
            else:
                command_parts.append(parts[i])
                i += 1
        
        if not command_parts:
            return "No command specified"
        
        job = self.scheduler.queue_job(command_parts, priority, preferred_gpus)
        
        lines = []
        lines.append("✅ Job queued successfully!")
        lines.append("")
        lines.append(f"  Job ID:    {job.id}")
        lines.append(f"  Command:   {' '.join(job.command)}")
        lines.append(f"  Priority:  {job.priority}")
        if preferred_gpus:
            lines.append(f"  Preferred: GPU {', GPU '.join(map(str, preferred_gpus))}")
        
        return "\n".join(lines)
    
    def _handle_cancel(self, args: str) -> str:
        """Handle cancel command"""
        job_id = args.strip()
        if not job_id:
            return "Usage: cancel <job_id>"
        
        if self.scheduler.cancel_job(job_id):
            return f"Job {job_id} cancelled"
        else:
            return f"Failed to cancel job {job_id} (not found or not queued)"
    
    def _handle_kill(self, args: str) -> str:
        """Handle kill command"""
        job_id = args.strip()
        if not job_id:
            return "Usage: kill <job_id>"
        
        if self.scheduler.kill_job(job_id):
            return f"Job {job_id} killed"
        else:
            return f"Failed to kill job {job_id} (not found or not running)"
    
    def _handle_status(self, _: str) -> str:
        """Handle status command"""
        status = self.scheduler.get_status()
        
        lines = []
        lines.append("╔" + "═" * 50 + "╗")
        lines.append("║" + " Scheduler Status ".center(50) + "║")
        lines.append("╚" + "═" * 50 + "╝")
        lines.append("")
        lines.append(f"  Queued jobs:    {status['queued_jobs']}")
        lines.append(f"  Running jobs:   {status['running_jobs']}")
        lines.append(f"  Completed jobs: {status['completed_jobs']}")
        lines.append("")
        lines.append("  GPU Status:")
        lines.append("  " + "-" * 48)
        
        for gpu_id, gpu_info in status['gpus'].items():
            state_str = gpu_info['state']
            if gpu_info['current_job']:
                state_str += f" (job {gpu_info['current_job']})"
            elif gpu_info['reserved_by']:
                state_str += f" (reserved by {gpu_info['reserved_by']})"
            lines.append(f"  GPU {gpu_id} [{gpu_info['name']}]: {state_str}")
        
        return "\n".join(lines)
    
    def _handle_jobs(self, _: str) -> str:
        """Handle jobs command"""
        lines = []
        lines.append("╔" + "═" * 50 + "╗")
        lines.append("║" + " Job List ".center(50) + "║")
        lines.append("╚" + "═" * 50 + "╝")
        
        # Queued jobs
        queued = self.scheduler.job_queue.get_queued_jobs()
        if queued:
            lines.append("")
            lines.append("  📋 Queued Jobs:")
            lines.append("  " + "-" * 48)
            for job in queued:
                lines.append(f"  {job.id} [priority={job.priority}]: {' '.join(job.command)}")
        
        # Running jobs
        if self.scheduler.running_jobs:
            lines.append("")
            lines.append("  🚀 Running Jobs:")
            lines.append("  " + "-" * 48)
            for job in self.scheduler.running_jobs.values():
                runtime = datetime.datetime.now() - job.started_at
                lines.append(f"  {job.id} [GPU {job.gpu_id}, {runtime}]: {' '.join(job.command)}")
        
        # Recent completed jobs
        recent_completed = self.scheduler.completed_jobs[-10:]
        if recent_completed:
            lines.append("")
            lines.append("  ✅ Recent Completed Jobs:")
            lines.append("  " + "-" * 48)
            for job in reversed(recent_completed):
                lines.append(f"  {job.id} [{job.state.value}]: {' '.join(job.command)}")
        
        if not queued and not self.scheduler.running_jobs and not recent_completed:
            lines.append("")
            lines.append("  No jobs in the system.")
        
        return "\n".join(lines)
    
    def _handle_gpus(self, _: str) -> str:
        """Handle gpus command"""
        lines = []
        lines.append("╔" + "═" * 50 + "╗")
        lines.append("║" + " GPU Information ".center(50) + "║")
        lines.append("╚" + "═" * 50 + "╝")
        lines.append("")
        
        for gpu_id, gpu in self.scheduler.gpu_manager.gpus.items():
            state_str = gpu.state.value
            state_icon = {
                "free": "🟢",
                "busy": "🔴",
                "reserved": "🟡",
                "occupied_by_other": "🟣"
            }.get(gpu.state.value, "⚪")
            
            if gpu.current_job_id:
                state_str += f" (job {gpu.current_job_id})"
            elif gpu.reserved_by:
                state_str += f" (reserved by {gpu.reserved_by})"
            
            lines.append(f"  {state_icon} GPU {gpu_id} [{gpu.name}]")
            lines.append(f"     Status: {state_str}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _handle_reserve(self, args: str) -> str:
        """Handle reserve command"""
        try:
            gpu_id = int(args.strip())
            if self.scheduler.reserve_gpu(gpu_id):
                return f"GPU {gpu_id} reserved"
            else:
                return f"Failed to reserve GPU {gpu_id}"
        except ValueError:
            return "Usage: reserve <gpu_id>"
    
    def _handle_unreserve(self, args: str) -> str:
        """Handle unreserve command"""
        try:
            gpu_id = int(args.strip())
            if self.scheduler.unreserve_gpu(gpu_id):
                return f"GPU {gpu_id} unreserved"
            else:
                return f"Failed to unreserve GPU {gpu_id}"
        except ValueError:
            return "Usage: unreserve <gpu_id>"
    
    def _handle_info(self, args: str) -> str:
        """Handle info command"""
        job_id = args.strip()
        if not job_id:
            return "Usage: info <job_id>"
        
        job = self.scheduler.get_job_info(job_id)
        if not job:
            return f"Job {job_id} not found"
        
        lines = []
        lines.append("╔" + "═" * 50 + "╗")
        lines.append("║" + f" Job {job_id} Details ".center(50) + "║")
        lines.append("╚" + "═" * 50 + "╝")
        lines.append("")
        
        state_icon = {
            "queued": "📋",
            "running": "🚀",
            "completed": "✅",
            "failed": "❌",
            "killed": "🛑",
            "cancelled": "🚫"
        }.get(job.state.value, "⚪")
        
        lines.append(f"  {state_icon} State:     {job.state.value}")
        lines.append(f"  📝 Command:   {' '.join(job.command)}")
        lines.append(f"  ⚡ Priority:  {job.priority}")
        lines.append(f"  🕐 Submitted: {job.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if job.gpu_id is not None:
            lines.append(f"  🖥️  GPU:       {job.gpu_id}")
        if job.started_at:
            lines.append(f"  ▶️  Started:   {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if job.finished_at:
            lines.append(f"  ⏹️  Finished:  {job.finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = job.finished_at - job.started_at
            lines.append(f"  ⏱️  Duration:  {duration}")
        if job.exit_code is not None:
            lines.append(f"  🔢 Exit code: {job.exit_code}")
        if job.log_file:
            lines.append(f"  📄 Log file:  {job.log_file}")
        
        return "\n".join(lines)
    
    def _handle_help(self, _: str) -> str:
        """Handle help command"""
        lines = []
        lines.append("╔" + "═" * 60 + "╗")
        lines.append("║" + " GPU Scheduler Commands ".center(60) + "║")
        lines.append("╚" + "═" * 60 + "╝")
        lines.append("")
        lines.append("  📋 Job Management:")
        lines.append("  " + "-" * 58)
        lines.append("  queue <command> [--priority N] [--gpus GPU1,GPU2,...]")
        lines.append("      Queue a new job for execution")
        lines.append("")
        lines.append("  cancel <job_id>")
        lines.append("      Cancel a queued job")
        lines.append("")
        lines.append("  kill <job_id>")
        lines.append("      Kill a running job")
        lines.append("")
        lines.append("  📊 Status & Information:")
        lines.append("  " + "-" * 58)
        lines.append("  status")
        lines.append("      Show scheduler status and GPU availability")
        lines.append("")
        lines.append("  jobs")
        lines.append("      List all jobs (queued, running, completed)")
        lines.append("")
        lines.append("  gpus")
        lines.append("      Show detailed GPU information")
        lines.append("")
        lines.append("  info <job_id>")
        lines.append("      Show detailed information about a specific job")
        lines.append("")
        lines.append("  🔧 GPU Management:")
        lines.append("  " + "-" * 58)
        lines.append("  reserve <gpu_id>")
        lines.append("      Reserve a GPU for exclusive use")
        lines.append("")
        lines.append("  unreserve <gpu_id>")
        lines.append("      Release a GPU reservation")
        lines.append("")
        lines.append("  🚪 Other:")
        lines.append("  " + "-" * 58)
        lines.append("  help")
        lines.append("      Show this help message")
        lines.append("")
        lines.append("  exit / quit")
        lines.append("      Exit the scheduler")
        
        return "\n".join(lines)
    
    def _handle_exit(self, _: str) -> str:
        """Handle exit command"""
        return "EXIT"


class InteractiveCLI:
    """Interactive command-line interface with rich UI"""
    
    def __init__(self, scheduler: GPUScheduler):
        self.scheduler = scheduler
        self.processor = CommandProcessor(scheduler)
        self.console = Console()
        self.history = InMemoryHistory()
        
    def run(self):
        """Run the interactive CLI"""
        self.console.print("[bold green]GPU Scheduler Started[/bold green]")
        self.console.print("Type 'help' for available commands, 'exit' to quit\n")
        
        # Show initial status
        self._show_status()
        
        # Create completer
        completer = WordCompleter([
            'queue', 'cancel', 'kill', 'status', 'jobs', 'gpus',
            'reserve', 'unreserve', 'info', 'help', 'exit', 'quit'
        ])
        
        try:
            while True:
                try:
                    # Get command
                    command = prompt(
                        "scheduler> ",
                        history=self.history,
                        completer=completer
                    )
                    
                    # Process command
                    result = self.processor.process_command(command)
                    
                    if result == "EXIT":
                        break
                    else:
                        self.console.print()  # Add blank line before output
                        self.console.print(result)
                        self.console.print()  # Add blank line after output
                        
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                    
        finally:
            self.console.print("\n[bold yellow]Shutting down scheduler...[/bold yellow]")
            self.scheduler.shutdown()
    
    def _show_status(self):
        """Display current status"""
        try:
            status = self.scheduler.get_status()
            
            # Create status table
            table = Table(title="GPU Status", show_header=True)
            table.add_column("GPU", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("State", style="green")
            table.add_column("Job/Reserved", style="yellow")
            
            for gpu_id, gpu_info in status['gpus'].items():
                state_color = {
                    "free": "green",
                    "busy": "red",
                    "reserved": "yellow",
                    "occupied_by_other": "magenta"
                }.get(gpu_info['state'], "white")
                
                job_info = ""
                if gpu_info['current_job']:
                    job_info = gpu_info['current_job']
                elif gpu_info['reserved_by']:
                    job_info = f"Reserved: {gpu_info['reserved_by']}"
                
                table.add_row(
                    str(gpu_id),
                    gpu_info['name'],
                    f"[{state_color}]{gpu_info['state']}[/{state_color}]",
                    job_info
                )
            
            # Create summary panel
            summary = Panel(
                f"Queued: {status['queued_jobs']}  Running: {status['running_jobs']}  Completed: {status['completed_jobs']}",
                title="Job Summary"
            )
            
            self.console.print(table)
            self.console.print(summary)
            self.console.print()
            
        except Exception as e:
            self.console.print(f"[red]Error displaying status: {e}[/red]")


class SchedulerServer:
    """TCP server for remote scheduler access"""
    
    def __init__(self, scheduler: GPUScheduler, host: str = "localhost", port: int = 8000, debug: bool = False):
        self.scheduler = scheduler
        self.processor = CommandProcessor(scheduler)
        self.host = host
        self.port = port
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.should_stop = threading.Event()
        
    def run(self):
        """Run the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            
            self.logger.info(f"Server listening on {self.host}:{self.port}")
            if self.debug:
                self.logger.debug("Debug mode enabled - verbose logging active")
            
            while not self.should_stop.is_set():
                try:
                    client_socket, client_addr = server_socket.accept()
                    self.logger.info(f"Client connected from {client_addr}")
                    
                    if self.debug:
                        self.logger.debug(f"Socket details - local: {client_socket.getsockname()}, remote: {client_addr}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket,),
                        daemon=True
                    )
                    client_thread.start()
                    
                    if self.debug:
                        self.logger.debug(f"Started handler thread for client {client_addr}")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Server error: {e}")
                    
        finally:
            server_socket.close()
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle a client connection"""
        client_addr = client_socket.getpeername()
        try:
            while True:
                # Receive command
                raw_data = client_socket.recv(4096)
                if not raw_data:
                    if self.debug:
                        self.logger.debug(f"Client {client_addr} disconnected - empty data")
                    break
                
                # Decode and strip data
                data = raw_data.decode('utf-8').strip()
                
                if self.debug:
                    self.logger.debug(f"Received from {client_addr}: Raw bytes: {raw_data!r}")
                    self.logger.debug(f"Received from {client_addr}: Decoded: '{data}'")
                
                if not data:
                    if self.debug:
                        self.logger.debug(f"Client {client_addr} sent empty string after stripping")
                    continue
                
                # Process command
                if self.debug:
                    self.logger.debug(f"Processing command from {client_addr}: '{data}'")
                
                result = self.processor.process_command(data)
                
                # Send response
                response = json.dumps({"result": result}) + "\n"
                
                if self.debug:
                    self.logger.debug(f"Sending response to {client_addr}: '{response.strip()}'")
                
                client_socket.send(response.encode('utf-8'))
                
                if result == "EXIT":
                    if self.debug:
                        self.logger.debug(f"Client {client_addr} requested exit")
                    break
                    
        except Exception as e:
            self.logger.error(f"Client handler error for {client_addr}: {e}")
            if self.debug:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        finally:
            if self.debug:
                self.logger.debug(f"Closing connection to {client_addr}")
            client_socket.close()
    
    def stop(self):
        """Stop the server"""
        self.should_stop.set()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPU Job Scheduler")
    parser.add_argument("--server", action="store_true", help="Run in server mode only")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--command", help="Execute a single command and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = args.log_level
    if args.debug:
        log_level = "DEBUG"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scheduler
    scheduler = GPUScheduler()
    
    # Handle single command
    if args.command:
        processor = CommandProcessor(scheduler)
        result = processor.process_command(args.command)
        print(result)
        scheduler.shutdown()
        return
    
    # Start server if requested
    server = None
    if args.server or args.host != "localhost" or args.port != 8000:
        server = SchedulerServer(scheduler, args.host, args.port, debug=args.debug)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
    
    # Run interactive CLI if not server-only mode
    if not args.server:
        cli = InteractiveCLI(scheduler)
        try:
            cli.run()
        except KeyboardInterrupt:
            pass
    else:
        # Server-only mode
        print(f"GPU Scheduler server running on {args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    # Cleanup
    if server:
        server.stop()
    scheduler.shutdown()


if __name__ == "__main__":
    main()