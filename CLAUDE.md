# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Schedule-GPU is a Python-based GPU job scheduler that manages compute jobs across multiple GPUs. It provides both an interactive CLI and a server mode for remote job submission.

## Development Commands

### Running the Application
```bash
# Interactive mode (default)
python gpu_scheduler.py

# Server mode only (no interactive shell)
python gpu_scheduler.py --server --host 0.0.0.0 --port 5555

# Execute single command
python gpu_scheduler.py --command "queue python train.py --epochs 100"
```

### Project Dependencies
- Python 3.10+ required
- Install dependencies: `pip install rich>=14.0.0`
- Requires NVIDIA GPU with nvidia-smi available

## Architecture

### Core Components

1. **GPUManager** (gpu_scheduler.py:91-186)
   - Handles GPU detection and state tracking
   - Monitors GPU usage via nvidia-smi
   - Manages GPU reservation system

2. **JobQueue** (gpu_scheduler.py:189-245)
   - Priority-based job queue implementation
   - Thread-safe job management
   - Tracks all job states and history

3. **GPUScheduler** (gpu_scheduler.py:248-402)
   - Main scheduler coordinating jobs and GPUs
   - Runs scheduler loop in background thread
   - Manages job execution lifecycle

4. **CommandProcessor** (gpu_scheduler.py:405-636)
   - Processes text commands for all operations
   - Handles argument parsing and validation
   - Returns user-friendly responses

5. **InteractiveCLI** (gpu_scheduler.py:639-733)
   - Rich terminal UI without auto-refresh (no prompt interference)
   - Command history and auto-completion
   - Shows status only on startup

6. **SchedulerServer** (gpu_scheduler.py:736-806)
   - TCP server for remote access
   - JSON-based message protocol
   - Multi-threaded client handling

### Key Design Patterns

- **Modular Architecture**: Clear separation between GPU management, job queue, and scheduling
- **Thread Safety**: Proper locking for concurrent operations
- **State Management**: Clean enum-based state tracking for GPUs and jobs
- **No Auto-refresh**: Status displayed only on demand to keep prompt visible

### Important Implementation Details

- GPU detection via `nvidia-smi -L` (gpu_scheduler.py:98)
- Job execution uses `CUDA_VISIBLE_DEVICES` for GPU isolation (gpu_scheduler.py:319)
- Socket communication uses UTF-8 encoded JSON messages (gpu_scheduler.py:790)
- Jobs logged to gpu_scheduler_logs/ directory with timestamps

## Common Tasks

### Adding New Commands
1. Add command to `supported_commands` list (gpu_scheduler.py:1871)
2. Implement handler method in GPUJobScheduler class
3. Add to command completer for interactive mode (gpu_scheduler.py:1927)

### Modifying Job Execution
- Job runner logic in `_gpu_worker` method (gpu_scheduler.py:518-628)
- Process management uses subprocess.Popen with proper cleanup

### Enhancing GPU Detection
- GPU listing in `list_gpus` method (gpu_scheduler.py:148-163)
- GPU status checking in `_check_gpu_status` (gpu_scheduler.py:165-195)