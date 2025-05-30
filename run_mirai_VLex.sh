#!/bin/bash

# Script to run multiple Python commands sequentially, and for each command,
# keep the CPU busy if its usage drops to prevent session termination.

# --- Configuration ---
# Define your list of commands to be executed sequentially.
# Each command will be monitored individually.
commands=(
    "python Main_Association_Rule.py --file_type MiraiBotnet --association RARM" # Example 1: Replace with your actual command
    "python Main_Association_Rule.py --file_type MiraiBotnet --association Eclat" # Example 2: Replace with your actual command
    # "python Wildfire_spread_graph.py --data_number 1" # Add more commands as needed
    # "python Wildfire_spread_graph.py --data_number 2"
)

# CPU usage threshold (integer percentage). If Python script's CPU usage is below this, start idle spin.
CPU_USAGE_THRESHOLD=10

# Command to generate CPU load for idle spin. 'yes > /dev/null' uses one core at 100%.
IDLE_CPU_COMMAND="yes > /dev/null"

# How often to check CPU usage (in seconds).
CHECK_INTERVAL=5
# --- End Configuration ---

idle_pid="" # PID of the idle spin process

# Function to start the idle spin process
start_idle_spin() {
    if [ -z "$idle_pid" ] || ! ps -p "$idle_pid" > /dev/null; then
        echo "[Keeper] Main script CPU usage low. Starting idle spin..."
        eval "$IDLE_CPU_COMMAND" &
        idle_pid=$!
        echo "[Keeper] Idle spin process started with PID: $idle_pid"
    else
        echo "[Keeper] Idle spin already running (PID: $idle_pid)."
    fi
}

# Function to stop the idle spin process
stop_idle_spin() {
    if [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null; then
        echo "[Keeper] Main script CPU usage recovered or script ended. Stopping idle spin (PID: $idle_pid)..."
        # Attempt to kill the process and its children more robustly
        pkill -P "$idle_pid" # Kill children of idle_pid first
        kill "$idle_pid" > /dev/null 2>&1
        sleep 0.1 # Give a moment for graceful termination
        if ps -p "$idle_pid" > /dev/null; then # Check if still alive
            kill -9 "$idle_pid" > /dev/null 2>&1 # Force kill if necessary
        fi
        wait "$idle_pid" 2>/dev/null # Suppress "No such process" error if already gone
        idle_pid=""
        echo "[Keeper] Idle spin process stopped."
    fi
}

# Function to monitor the main script's CPU usage and manage idle spin
monitor_and_manage_cpu_load() {
    local main_script_pid=$1
    local command_being_monitored="$2"
    echo "[Keeper] Monitoring CPU usage for: $command_being_monitored (PID: $main_script_pid)"

    while ps -p "$main_script_pid" > /dev/null; do
        current_cpu_usage=$(ps -p "$main_script_pid" -o %cpu --no-headers | awk '{print int($1)}')
        if [ -z "$current_cpu_usage" ]; then
            echo "[Keeper] Could not retrieve CPU usage for PID $main_script_pid ($command_being_monitored) or PID disappeared."
            stop_idle_spin
            break
        fi
        echo "[Keeper] Script '$command_being_monitored' (PID: $main_script_pid) current CPU: $current_cpu_usage%"
        if (( current_cpu_usage < CPU_USAGE_THRESHOLD )); then
            start_idle_spin
        else
            stop_idle_spin
        fi
        sleep "$CHECK_INTERVAL"
    done
    echo "[Keeper] Script '$command_being_monitored' (PID: $main_script_pid) appears to have finished."
    stop_idle_spin
    echo "[Keeper] CPU monitoring for PID $main_script_pid ($command_being_monitored) ended."
}

# --- Main Execution ---
overall_exit_code=0

for cmd in "${commands[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Starting Python script: $cmd"
    # Execute the Python command in the background
    eval "$cmd" &
    MAIN_SCRIPT_PID=$!
    echo "Python script started with PID: $MAIN_SCRIPT_PID"

    # Start the CPU monitor and load manager in the background
    monitor_and_manage_cpu_load "$MAIN_SCRIPT_PID" "$cmd" &
    MONITOR_PID=$!
    echo "CPU keeper process for '$cmd' started with PID: $MONITOR_PID"

    # Wait for the current Python script to complete
    echo "Waiting for Python script (PID: $MAIN_SCRIPT_PID) '$cmd' to complete..."
    wait "$MAIN_SCRIPT_PID"
    SCRIPT_EXIT_CODE=$?

    echo "Python script '$cmd' (PID: $MAIN_SCRIPT_PID) completed with exit code: $SCRIPT_EXIT_CODE."
    if (( SCRIPT_EXIT_CODE != 0 )); then
        overall_exit_code=$SCRIPT_EXIT_CODE
        echo "Warning: Command '$cmd' exited with error code $SCRIPT_EXIT_CODE."
    fi

    # Terminate the monitor process for this command if it's still running
    echo "Ensuring CPU keeper for '$cmd' (PID: $MONITOR_PID) is stopped..."
    if ps -p "$MONITOR_PID" > /dev/null; then
        kill "$MONITOR_PID" > /dev/null 2>&1 &
    fi
    stop_idle_spin # Ensure idle spin for the completed command is stopped
    echo "----------------------------------------------------------------------"
    echo ""

done

echo "All commands in the list have been processed."
echo "Overall script finished. Last non-zero exit code (if any): $overall_exit_code"
exit "$overall_exit_code"
