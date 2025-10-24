"""
Management script for the Log Analyzer project.

Provides a single command-line interface to start, stop, diagnose,
and manage the application components.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# --- Configuration ---
PORTS_TO_CHECK: Dict[int, str] = {
    8000: "Backend API",
    8501: "Frontend",
    9092: "Kafka",
    6379: "Redis",
    2181: "Zookeeper"
}
INFRASTRUCTURE_COMPOSE_FILE = "docker-compose.infrastructure.yml"
IS_WINDOWS = platform.system() == "Windows"


# --- Helper Functions ---

def print_header(title: str):
    """Prints a formatted header to the console."""
    print("=" * 60)
    print(f"🚀 {title.upper()}")
    print("=" * 60)


def run_command(command: List[str], capture_output=True, check=False, **kwargs) -> subprocess.CompletedProcess:
    """A wrapper around subprocess.run for consistency."""
    return subprocess.run(command, capture_output=capture_output, text=True, check=check, **kwargs)


def find_processes_on_port(port: int) -> List[str]:
    """Finds PIDs of processes listening on a given port."""
    pids = []
    try:
        if IS_WINDOWS:
            result = run_command(["netstat", "-ano", "-p", "tcp"], timeout=10)
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pids.append(line.split()[-1])
        else:
            result = run_command(["lsof", "-ti", f":{port}"], timeout=10)
            pids.extend(result.stdout.strip().split())
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Could not check port {port}: {e}")
    return [pid for pid in pids if pid]


def kill_processes(pids: List[str]):
    """Forcibly kills a list of processes by PID."""
    for pid in pids:
        try:
            if IS_WINDOWS:
                run_command(["taskkill", "/f", "/pid", pid], timeout=5)
            else:
                run_command(["kill", "-9", pid], timeout=5)
            print(f"✅ Killed process with PID {pid}")
        except Exception as e:
            print(f"⚠️ Failed to kill process {pid}: {e}")


# --- Core Logic Functions ---

def handle_stop(*args, **kwargs):
    """Stops all running services, both Docker and local."""
    print_header("Stopping All Services")
    
    print("\n🐳 Stopping Docker infrastructure...")
    run_command(["docker", "compose", "-f", INFRASTRUCTURE_COMPOSE_FILE, "down", "--remove-orphans"])
    print("✅ Docker infrastructure stopped.")

    print("\n💻 Stopping local processes...")
    ports_to_kill = [8000, 8501]
    for port in ports_to_kill:
        pids = find_processes_on_port(port)
        if pids:
            print(f"Found processes on port {port}: {', '.join(pids)}. Terminating...")
            kill_processes(pids)
        else:
            print(f"✅ Port {port} is free.")
    print("✅ Local processes stopped.")
    return True


def handle_prepare_logs(*args, **kwargs):
    """Prepares log files for the replay service."""
    print_header("Preparing Replay Logs")
    replay_logs_dir = Path("data/logs")
    replay_logs_dir.mkdir(parents=True, exist_ok=True)

    if any(replay_logs_dir.glob("*.txt")) or any(replay_logs_dir.glob("*.log")):
        print(f"✅ Log files already exist in '{replay_logs_dir}'. Skipping preparation.")
        return True

    print(f"📂 No logs found in '{replay_logs_dir}'. Creating test logs...")
    test_logs_content = """2024-01-15T10:30:00 INFO app: Application started
2024-01-15T10:30:01 WARNING app: High memory usage detected
2024-01-15T10:30:02 ERROR app: Database connection failed"""
    
    (replay_logs_dir / "app_server1_log.txt").write_text(test_logs_content)
    print("✅ Created 'app_server1_log.txt'.")
    return True


def handle_start(*args, **kwargs):
    """Starts all services in hybrid mode."""
    print_header("Starting System (Hybrid Mode)")

    # 1. Prepare Logs
    if not handle_prepare_logs():
        return False

    # 2. Start Docker Infrastructure
    print("\n🐳 Starting Docker infrastructure (Kafka, Redis, Replay)...")
    compose_cmd = ["docker", "compose", "-f", INFRASTRUCTURE_COMPOSE_FILE, "up", "-d", "--build"]
    result = run_command(compose_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ Failed to start Docker infrastructure. Error:\n{result.stderr}")
        return False
    print("✅ Docker infrastructure starting in the background.")
    print("⏳ Waiting for services to become healthy (this may take a moment)...")
    time.sleep(20) # Give containers time to start and become healthy

    # 3. Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    for component in ["backend", "frontend"]:
        req_path = Path(component) / "requirements.txt"
        if req_path.exists():
            print(f"Installing dependencies for '{component}'...")
            run_command([sys.executable, "-m", "pip", "install", "-r", str(req_path)], check=True)
    
    # 4. Start Local Services
    print("\n💻 Starting local services (Backend, Frontend, Consumer)...")
    env = os.environ.copy()
    env.update({
        "REDIS_HOST": "localhost",
        "KAFKA_BOOTSTRAP": "localhost:9092",
        "API_BASE": "http://localhost:8000"
    })

    try:
        # Start Backend API
        print(" -> Starting Backend API...")
        subprocess.Popen([sys.executable, "app.py"], cwd="backend", env=env)
        
        # Start Kafka Consumer
        print(" -> Starting Kafka Consumer...")
        subprocess.Popen([sys.executable, "kafka_consumer.py"], cwd="backend", env=env)
        
        # Start Frontend
        print(" -> Starting Frontend...")
        subprocess.Popen(["streamlit", "run", "main.py"], cwd="frontend", env=env)
    except Exception as e:
        print(f"❌ Failed to start local services: {e}")
        return False

    print("\n" + "="*60)
    print("🎉 System started successfully!")
    print(f"✅ Frontend running at: http://localhost:8501")
    print(f"✅ Backend running at:  http://localhost:8000")
    print("="*60)
    return True


def handle_restart(*args, **kwargs):
    """Restarts all services."""
    print_header("Restarting System")
    if handle_stop():
        print("\n⏳ Waiting a few seconds before starting...")
        time.sleep(5)
        return handle_start()
    return False


def handle_diagnose(*args, **kwargs):
    """Runs diagnostic checks on the entire system."""
    print_header("Running System Diagnostics")

    # Check Docker containers
    print("\n🐳 Checking Docker containers...")
    run_command(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"], check=True, capture_output=False)
    
    # Check ports
    print("\n🔌 Checking ports...")
    all_ports_free = True
    for port, service in PORTS_TO_CHECK.items():
        if find_processes_on_port(port):
            print(f"⚠️ Port {port} ({service}) is IN USE.")
            all_ports_free = False
        else:
            print(f"✅ Port {port} ({service}) is FREE.")
            
    # Check API endpoints
    print("\n🔧 Checking API endpoints...")
    time.sleep(5) # Give API time to start if recently launched
    try:
        import requests
        endpoints = {
            "Backend Health": "http://localhost:8000/health",
            "Frontend": "http://localhost:8501",
        }
        for name, url in endpoints.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {name} is accessible (Status: {response.status_code})")
                else:
                    print(f"❌ {name} is not accessible (Status: {response.status_code})")
            except requests.ConnectionError:
                print(f"❌ {name} is not accessible (Connection failed)")
    except ImportError:
        print("⚠️ 'requests' library not found. Skipping API checks. (pip install requests)")

    # Check data files
    print("\n📁 Checking data files...")
    if Path("data/anomalies_problems.csv").exists():
        print("✅ 'data/anomalies_problems.csv' found.")
    else:
        print("❌ 'data/anomalies_problems.csv' NOT found.")
    
    if any(Path("data/logs").glob("*.txt")):
        print("✅ Log files found in 'data/logs/'.")
    else:
        print("⚠️ No log files found in 'data/logs/'. Consider running 'prepare-logs'.")
        
    print("\n" + "="*60)
    print("🔍 Diagnostics complete.")
    print("="*60)
    return True


def main():
    """Main function to parse arguments and execute commands."""
    # Setup for Windows stdout encoding if needed
    if IS_WINDOWS and sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description="Log Analyzer Management Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Define commands
    subparsers.add_parser("start", help="Start all services in hybrid mode.").set_defaults(func=handle_start)
    subparsers.add_parser("stop", help="Stop all services and free up ports.").set_defaults(func=handle_stop)
    subparsers.add_parser("restart", help="Restart all services.").set_defaults(func=handle_restart)
    subparsers.add_parser("diagnose", help="Run diagnostic checks on the system.").set_defaults(func=handle_diagnose)
    subparsers.add_parser("prepare-logs", help="Prepare log files for the replay service.").set_defaults(func=handle_prepare_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # Add a global logger for simplicity, although it's not heavily used
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
