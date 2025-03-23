
"""
Launcher script for the Quantum Trading System.
Handles initialization and coordination of backend and dashboard components.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_trading_system() -> subprocess.Popen:
    """
    Start the trading system backend.
    Returns:
        subprocess.Popen: Process handle for the backend
    """
    try:
        logger.info("Starting trading system backend...")
        backend_process = subprocess.Popen(
            [sys.executable, "api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Allow server startup
        return backend_process
    except Exception as e:
        logger.error(f"Failed to start trading system: {e}")
        raise

def start_dashboard() -> subprocess.Popen:
    """
    Start the trading dashboard.
    Returns:
        subprocess.Popen: Process handle for the dashboard
    """
    try:
        logger.info("Starting trading dashboard...")
        os.environ['BACKEND_URL'] = 'http://0.0.0.0:8000'
        dashboard_process = subprocess.Popen(
            [sys.executable, "dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Allow dashboard startup
        return dashboard_process
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise

def main():
    """
    Main launcher function that coordinates system components.
    """
    try:
        backend_process = start_trading_system()
        logger.info("Trading system backend started")

        dashboard_process = start_dashboard()
        logger.info("Trading dashboard started")

        # Keep system running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if 'backend_process' in locals():
            backend_process.terminate()
        if 'dashboard_process' in locals():
            dashboard_process.terminate()
    except Exception as e:
        logger.error(f"Error running system: {e}")
        raise

if __name__ == "__main__":
    main()
