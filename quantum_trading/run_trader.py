"""
FastAPI setup for trading system with core trading functionality
"""
import logging
import traceback
from datetime import datetime
import os
from dotenv import load_dotenv
import signal as signal_module
import sys
import time
import uvicorn
from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socket

from qualia.quantum_trader import QuantumTrader
from qualia.utils.logging import setup_logger

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
trader = None
running = True

def check_port_available(port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            return True
    except Exception as e:
        logger.error(f"Port {port} is not available: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown events"""
    try:
        logger.info("Starting FastAPI application...")
        # We'll initialize the trader in a background task later
        yield
    finally:
        logger.info("Shutting down FastAPI application...")
        cleanup()

# Initialize FastAPI app with explicit settings
app = FastAPI(
    title="Quantum Trading System",
    description="Advanced Quantum-Inspired Trading System API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    """Basic root endpoint"""
    try:
        return {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'trader_status': 'initialized' if trader else 'not initialized'
        }
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return {'error': str(e)}, 500

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'trader': {
                'initialized': trader is not None,
                'simulation_mode': trader.simulation_mode if trader else True
            }
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {'status': 'error', 'message': str(e)}, 500

@app.get("/trader/status")
async def trader_status():
    """Trading system status endpoint"""
    try:
        if not trader:
            return {
                'status': 'not_initialized',
                'timestamp': datetime.now().isoformat()
            }

        return {
            'status': 'running',
            'simulation_mode': trader.simulation_mode,
            'trading_pairs': trader.trading_pairs,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trader status: {e}")
        return {'status': 'error', 'message': str(e)}, 500

@app.post("/initialize")
async def initialize_trader_endpoint(background_tasks: BackgroundTasks):
    """Endpoint to initialize trader in background"""
    try:
        background_tasks.add_task(initialize_trader)
        return {"status": "initialization_started"}
    except Exception as e:
        logger.error(f"Failed to start initialization: {e}")
        return {"status": "error", "message": str(e)}

def initialize_trader():
    """Initialize trading system with detailed logging"""
    global trader
    try:
        logger.info("Starting trading system initialization...")

        # Log environment setup
        logger.debug("Current working directory: %s", os.getcwd())
        logger.debug("Python path: %s", sys.path)

        # Initialize trader with detailed logging of each step
        logger.info("Creating QuantumTrader instance...")
        new_trader = QuantumTrader(simulation_mode=True)

        # Verify trader initialization
        if not new_trader:
            logger.error("QuantumTrader instance is None after initialization")
            return None

        # Verify critical components
        logger.info("Verifying trader components...")
        critical_components = ['exchange', 'market_data', 'analyzer', 'risk_analyzer']
        for component in critical_components:
            if not hasattr(new_trader, component):
                logger.error(f"Trader missing {component} component")
                return None
            logger.debug(f"Component {component} verified")

        logger.info("Trading system initialization completed successfully")
        trader = new_trader
        return trader

    except Exception as e:
        logger.error(f"Failed to initialize trading system: {e}")
        logger.error(f"Initialization error details: {traceback.format_exc()}")
        return None

def cleanup():
    """Cleanup resources with proper error handling"""
    global running, trader
    logger.info("Starting cleanup process...")
    running = False
    try:
        if trader:
            logger.info("Cleaning up trader instance...")
            # Add specific cleanup if needed
            trader = None
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        logger.info("Cleanup completed")

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    logger.info(f"Received signal {signum}")
    cleanup()
    sys.exit(0)

def main():
    """Main entry point with comprehensive error handling"""
    try:
        # Set up signal handlers for graceful shutdown
        signal_module.signal(signal_module.SIGINT, signal_handler)
        signal_module.signal(signal_module.SIGTERM, signal_handler)

        logger.info("Starting server initialization...")

        # Check if port is available
        port = 8000
        if not check_port_available(port):
            logger.error(f"Port {port} is not available")
            sys.exit(1)

        logger.info("All components initialized successfully")

        # Start server using uvicorn with proper settings
        logger.info("Starting server with uvicorn on port 8000...")
        try:
            uvicorn.run(
                app,
                host='0.0.0.0',
                port=8000,
                log_level="debug",
                access_log=True,
                use_colors=True
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            logger.error(traceback.format_exc())
            raise

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        cleanup()
        sys.exit(1)

if __name__ == '__main__':
    main()