import asyncio
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import socket
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkSimulator:
    def __init__(self, base_difficulty: float = 1000.0, base_latency: float = 50.0):
        self.base_difficulty = base_difficulty
        self.base_latency = base_latency  # ms
        self.current_difficulty = base_difficulty
        self.current_latency = base_latency
        self.volatility = 0.2  # Difficulty change factor
        self.running = False
        self.clients = []
        self.server_socket = None
        self._lock = threading.Lock()
        
    def start(self, port: int = 0):
        """Start network simulator on specified port."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen(5)
        self.actual_port = self.server_socket.getsockname()[1]
        self.running = True
        
        # Start network condition simulator
        self.simulator_thread = threading.Thread(target=self._simulate_network_conditions)
        self.simulator_thread.daemon = True
        self.simulator_thread.start()
        
        # Start client handler
        self.handler_thread = threading.Thread(target=self._handle_clients)
        self.handler_thread.daemon = True
        self.handler_thread.start()
        
        logger.info(f"Network simulator started on port {self.actual_port}")
        
    def _simulate_network_conditions(self):
        """Simulate network condition changes."""
        while self.running:
            try:
                # Update difficulty using sine wave + random noise
                t = time.time() / 30  # 30-second cycle
                noise = np.random.normal(0, 0.1)
                difficulty_factor = 1 + self.volatility * (np.sin(t) + noise)
                
                # Update latency using different frequency
                latency_factor = 1 + 0.3 * np.sin(t * 0.7) + 0.1 * noise
                
                with self._lock:
                    self.current_difficulty = self.base_difficulty * difficulty_factor
                    self.current_latency = max(10, self.base_latency * latency_factor)
                
                # Log significant changes
                if abs(difficulty_factor - 1) > 0.2:
                    logger.info(f"Network difficulty changed: {self.current_difficulty:.2f}")
                if abs(latency_factor - 1) > 0.3:
                    logger.info(f"Network latency changed: {self.current_latency:.2f}ms")
                    
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in network simulation: {str(e)}")
                
    def _handle_clients(self):
        """Accept and handle client connections."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_handler = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr)
                )
                client_handler.daemon = True
                client_handler.start()
                self.clients.append(client_handler)
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting client: {str(e)}")
                    
    def _handle_client(self, client_socket: socket.socket, addr: tuple):
        """Handle individual client connection."""
        try:
            while self.running:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                    
                # Simulate network latency
                time.sleep(self.current_latency / 1000)
                
                try:
                    request = json.loads(data)
                    response = self._process_request(request)
                    
                    # Send response with current network conditions
                    with self._lock:
                        response.update({
                            "network_difficulty": self.current_difficulty,
                            "network_latency": self.current_latency
                        })
                    
                    client_socket.send(json.dumps(response).encode() + b'\n')
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received from {addr}")
                    
        except Exception as e:
            logger.error(f"Error handling client {addr}: {str(e)}")
        finally:
            client_socket.close()
            
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process client request and generate response."""
        if request.get("method") == "login":
            return {
                "id": request["id"],
                "result": {
                    "status": "OK",
                    "id": f"test_worker_{request['id']}"
                }
            }
            
        elif request.get("method") == "submit":
            # Validate share based on current difficulty
            share_difficulty = float(request.get("params", {}).get("difficulty", 0))
            with self._lock:
                accepted = share_difficulty >= self.current_difficulty
                
            return {
                "id": request["id"],
                "result": {
                    "status": "OK" if accepted else "REJECTED",
                    "reason": None if accepted else "Difficulty too low"
                }
            }
            
        return {
            "id": request.get("id"),
            "error": "Unknown method"
        }
        
    def stop(self):
        """Stop network simulator."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Network simulator stopped")
