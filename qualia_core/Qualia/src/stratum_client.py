# src/stratum_client.py
import asyncio
import json
import logging
from typing import Optional, Dict
import socket

class StratumClient:
    def __init__(self, host: str, port: int, wallet_address: str):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.wallet_address = wallet_address
        self.reader = None
        self.writer = None
        self.id = 1
        self.current_job: Optional[Dict] = None

    async def connect(self):
        """Establish connection to mining pool"""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, 
                self.port
            )
            self.logger.info(f"Connected to pool: {self.host}:{self.port}")
            await self._subscribe()
            await self._authorize()
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise

    async def _subscribe(self):
        """Subscribe to mining pool"""
        subscribe_request = {
            "id": self.id,
            "method": "mining.subscribe",
            "params": ["QuantumBitcoinMiner/1.0.0"]
        }
        await self._send_request(subscribe_request)
        response = await self._receive_response()
        if not response or "error" in response:
            raise ConnectionError("Subscription failed")

    async def _authorize(self):
        """Authorize with mining pool"""
        auth_request = {
            "id": self.id + 1,
            "method": "mining.authorize",
            "params": [self.wallet_address, "x"]
        }
        await self._send_request(auth_request)
        response = await self._receive_response()
        if not response or "error" in response:
            raise ConnectionError("Authorization failed")

    async def get_work(self) -> Optional[Dict]:
        """Get mining work from pool"""
        try:
            response = await self._receive_response()
            if response and response.get("method") == "mining.notify":
                self.current_job = {
                    "job_id": response["params"][0],
                    "prevhash": response["params"][1],
                    "coinbase1": response["params"][2],
                    "coinbase2": response["params"][3],
                    "merkle_branch": response["params"][4],
                    "version": response["params"][5],
                    "bits": response["params"][6],
                    "time": response["params"][7],
                    "clean_jobs": response["params"][8]
                }
                return self.current_job
        except Exception as e:
            self.logger.error(f"Error getting work: {e}")
        return None

    async def submit_work(self, result: Dict):
        """Submit mining result to pool"""
        submit_request = {
            "id": self.id + 2,
            "method": "mining.submit",
            "params": [
                self.wallet_address,
                result["job_id"],
                result["extra_nonce2"],
                result["ntime"],
                result["nonce"]
            ]
        }
        await self._send_request(submit_request)
        response = await self._receive_response()
        if response and "error" in response:
            self.logger.warning(f"Share rejected: {response['error']}")
        else:
            self.logger.info("Share accepted")

    async def _send_request(self, request: Dict):
        """Send JSON-RPC request to pool"""
        message = json.dumps(request) + "\n"
        self.writer.write(message.encode())
        await self.writer.drain()

    async def _receive_response(self) -> Optional[Dict]:
        """Receive and parse response from pool"""
        try:
            data = await self.reader.readline()
            if data:
                return json.loads(data.decode())
        except Exception as e:
            self.logger.error(f"Error receiving response: {e}")
        return None

    async def close(self):
        """Close connection to pool"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()