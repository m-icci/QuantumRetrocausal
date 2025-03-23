# src/block_processor.py
import hashlib
import struct
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from src.qualia_engine import QualiaEngine
from src.gpu_manager import GPUManager

class BlockProcessor:
    def __init__(self, qualia_engine: QualiaEngine, gpu_manager: GPUManager):
        self.logger = logging.getLogger(__name__)
        self.qualia_engine = qualia_engine
        self.gpu_manager = gpu_manager
        self.target = None
        
    def _calculate_target(self, bits: str) -> bytes:
        """Calculate target from compact bits representation"""
        bits = int(bits, 16)
        shift = bits >> 24
        target = (bits & 0x00ffffff) * (2 ** (8 * (shift - 3)))
        return target.to_bytes(32, byteorder='little')

    def _create_block_header(self, job: Dict, nonce: int) -> bytes:
        """Create block header for mining"""
        version = struct.pack("<I", int(job['version'], 16))
        prev_hash = bytes.fromhex(job['prevhash'])[::-1]
        merkle_root = bytes.fromhex(job['merkle_root'])[::-1]
        timestamp = struct.pack("<I", int(job['time'], 16))
        bits = struct.pack("<I", int(job['bits'], 16))
        nonce = struct.pack("<I", nonce)
        
        return version + prev_hash + merkle_root + timestamp + bits + nonce

    async def process_block(self, job: Dict, mining_params: Dict) -> Optional[Dict]:
        """Process a mining job with QUALIA optimization"""
        try:
            self.target = self._calculate_target(job['bits'])
            batch_size = mining_params['batch_size']
            
            # Get entropy-optimized nonce predictions
            current_hash = hashlib.sha256(self._create_block_header(job, 0)).digest()
            entropy_score, nonce_predictions = self.qualia_engine.entropy_optimization(current_hash)
            
            # Prepare batch of nonces around predictions
            nonce_batches = []
            for pred_nonce in nonce_predictions:
                nonce_range = range(pred_nonce, pred_nonce + batch_size)
                headers = [self._create_block_header(job, n) for n in nonce_range]
                nonce_batches.extend(headers)
            
            # Process headers on GPU
            hashes = await self.gpu_manager.process_batch(nonce_batches)
            
            # Check for valid solution
            for i, hash_result in enumerate(hashes):
                if int.from_bytes(hash_result, 'little') < int.from_bytes(self.target, 'little'):
                    nonce = nonce_predictions[i // batch_size] + (i % batch_size)
                    self.qualia_engine.update_quantum_state(True)
                    return {
                        'job_id': job['job_id'],
                        'nonce': hex(nonce)[2:],
                        'extra_nonce2': job['extra_nonce2'],
                        'ntime': job['time']
                    }
            
            self.qualia_engine.update_quantum_state(False)
            return None
            
        except Exception as e:
            self.logger.error(f"Block processing error: {e}")
            return None