"""
Unified Quantum System
Integrates worker pool with quantum state management
"""
from typing import Dict, Optional, Callable
import logging
from datetime import datetime
from uuid import uuid4

from .utils.worker_pool import QuantumWorkerPool, QuantumTask

class UnifiedQuantumSystem:
    def __init__(self, num_workers: int = 4):
        self.worker_pool = QuantumWorkerPool(num_workers)
        self.active_computations: Dict[str, Dict] = {}
        
    def compute_quantum_state(self, state: Dict, callback: Optional[Callable] = None) -> str:
        """
        Submit a quantum state computation
        Returns a computation ID for tracking
        """
        computation_id = str(uuid4())
        task = QuantumTask(
            task_id=computation_id,
            state=state,
            priority=1,
            created_at=datetime.now(),
            callback=callback or self._default_callback
        )
        
        # Register computation
        self.active_computations[computation_id] = {
            'status': 'submitted',
            'submitted_at': datetime.now(),
            'state': state
        }
        
        # Submit to worker pool
        self.worker_pool.submit_task(task)
        return computation_id
    
    def _default_callback(self, result: Dict):
        """Default callback for handling computation results"""
        computation_id = result.get('computation_id')
        if computation_id in self.active_computations:
            self.active_computations[computation_id].update({
                'status': 'completed',
                'completed_at': datetime.now(),
                'result': result
            })
    
    def get_computation_status(self, computation_id: str) -> Dict:
        """Get status of a quantum computation"""
        return self.active_computations.get(computation_id, {
            'status': 'not_found',
            'error': 'Computation ID not found'
        })
    
    def register_error_handler(self, handler: Callable):
        """Register an error handler for quantum computations"""
        self.worker_pool.add_error_handler(handler)
    
    def shutdown(self):
        """Gracefully shutdown the quantum system"""
        self.worker_pool.shutdown()

# Initialize global quantum system
quantum_system = UnifiedQuantumSystem()
