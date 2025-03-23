"""
Tests for the mining system component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.mining.mining_system import MiningSystem
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'mining': {
            'algorithm': 'randomx',
            'threads': 4,
            'intensity': 0.8,
            'batch_size': 256,
            'optimization': {
                'auto_tune': True,
                'target_temperature': 75,
                'power_limit': 100
            }
        },
        'quantum': {
            'qubits': 20,
            'circuit_depth': 5,
            'measurement_basis': 'computational',
            'optimization': {
                'quantum_annealing': True,
                'annealing_cycles': 1000
            }
        },
        'network': {
            'pool': 'monero.pool.com:3333',
            'wallet': '4xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'worker_id': 'worker1',
            'password': 'x'
        }
    }

@pytest.fixture
async def mock_data_loader():
    loader = AsyncMock()
    
    # Mock get_mining_stats method
    async def get_mining_stats():
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'hashrate': [1000],
            'shares_accepted': [10],
            'shares_rejected': [1],
            'temperature': [65],
            'power_usage': [150],
            'efficiency': [6.67]
        })
    loader.get_mining_stats = get_mining_stats
    
    return loader

@pytest.fixture
async def mining_system(config, mock_data_loader):
    """Create a MiningSystem instance for testing."""
    system = MiningSystem(config)
    system.data_loader = mock_data_loader
    return system

@pytest.mark.asyncio
async def test_mining_system_initialization(config):
    """Test mining system initialization."""
    mining_system = MiningSystem(config)
    assert mining_system.config == config
    assert mining_system.algorithm == config['mining']['algorithm']
    assert mining_system.threads == config['mining']['threads']
    assert not mining_system.is_mining

@pytest.mark.asyncio
async def test_mining_start_stop(config):
    """Test mining start and stop."""
    mining_system = MiningSystem(config)
    
    # Start mining
    await mining_system.start()
    assert mining_system.is_mining
    assert mining_system.start_time is not None
    
    # Stop mining
    await mining_system.stop()
    assert not mining_system.is_mining
    assert mining_system.start_time is None

@pytest.mark.asyncio
async def test_hashrate_monitoring(config, mock_data_loader):
    """Test hashrate monitoring."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Get hashrate
    hashrate = await mining_system.get_hashrate()
    assert isinstance(hashrate, dict)
    assert 'current' in hashrate
    assert 'average' in hashrate
    assert 'peak' in hashrate
    assert all(v >= 0 for v in hashrate.values())

@pytest.mark.asyncio
async def test_quantum_optimization(config):
    """Test quantum optimization."""
    mining_system = MiningSystem(config)
    
    # Optimize parameters
    params = await mining_system.optimize_quantum_params()
    assert isinstance(params, dict)
    assert 'circuit_params' in params
    assert 'annealing_schedule' in params
    assert 'optimization_score' in params

@pytest.mark.asyncio
async def test_performance_monitoring(config, mock_data_loader):
    """Test performance monitoring."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Get performance metrics
    metrics = await mining_system.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert 'hashrate' in metrics
    assert 'shares' in metrics
    assert 'temperature' in metrics
    assert 'power_usage' in metrics
    assert 'efficiency' in metrics

@pytest.mark.asyncio
async def test_network_interaction(config, mock_data_loader):
    """Test network interaction."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Get network stats
    stats = await mining_system.get_network_stats()
    assert isinstance(stats, dict)
    assert 'difficulty' in stats
    assert 'height' in stats
    assert 'hashrate' in stats
    assert 'reward' in stats

@pytest.mark.asyncio
async def test_share_submission(config):
    """Test share submission."""
    mining_system = MiningSystem(config)
    
    # Submit share
    share = {
        'nonce': '1234567890',
        'result': 'a' * 64,
        'job_id': '987654321'
    }
    result = await mining_system.submit_share(share)
    assert isinstance(result, dict)
    assert 'accepted' in result
    assert 'difficulty' in result

@pytest.mark.asyncio
async def test_auto_tuning(config, mock_data_loader):
    """Test auto-tuning."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Run auto-tuning
    tuning = await mining_system.auto_tune()
    assert isinstance(tuning, dict)
    assert 'threads' in tuning
    assert 'intensity' in tuning
    assert 'batch_size' in tuning
    assert tuning['threads'] > 0
    assert 0 < tuning['intensity'] <= 1

@pytest.mark.asyncio
async def test_thermal_management(config, mock_data_loader):
    """Test thermal management."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Get thermal status
    status = await mining_system.get_thermal_status()
    assert isinstance(status, dict)
    assert 'temperature' in status
    assert 'fan_speed' in status
    assert 'throttling' in status
    assert 0 <= status['temperature'] <= 100

@pytest.mark.asyncio
async def test_power_management(config, mock_data_loader):
    """Test power management."""
    mining_system = MiningSystem(config)
    mining_system.data_loader = mock_data_loader
    
    # Get power metrics
    metrics = await mining_system.get_power_metrics()
    assert isinstance(metrics, dict)
    assert 'current_usage' in metrics
    assert 'average_usage' in metrics
    assert 'efficiency' in metrics
    assert all(v >= 0 for v in metrics.values())

@pytest.mark.asyncio
async def test_error_handling(mining_system):
    """Test error handling for invalid configurations"""
    with pytest.raises(ValueError):
        mining_system.threads = -1
    
    with pytest.raises(ValueError):
        mining_system.quantum_depth = -1
        
    with pytest.raises(ValueError):
        mining_system.memory_pool = 0

@pytest.mark.asyncio
async def test_quantum_circuit_generation(config):
    """Test quantum circuit generation."""
    mining_system = MiningSystem(config)
    
    # Generate quantum circuit
    circuit = mining_system.generate_quantum_circuit()
    assert isinstance(circuit, dict)
    assert 'qubits' in circuit
    assert 'gates' in circuit
    assert 'measurements' in circuit
    assert len(circuit['qubits']) == config['quantum']['qubits']

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 