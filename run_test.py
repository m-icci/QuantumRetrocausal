"""
Script para executar os testes do sistema de mineração.
"""

import os
import sys
import unittest
import logging
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Agora podemos importar nossos módulos
from quantum_trading.mining.mining_system import MiningSystem

class TestMiningSystem(unittest.TestCase):
    """Testes para o sistema de mineração."""
    
    def setUp(self):
        """Configuração dos testes."""
        self.config = {
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
        self.mining_system = MiningSystem(self.config)
    
    def test_initialization(self):
        """Testa a inicialização do sistema de mineração."""
        self.assertEqual(self.mining_system.algorithm, 'randomx')
        self.assertEqual(self.mining_system.threads, 4)
        self.assertEqual(self.mining_system.intensity, 0.8)
        self.assertFalse(self.mining_system.is_mining)
    
    @patch('builtins.print')
    def test_quantum_circuit_generation(self, mock_print):
        """Testa a geração de circuitos quânticos."""
        circuit = self.mining_system.generate_quantum_circuit()
        self.assertIsInstance(circuit, dict)
        self.assertIn('qubits', circuit)
        self.assertIn('gates', circuit)
        self.assertIn('measurements', circuit)
        self.assertEqual(len(circuit['qubits']), self.config['quantum']['qubits'])
        
        # Exibe informações sobre o circuito
        print(f"Circuito quântico gerado: {circuit}")

if __name__ == '__main__':
    print("Iniciando testes do sistema de mineração...")
    unittest.main() 