import numpy as np
import pytest
from qualia.core.metrics import QuantumMetrics

class TestQuantumMetrics:
    def setup_method(self):
        self.metrics = QuantumMetrics()
        
    def test_calculate_entropy_valid_input(self):
        # Teste com estado quântico simples
        quantum_state = np.random.rand(10, 10)
        entropy = self.metrics.calculate_entropy(quantum_state)
        
        assert 0 <= entropy <= np.log2(len(quantum_state)), \
            "Entropia deve estar entre 0 e log2(dimensão)"
        
    def test_calculate_entropy_edge_cases(self):
        # Caso de matriz identidade
        identity = np.eye(10)
        entropy_identity = self.metrics.calculate_entropy(identity)
        assert entropy_identity == 0, "Entropia de matriz identidade deve ser zero"
        
        # Caso de matriz aleatória
        random_matrix = np.random.rand(10, 10)
        entropy_random = self.metrics.calculate_entropy(random_matrix)
        assert entropy_random >= 0, "Entropia não pode ser negativa"
        
    def test_get_quantum_metrics_consistency(self):
        # Teste de consistência entre métricas
        quantum_state = np.random.rand(100)
        metrics = self.metrics.get_quantum_metrics(quantum_state)
        
        assert "entropia" in metrics, "Métrica de entropia ausente"
        assert "complexidade" in metrics, "Métrica de complexidade ausente"
        assert "coerencia" in metrics, "Métrica de coerência ausente"
        
        assert 0 <= metrics["entropia"] <= np.log2(len(quantum_state)), \
            "Entropia fora do intervalo esperado"
        assert metrics["complexidade"] >= 0, "Complexidade não pode ser negativa"
        assert 0 <= metrics["coerencia"] <= 1, "Coerência deve estar entre 0 e 1"
        
    def test_calculate_coherence(self):
        # Teste de coerência
        quantum_state = np.random.rand(10, 10)
        coherence = self.metrics.calculate_coherence(quantum_state)
        
        assert 0 <= coherence <= 1, "Coerência deve estar entre 0 e 1"
        
    def test_retrocausality_index(self):
        # Teste de índice de retrocausalidade
        current_field = np.random.rand(10, 10)
        past_field = np.random.rand(10, 10)
        
        retrocausality = self.metrics.calculate_retrocausality_index(current_field, past_field)
        
        assert isinstance(retrocausality, float), "Retrocausalidade deve ser um float"
        assert retrocausality >= 0, "Retrocausalidade não pode ser negativa"
