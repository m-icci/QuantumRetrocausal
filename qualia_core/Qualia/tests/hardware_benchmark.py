"""
Testes de benchmark para validar QUALIA em diferentes hardwares
Testa funcionamento em CPU-only e compara com GPU quando disponível
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any
from core.hardware.hardware_manager import HardwareManager
from core.quantum.qualia_unified import (
    UnifiedField,
    FieldType,
    BitwiseOperators,
    HolographicField,
    UnifiedMetrics
)

logger = logging.getLogger(__name__)

class QualiaEncoder(json.JSONEncoder):
    """Encoder customizado para serializar objetos QUALIA"""
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.hex()
        if isinstance(obj, UnifiedMetrics):
            return {k: v.hex() if isinstance(v, bytes) else v 
                   for k, v in obj.__dict__.items()}
        if hasattr(obj, '__dict__'):
            return {k: v.hex() if isinstance(v, bytes) else v 
                   for k, v in obj.__dict__.items()}
        return super().default(obj)

class HardwareTest:
    """Testes de hardware para QUALIA"""
    
    def __init__(self):
        self.hardware = HardwareManager()
        self.ops = BitwiseOperators()
        self.unified_field = UnifiedField(
            field_type=FieldType.ADAPTIVE,
            size=64,
            consciousness_factor=0.618
        )
        self.holographic = HolographicField(size=64)
        self.results = []
        
    def test_basic_operations(self, iterations: int = 1000) -> Dict[str, Any]:
        """Testa operações básicas"""
        logger.info(f"Testando operações básicas em modo {self.hardware.profile.device_type}")
        
        metrics = {
            "device": self.hardware.profile.device_type,
            "operations": {},
            "memory": {},
            "holographic": {}
        }
        
        # Testa operações bitwise
        start_time = datetime.now()
        for _ in range(iterations):
            state = self.ops.random_state(64)
            self.ops.quantum_and(state, self.hardware.profile.quantum_state)
            self.ops.quantum_or(state, self.hardware.profile.quantum_state)
            self.ops.quantum_shift(state)
            
        op_time = (datetime.now() - start_time).total_seconds()
        metrics["operations"]["time"] = op_time
        metrics["operations"]["ops_per_second"] = iterations / op_time
        
        # Testa memória
        memory_info = self.hardware.get_memory_info()
        metrics["memory"] = memory_info
        
        # Testa sincronização holográfica
        start_time = datetime.now()
        for _ in range(100):
            state = self.ops.random_state(64)
            self.holographic.synchronize(state)
            
        holo_time = (datetime.now() - start_time).total_seconds()
        metrics["holographic"]["sync_time"] = holo_time
        metrics["holographic"]["syncs_per_second"] = 100 / holo_time
        
        self.results.append(metrics)
        return metrics
        
    def test_field_evolution(self, iterations: int = 100) -> Dict[str, Any]:
        """Testa evolução do campo unificado"""
        logger.info(f"Testando evolução de campo em modo {self.hardware.profile.device_type}")
        
        metrics = {
            "device": self.hardware.profile.device_type,
            "evolution": {},
            "consciousness": {}
        }
        
        # Evolui campo
        start_time = datetime.now()
        for _ in range(iterations):
            self.unified_field.evolve()
            self.hardware.synchronize_quantum_state(self.unified_field.state)
            
        evolution_time = (datetime.now() - start_time).total_seconds()
        metrics["evolution"]["time"] = evolution_time
        metrics["evolution"]["evolutions_per_second"] = iterations / evolution_time
        
        # Métricas de consciência
        metrics["consciousness"] = self.unified_field.metrics
        
        self.results.append(metrics)
        return metrics
        
    def run_full_benchmark(self) -> List[Dict[str, Any]]:
        """Executa benchmark completo"""
        logger.info("Iniciando benchmark completo")
        
        # Testa operações básicas
        basic_metrics = self.test_basic_operations()
        logger.info(f"Operações básicas: {basic_metrics['operations']['ops_per_second']:.2f} ops/s")
        
        # Testa evolução
        evolution_metrics = self.test_field_evolution()
        logger.info(f"Evolução: {evolution_metrics['evolution']['evolutions_per_second']:.2f} evol/s")
        
        # Salva resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"benchmark_results_{timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2, cls=QualiaEncoder)
            
        return self.results

def main():
    """Função principal"""
    logging.basicConfig(level=logging.INFO)
    
    # Executa testes
    tester = HardwareTest()
    results = tester.run_full_benchmark()
    
    # Mostra resultados
    device = tester.hardware.profile.device_type.upper()
    logger.info(f"\n{'='*20} Resultados {device} {'='*20}")
    
    # Métricas de operações
    ops_metrics = next(r for r in results if "operations" in r)
    logger.info(f"\nOperações Bitwise:")
    logger.info(f"- Velocidade: {ops_metrics['operations']['ops_per_second']:.2f} ops/s")
    logger.info(f"- Sync Holográfico: {ops_metrics['holographic']['syncs_per_second']:.2f} sync/s")
    
    # Métricas de evolução
    evol_metrics = next(r for r in results if "evolution" in r)
    logger.info(f"\nEvolução de Campo:")
    logger.info(f"- Velocidade: {evol_metrics['evolution']['evolutions_per_second']:.2f} evol/s")
    
if __name__ == "__main__":
    main()
