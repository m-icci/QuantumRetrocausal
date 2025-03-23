#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qualia.quantum.merge import QuantumMergeSimulator
from qualia.utils.logging import setup_logger

def run_quantum_merge_test():
    """
    Executa teste do sistema de merge quântico com monitoramento
    """
    # Configurar logger
    logger = setup_logger('quantum_merge_test')
    logger.info("Iniciando teste do sistema de merge quântico...")
    
    # Inicializar simulador com monitoramento
    simulator = QuantumMergeSimulator(file_merge_mode=True)
    
    # Definir diretórios para teste
    source_dir = "/Users/infrastructure/Desktop/QC"
    target_dir = "/Users/infrastructure/Documents/GitHub/QuantumConsciousness/qualia/quantum/merge"
    
    logger.info(f"Configuração do teste:")
    logger.info(f"- Fonte: {source_dir}")
    logger.info(f"- Destino: {target_dir}")
    
    try:
        # Registrar início do teste
        start_time = time.time()
        
        # Executar merge
        simulator.merge_directories(source_dir, target_dir)
        
        # Calcular duração
        duration = time.time() - start_time
        
        # Registrar sucesso
        simulator.monitor.record_merge_attempt(success=True, duration=duration)
        
        logger.info("\nMerge concluído com sucesso!")
        logger.info(f"Duração: {duration:.2f} segundos")
        
        # Obter métricas
        metrics = simulator.monitor.get_metrics_summary()
        logger.info("\nMétricas do sistema:")
        logger.info(f"- Coerência média: {metrics.get('coherence_mean', 0):.3f}")
        logger.info(f"- Desvio padrão da coerência: {metrics.get('coherence_std', 0):.3f}")
        logger.info(f"- Entropia média: {metrics.get('entropy_mean', 0):.3f}")
        logger.info(f"- Taxa de sucesso: {metrics.get('success_rate', 0)*100:.1f}%")
        
        # Verificar anomalias
        anomalies = simulator.monitor.detect_anomalies()
        if anomalies:
            logger.warning("\nAnomalias detectadas:")
            for anomaly in anomalies:
                logger.warning(
                    f"- {anomaly['metric']}: {anomaly['value']:.3f} "
                    f"(desvio: {anomaly['deviation']:.1f}σ)"
                )
        
    except Exception as e:
        # Registrar falha
        simulator.monitor.record_merge_attempt(success=False, duration=time.time() - start_time)
        logger.error(f"\nErro durante o teste: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_quantum_merge_test()
