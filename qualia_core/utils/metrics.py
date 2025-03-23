"""
Métricas e Cálculos Utilitários
------------------------------
"""

def calculate_hashrate(total_hashes: int, elapsed_time: float) -> float:
    """
    Calcula o hashrate em MH/s
    
    Args:
        total_hashes: Total de hashes calculados
        elapsed_time: Tempo decorrido em segundos
        
    Returns:
        float: Hashrate em MH/s
    """
    if elapsed_time <= 0:
        return 0.0
        
    return (total_hashes / elapsed_time) / 1_000_000  # Converte para MH/s
