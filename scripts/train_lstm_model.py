#!/usr/bin/env python
"""
Script para treinar o modelo LSTM de predição de momentos ótimos de entrada
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Adicionar diretório principal ao path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Importar módulos QUALIA
from quantum_trading.neural.lstm_predictor import LSTMPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(root_dir, 'logs', 'lstm_training.log'))
    ]
)

logger = logging.getLogger("lstm_training")

def load_historical_data(data_path, min_records=1000):
    """
    Carrega dados históricos para treinamento
    
    Args:
        data_path: Caminho para o arquivo de dados
        min_records: Número mínimo de registros requeridos
        
    Returns:
        DataFrame com os dados históricos
    """
    logger.info(f"Carregando dados de {data_path}")
    
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {data_path}")
        
        logger.info(f"Carregados {len(df)} registros")
        
        if len(df) < min_records:
            logger.warning(f"Poucos registros ({len(df)}). O ideal é ter pelo menos {min_records}.")
        
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None

def generate_synthetic_data(num_records=5000):
    """
    Gera dados sintéticos para treinamento quando não há dados reais suficientes
    
    Args:
        num_records: Número de registros a gerar
        
    Returns:
        DataFrame com dados sintéticos
    """
    logger.info(f"Gerando {num_records} registros sintéticos para treinamento")
    
    # Timestamp base
    base_time = datetime.now() - timedelta(days=30)
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(num_records)]
    
    # Gerar dados
    data = {
        'timestamp': timestamps,
        'spread': np.random.uniform(0.001, 0.03, num_records),
        'volume_a': np.random.uniform(1000, 100000, num_records),
        'volume_b': np.random.uniform(1000, 100000, num_records),
        'entropy': np.random.uniform(0.1, 0.9, num_records),
        'fractal_dimension': np.random.uniform(1.1, 1.9, num_records),
        'volatility_a': np.random.uniform(0.01, 0.1, num_records),
        'volatility_b': np.random.uniform(0.01, 0.1, num_records),
    }
    
    # Criar tendências artificiais para tornar os dados mais realistas
    for i in range(1, num_records):
        random_walk = np.random.uniform(-0.05, 0.05)
        data['spread'][i] = max(0.001, min(0.03, data['spread'][i-1] + random_walk))
        
        data['entropy'][i] = max(0.1, min(0.9, data['entropy'][i-1] + np.random.uniform(-0.02, 0.02)))
        
        data['fractal_dimension'][i] = max(1.1, min(1.9, data['fractal_dimension'][i-1] + np.random.uniform(-0.01, 0.01)))
    
    # Gerar target (profitable) - uma combinação de features que simula resultados realistas
    # Exemplo: operações tendem a ser lucrativas quando spread é alto e volatilidade baixa
    profitable = []
    for i in range(num_records):
        # Fórmula sintética: spread alto e volatilidade baixa indicam maior chance de lucro
        prob = (data['spread'][i] / 0.03) * 0.7 + (1 - data['volatility_a'][i] / 0.1) * 0.3
        profitable.append(1 if np.random.random() < prob else 0)
    
    data['profitable'] = profitable
    
    # Adicionar coluna de lucro sintético
    profit = []
    for i in range(num_records):
        if profitable[i] == 1:
            # Lucro positivo quando profitable=1, relacionado ao spread
            profit.append(data['spread'][i] * np.random.uniform(50, 200))
        else:
            # Prejuízo ou lucro mínimo quando profitable=0
            profit.append(np.random.uniform(-1.0, 0.5))
    
    data['profit'] = profit
    
    # Converter para DataFrame
    df = pd.DataFrame(data)
    
    logger.info("Dados sintéticos gerados com sucesso")
    return df

def train_lstm_model(data_path=None, 
                    model_path=None, 
                    lookback_periods=10,
                    epochs=100,
                    batch_size=32,
                    synthetic=False,
                    synthetic_records=5000):
    """
    Treina o modelo LSTM
    
    Args:
        data_path: Caminho para os dados históricos
        model_path: Caminho para salvar o modelo
        lookback_periods: Número de períodos para análise retrospectiva
        epochs: Número máximo de épocas para treinamento
        batch_size: Tamanho do batch para treinamento
        synthetic: Se True, usa dados sintéticos
        synthetic_records: Número de registros sintéticos a gerar
    
    Returns:
        Modelo treinado e métricas de performance
    """
    # Definir caminho do modelo
    if model_path is None:
        model_path = os.path.join(root_dir, 'models', 'lstm_predictor.h5')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Carregar dados
    if synthetic or data_path is None:
        df = generate_synthetic_data(synthetic_records)
    else:
        df = load_historical_data(data_path)
        if df is None:
            logger.warning("Falha ao carregar dados. Usando dados sintéticos.")
            df = generate_synthetic_data(synthetic_records)
    
    # Inicializar modelo
    lstm_predictor = LSTMPredictor(
        model_path=model_path,
        lookback_periods=lookback_periods,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Treinar modelo
    logger.info("Iniciando treinamento do modelo LSTM...")
    metrics = lstm_predictor.train(df)
    
    # Salvar métricas
    metrics_path = model_path.replace('.h5', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Treinamento concluído. Métricas: {metrics}")
    logger.info(f"Modelo salvo em {model_path}")
    logger.info(f"Métricas salvas em {metrics_path}")
    
    return lstm_predictor, metrics

def main():
    parser = argparse.ArgumentParser(description='Treinamento de modelo LSTM para QUALIA')
    parser.add_argument('--data-path', type=str, 
                        help='Caminho para os dados históricos (CSV, JSON ou Parquet)')
    parser.add_argument('--model-path', type=str, 
                        help='Caminho para salvar o modelo treinado')
    parser.add_argument('--lookback', type=int, default=10,
                        help='Número de períodos para análise retrospectiva')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número máximo de épocas para treinamento')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help='Usar dados sintéticos para treinamento')
    parser.add_argument('--synthetic-records', type=int, default=5000,
                        help='Número de registros sintéticos a gerar')
    
    args = parser.parse_args()
    
    try:
        train_lstm_model(
            data_path=args.data_path,
            model_path=args.model_path,
            lookback_periods=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size,
            synthetic=args.synthetic,
            synthetic_records=args.synthetic_records
        )
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 