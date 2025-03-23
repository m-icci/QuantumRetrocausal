#!/usr/bin/env python3
"""
Wrapper para o QuantumCNNAnalyzer para integração com o sistema QUALIA.
Facilita a conversão de dados e a interação com o modelo de consciência quântica.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List

# Importar o QuantumCNNAnalyzer do módulo quantum_cnn.py
from quantum_cnn import QuantumCNNAnalyzer

class QCNNWrapper:
    """
    Wrapper para o QuantumCNNAnalyzer que facilita seu uso no sistema QUALIA.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o wrapper para o QuantumCNNAnalyzer.
        
        Args:
            config: Configurações opcionais para o QuantumCNNAnalyzer.
        """
        self.analyzer = QuantumCNNAnalyzer(config=config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def process(self, 
               input_data: np.ndarray, 
               quantum_state: Optional[np.ndarray] = None, 
               consciousness_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa dados de mercado usando o QuantumCNNAnalyzer.
        
        Args:
            input_data: Dados de mercado formatados como imagem (shape: [batch, channels, height, width]).
            quantum_state: Estado quântico opcional (vetor de flutuantes).
            consciousness_state: Estado de consciência opcional (dicionário com 'coherence', 'complexity', etc.).
            
        Returns:
            Dicionário com resultados da análise quântica.
        """
        # Validar e preparar os dados
        if input_data.ndim < 4:
            # Converter para o formato esperado (batch, channels, height, width)
            if input_data.ndim == 3:
                input_data = input_data.reshape(1, *input_data.shape)
            else:
                raise ValueError(f"Formato de dados inválido. Esperado 3D ou 4D, recebido {input_data.ndim}D")
        
        # Se quantum_state não for fornecido, criar um estado padrão
        if quantum_state is None:
            quantum_state = np.zeros(16)  # Dimensão padrão para estado quântico
        
        # Preparar o estado de consciência
        if consciousness_state is None:
            consciousness_state = {
                'coherence': 0.5,
                'complexity': 0.3,
                'batch_size': input_data.shape[0]
            }
        else:
            if 'batch_size' not in consciousness_state:
                consciousness_state['batch_size'] = input_data.shape[0]
        
        # Converter dados para tensores
        input_tensor = torch.from_numpy(input_data).float().to(self.device)
        
        # Processar com o QuantumCNNAnalyzer
        results = self.analyzer.analyze_patterns(
            input_tensor,
            quantum_state,
            consciousness_state
        )
        
        return results
    
    def process_candlestick_data(self, 
                                candlesticks: np.ndarray, 
                                window_size: int = 32, 
                                quantum_state: Optional[np.ndarray] = None,
                                consciousness_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa dados de candlestick convertendo-os em uma imagem para o QCNN.
        
        Args:
            candlesticks: Array de candlesticks [open, high, low, close, volume].
            window_size: Tamanho da janela para criar a imagem.
            quantum_state: Estado quântico opcional.
            consciousness_state: Estado de consciência opcional.
            
        Returns:
            Resultados da análise quântica.
        """
        # Converter candlesticks em imagem (3 canais: preço, volume, volatilidade)
        image = self._convert_candlesticks_to_image(candlesticks, window_size)
        return self.process(image, quantum_state, consciousness_state)
    
    def _convert_candlesticks_to_image(self, 
                                      candlesticks: np.ndarray, 
                                      window_size: int = 32) -> np.ndarray:
        """
        Converte dados de candlestick em uma imagem para processamento pelo QCNN.
        
        Args:
            candlesticks: Array de candlesticks [open, high, low, close, volume].
            window_size: Tamanho da janela para criar a imagem.
            
        Returns:
            Imagem em formato NumPy (1, 3, window_size, window_size).
        """
        if len(candlesticks) < window_size:
            pad_size = window_size - len(candlesticks)
            candlesticks = np.pad(candlesticks, ((pad_size, 0), (0, 0)), mode='edge')
        
        candles = candlesticks[-window_size:]
        # Normalizar preços (utilizando apenas o preço de fechamento)
        normalized_prices = (candles[:, :4] - candles[:, :4].min()) / (candles[:, :4].max() - candles[:, :4].min() + 1e-8)
        normalized_volume = (candles[:, 4] - candles[:, 4].min()) / (candles[:, 4].max() - candles[:, 4].min() + 1e-8)
        volatility = candles[:, 1] - candles[:, 2]  # High - Low
        normalized_volatility = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
        
        price_map = self._create_heatmap(normalized_prices[:, 3], window_size)  # Usar o close price
        volume_map = self._create_heatmap(normalized_volume, window_size)
        volatility_map = self._create_heatmap(normalized_volatility, window_size)
        
        image = np.stack([price_map, volume_map, volatility_map], axis=0)
        return image.reshape(1, 3, window_size, window_size)
    
    def _create_heatmap(self, values: np.ndarray, size: int) -> np.ndarray:
        """
        Cria um mapa de calor a partir de valores unidimensionais.
        
        Args:
            values: Array de valores.
            size: Tamanho do mapa de calor.
            
        Returns:
            Mapa de calor como array 2D.
        """
        heatmap = np.zeros((size, size))
        for i, val in enumerate(values):
            heatmap[i, :] = val
        return heatmap 