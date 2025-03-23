"""
Sistema de Trading em Tempo Real para QUALIA
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from .market_api import MarketAPI
from .quantum_pattern_analyzer_v2 import QuantumPatternAnalyzer
from .consciousness import MarketConsciousness

# Configure logging
logger = logging.getLogger("real_time_trader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class RealTimeTrader:
    """
    Trader em tempo real com proteção quântica
    """
    
    def __init__(self, 
                 market_api: MarketAPI,
                 max_memory_capacity: int = 2048,
                 min_confidence: float = 0.7,
                 max_position_size: float = 0.1):
        """
        Inicializa o trader
        
        Args:
            market_api: API do mercado
            max_memory_capacity: Capacidade máxima da memória
            min_confidence: Confiança mínima para operar
            max_position_size: Tamanho máximo da posição
        """
        logger.info("Inicializando RealTimeTrader")
        
        # Componentes
        self.market_api = market_api
        self.analyzer = QuantumPatternAnalyzer(
            memory_dimension=max_memory_capacity,
            max_memory_states=max_memory_capacity,
            similarity_threshold=min_confidence
        )
        self.consciousness = MarketConsciousness()
        
        # Parâmetros
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        
        # Estado
        self.current_state = {
            'position': None,
            'last_trade': None,
            'balance': 0.0,
            'trades': []
        }
        
        logger.info("Trader inicializado com sucesso")
    
    def analyze_trading_opportunity(self,
                                  symbol: str,
                                  price: float,
                                  volume: float) -> Dict[str, Any]:
        """
        Analisa oportunidade de trading
        
        Args:
            symbol: Par de trading
            price: Preço atual
            volume: Volume atual
            
        Returns:
            Resultado da análise
        """
        try:
            # Preparar dados
            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'timestamp': datetime.now().timestamp()
            }
            
            # Analisar padrão
            analysis = self.analyzer.analyze_pattern(market_data)
            
            # Validar proteção
            protection = self._validate_quantum_protection(analysis)
            if not protection['is_safe']:
                logger.warning(f"Proteção quântica ativada: {protection['reason']}")
                return self._get_default_opportunity()
            
            # Calcular tamanho da posição
            position_size = self._calculate_position_size(
                price,
                analysis['recommendation']['confidence']
            )
            
            # Preparar resultado
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': price,
                'volume': volume,
                'analysis': analysis,
                'protection': protection,
                'position_size': position_size,
                'action': analysis['recommendation']['signal'],
                'confidence': analysis['recommendation']['confidence']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise: {str(e)}")
            return self._get_default_opportunity()
    
    def execute_trade(self,
                     opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa trade
        
        Args:
            opportunity: Oportunidade de trading
            
        Returns:
            Resultado da execução
        """
        try:
            # Validar oportunidade
            if not self._validate_opportunity(opportunity):
                logger.warning("Oportunidade inválida")
                return self._get_default_execution()
            
            # Extrair dados
            symbol = opportunity['symbol']
            action = opportunity['action']
            price = opportunity['price']
            size = opportunity['position_size']
            
            # Executar ordem
            if action == 'buy':
                order = self.market_api.create_buy_order(
                    symbol=symbol,
                    quantity=size,
                    price=price
                )
                
            elif action == 'sell':
                order = self.market_api.create_sell_order(
                    symbol=symbol,
                    quantity=size,
                    price=price
                )
                
            else:
                logger.info("Nenhuma ação necessária")
                return self._get_default_execution()
            
            # Atualizar estado
            self.current_state.update({
                'position': {
                    'symbol': symbol,
                    'side': action,
                    'size': size,
                    'price': price
                },
                'last_trade': datetime.now().isoformat(),
                'trades': self.current_state['trades'] + [{
                    'symbol': symbol,
                    'action': action,
                    'price': price,
                    'size': size,
                    'timestamp': datetime.now().isoformat()
                }]
            })
            
            # Preparar resultado
            result = {
                'success': True,
                'order': order,
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'symbol': symbol,
                'price': price,
                'size': size
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na execução: {str(e)}")
            return self._get_default_execution()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna estado atual do trader
        
        Returns:
            Estado atual
        """
        return {
            'position': self.current_state['position'],
            'last_trade': self.current_state['last_trade'],
            'balance': float(self.current_state['balance']),
            'total_trades': len(self.current_state['trades']),
            'analyzer_state': self.analyzer.get_current_state()
        }
    
    def _validate_quantum_protection(self,
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida proteção quântica
        
        Args:
            analysis: Análise do padrão
            
        Returns:
            Resultado da validação
        """
        try:
            # Extrair métricas
            coherence = analysis['metrics']['coherence']
            entropy = analysis['metrics']['entropy']
            confidence = analysis['recommendation']['confidence']
            
            # Validar coerência
            if coherence < 0.5:
                return {
                    'is_safe': False,
                    'reason': 'Baixa coerência quântica'
                }
            
            # Validar entropia
            if entropy > 0.8:
                return {
                    'is_safe': False,
                    'reason': 'Alta entropia do sistema'
                }
            
            # Validar confiança
            if confidence < self.min_confidence:
                return {
                    'is_safe': False,
                    'reason': 'Baixa confiança na previsão'
                }
            
            return {
                'is_safe': True,
                'reason': None
            }
        
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return {
                'is_safe': False,
                'reason': 'Erro na validação'
            }
    
    def _calculate_position_size(self,
                               price: float,
                               confidence: float) -> float:
        """
        Calcula tamanho da posição
        
        Args:
            price: Preço atual
            confidence: Confiança na previsão
            
        Returns:
            Tamanho da posição
        """
        try:
            # Obter saldo disponível
            balance = self.market_api.get_balance()
            
            # Calcular tamanho base
            base_size = balance * self.max_position_size
            
            # Ajustar por confiança
            adjusted_size = base_size * confidence
            
            # Calcular quantidade
            quantity = adjusted_size / price
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erro no cálculo: {str(e)}")
            return 0.0
    
    def _validate_opportunity(self,
                            opportunity: Dict[str, Any]) -> bool:
        """
        Valida oportunidade de trading
        
        Args:
            opportunity: Oportunidade a validar
            
        Returns:
            Se a oportunidade é válida
        """
        try:
            # Verificar campos obrigatórios
            required = ['symbol', 'action', 'price', 'position_size']
            if not all(k in opportunity for k in required):
                logger.warning("Oportunidade não contém campos obrigatórios")
                return False
            
            # Verificar ação
            if opportunity['action'] not in ['buy', 'sell', 'hold']:
                logger.warning("Ação inválida")
                return False
            
            # Verificar preço
            if not isinstance(opportunity['price'], (int, float)):
                logger.warning("Preço inválido")
                return False
            
            # Verificar tamanho
            if not isinstance(opportunity['position_size'], (int, float)):
                logger.warning("Tamanho inválido")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def _get_default_opportunity(self) -> Dict[str, Any]:
        """
        Retorna oportunidade padrão
        
        Returns:
            Oportunidade padrão
        """
        return {
            'symbol': None,
            'timestamp': datetime.now().isoformat(),
            'price': 0.0,
            'volume': 0.0,
            'analysis': None,
            'protection': {
                'is_safe': False,
                'reason': 'Análise padrão'
            },
            'position_size': 0.0,
            'action': 'hold',
            'confidence': 0.0
        }
    
    def _get_default_execution(self) -> Dict[str, Any]:
        """
        Retorna execução padrão
        
        Returns:
            Execução padrão
        """
        return {
            'success': False,
            'order': None,
            'timestamp': datetime.now().isoformat(),
            'action': 'hold',
            'symbol': None,
            'price': 0.0,
            'size': 0.0
        }
