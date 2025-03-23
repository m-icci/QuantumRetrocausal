"""
Sistema de Trading Quântico Integrado
Utiliza análise de padrões quânticos para execução de trades
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
from quantum_trading.quantum_pattern_analyzer import QuantumPatternAnalyzer
from quantum_trading.consciousness import MarketConsciousness

logger = logging.getLogger("quantum_trader")

class QuantumTrader:
    """
    Sistema de trading que utiliza análise quântica para execução de operações
    """
    def __init__(
            self,
            market_api,
            max_memory_capacity: int = 1000,
            min_confidence: float = 0.7,
            max_position_size: float = 0.05  # 5% do balanço
        ):
        self.market_api = market_api
        self.analyzer = QuantumPatternAnalyzer(
            max_memory_capacity=max_memory_capacity
        )
        self.consciousness = MarketConsciousness()
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.active_positions = {}
        
        logger.info("Inicializando QuantumTrader com Consciência de Mercado")
        
    def _validate_consciousness(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida condições de consciência do mercado"""
        try:
            consciousness_field = self.consciousness.calculate_consciousness_field(
                data['close'].values
            )
            
            market_prediction = self.consciousness.get_market_prediction(symbol)
            
            return {
                'field': consciousness_field,
                'prediction': market_prediction,
                'is_valid': (
                    consciousness_field['coherence'] > 0.6 and
                    consciousness_field['quantum_entropy'] < 0.7 and
                    market_prediction['confidence'] > 0.65
                )
            }
        except Exception as e:
            logger.error(f"Erro na validação da consciência: {e}")
            return {
                'field': {},
                'prediction': {},
                'is_valid': False
            }

    def analyze_trading_opportunity(
            self,
            symbol: str,
            timeframe: str = '1h',
            window_size: int = 24
        ) -> Dict[str, Any]:
        """
        Analisa oportunidade de trading usando análise quântica
        """
        try:
            # Obter dados do mercado
            data = self.market_api.get_ohlcv(symbol, timeframe, limit=window_size)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Validar consciência do mercado
            consciousness_check = self._validate_consciousness(symbol, data)
            if not consciousness_check['is_valid']:
                return {
                    'should_trade': False,
                    'reason': 'invalid_consciousness',
                    'consciousness': consciousness_check
                }
            
            # Obter sinais de trading
            signals = self.analyzer.get_trading_signals(symbol, data)
            
            # Validar sinal com consciência
            market_prediction = consciousness_check['prediction']
            if signals['signal'] != market_prediction['signal']:
                return {
                    'should_trade': False,
                    'reason': 'signal_consciousness_mismatch',
                    'signals': signals,
                    'consciousness': consciousness_check
                }
            
            if not signals['signal']:
                return {
                    'should_trade': False,
                    'reason': 'no_signal',
                    'analysis': signals
                }
            
            # Verificar confiança mínima ajustada pela consciência
            adjusted_confidence = signals['confidence'] * market_prediction['confidence']
            if adjusted_confidence < self.min_confidence:
                return {
                    'should_trade': False,
                    'reason': 'low_confidence',
                    'analysis': signals,
                    'adjusted_confidence': adjusted_confidence
                }
            
            # Obter balanço disponível
            balance = self.market_api.get_balance('USDT')
            current_price = self.market_api.get_price(symbol)
            
            # Otimizar tamanho da posição
            position = self.analyzer.optimize_position_sizing(
                symbol=symbol,
                available_balance=balance,
                current_price=current_price,
                analysis=signals
            )
            
            # Ajustar tamanho baseado na força da consciência
            position['quantity'] *= consciousness_check['field']['coherence']
            
            # Verificar se o tamanho da posição é válido
            if position['position_size'] <= 0:
                return {
                    'should_trade': False,
                    'reason': 'invalid_position_size',
                    'analysis': signals,
                    'position': position
                }
            
            return {
                'should_trade': True,
                'signal': signals['signal'],
                'analysis': signals,
                'position': position,
                'consciousness': consciousness_check,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar oportunidade: {e}")
            return {
                'should_trade': False,
                'reason': 'error',
                'error': str(e)
            }
    
    def execute_trade(
            self,
            symbol: str,
            opportunity: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Executa uma operação de trading com proteção quântica
        """
        try:
            if not opportunity['should_trade']:
                return {
                    'success': False,
                    'reason': opportunity['reason']
                }
            
            signal = opportunity['signal']
            position = opportunity['position']
            analysis = opportunity['analysis']
            
            # Criar ordem com proteção
            order = {
                'symbol': symbol,
                'side': signal,
                'type': 'market',
                'amount': position['quantity']
            }
            
            # Adicionar stop loss e take profit
            protection = analysis['protection']
            current_price = self.market_api.get_price(symbol)
            
            if signal == 'buy':
                stop_loss = current_price * (1 - (0.02 * protection['stop_multiplier']))
                take_profit = current_price * (1 + (0.03 * protection['take_multiplier']))
            else:
                stop_loss = current_price * (1 + (0.02 * protection['stop_multiplier']))
                take_profit = current_price * (1 - (0.03 * protection['take_multiplier']))
            
            order['stop_loss'] = stop_loss
            order['take_profit'] = take_profit
            
            # Executar ordem
            result = self.market_api.create_order(**order)
            
            if result and 'id' in result:
                # Registrar posição ativa
                self.active_positions[result['id']] = {
                    'symbol': symbol,
                    'entry_price': current_price,
                    'quantity': position['quantity'],
                    'side': signal,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'protection': protection,
                    'timestamp': datetime.now().isoformat()
                }
                
                return {
                    'success': True,
                    'order': result,
                    'position': self.active_positions[result['id']]
                }
            
            return {
                'success': False,
                'reason': 'order_failed',
                'order_result': result
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return {
                'success': False,
                'reason': 'error',
                'error': str(e)
            }
    
    def monitor_positions(self) -> Dict[str, Any]:
        """
        Monitora posições ativas e ajusta proteções
        """
        try:
            updates = {}
            
            for position_id, position in self.active_positions.items():
                symbol = position['symbol']
                current_price = self.market_api.get_price(symbol)
                
                # Analisar mercado atual
                data = self.market_api.get_ohlcv(symbol, '1h', limit=24)
                analysis = self.analyzer.analyze_market_pattern(symbol, data)
                
                # Calcular novos níveis de proteção
                protection = analysis['protection']
                
                if position['side'] == 'buy':
                    new_stop = current_price * (1 - (0.02 * protection['stop_multiplier']))
                    if new_stop > position['stop_loss']:  # Trailing stop
                        updates[position_id] = {
                            'type': 'stop_update',
                            'old_stop': position['stop_loss'],
                            'new_stop': new_stop,
                            'reason': 'trailing_stop'
                        }
                        position['stop_loss'] = new_stop
                else:
                    new_stop = current_price * (1 + (0.02 * protection['stop_multiplier']))
                    if new_stop < position['stop_loss']:  # Trailing stop
                        updates[position_id] = {
                            'type': 'stop_update',
                            'old_stop': position['stop_loss'],
                            'new_stop': new_stop,
                            'reason': 'trailing_stop'
                        }
                        position['stop_loss'] = new_stop
                
                # Verificar condições de saída
                if protection['risk_level'] > 0.8:  # Alto risco
                    updates[position_id] = {
                        'type': 'exit_signal',
                        'reason': 'high_risk',
                        'risk_level': protection['risk_level']
                    }
            
            return {
                'updates': updates,
                'positions': self.active_positions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao monitorar posições: {e}")
            return {
                'updates': {},
                'positions': self.active_positions,
                'error': str(e)
            }
    
    def close_position(
            self,
            position_id: str,
            reason: str = 'manual'
        ) -> Dict[str, Any]:
        """
        Fecha uma posição específica
        """
        try:
            if position_id not in self.active_positions:
                return {
                    'success': False,
                    'reason': 'position_not_found'
                }
            
            position = self.active_positions[position_id]
            
            # Criar ordem de fechamento
            order = {
                'symbol': position['symbol'],
                'side': 'sell' if position['side'] == 'buy' else 'buy',
                'type': 'market',
                'amount': position['quantity']
            }
            
            # Executar ordem
            result = self.market_api.create_order(**order)
            
            if result and 'id' in result:
                # Remover da lista de posições ativas
                closed_position = self.active_positions.pop(position_id)
                
                return {
                    'success': True,
                    'order': result,
                    'position': closed_position,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'success': False,
                'reason': 'close_failed',
                'order_result': result
            }
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
            return {
                'success': False,
                'reason': 'error',
                'error': str(e)
            } 