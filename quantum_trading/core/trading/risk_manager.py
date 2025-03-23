"""
Gerenciador de risco
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ...data.data_loader import DataLoader

logger = logging.getLogger(__name__)

class RiskManager:
    """Gerenciador de risco"""
    
    def __init__(self, config: Dict):
        """
        Inicializa gerenciador.
        
        Args:
            config: Configuração.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        
        # Métricas
        self.metrics = {
            'daily_pnl': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_duration': 0,
            'avg_profit_per_trade': 0,
            'avg_loss_per_trade': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_consecutive_losses': 0
        }
        
        # Histórico
        self.trade_history = []
        self.daily_history = []
        self.position_history = []
        
        # Cache
        self.volatility_cache = {}
        self.correlation_cache = {}
        
        # Limites dinâmicos
        self.dynamic_limits = {
            'max_position_size': self.config['risk']['max_position_size'],
            'max_daily_loss': self.config['risk']['max_daily_loss'],
            'max_drawdown': self.config['risk']['max_drawdown']
        }
        
    async def initialize(self) -> None:
        """Inicializa gerenciador"""
        try:
            logger.info("Inicializando gerenciador de risco")
            await self.data_loader.connect()
            
            # Carrega histórico
            await self._load_history()
            
            # Inicializa métricas
            await self.update()
            
            # Ajusta limites iniciais
            self._adjust_risk_limits()
            
            logger.info("Gerenciador de risco inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar gerenciador de risco: {str(e)}")
            raise
            
    async def update(self) -> None:
        """Atualiza métricas"""
        try:
            # Atualiza histórico diário
            await self._update_daily_history()
            
            # Atualiza métricas
            self._update_metrics()
            
            # Ajusta limites dinâmicos
            self._adjust_risk_limits()
            
            # Atualiza cache
            await self._update_cache()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas: {str(e)}")
            raise
            
    def check_risk_limits(self) -> bool:
        """
        Verifica limites de risco.
        
        Returns:
            True se dentro dos limites.
        """
        try:
            # Verifica perda diária máxima
            if self.metrics['daily_pnl'] <= -self.dynamic_limits['max_daily_loss']:
                logger.warning(f"Perda diária máxima atingida: {self.metrics['daily_pnl']:.2%}")
                return False
                
            # Verifica drawdown máximo
            if self.metrics['max_drawdown'] >= self.dynamic_limits['max_drawdown']:
                logger.warning(f"Drawdown máximo atingido: {self.metrics['max_drawdown']:.2%}")
                return False
                
            # Verifica perdas consecutivas
            if self.metrics['max_consecutive_losses'] >= 5:
                logger.warning(f"Máximo de perdas consecutivas atingido: {self.metrics['max_consecutive_losses']}")
                return False
                
            # Verifica Sharpe ratio
            min_sharpe = 0.5  # Mínimo aceitável para scalping
            if self.metrics['sharpe_ratio'] < min_sharpe:
                logger.warning(f"Sharpe ratio abaixo do mínimo: {self.metrics['sharpe_ratio']:.2f}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar limites de risco: {str(e)}")
            return False
            
    def calculate_position_size(self, symbol: str, direction: str) -> float:
        """
        Calcula tamanho da posição.
        
        Args:
            symbol: Símbolo.
            direction: Direção.
            
        Returns:
            Tamanho da posição.
        """
        try:
            # Obtém configuração
            max_position_size = self.dynamic_limits['max_position_size']
            min_trade_size = self.config['scalping']['min_trade_size']
            
            # Calcula tamanho base
            base_size = max_position_size
            
            # Ajusta por volatilidade
            volatility = self._calculate_volatility(symbol)
            if volatility > 0:
                volatility_factor = 1 / (1 + volatility)
                base_size *= volatility_factor
                
            # Ajusta por correlação
            correlation = self._calculate_market_correlation(symbol)
            if correlation is not None:
                correlation_factor = 1 - abs(correlation)
                base_size *= correlation_factor
                
            # Ajusta por drawdown
            if self.metrics['max_drawdown'] > 0:
                drawdown_factor = 1 - self.metrics['max_drawdown']
                base_size *= drawdown_factor
                
            # Ajusta por win rate
            if self.metrics['win_rate'] > 0:
                winrate_factor = self.metrics['win_rate']
                base_size *= winrate_factor
                
            # Ajusta por profit factor
            if self.metrics['profit_factor'] > 1:
                profit_factor = min(self.metrics['profit_factor'], 2)
                base_size *= (profit_factor / 2)
                
            # Limita ao mínimo
            size = max(base_size, min_trade_size)
            
            # Limita ao máximo
            size = min(size, max_position_size)
            
            # Arredonda para precisão do ativo
            size = round(size, 8)  # 8 casas decimais para crypto
            
            return size
            
        except Exception as e:
            logger.error(f"Erro ao calcular tamanho da posição: {str(e)}")
            return min_trade_size
            
    def calculate_pnl(self, position: Dict, current_price: float) -> float:
        """
        Calcula P&L.
        
        Args:
            position: Posição.
            current_price: Preço atual.
            
        Returns:
            P&L.
        """
        try:
            # Calcula P&L bruto
            if position['direction'] == 'long':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                
            # Calcula custos totais
            entry_costs = position.get('costs', 0)
            exit_costs = self._calculate_exit_costs(position, current_price)
            total_costs = entry_costs + exit_costs
            
            # Subtrai custos
            net_pnl = pnl - total_costs
            
            return net_pnl
            
        except Exception as e:
            logger.error(f"Erro ao calcular P&L: {str(e)}")
            return 0
            
    def _calculate_exit_costs(self, position: Dict, current_price: float) -> float:
        """
        Calcula custos de saída.
        
        Args:
            position: Posição.
            current_price: Preço atual.
            
        Returns:
            Custos de saída.
        """
        try:
            # Calcula valor
            value = current_price * position['size']
            
            # Calcula taxa
            fee_rate = self.config['scalping']['exchange_fee']
            fee = value * fee_rate
            
            # Calcula slippage
            slippage_rate = self.config['scalping']['slippage']
            slippage = value * slippage_rate
            
            # Retorna custos totais
            return fee + slippage
            
        except Exception as e:
            logger.error(f"Erro ao calcular custos de saída: {str(e)}")
            return 0
            
    def _adjust_risk_limits(self) -> None:
        """Ajusta limites dinâmicos"""
        try:
            # Ajusta por win rate
            if self.metrics['win_rate'] < 0.4:  # Win rate muito baixo
                self.dynamic_limits['max_position_size'] *= 0.8
                self.dynamic_limits['max_daily_loss'] *= 0.8
            elif self.metrics['win_rate'] > 0.6:  # Win rate bom
                self.dynamic_limits['max_position_size'] *= 1.2
                self.dynamic_limits['max_daily_loss'] *= 1.2
                
            # Ajusta por profit factor
            if self.metrics['profit_factor'] < 1.2:  # Profit factor baixo
                self.dynamic_limits['max_position_size'] *= 0.8
            elif self.metrics['profit_factor'] > 1.5:  # Profit factor bom
                self.dynamic_limits['max_position_size'] *= 1.2
                
            # Ajusta por drawdown
            if self.metrics['max_drawdown'] > self.config['risk']['max_drawdown'] * 0.8:
                self.dynamic_limits['max_position_size'] *= 0.8
                self.dynamic_limits['max_daily_loss'] *= 0.8
                
            # Limita ajustes
            for key, value in self.dynamic_limits.items():
                original = self.config['risk'][key]
                min_value = original * 0.5  # Não reduz mais que 50%
                max_value = original * 1.5  # Não aumenta mais que 50%
                self.dynamic_limits[key] = min(max(value, min_value), max_value)
                
        except Exception as e:
            logger.error(f"Erro ao ajustar limites de risco: {str(e)}")
            
    async def _update_cache(self) -> None:
        """Atualiza cache"""
        try:
            # Limpa cache antigo
            current_time = datetime.now()
            cache_ttl = timedelta(minutes=5)  # 5 minutos
            
            # Limpa volatilidade
            self.volatility_cache = {
                k: v for k, v in self.volatility_cache.items()
                if current_time - v['timestamp'] < cache_ttl
            }
            
            # Limpa correlação
            self.correlation_cache = {
                k: v for k, v in self.correlation_cache.items()
                if current_time - v['timestamp'] < cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Erro ao atualizar cache: {str(e)}")
            
    def _calculate_market_correlation(self, symbol: str) -> Optional[float]:
        """
        Calcula correlação com mercado.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Correlação ou None.
        """
        try:
            # Verifica cache
            if symbol in self.correlation_cache:
                cache = self.correlation_cache[symbol]
                if datetime.now() - cache['timestamp'] < timedelta(minutes=5):
                    return cache['value']
                    
            # Obtém dados
            market_data = self.data_loader.get_market_data()
            if market_data is None:
                return None
                
            # Calcula correlação
            correlation = np.corrcoef(
                market_data['market_returns'],
                market_data['symbol_returns']
            )[0, 1]
            
            # Atualiza cache
            self.correlation_cache[symbol] = {
                'value': correlation,
                'timestamp': datetime.now()
            }
            
            return correlation
            
        except Exception as e:
            logger.error(f"Erro ao calcular correlação: {str(e)}")
            return None
            
    def _calculate_volatility(self, symbol: str) -> float:
        """
        Calcula volatilidade.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Volatilidade.
        """
        try:
            # Verifica cache
            if symbol in self.volatility_cache:
                cache = self.volatility_cache[symbol]
                if datetime.now() - cache['timestamp'] < timedelta(minutes=5):
                    return cache['value']
                    
            # Obtém dados recentes
            recent_data = self.data_loader.get_recent_data(
                symbol,
                limit=30
            )
            
            if recent_data is None or len(recent_data) < 2:
                return 0
                
            # Calcula retornos
            returns = np.diff(np.log(recent_data['close']))
            
            # Calcula volatilidade
            volatility = np.std(returns)
            
            # Atualiza cache
            self.volatility_cache[symbol] = {
                'value': volatility,
                'timestamp': datetime.now()
            }
            
            return volatility
            
        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade: {str(e)}")
            return 0
            
    async def _load_history(self) -> None:
        """Carrega histórico"""
        try:
            # Carrega trades
            trades = await self.data_loader.get_trade_history()
            if trades:
                self.trade_history = trades
                
            # Carrega histórico diário
            daily = await self.data_loader.get_daily_history()
            if daily:
                self.daily_history = daily
                
        except Exception as e:
            logger.error(f"Erro ao carregar histórico: {str(e)}")
            raise
            
    async def _update_daily_history(self) -> None:
        """Atualiza histórico diário"""
        try:
            # Obtém data atual
            current_date = datetime.now().date()
            
            # Verifica se já tem registro do dia
            if not self.daily_history or self.daily_history[-1]['date'] != current_date:
                # Cria novo registro
                self.daily_history.append({
                    'date': current_date,
                    'pnl': 0,
                    'trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                })
                
            # Remove registros antigos (mais de 30 dias)
            cutoff_date = current_date - timedelta(days=30)
            self.daily_history = [
                day for day in self.daily_history 
                if day['date'] >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Erro ao atualizar histórico diário: {str(e)}")
            raise
            
    def _update_metrics(self) -> None:
        """Atualiza métricas"""
        try:
            # Atualiza métricas diárias
            if self.daily_history:
                today = self.daily_history[-1]
                self.metrics['daily_pnl'] = today['pnl']
                
            # Atualiza métricas totais
            if self.trade_history:
                # P&L total
                self.metrics['total_pnl'] = sum(
                    trade['pnl'] for trade in self.trade_history
                )
                
                # Total de trades
                self.metrics['total_trades'] = len(self.trade_history)
                
                # Trades ganhos/perdidos
                self.metrics['winning_trades'] = len([
                    trade for trade in self.trade_history
                    if trade['pnl'] > 0
                ])
                self.metrics['losing_trades'] = len([
                    trade for trade in self.trade_history
                    if trade['pnl'] <= 0
                ])
                
                # Win rate
                if self.metrics['total_trades'] > 0:
                    self.metrics['win_rate'] = (
                        self.metrics['winning_trades'] / 
                        self.metrics['total_trades']
                    )
                    
                # Drawdown
                equity_curve = np.cumsum([
                    trade['pnl'] for trade in self.trade_history
                ])
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak
                self.metrics['max_drawdown'] = np.max(drawdown)
                
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas: {str(e)}")
            raise 