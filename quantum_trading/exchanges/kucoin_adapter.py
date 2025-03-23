#!/usr/bin/env python3
"""
KuCoin Adapter
==============
Adaptador específico para a exchange KuCoin, estendendo a classe base ExchangeBase
e implementando funcionalidades específicas para essa exchange.
"""

import logging
from typing import Dict, List, Any, Optional

from quantum_trading.exchanges.exchange_base import ExchangeBase

logger = logging.getLogger("kucoin_adapter")

class KuCoinAdapter(ExchangeBase):
    """Adaptador para a exchange KuCoin."""
    
    def __init__(self, config_path: str = "exchange_config.json"):
        """
        Inicializa o adaptador KuCoin.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        super().__init__("kucoin", config_path)
        logger.info("KuCoin adapter inicializado.")
        
        # Verifica se a API da KuCoin está funcionando
        self._check_api_status()
    
    def _check_api_status(self) -> bool:
        """
        Verifica se a API da KuCoin está funcionando.
        
        Returns:
            True se a API estiver funcionando, False caso contrário
        """
        try:
            # Tenta obter informações de mercado para verificar a conexão com a API
            self.exchange.fetch_status()
            logger.info("KuCoin API está funcionando normalmente.")
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar à API da KuCoin: {e}")
            return False
    
    def get_futures_position(self, symbol: str) -> Dict[str, Any]:
        """
        Obtém a posição atual em futuros para um determinado símbolo.
        
        Args:
            symbol: Símbolo do contrato (e.g., 'XBTUSDTM')
            
        Returns:
            Informações da posição
        """
        try:
            # Implementação específica para KuCoin Futures API
            # Nota: Pode requer configuração adicional na inicialização da exchange
            positions = self.exchange.fetch_positions([symbol])
            if positions and len(positions) > 0:
                return positions[0]
            return {}
        except Exception as e:
            logger.error(f"Erro ao obter posição em futuros para {symbol}: {e}")
            return {}
    
    def get_funding_rate(self, symbol: str) -> float:
        """
        Obtém a taxa de financiamento atual para um par de futuros.
        
        Args:
            symbol: Símbolo do contrato (e.g., 'XBTUSDTM')
            
        Returns:
            Taxa de financiamento atual
        """
        try:
            funding_info = self.exchange.fetch_funding_rate(symbol)
            return float(funding_info.get('fundingRate', 0.0))
        except Exception as e:
            logger.error(f"Erro ao obter taxa de financiamento para {symbol}: {e}")
            return 0.0
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Obtém o livro de ordens para um determinado símbolo.
        
        Args:
            symbol: Símbolo do par (e.g., 'BTC/USDT')
            limit: Número de ordens a recuperar
            
        Returns:
            Livro de ordens com ofertas de compra e venda
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Erro ao obter livro de ordens para {symbol}: {e}")
            return {"bids": [], "asks": []}
    
    def get_spread(self, symbol: str) -> float:
        """
        Calcula o spread atual (diferença entre menor venda e maior compra).
        
        Args:
            symbol: Símbolo do par (e.g., 'BTC/USDT')
            
        Returns:
            Spread percentual
        """
        try:
            orderbook = self.get_orderbook(symbol, 1)
            if not orderbook["bids"] or not orderbook["asks"]:
                return 0.0
                
            best_bid = orderbook["bids"][0][0]
            best_ask = orderbook["asks"][0][0]
            
            # Calcula o spread em percentual
            spread_pct = (best_ask - best_bid) / best_bid * 100
            
            return spread_pct
        except Exception as e:
            logger.error(f"Erro ao calcular spread para {symbol}: {e}")
            return 0.0 