#!/usr/bin/env python3
"""
Kraken Adapter
=============
Adaptador específico para a exchange Kraken, estendendo a classe base ExchangeBase
e implementando funcionalidades específicas para essa exchange.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from quantum_trading.exchanges.exchange_base import ExchangeBase

logger = logging.getLogger("kraken_adapter")

class KrakenAdapter(ExchangeBase):
    """Adaptador para a exchange Kraken."""
    
    def __init__(self, config_path: str = "exchange_config.json"):
        """
        Inicializa o adaptador Kraken.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        super().__init__("kraken", config_path)
        logger.info("Kraken adapter inicializado.")
        
        # Mapeamento de símbolos - a Kraken tem peculiaridades em alguns pares
        self.symbol_mapping = {
            'XBT/USD': 'BTC/USD',  # Kraken usa XBT para Bitcoin
            'XBT/USDT': 'BTC/USDT',
            'XMR/XBT': 'XMR/BTC'
        }
        
        # Verifica se a API da Kraken está funcionando
        self._check_api_status()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normaliza o símbolo para o formato usado pela Kraken.
        
        Args:
            symbol: Símbolo padronizado (e.g., 'BTC/USD')
            
        Returns:
            Símbolo no formato da Kraken
        """
        # Mapeamento inverso para conversão de símbolos padronizados para formato da Kraken
        inverse_mapping = {v: k for k, v in self.symbol_mapping.items()}
        return inverse_mapping.get(symbol, symbol)
    
    def _denormalize_symbol(self, kraken_symbol: str) -> str:
        """
        Converte o símbolo do formato da Kraken para o formato padronizado.
        
        Args:
            kraken_symbol: Símbolo no formato da Kraken
            
        Returns:
            Símbolo padronizado
        """
        return self.symbol_mapping.get(kraken_symbol, kraken_symbol)
    
    def _check_api_status(self) -> bool:
        """
        Verifica se a API da Kraken está funcionando.
        
        Returns:
            True se a API estiver funcionando, False caso contrário
        """
        try:
            system_status = self.exchange.fetch_status()
            if system_status.get('status') == 'online':
                logger.info("Kraken API está funcionando normalmente.")
                return True
            else:
                logger.warning(f"Kraken API status: {system_status.get('status')}")
                return False
        except Exception as e:
            logger.error(f"Erro ao conectar à API da Kraken: {e}")
            return False
    
    def get_price(self, pair: str) -> float:
        """
        Obtém o preço atual de um par com tratamento especial para símbolos da Kraken.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            
        Returns:
            Preço atual
        """
        try:
            # Normaliza o símbolo para o formato da Kraken
            kraken_pair = self._normalize_symbol(pair)
            ticker = self.exchange.fetch_ticker(kraken_pair)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Erro ao obter preço de {pair}: {e}")
            return 0.0
    
    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Obtém o saldo disponível de um ativo com tratamento para peculiaridades da Kraken.
        
        Args:
            asset: Símbolo do ativo a verificar
            
        Returns:
            Saldo disponível
        """
        try:
            # A Kraken pode adicionar prefixos a alguns ativos, como 'X' para criptomoedas e 'Z' para moedas fiduciárias
            balance = self.exchange.fetch_balance()
            
            # Verifica diferentes possibilidades de nome de ativo na Kraken
            asset = asset.upper()
            asset_mappings = [asset]
            
            # Adiciona possíveis prefixos para o ativo
            if asset in ['BTC', 'ETH', 'XMR']:
                asset_mappings.append(f"X{asset}")
            elif asset in ['USD', 'EUR', 'JPY']:
                asset_mappings.append(f"Z{asset}")
            
            # Procura o ativo em qualquer um dos formatos possíveis
            for mapped_asset in asset_mappings:
                if mapped_asset in balance:
                    return float(balance[mapped_asset]['free'])
            
            return 0.0
        except Exception as e:
            logger.error(f"Erro ao obter saldo de {asset}: {e}")
            return 0.0
    
    def get_maker_taker_fees(self, pair: str = None) -> Tuple[float, float]:
        """
        Obtém as taxas maker e taker para o par especificado ou as taxas padrão.
        
        Args:
            pair: Par de trading (opcional)
            
        Returns:
            Tuple contendo (maker_fee, taker_fee)
        """
        try:
            # A Kraken tem uma estrutura de taxas baseada em volume
            # Aqui obtemos as taxas para o par específico ou as taxas padrão
            fees = self.exchange.fetch_trading_fees()
            
            if pair and pair in fees:
                pair_fees = fees[pair]
                return pair_fees.get('maker', 0.0), pair_fees.get('taker', 0.0)
            
            # Retorna taxas padrão se o par não for encontrado ou não for especificado
            return fees.get('maker', 0.0), fees.get('taker', 0.0)
        except Exception as e:
            logger.error(f"Erro ao obter taxas para {pair}: {e}")
            return 0.0, 0.0
    
    def get_orderbook_depth(self, symbol: str) -> Dict[str, Any]:
        """
        Obtém a profundidade do livro de ordens com informações mais detalhadas.
        
        Args:
            symbol: Símbolo do par (e.g., 'BTC/USDT')
            
        Returns:
            Estatísticas de profundidade do livro
        """
        try:
            orderbook = self.get_orderbook(symbol, 100)
            
            # Calcula métricas de profundidade do livro
            bid_volumes = sum(bid[1] for bid in orderbook["bids"])
            ask_volumes = sum(ask[1] for ask in orderbook["asks"])
            
            # Calcula o preço médio ponderado pelo volume
            bid_vwap = sum(bid[0] * bid[1] for bid in orderbook["bids"]) / bid_volumes if bid_volumes > 0 else 0
            ask_vwap = sum(ask[0] * ask[1] for ask in orderbook["asks"]) / ask_volumes if ask_volumes > 0 else 0
            
            # Calcula a pressão de compra/venda (razão entre volumes de compra e venda)
            buy_sell_ratio = bid_volumes / ask_volumes if ask_volumes > 0 else 0
            
            return {
                "bid_volumes": bid_volumes,
                "ask_volumes": ask_volumes,
                "bid_vwap": bid_vwap,
                "ask_vwap": ask_vwap,
                "buy_sell_ratio": buy_sell_ratio,
                "spread": self.get_spread(symbol)
            }
        except Exception as e:
            logger.error(f"Erro ao calcular profundidade do livro para {symbol}: {e}")
            return {} 