#!/usr/bin/env python3
"""
Exchange Factory
===============
Fábrica para criar instâncias de adaptadores de exchange conforme necessário.
"""

import logging
from typing import Dict, List, Any, Optional, Type
import os

try:
    import ccxt
except ImportError:
    logging.error("Biblioteca ccxt não encontrada. Instale com: pip install ccxt")
    raise

from quantum_trading.exchanges.exchange_base import ExchangeBase
from quantum_trading.exchanges.kucoin_adapter import KuCoinAdapter
from quantum_trading.exchanges.kraken_adapter import KrakenAdapter

logger = logging.getLogger("exchange_factory")

class ExchangeFactory:
    """Fábrica para criar instâncias de adaptadores de exchange."""
    
    # Mapeamento de IDs de exchange para suas respectivas classes de adaptador
    _EXCHANGE_MAPPING: Dict[str, Type[ExchangeBase]] = {
        'kucoin': KuCoinAdapter,
        'kraken': KrakenAdapter
        # Adicionar mais exchanges conforme implementadas
    }
    
    @classmethod
    def create_exchange(cls, exchange_id: str, config_path: str = "exchange_config.json") -> Optional[ExchangeBase]:
        """
        Cria uma instância do adaptador apropriado para a exchange especificada.
        
        Args:
            exchange_id: ID da exchange a ser instanciada
            config_path: Caminho para o arquivo de configuração
            
        Returns:
            Instância do adaptador de exchange ou None se não suportado
        """
        exchange_id = exchange_id.lower()
        
        if exchange_id not in cls._EXCHANGE_MAPPING:
            logger.error(f"Exchange não suportada: {exchange_id}")
            return None
        
        try:
            adapter_class = cls._EXCHANGE_MAPPING[exchange_id]
            return adapter_class(config_path)
        except Exception as e:
            logger.error(f"Erro ao criar adaptador para {exchange_id}: {e}")
            return None
    
    @classmethod
    def create_all_enabled_exchanges(cls, config_path: str = "exchange_config.json") -> List[ExchangeBase]:
        """
        Cria instâncias para todas as exchanges habilitadas no arquivo de configuração.
        
        Args:
            config_path: Caminho para o arquivo de configuração
            
        Returns:
            Lista de instâncias de adaptadores de exchange
        """
        import json
        from pathlib import Path
        
        exchanges = []
        
        try:
            # Carrega o arquivo de configuração
            with open(Path(config_path), 'r') as f:
                config = json.load(f)
            
            # Itera sobre todas as exchanges configuradas
            for exchange_id, exchange_config in config.get("exchanges", {}).items():
                # Verifica se a exchange está habilitada
                if exchange_config.get("enabled", False):
                    exchange = cls.create_exchange(exchange_id, config_path)
                    if exchange:
                        exchanges.append(exchange)
                        logger.info(f"Exchange {exchange_id} habilitada e inicializada.")
                else:
                    logger.info(f"Exchange {exchange_id} desabilitada na configuração.")
            
            return exchanges
        except Exception as e:
            logger.error(f"Erro ao criar exchanges habilitadas: {e}")
            return []
    
    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """
        Retorna a lista de exchanges suportadas pelo sistema.
        
        Returns:
            Lista de IDs de exchanges suportadas
        """
        return list(ExchangeFactory._EXCHANGE_MAPPING.keys())

def create_exchange(exchange_id: str, api_key: Optional[str] = None, 
                   api_secret: Optional[str] = None, sandbox: bool = False,
                   additional_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Cria uma instância de exchange com base no ID fornecido.
    
    Args:
        exchange_id: ID da exchange (ex: 'kucoin', 'kraken')
        api_key: Chave de API (opcional, pode ser fornecida via .env)
        api_secret: Segredo de API (opcional, pode ser fornecido via .env)
        sandbox: Se True, usa o modo sandbox/testnet quando disponível
        additional_params: Parâmetros adicionais específicos da exchange
        
    Returns:
        Instância da exchange
    """
    exchange_id = exchange_id.lower()
    
    # Configuração padrão
    config = {
        'enableRateLimit': True,
        'timeout': 30000,  # 30 segundos
    }
    
    # Adiciona credenciais de API se fornecidas
    if api_key is not None and api_secret is not None:
        config['apiKey'] = api_key
        config['secret'] = api_secret
    else:
        # Tenta carregar credenciais do ambiente
        env_api_key = os.getenv(f"{exchange_id.upper()}_API_KEY")
        env_api_secret = os.getenv(f"{exchange_id.upper()}_API_SECRET")
        
        if env_api_key and env_api_secret:
            config['apiKey'] = env_api_key
            config['secret'] = env_api_secret
    
    # Configurações específicas para cada exchange
    if exchange_id == 'kucoin':
        # KuCoin requer passphrase
        passphrase = additional_params.get('passphrase') if additional_params else None
        
        if passphrase is None:
            # Tenta obter passphrase do ambiente
            passphrase = os.getenv('KUCOIN_PASSPHRASE') or os.getenv('KUCOIN_API_PASSPHRASE')
            
        if passphrase:
            config['password'] = passphrase
        
        # Configura sandbox se solicitado
        if sandbox:
            config['sandbox'] = True
    
    elif exchange_id == 'kraken':
        # Configurações específicas para Kraken
        if sandbox:
            logging.warning("Kraken não possui modo sandbox oficial. Usando conta real com credenciais de teste.")
    
    elif exchange_id == 'binance':
        # Configurações específicas para Binance
        if sandbox:
            config['urls'] = {
                'api': 'https://testnet.binance.vision/api',
            }
    
    # Adiciona parâmetros adicionais
    if additional_params:
        for key, value in additional_params.items():
            if key != 'passphrase':  # Já tratado separadamente para kucoin
                config[key] = value
    
    # Verifica se a exchange é suportada pelo ccxt
    if exchange_id not in ccxt.exchanges:
        supported = ', '.join(ccxt.exchanges)
        raise ValueError(f"Exchange '{exchange_id}' não suportada. Exchanges suportadas: {supported}")
    
    # Cria e retorna a instância da exchange
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(config)
        
        logging.info(f"Exchange {exchange_id} criada com sucesso")
        if sandbox:
            logging.info(f"Modo sandbox ativado para {exchange_id}")
        
        return exchange
        
    except Exception as e:
        logging.error(f"Erro ao criar exchange {exchange_id}: {str(e)}")
        raise 