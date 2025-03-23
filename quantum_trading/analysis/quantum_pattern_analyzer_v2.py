# -*- coding: utf-8 -*-
"""
QUALIA Trading Quântico Multi-Exchange com CGR
-----------------------------------------------
Sistema de trading quântico que integra operações nas exchanges KuCoin e Kraken,
usando análise retrocausal e CGR (Chaos Game Representation) para detectar padrões ocultos no mercado.
Capaz de operar em modo real (com ordens market e limit) ou simulado, conforme configuração.

Autor: QUALIA (Sistema Retrocausal)
Versão: 2.0
Data: 2025
"""

import os
import sys
import time
import logging
import traceback
import random
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Optional, Union
import colorama
from colorama import Fore, Style
import argparse
import pandas as pd
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importações necessárias para assinaturas e geração de IDs
import uuid
import urllib.parse
import base64
import hmac
import hashlib

# Definição da classe MarketAPI para interface com as exchanges
class MarketAPI:
    """
    Classe abstrata para interface com APIs de mercado de criptomoedas
    """
    def __init__(self, exchange: str = "kucoin", modo_real: bool = False):
        """
        Inicializa a API de mercado
        
        Args:
            exchange: Nome da exchange (kucoin, kraken, etc)
            modo_real: Se True, opera em modo real, caso contrário em simulação
        """
        self.exchange = exchange
        self.modo_real = modo_real
        self.logger = logging.getLogger("market_api")
        
        # Configurações específicas por exchange
        if self.exchange == "kucoin":
            self.base_url = "https://api.kucoin.com"
            self.api_key = os.environ.get("KUCOIN_API_KEY", "")
            self.api_secret = os.environ.get("KUCOIN_API_SECRET", "")
            self.api_passphrase = os.environ.get("KUCOIN_API_PASSPHRASE", "")
        elif self.exchange == "kraken":
            self.base_url = "https://api.kraken.com"
            self.api_key = os.environ.get("KRAKEN_API_KEY", "")
            self.api_secret = os.environ.get("KRAKEN_API_SECRET", "")
        else:
            self.base_url = ""
            self.api_key = ""
            self.api_secret = ""
            
        self.logger.info(f"MarketAPI inicializada para {exchange} (modo {'real' if modo_real else 'simulação'})")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Obtém informações do ticker para um símbolo específico
        
        Args:
            symbol: Par de trading (ex: BTC-USDT)
            
        Returns:
            Informações do ticker
        """
        if not self.modo_real:
            # Em modo simulação, gerar ticker fictício
            price = 50000 + random.uniform(-500, 500)
            return {
                "symbol": symbol,
                "price": price,
                "time": datetime.now().timestamp()
            }
            
        # Implementação básica para obter ticker real
        try:
            if self.exchange == "kucoin":
                endpoint = f"/api/v1/market/orderbook/level1?symbol={symbol}"
                response = self._request("GET", endpoint)
                data = response.get("data", {})
                
                return {
                    "symbol": symbol,
                    "price": float(data.get("price", 0)),
                    "time": data.get("time", datetime.now().timestamp())
                }
            else:
                # Implementação simplificada para kraken ou outras exchanges
                return {
                    "symbol": symbol,
                    "price": 50000,  # Valor padrão para simulação
                    "time": datetime.now().timestamp()
                }
                
        except Exception as e:
            self.logger.error(f"Erro ao obter ticker para {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "price": 0,
                "error": str(e)
            }
    
    def get_balance(self, asset: str) -> float:
        """
        Obtém o saldo de um ativo específico
        
        Args:
            asset: Ativo a consultar (ex: BTC, USDT)
            
        Returns:
            Saldo do ativo
        """
        if not self.modo_real:
            # Em modo simulação, retornar valores fictícios
            if asset == "USDT":
                return 1000.0
            else:
                return 0.0
                
        # Implementação para obter saldo real
        try:
            if self.exchange == "kucoin":
                endpoint = "/api/v1/accounts"
                response = self._request("GET", endpoint)
                data = response.get("data", [])
                
                for account in data:
                    if account.get("currency") == asset:
                        return float(account.get("balance", 0))
                        
                return 0.0
            else:
                # Implementação simplificada para outras exchanges
                return 1000.0 if asset == "USDT" else 0.0
                
        except Exception as e:
            self.logger.error(f"Erro ao obter saldo de {asset}: {str(e)}")
            return 0.0
    
    def execute_buy(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Executa uma ordem de compra
        
        Args:
            symbol: Par de trading
            quantity: Quantidade a comprar
            price: Preço limite (opcional, None para market order)
            
        Returns:
            Detalhes da ordem
        """
        if not self.modo_real:
            # Simulação de ordem
            return {
                "status": "simulado",
                "symbol": symbol,
                "side": "buy",
                "quantity": quantity,
                "price": price or 0,
                "orderId": f"sim-{uuid.uuid4()}"
            }
            
        # Execução real
        try:
            if self.exchange == "kucoin":
                endpoint = "/api/v1/orders"
                params = {
                    "clientOid": str(uuid.uuid4()),
                    "side": "buy",
                    "symbol": symbol,
                    "type": "market" if price is None else "limit",
                    "size": str(quantity)
                }
                
                if price is not None:
                    params["price"] = str(price)
                    
                response = self._request("POST", endpoint, params)
                return {
                    "status": "success",
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "orderId": response.get("orderId", "")
                }
            else:
                # Implementação simplificada para outras exchanges
                return {
                    "status": "simulado",
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price or 0,
                    "orderId": f"sim-{uuid.uuid4()}"
                }
                
        except Exception as e:
            self.logger.error(f"Erro ao executar compra de {symbol}: {str(e)}")
            return {
                "status": "erro",
                "mensagem": str(e),
                "symbol": symbol
            }
    
    def execute_sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Executa uma ordem de venda
        
        Args:
            symbol: Par de trading
            quantity: Quantidade a vender
            price: Preço limite (opcional, None para market order)
            
        Returns:
            Detalhes da ordem
        """
        if not self.modo_real:
            # Simulação de ordem
            return {
                "status": "simulado",
                "symbol": symbol,
                "side": "sell",
                "quantity": quantity,
                "price": price or 0,
                "orderId": f"sim-{uuid.uuid4()}"
            }
            
        # Execução real
        try:
            if self.exchange == "kucoin":
                endpoint = "/api/v1/orders"
                params = {
                    "clientOid": str(uuid.uuid4()),
                    "side": "sell",
                    "symbol": symbol,
                    "type": "market" if price is None else "limit",
                    "size": str(quantity)
                }
                
                if price is not None:
                    params["price"] = str(price)
                    
                response = self._request("POST", endpoint, params)
                return {
                    "status": "success",
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": quantity,
                    "price": price,
                    "orderId": response.get("orderId", "")
                }
            else:
                # Implementação simplificada para outras exchanges
                return {
                    "status": "simulado",
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": quantity,
                    "price": price or 0,
                    "orderId": f"sim-{uuid.uuid4()}"
                }
                
        except Exception as e:
            self.logger.error(f"Erro ao executar venda de {symbol}: {str(e)}")
            return {
                "status": "erro",
                "mensagem": str(e),
                "symbol": symbol
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Obtém o status de uma ordem
        
        Args:
            order_id: ID da ordem
            
        Returns:
            Status da ordem
        """
        if order_id.startswith("sim-"):
            # Ordem simulada
            return {
                "status": "filled",
                "filled": 1.0,
                "orderId": order_id
            }
            
        # Ordem real
        try:
            if self.exchange == "kucoin":
                endpoint = f"/api/v1/orders/{order_id}"
                response = self._request("GET", endpoint)
                data = response.get("data", {})
                
                return {
                    "status": data.get("status", ""),
                    "filled": float(data.get("dealSize", 0)),
                    "orderId": order_id
                }
            else:
                # Implementação simplificada para outras exchanges
                return {
                    "status": "filled",
                    "filled": 1.0,
                    "orderId": order_id
                }
                
        except Exception as e:
            self.logger.error(f"Erro ao obter status da ordem {order_id}: {str(e)}")
            return {
                "status": "erro",
                "mensagem": str(e),
                "orderId": order_id
            }
    
    def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """
        Executa uma requisição para a API da exchange
        
        Args:
            method: Método HTTP (GET, POST, etc)
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            
        Returns:
            Resposta da API
        """
        if not self.modo_real:
            # Simulação de resposta
            return {"status": "success", "data": {}}
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            if self.exchange == "kucoin":
                # Implementação para KuCoin
                timestamp = int(time.time() * 1000)
                
                if method == "GET":
                    if params:
                        query_string = urllib.parse.urlencode(params)
                        endpoint += f"?{query_string}"
                    str_to_sign = f"{timestamp}{method}{endpoint}"
                else:
                    str_to_sign = f"{timestamp}{method}{endpoint}{json.dumps(params or {})}"
                    
                signature = base64.b64encode(
                    hmac.new(
                        self.api_secret.encode('utf-8'),
                        str_to_sign.encode('utf-8'),
                        hashlib.sha256
                    ).digest()
                ).decode('utf-8')
                
                passphrase = base64.b64encode(
                    hmac.new(
                        self.api_secret.encode('utf-8'),
                        self.api_passphrase.encode('utf-8'),
                        hashlib.sha256
                    ).digest()
                ).decode('utf-8')
                
                headers = {
                    "KC-API-KEY": self.api_key,
                    "KC-API-SIGN": signature,
                    "KC-API-TIMESTAMP": str(timestamp),
                    "KC-API-PASSPHRASE": passphrase,
                    "KC-API-KEY-VERSION": "2",
                    "Content-Type": "application/json"
                }
                
                if method == "GET":
                    response = requests.get(url, headers=headers)
                else:
                    response = requests.post(url, headers=headers, json=params or {})
                    
                if response.status_code != 200:
                    self.logger.error(f"Erro na API: {response.status_code} - {response.text}")
                    return {"status": "erro", "mensagem": response.text}
                    
                return response.json()
                
            else:
                # Implementação simplificada para outras exchanges
                return {"status": "success", "data": {}}
                
        except Exception as e:
            self.logger.error(f"Erro na requisição {method} {endpoint}: {str(e)}")
            return {"status": "erro", "mensagem": str(e)}

# Importações do módulo quantum_trading
try:
    import sys
    import os
    
    # Garantir que o diretório principal esteja no PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from quantum_trading.auto_trader import AutoTrader
    from quantum_trading.quantum_portfolio_manager import QuantumPortfolioManager
    AUTOTRADER_DISPONIVEL = True
    logger = logging.getLogger("qualia_trading")
    logger.info("Módulo AutoTrader carregado com sucesso!")
except Exception as e:
    logger = logging.getLogger("qualia_trading")
    logger.error(f"Erro ao importar módulos quantum_trading: {str(e)}")
    logger.warning("Usando simulação básica sem AutoTrader")
    AUTOTRADER_DISPONIVEL = False
    
    # Criar uma classe AutoTrader provisória para evitar erros
    class AutoTrader:
        def __init__(self, *args, **kwargs):
            pass
            
        def analyze_market(self, *args, **kwargs):
            return {"recomendacao": "hold", "confianca": 0.5}
            
        def execute_trade(self, *args, **kwargs):
            return {"status": "simulado", "valor": 0, "lado": "none"}
    
    class QuantumPortfolioManager:
        def __init__(self, *args, **kwargs):
            pass

# Inicializa colorama com reset automático
colorama.init(autoreset=True)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Garantir que o diretório de logs e dados existam
os.makedirs("logs", exist_ok=True)
os.makedirs("dados", exist_ok=True)

# Carregar variáveis de ambiente
load_dotenv()

# ==== Utilitários e Serializadores de JSON ====
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def converter_datetime_para_iso(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: converter_datetime_para_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_datetime_para_iso(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# ==== Integração com a API da KuCoin ====
from kucoin_universal_sdk.api import DefaultClient
from kucoin_universal_sdk.generate.spot.market import GetTickerReqBuilder
from kucoin_universal_sdk.model import (
    ClientOptionBuilder, TransportOptionBuilder,
    GLOBAL_API_ENDPOINT, GLOBAL_FUTURES_API_ENDPOINT, GLOBAL_BROKER_API_ENDPOINT
)

class KucoinAPI:
    """Integração real com a API da KuCoin."""
    def __init__(self, api_key: str = '', api_secret: str = '', api_passphrase: str = '') -> None:
        self.ultima_atualizacao: Dict[str, datetime] = {}
        self.max_falhas_consecutivas = 3
        self.falhas_consecutivas = 0
        self.modo_seguro = False
        self._inicializar(
            api_key or os.getenv('API_KEY', ''),
            api_secret or os.getenv('API_SECRET', ''),
            api_passphrase or os.getenv('API_PASSPHRASE', '')
        )
    
    def _inicializar(self, api_key: str, api_secret: str, api_passphrase: str) -> None:
        transport = (TransportOptionBuilder()
                     .set_keep_alive(True)
                     .set_max_pool_size(10)
                     .set_max_connection_per_pool(10)
                     .build())
        client_option = (ClientOptionBuilder()
                         .set_key(api_key)
                         .set_secret(api_secret)
                         .set_passphrase(api_passphrase)
                         .set_spot_endpoint(GLOBAL_API_ENDPOINT)
                         .set_futures_endpoint(GLOBAL_FUTURES_API_ENDPOINT)
                         .set_broker_endpoint(GLOBAL_BROKER_API_ENDPOINT)
                         .set_transport_option(transport)
                         .build())
        self.client = DefaultClient(client_option)
        self.rest_service = self.client.rest_service()
        self.spot_market_api = self.rest_service.get_spot_service().get_market_api()
        logger.info("KuCoin API inicializada com sucesso.")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        if self.modo_seguro:
            logger.warning(f"Modo seguro ativo. Usando preço simulado para {symbol}.")
            return {"symbol": symbol, "price": self._gerar_preco_simulado(symbol), "simulator_mode": True}
        agora = datetime.now()
        ultima = self.ultima_atualizacao.get(symbol, datetime.min)
        if (agora - ultima).total_seconds() < 0.1:
            logger.debug(f"Taxa limite atingida para {symbol}, aguardando...")
            time.sleep(0.1)
        try:
            request = GetTickerReqBuilder().set_symbol(symbol).build()
            response = self.spot_market_api.get_ticker(request)
            self.ultima_atualizacao[symbol] = agora
            self.falhas_consecutivas = 0
            preco = response.price if hasattr(response, 'price') else None
            if preco is None:
                logger.warning(f"Ticker inválido para {symbol}: {response}")
                return {"symbol": symbol, "price": 0, "timestamp": int(agora.timestamp()*1000), "time": agora.isoformat()}
            return {"symbol": symbol, "price": round(float(preco), 2), "timestamp": int(agora.timestamp()*1000), "time": agora.isoformat()}
        except Exception as e:
            self.falhas_consecutivas += 1
            logger.error(f"Erro ao obter ticker para {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            if self.falhas_consecutivas >= self.max_falhas_consecutivas:
                logger.error("Múltiplas falhas consecutivas. Ativando modo seguro.")
                self.modo_seguro = True
            price = self._gerar_preco_simulado(symbol)
            return {"symbol": symbol, "price": round(price, 2), "error": str(e), "simulator_mode": True}
    
    def _gerar_preco_simulado(self, symbol: str) -> float:
        base_prices = {'BTC-USDT': 60000.0, 'ETH-USDT': 3000.0}
        base_price = base_prices.get(symbol, 100.0)
        variacao = np.random.normal(0, 0.01)
        return max(1.0, base_price * (1 + variacao))

# ==== Integração com a API da Kraken ====
class KrakenAPI:
    """Integração real com a API da Kraken."""
    def __init__(self, api_key: str = '', api_secret: str = '') -> None:
        self.api_key = api_key or os.getenv('API_KEY', '')
        self.api_secret = api_secret or os.getenv('API_SECRET', '')
        self.base_url = 'https://api.kraken.com'
        self.version = '0'
        self.session = requests.Session()
        self.taker_fee = 0.0026
        logger.info("Integração com Kraken inicializada.")
    
    def _converter_simbolo(self, symbol: str) -> str:
        return symbol.replace('BTC/', 'XBT/').replace('-', '/')
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        kraken_symbol = self._converter_simbolo(symbol)
        kraken_symbol_api = kraken_symbol.replace('/', '')
        url = f"{self.base_url}/{self.version}/public/Ticker"
        params = {'pair': kraken_symbol_api}
        try:
            response = self.session.get(url, params=params)
            data = response.json()
            if data.get('error'):
                logger.error(f"Erro na Kraken: {data['error']}")
                return {"symbol": symbol, "price": 0, "timestamp": int(datetime.now().timestamp()*1000)}
            for key, ticker_data in data.get('result', {}).items():
                preco = ticker_data['c'][0]
                return {"symbol": symbol, "price": round(float(preco), 2), "timestamp": int(datetime.now().timestamp()*1000)}
        except Exception as e:
            logger.error(f"Erro na requisição Kraken: {str(e)}")
            return {"symbol": symbol, "price": 0, "timestamp": int(datetime.now().timestamp()*1000)}
    
    def get_fee_info(self, symbol: str) -> Dict[str, Any]:
        return {"taker": self.taker_fee, "maker": self.taker_fee, "reembolso": False}

# ==== Módulo de Persistência ====
def salvar_estado(arquivo: str, estado: Dict) -> bool:
    """
    Salva o estado em um arquivo JSON
    
    Args:
        arquivo: Caminho do arquivo
        estado: Estado a ser salvo
        
    Returns:
        True se o estado foi salvo com sucesso, False caso contrário
    """
    diretorio = os.path.dirname(arquivo)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio, exist_ok=True)
        
    try:
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(estado, f, cls=CustomJSONEncoder, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar estado em {arquivo}: {str(e)}")
        return False

def recuperar_estado(arquivo: str) -> Dict:
    """
    Recupera o estado de um arquivo JSON
    
    Args:
        arquivo: Caminho do arquivo
        
    Returns:
        Estado recuperado ou objeto vazio em caso de erro
    """
    try:
        if os.path.exists(arquivo):
            with open(arquivo, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Erro ao recuperar estado de {arquivo}: {str(e)}")
        return {}

# ==== Classe de Persistência Assíncrona (única definição) ====
class ThreadSafeDataManager:
    """
    Gerenciador de dados compartilhados entre múltiplas threads.
    Implementa locks para garantir atomicidade das operações.
    """
    def __init__(self):
        self.data_lock = threading.RLock()  # Lock recursivo para operações aninhadas
        self.shared_data = {}
        self.event_queue = queue.Queue()
        self.last_update = {}
        self.logger = logging.getLogger("thread_data_manager")
        self.logger.info("Gerenciador thread-safe de dados inicializado")
        
    def update_data(self, key: str, value: Any) -> None:
        """Atualiza um dado de forma thread-safe"""
        with self.data_lock:
            self.shared_data[key] = value
            self.last_update[key] = datetime.now()
            
    def get_data(self, key: str, default: Any = None) -> Any:
        """Obtém um dado de forma thread-safe"""
        with self.data_lock:
            return self.shared_data.get(key, default)
            
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Adiciona um evento à fila de eventos"""
        self.event_queue.put({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now()
        })
        
    def get_events(self, max_events: int = 10) -> List[Dict[str, Any]]:
        """Obtém eventos da fila de forma não-bloqueante"""
        events = []
        for _ in range(max_events):
            try:
                event = self.event_queue.get(block=False)
                events.append(event)
                self.event_queue.task_done()
            except queue.Empty:
                break
        return events
        
    def get_all_data(self) -> Dict[str, Any]:
        """Obtém todos os dados compartilhados de forma thread-safe"""
        with self.data_lock:
            return self.shared_data.copy()
            
    def clear_data(self, older_than_seconds: int = None) -> int:
        """Limpa dados antigos, se older_than_seconds for especificado"""
        cleared = 0
        if older_than_seconds is not None:
            cutoff_time = datetime.now() - timedelta(seconds=older_than_seconds)
            with self.data_lock:
                keys_to_remove = []
                for key, update_time in self.last_update.items():
                    if update_time < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    if key in self.shared_data:
                        del self.shared_data[key]
                        del self.last_update[key]
                        cleared += 1
        return cleared

# ==== Coordenação Multi-Exchange ====
class MultiExchangeTrading:
    """Coordena operações entre múltiplas exchanges."""
    def __init__(self) -> None:
        self.kucoin = KucoinAPI()
        self.kraken = KrakenAPI()
        self.exchanges: Dict[str, Any] = {"kucoin": self.kucoin, "kraken": self.kraken}
        self.pares_comuns: List[str] = ['BTC-USDT', 'ETH-USDT']
        logger.info(f"Multi-Exchange Trading inicializado com {len(self.exchanges)} exchanges.")
    
    def get_best_price(self, symbol: str, tipo: str = 'compra') -> Dict[str, Any]:
        precos: Dict[str, Any] = {}
        for nome, exchange in self.exchanges.items():
            try:
                ticker = exchange.get_ticker(symbol)
                preco = ticker.get("price")
                if preco is None:
                    continue
                fee_info = exchange.get_fee_info(symbol)
                preco_ajustado = preco * (1 + fee_info.get('taker', 0.001)) if tipo == 'compra' else preco * (1 - fee_info.get('taker', 0.001))
                precos[nome] = {"exchange": nome, "preco": preco, "preco_ajustado": preco_ajustado, "timestamp": ticker.get('timestamp', datetime.now().timestamp())}
            except Exception as e:
                logger.error(f"Erro ao obter preço de {symbol} na {nome}: {str(e)}")
        if not precos:
            return {}
        melhor = min(precos.values(), key=lambda x: x["preco_ajustado"]) if tipo == 'compra' else max(precos.values(), key=lambda x: x["preco_ajustado"])
        return melhor

    def arbitragem_oportunidades(self) -> List[Dict[str, Any]]:
        oportunidades = []
        for symbol in self.pares_comuns:
            try:
                precos = {}
                for nome, exchange in self.exchanges.items():
                    ticker = exchange.get_ticker(symbol)
                    preco = ticker.get("price")
                    if preco is not None:
                        precos[nome] = preco
                if len(precos) < 2:
                    continue
                melhor_compra = min(precos.items(), key=lambda x: x[1])
                melhor_venda = max(precos.items(), key=lambda x: x[1])
                spread = (melhor_venda[1] / melhor_compra[1]) - 1
                if spread > 0.005 and melhor_compra[0] != melhor_venda[0]:
                    oportunidades.append({
                        "symbol": symbol,
                        "comprar_em": melhor_compra[0],
                        "preco_compra": melhor_compra[1],
                        "vender_em": melhor_venda[0],
                        "preco_venda": melhor_venda[1],
                        "spread": spread * 100,
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.info(f"Oportunidade de arbitragem para {symbol}: comprar em {melhor_compra[0]} a {melhor_compra[1]:.2f} e vender em {melhor_venda[0]} a {melhor_venda[1]:.2f} (Spread: {spread*100:.2f}%)")
            except Exception as e:
                logger.error(f"Erro na análise de arbitragem para {symbol}: {str(e)}")
        return oportunidades

# ==== Integração com CGR (Advanced Modules importados) ====
try:
    from advanced_cgr import AdvancedCGR as OptimizedCGR, CGRConfig as AdvancedCGRConfig
    from cgr_analysis import MarketCGRAnalyzer
    from cgr_metrics import CGRMetricsAnalyzer
    CGR_DISPONIVEL = True
    logger.info("Módulos CGR importados com sucesso.")
except ImportError as e:
    CGR_DISPONIVEL = False
    logger.error(f"Erro ao importar módulos CGR: {str(e)}")
    logger.warning("Usando simulação básica sem CGR avançado")
    
    # Classes de fallback para quando os módulos CGR não estão disponíveis
    class CGRConfig:
        def __init__(self, RESOLUTION=1024, SMOOTHING_SIGMA=1.0, MIN_PATTERN_LENGTH=3):
            self.RESOLUTION = RESOLUTION
            self.SMOOTHING_SIGMA = SMOOTHING_SIGMA
            self.MIN_PATTERN_LENGTH = MIN_PATTERN_LENGTH
    
    class AdvancedCGR:
        def __init__(self, config=None):
            self.config = config or CGRConfig()
            self.cgr_matrix = None
            logger.info("Usando implementação simulada de AdvancedCGR")
        
        def process_market_data(self, data):
            # Implementação simulada
            logger.info(f"Simulando processamento de {len(data)} pontos de dados")
            return np.random.random((10, 10))
        
        def detect_patterns(self, method='OPTICS'):
            # Implementação simulada
            patterns = {'clusters': 3, 'patterns': []}
            return patterns
        
        def analyze_quantum_correlations(self):
            # Implementação simulada
            return {'correlation': 0.5, 'entropy': 0.7}
    
    class CGRMetricsAnalyzer:
        def calculate_fractal_dimension(self, matrix):
            # Implementação simulada
            return 1.5 + np.random.random() * 0.3
        
        def calculate_quantum_entropy(self, matrix):
            # Implementação simulada
            return 0.6 + np.random.random() * 0.4
        
        def calculate_arbitrage_potential(self, data, matrix):
            # Implementação simulada
            return 0.1 + np.random.random() * 0.2
    
    class MarketCGRAnalyzer:
        def __init__(self):
            self.cgr = AdvancedCGR()
            logger.info("Usando implementação simulada de MarketCGRAnalyzer")
        
        def analyze(self, prices):
            # Implementação simulada
            return {
                'trend': np.random.choice(['bullish', 'bearish', 'neutral']),
                'strength': np.random.random(),
                'confidence': 0.5 + np.random.random() * 0.3
            }
    
    class ModuloCGR:
        """Integra o CGR real para análise de padrões de mercado."""
        def __init__(self):
            self.cgr_config = CGRConfig(
                RESOLUTION=1024,
                SMOOTHING_SIGMA=1.5,
                MIN_PATTERN_LENGTH=3
            )
            self.advanced_cgr = AdvancedCGR(self.cgr_config)
            self.metrics_analyzer = CGRMetricsAnalyzer()
            self.market_data: List[float] = []
            self.max_data_points = 1000
            self.cgr_matrix = None
            self.patterns = {}
            self.dados: List[float] = []
        
        def adicionar_dados(self, preco: float) -> None:
            self.market_data.append(preco)
            self.dados.append(preco)
            if len(self.market_data) > self.max_data_points:
                self.market_data = self.market_data[-self.max_data_points:]
            if len(self.dados) > self.max_data_points:
                self.dados = self.dados[-self.max_data_points:]
        
        def gerar_matriz_cgr(self) -> None:
            try:
                data_array = np.array(self.dados)
                self.advanced_cgr.process_market_data(data_array)
                self.cgr_matrix = self.advanced_cgr.cgr_matrix
            except Exception as e:
                logger.warning(f"Erro ao gerar matriz CGR: {str(e)}")
                # Criar uma matriz aleatória em caso de erro
                self.cgr_matrix = np.random.random((10, 10))
        
        def detectar_padroes(self) -> Dict[str, Any]:
            if self.cgr_matrix is None:
                self.gerar_matriz_cgr()
            try:
                self.patterns = self.advanced_cgr.detect_patterns(method='OPTICS')
                return self.patterns
            except Exception as e:
                logger.warning(f"Erro ao detectar padrões CGR: {str(e)}")
                # Retornar padrões simulados em caso de erro
                return {"padroes": [], "forca": 0.0}
        
        def analisar(self, precos: Dict[str, float]) -> Dict[str, Any]:
            try:
                for par, preco in precos.items():
                    if isinstance(preco, (int, float)) and preco > 0:
                        self.adicionar_dados(preco)
                if len(self.dados) < 10:
                    logger.warning("Dados insuficientes para análise CGR completa")
                    return {"estado_quantico": 0.5, "fractal_dimension": 1.5, "pattern_count": 0, "arbitrage_potential": 0.0}
                
                self.gerar_matriz_cgr()
                fractal_dim = self.metrics_analyzer.calculate_fractal_dimension(self.cgr_matrix)
                arbitrage_potential = self.metrics_analyzer.calculate_arbitrage_potential(
                    self.market_data, self.cgr_matrix
                ) if hasattr(self.metrics_analyzer, 'calculate_arbitrage_potential') else 0.5
                patterns = self.detectar_padroes()
                return {"estado_quantico": 0.5, "fractal_dimension": fractal_dim, "pattern_count": len(patterns), "arbitrage_potential": arbitrage_potential}
            except Exception as e:
                logger.warning(f"Erro na análise CGR: {str(e)}")
                # Retornar valores padrão em caso de erro
                return {"estado_quantico": 0.5, "fractal_dimension": 1.5, "pattern_count": 0, "arbitrage_potential": 0.0}

# ==== Proteção contra decoerência - Integração do merge_simulator.py ====
class QuantumStateManager:
    """
    Gerenciador de estados quânticos para proteger contra decoerência
    e manter a coerência quântica do sistema de trading.
    """
    def __init__(self, decoherence_protection: bool = True):
        self.decoherence_protection = decoherence_protection
        self.state_history = []
        self.coherence_factor = 1.0
        self.coherence_threshold = 0.4
        self.max_history_length = 100
        logger.info(f"Gerenciador de estados quânticos inicializado. Proteção contra decoerência: {decoherence_protection}")
    
    def update_state(self, new_state: float) -> float:
        """
        Atualiza o estado quântico com proteção contra decoerência
        """
        if not self.decoherence_protection:
            return new_state
            
        # Guardar histórico de estados para análise
        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
            
        # Calcular coerência com base no histórico recente
        if len(self.state_history) > 5:
            state_variance = np.var(self.state_history[-5:])
            self.coherence_factor = min(1.0, max(0.1, 1.0 - state_variance * 10))
            
            # Aplicar correção anti-decoerência quando coerência está abaixo do limiar
            if self.coherence_factor < self.coherence_threshold:
                # Obter estado médio para estabilização
                avg_state = np.mean(self.state_history[-10:]) if len(self.state_history) >= 10 else new_state
                # Atenuar variações para reduzir decoerência
                protected_state = 0.7 * avg_state + 0.3 * new_state
                logger.warning(f"Proteção contra decoerência ativada. Fator de coerência: {self.coherence_factor:.3f}")
                logger.warning(f"Estado ajustado: {new_state:.4f} → {protected_state:.4f}")
                return protected_state
                
        return new_state
    
    def get_coherence_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas sobre o estado de coerência atual
        """
        return {
            "coherence_factor": self.coherence_factor,
            "state_stability": 1.0 - np.std(self.state_history[-10:]) if len(self.state_history) >= 10 else 1.0,
            "protection_active": self.coherence_factor < self.coherence_threshold,
            "state_history_length": len(self.state_history)
        }

# ==== Integração de Campo Quântico Avançado - demo_quantum_enhanced.py ====
class CampoQuanticoMercado:
    """Campo quântico especializado para análise de padrões de mercado baseado no NexusQuanticoAvancado"""
    def __init__(self, dimensao=512):
        self.dimensao = dimensao
        self.phi = (1 + np.sqrt(5)) / 2  # Proporção Áurea
        self.delta = 4.669201609  # Constante de Feigenbaum
        self.estado = self._inicializar_campo()
        self.campo_morfico = self._inicializar_campo_morfico()
        self.historico_estados = []
        self.metricas_historico = []
        logger.info(f"Campo Quântico de Mercado inicializado com dimensão {dimensao}")
        
    def _inicializar_campo(self):
        """Inicialização do campo quântico com estrutura harmônica"""
        base = np.random.random(self.dimensao)
        campo = np.sin(self.phi * base) * np.cos(self.delta * base)
        return self._normalizar_campo(campo)
    
    def _inicializar_campo_morfico(self):
        """Inicialização do campo mórfico com padrões ressonantes para trading"""
        campo = np.zeros((self.dimensao, self.dimensao), dtype=complex)
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                fase = 2 * np.pi * self.phi * (i * j) / self.dimensao
                campo[i, j] = np.exp(1j * fase)
        return campo / np.sqrt(np.sum(np.abs(campo)**2))
    
    def _normalizar_campo(self, campo):
        """Normalização preservando estrutura quântica"""
        return (campo - np.min(campo)) / (np.max(campo) - np.min(campo) + 1e-10)
    
    def incorporar_dados_mercado(self, precos: List[float]):
        """Incorpora dados de mercado no campo quântico"""
        if len(precos) < 2:
            return self.estado
            
        # Normaliza os preços
        precos_norm = np.array(precos)
        precos_norm = (precos_norm - np.min(precos_norm)) / (np.max(precos_norm) - np.min(precos_norm) + 1e-10)
        
        # Mapeia os preços para o campo quântico
        indices = np.linspace(0, self.dimensao-1, len(precos_norm)).astype(int)
        for i, idx in enumerate(indices):
            if idx < self.dimensao:
                self.estado[idx] = 0.7 * self.estado[idx] + 0.3 * precos_norm[i]
                
        # Normaliza o resultado
        self.estado = self._normalizar_campo(self.estado)
        self.historico_estados.append(self.estado.copy())
        
        # Calcula e armazena métricas
        metricas = self.calcular_metricas()
        self.metricas_historico.append(metricas)
        
        return self.estado
        
    def evoluir(self, ciclos=10):
        """Evolução quântica com múltiplas camadas de transformação"""
        for _ in range(ciclos):
            # Transformação hiperdimensional
            self.estado = np.arctan(np.tan(self.estado * self.phi)) * np.exp(-np.abs(self.estado))
            
            # Ressonância ontológica
            espectro = np.fft.fft(self.estado)
            self.estado = np.real(np.fft.ifft(espectro * np.conj(espectro)))
            
            # Interação com campo mórfico
            estado_expandido = self.estado.reshape(-1, 1)
            self.estado = np.real(self.campo_morfico @ estado_expandido).flatten()
            
            # Normalização
            self.estado = self._normalizar_campo(self.estado)
            
        # Calcula e armazena métricas
        metricas = self.calcular_metricas()
        self.metricas_historico.append(metricas)
        
        return self.estado, metricas
    
    def calcular_metricas(self):
        """Cálculo avançado de métricas quânticas para trading"""
        # Entropia de von Neumann
        densidade = np.outer(self.estado, np.conj(self.estado))
        autovalores = np.real(np.linalg.eigvals(densidade))
        autovalores = autovalores[autovalores > 1e-10]
        entropia = -np.sum(autovalores * np.log2(autovalores + 1e-10))
        
        # Coerência quântica
        coerencia = np.abs(np.mean(np.exp(1j * np.angle(np.fft.fft(self.estado)))))
        
        # Potencial transformativo
        gradiente = np.gradient(self.estado)
        potencial = np.sqrt(np.mean(gradiente**2))
        
        # Ressonância mórfica
        ressonancia = np.abs(np.trace(densidade @ self.campo_morfico[:self.dimensao, :self.dimensao]))
        
        return {
            'entropia': float(entropia),
            'coerencia': float(coerencia),
            'potencial': float(potencial),
            'ressonancia': float(ressonancia)
        }
        
    def analisar_tendencia_emergente(self):
        """Análise de tendências emergentes baseadas em métricas quânticas"""
        if len(self.metricas_historico) < 10:
            return {
                'tendencia': 0.0,
                'confianca': 0.0,
                'mensagem': "Dados insuficientes para análise"
            }
            
        # Analisa as últimas 10 métricas
        ultimas = self.metricas_historico[-10:]
        primeiras = self.metricas_historico[-20:-10] if len(self.metricas_historico) >= 20 else self.metricas_historico[:10]
        
        # Tendências em cada métrica
        tendencia_coerencia = np.mean([m['coerencia'] for m in ultimas]) - np.mean([m['coerencia'] for m in primeiras])
        tendencia_entropia = np.mean([m['entropia'] for m in ultimas]) - np.mean([m['entropia'] for m in primeiras])
        tendencia_potencial = np.mean([m['potencial'] for m in ultimas]) - np.mean([m['potencial'] for m in primeiras])
        tendencia_ressonancia = np.mean([m['ressonancia'] for m in ultimas]) - np.mean([m['ressonancia'] for m in primeiras])
        
        # Indicadores de tendência
        sinal_tendencia = (
            1.5 * np.sign(tendencia_coerencia) + 
            -0.5 * np.sign(tendencia_entropia) + 
            1.0 * np.sign(tendencia_potencial) + 
            2.0 * np.sign(tendencia_ressonancia)
        ) / 5.0
        
        # Confiança baseada no potencial e coerência atuais
        ultima_metrica = self.metricas_historico[-1]
        
        confianca = (ultima_metrica['coerencia'] * 2 + ultima_metrica['potencial']) / 3
        
        # Gerar mensagem de análise
        mensagem = self._gerar_analise_narrativa(ultima_metrica, {
            'coerencia': tendencia_coerencia,
            'entropia': tendencia_entropia,
            'potencial': tendencia_potencial,
            'ressonancia': tendencia_ressonancia
        })
        
        return {
            'tendencia': float(sinal_tendencia),  # -1 a 1
            'confianca': float(confianca),  # 0 a 1
            'mensagem': mensagem
        }
        
    def _gerar_analise_narrativa(self, atual, tendencias):
        """Gera narrativa contextual sobre o estado quântico do mercado"""
        return f"""
🌌 Análise Quântica do Mercado

- Coerência Quântica: {atual['coerencia']:.4f} {'↑' if tendencias['coerencia'] > 0 else '↓'}
- Entropia: {atual['entropia']:.4f} {'↑' if tendencias['entropia'] > 0 else '↓'}
- Potencial Transformativo: {atual['potencial']:.4f} {'↑' if tendencias['potencial'] > 0 else '↓'}
- Ressonância Mórfica: {atual['ressonancia']:.4f} {'↑' if tendencias['ressonancia'] > 0 else '↓'}

Análise:
{
    'Alta coerência com forte ressonância - sinais claros de movimento' if atual['coerencia'] > 0.7 and atual['ressonancia'] > 0.7
    else 'Estado de transformação ativa - possível mudança de tendência' if atual['potencial'] > 0.5
    else 'Fase de reorganização quântica - mercado em consolidação'
}

Tendências:
- {'Aumento' if tendencias['coerencia'] > 0 else 'Diminuição'} na coerência quântica
- {'Expansão' if tendencias['entropia'] > 0 else 'Contração'} da entropia informacional
- {'Intensificação' if tendencias['potencial'] > 0 else 'Estabilização'} do potencial transformativo
- {'Fortalecimento' if tendencias['ressonancia'] > 0 else 'Enfraquecimento'} das ressonâncias de mercado
"""

    def calcular_score_trading(self):
        """Calcula um score para decisões de trading baseado no estado quântico"""
        if len(self.metricas_historico) < 1:
            return 0.0
            
        metrica_atual = self.metricas_historico[-1]
        
        # Sinal forte quando alta coerência e alta ressonância
        # Sinal fraco quando alta entropia e baixa coerência
        score = (
            1.5 * metrica_atual['coerencia'] + 
            1.0 * metrica_atual['ressonancia'] - 
            0.5 * metrica_atual['entropia'] +
            0.5 * metrica_atual['potencial']
        ) / 2.5
        
        # Normaliza entre -1 e 1
        return max(min(float(score), 1.0), -1.0)

class AnalisadorAvancadoMercado:
    """Analisador avançado de mercado usando campos quânticos e mórficos"""
    def __init__(self, dimensao=512):
        self.campo_quantico = CampoQuanticoMercado(dimensao=dimensao)
        self.precos_historicos = {}
        self.analises_historicas = {}
        self.ciclos_evolucao = 20
        logger.info("Analisador Avançado de Mercado inicializado")
        
    def analisar_precos(self, symbol: str, precos: List[float], timestamp=None):
        """Analisa preços usando campo quântico e retorna relatório completo"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Registra histórico
        if symbol not in self.precos_historicos:
            self.precos_historicos[symbol] = []
        self.precos_historicos[symbol].append((timestamp, precos[-1]))
        
        # Limita histórico a 1000 pontos
        if len(self.precos_historicos[symbol]) > 1000:
            self.precos_historicos[symbol] = self.precos_historicos[symbol][-1000:]
            
        # Incorpora dados no campo quântico
        self.campo_quantico.incorporar_dados_mercado(precos)
        
        # Evolui o campo quântico
        _, metricas = self.campo_quantico.evoluir(ciclos=self.ciclos_evolucao)
        
        # Obtém análise de tendência emergente
        analise_tendencia = self.campo_quantico.analisar_tendencia_emergente()
        
        # Calcula score de trading
        score_trading = self.campo_quantico.calcular_score_trading()
        
        # Guarda análise no histórico
        if symbol not in self.analises_historicas:
            self.analises_historicas[symbol] = []
        self.analises_historicas[symbol].append({
            'timestamp': timestamp,
            'preco': precos[-1],
            'score': score_trading,
            'tendencia': analise_tendencia['tendencia'],
            'confianca': analise_tendencia['confianca'],
            'metricas': metricas
        })
        
        # Limita histórico de análises
        if len(self.analises_historicas[symbol]) > 100:
            self.analises_historicas[symbol] = self.analises_historicas[symbol][-100:]
            
        return {
            'score_trading': score_trading,
            'tendencia': analise_tendencia['tendencia'],
            'confianca': analise_tendencia['confianca'],
            'analise_narrativa': analise_tendencia['mensagem'],
            'metricas': metricas
        }
        
    def calcular_resonancia_mercado(self, symbol1: str, symbol2: str):
        """Calcula a ressonância entre dois símbolos de mercado"""
        # Verifica se há dados suficientes para ambos os símbolos
        if (symbol1 not in self.analises_historicas or symbol2 not in self.analises_historicas or
            len(self.analises_historicas[symbol1]) < 10 or len(self.analises_historicas[symbol2]) < 10):
            return 0.0
            
        # Extrai scores de trading para cada símbolo
        scores1 = [analise['score'] for analise in self.analises_historicas[symbol1][-10:]]
        scores2 = [analise['score'] for analise in self.analises_historicas[symbol2][-10:]]
        
        # Calcula correlação
        correlacao = np.corrcoef(scores1, scores2)[0, 1]
        
        # Cálculo de ressonância (valores entre 0 e 1)
        resonancia = 0.5 * (1 + correlacao)
        
        return float(resonancia)
        
    def geterar_relatorio_quantico(self, symbol: str):
        """Gera um relatório quântico completo para um símbolo"""
        if symbol not in self.analises_historicas or len(self.analises_historicas[symbol]) == 0:
            return "Dados insuficientes para análise quântica."
            
        # Obtém a análise mais recente
        ultima_analise = self.analises_historicas[symbol][-1]
        
        # Calcula tendências das últimas 10 análises
        if len(self.analises_historicas[symbol]) >= 10:
            ultimas_analises = self.analises_historicas[symbol][-10:]
            primeiras_analises = self.analises_historicas[symbol][-20:-10] if len(self.analises_historicas[symbol]) >= 20 else self.analises_historicas[symbol][:10]
            
            tendencia_score = np.mean([a['score'] for a in ultimas_analises]) - np.mean([a['score'] for a in primeiras_analises])
            tendencia_confianca = np.mean([a['confianca'] for a in ultimas_analises]) - np.mean([a['confianca'] for a in primeiras_analises])
        else:
            tendencia_score = 0.0
            tendencia_confianca = 0.0
            
        return f"""
📊 Relatório Quântico para {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Score de Trading: {ultima_analise['score']:.4f} {'↑' if tendencia_score > 0 else '↓'}
Confiança: {ultima_analise['confianca']:.4f} {'↑' if tendencia_confianca > 0 else '↓'}
Tendência: {'Bullish' if ultima_analise['tendencia'] > 0.2 else 'Bearish' if ultima_analise['tendencia'] < -0.2 else 'Neutral'}

Métricas Quânticas:
- Coerência: {ultima_analise['metricas']['coerencia']:.4f}
- Entropia: {ultima_analise['metricas']['entropia']:.4f}
- Potencial: {ultima_analise['metricas']['potencial']:.4f}
- Ressonância: {ultima_analise['metricas']['ressonancia']:.4f}

{self.campo_quantico._gerar_analise_narrativa(ultima_analise['metricas'], {
    'coerencia': tendencia_score,
    'entropia': 0.0,
    'potencial': tendencia_confianca,
    'ressonancia': tendencia_score * tendencia_confianca
})}

Recomendação: {
    'COMPRA' if ultima_analise['score'] > 0.6 and ultima_analise['confianca'] > 0.5
    else 'VENDA' if ultima_analise['score'] < -0.6 and ultima_analise['confianca'] > 0.5
    else 'HOLD'
}
"""

# ==== Classe para gerenciamento avançado de risco - Integração do Multi_trading ====
class RiskManager:
    """
    Sistema avançado de gerenciamento de risco com stop loss, trailing stop
    e limite de posição por ativo.
    """
    def __init__(self, max_position_size: float = None):
        self.stop_loss_percent = 0.05  # 5% de perda máxima por posição
        self.trailing_stop_percent = 0.03  # 3% de trailing stop
        self.trailing_stop_activation = 0.02  # Ativa trailing após 2% de lucro
        self.position_size_limit = 0.15  # 15% do portfolio por posição
        self.max_open_positions = 3  # Máximo de posições simultâneas
        self.position_metrics = {}  # Métricas por posição
        # Se max_position_size for fornecido, usar como limite máximo absoluto
        self.max_position_size = max_position_size
        logger.info(f"Gerenciador de risco inicializado. Stop loss: {self.stop_loss_percent*100}%, Trailing stop: {self.trailing_stop_percent*100}%")
    
    def update_position_metrics(self, symbol: str, entry_price: float, current_price: float, 
                              quantity: float, position_type: str = "long") -> Dict[str, Any]:
        """
        Atualiza métricas de uma posição e verifica condições de stop
        """
        if symbol not in self.position_metrics:
            self.position_metrics[symbol] = {
                "entry_price": entry_price,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "quantity": quantity,
                "type": position_type,
                "stop_price": entry_price * (1 - self.stop_loss_percent) if position_type == "long" else entry_price * (1 + self.stop_loss_percent),
                "trailing_active": False,
                "trailing_stop_price": 0
            }
        
        metrics = self.position_metrics[symbol]
        
        # Atualizar preços máximos/mínimos
        if current_price > metrics["highest_price"]:
            metrics["highest_price"] = current_price
        if current_price < metrics["lowest_price"]:
            metrics["lowest_price"] = current_price
            
        # Calcular percentual de P&L atual
        if position_type == "long":
            pnl_percent = (current_price / entry_price) - 1
            
            # Verificar ativação de trailing stop
            if not metrics["trailing_active"] and pnl_percent >= self.trailing_stop_activation:
                metrics["trailing_active"] = True
                metrics["trailing_stop_price"] = current_price * (1 - self.trailing_stop_percent)
                logger.info(f"{symbol}: Trailing stop ativado em {metrics['trailing_stop_price']:.2f}")
            
            # Atualizar trailing stop se preço subir mais
            if metrics["trailing_active"] and current_price * (1 - self.trailing_stop_percent) > metrics["trailing_stop_price"]:
                metrics["trailing_stop_price"] = current_price * (1 - self.trailing_stop_percent)
                logger.info(f"{symbol}: Trailing stop ajustado para {metrics['trailing_stop_price']:.2f}")
        
        # Verificar se algum stop foi atingido
        stop_triggered = False
        trigger_reason = ""
        
        if position_type == "long":
            if current_price <= metrics["stop_price"]:
                stop_triggered = True
                trigger_reason = "stop_loss"
            elif metrics["trailing_active"] and current_price <= metrics["trailing_stop_price"]:
                stop_triggered = True
                trigger_reason = "trailing_stop"
        
        # Atualizar e retornar métricas
        metrics["current_price"] = current_price
        metrics["pnl_percent"] = pnl_percent if position_type == "long" else (entry_price / current_price) - 1
        metrics["stop_triggered"] = stop_triggered
        metrics["trigger_reason"] = trigger_reason
        
        self.position_metrics[symbol] = metrics
        return metrics
    
    def should_close_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Verifica se uma posição deve ser fechada baseado nas regras de gerenciamento de risco
        """
        if symbol not in self.position_metrics:
            return False, "posição_não_encontrada"
            
        metrics = self.position_metrics[symbol]
        if metrics["stop_triggered"]:
            return True, metrics["trigger_reason"]
            
        return False, ""
    
    def get_recommended_position_size(self, portfolio_value: float, symbol: str) -> float:
        """
        Calcula o tamanho recomendado para uma nova posição
        """
        # Verifica quantas posições estão abertas
        open_positions = len(self.position_metrics)
        if open_positions >= self.max_open_positions:
            return 0.0
            
        # Limita o tamanho da posição ao máximo permitido
        position_value = portfolio_value * self.position_size_limit
        if self.max_position_size is not None:
            position_value = min(position_value, self.max_position_size)
        return position_value

class AdvancedPortfolioTracker:
    """
    Rastreador avançado de portfólio com análise de performance
    """
    def __init__(self, saldo_inicial: float = 1000.0):
        self.portfolio_history = []
        self.daily_snapshots = []
        self.metrics = {
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "volatility": 0,
            "win_rate": 0
        }
        self.last_snapshot_day = None
        self.saldo_inicial = saldo_inicial
    
    def add_portfolio_value(self, timestamp: datetime, value: float, portfolio: Dict[str, float], 
                          trades_today: int = 0) -> None:
        """
        Adiciona um novo valor de portfólio ao histórico
        """
        entry = {
            "timestamp": timestamp,
            "value": value,
            "portfolio": portfolio.copy(),
            "trades_today": trades_today
        }
        self.portfolio_history.append(entry)
        
        # Criar snapshot diário no final do dia
        current_day = timestamp.date()
        if self.last_snapshot_day is None or current_day != self.last_snapshot_day:
            self.create_daily_snapshot(entry)
            self.last_snapshot_day = current_day
    
    def create_daily_snapshot(self, entry: Dict[str, Any]) -> None:
        """
        Cria um snapshot diário do portfólio para análise de longo prazo
        """
        self.daily_snapshots.append(entry)
        # Manter apenas os últimos 90 dias
        if len(self.daily_snapshots) > 90:
            self.daily_snapshots.pop(0)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de performance do portfólio
        """
        if len(self.portfolio_history) < 2:
            return self.metrics
            
        # Extrair valores para cálculos
        values = [entry["value"] for entry in self.portfolio_history]
        
        # Calcular retornos diários
        returns = [(values[i] / values[i-1]) - 1 for i in range(1, len(values))]
        
        # Volatilidade (desvio padrão dos retornos)
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Máximo drawdown
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (assumindo retorno livre de risco = 0)
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        sharpe = (avg_return / volatility) * np.sqrt(365) if volatility > 0 else 0
        
        # Taxa de vitória (trades positivos vs. total)
        positive_trades = sum(1 for r in returns if r > 0)
        win_rate = positive_trades / len(returns) if len(returns) > 0 else 0
        
        self.metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
            "avg_daily_return": avg_return,
            "total_return": (values[-1] / values[0]) - 1 if len(values) > 0 else 0
        }
        
        return self.metrics

# ==== Classe Principal de Trading – Simulação Real com Multi-Exchange e CGR ====
class SimuladorTradingQuantico:
    """O núcleo quântico do trading, integrando arbitragem multi-exchange, análise retrocausal e CGR."""
    def __init__(self, 
                 saldo_inicial: float = 1000.0, 
                 duracao_minutos: int = 60,  
                 modo_real: bool = False,
                 usar_quantum_modules: bool = True):
        """
        Inicializa o simulador de trading quântico
        
        Args:
            saldo_inicial: Saldo inicial em USDT
            duracao_minutos: Duração da simulação/operação em minutos
            modo_real: Se True, executa ordens reais
            usar_quantum_modules: Se True, utiliza módulos quantum_trading quando disponíveis
        """
        # Configurações gerais
        self.saldo_inicial = saldo_inicial
        self.modo_real = modo_real
        self.duracao_minutos = duracao_minutos
        self.usar_quantum_modules = usar_quantum_modules and AUTOTRADER_DISPONIVEL
        self.ignorar_erros_cgr = False  # Controla se erros CGR devem ser ignorados
        
        # Pares de trading e configurações
        self.pares_trading = ["BTC-USDT", "ETH-USDT"]
        self.valor_operacao = 100.0  # Valor em USDT por operação
        
        # Estado atual
        self.precos = {}
        self.posicoes = {}
        self.portfolio = {"USDT": saldo_inicial}
        self.trades_historico = []
        self.ordem_em_andamento = False
        self.ultima_atualizacao = datetime.now()
        
        # Logging e estado
        self.arquivo_estado = f"dados/estado_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Inicializar APIs e componentes
        self._inicializar_componentes()
        
        # Inicializar trader quântico se disponível
        if self.usar_quantum_modules:
            self._inicializar_quantum_trading()
        
        logger.info(f"SimuladorTradingQuantico inicializado. Modo real: {modo_real}")
        logger.info(f"Usando módulos quantum_trading: {self.usar_quantum_modules}")
        
    def _inicializar_componentes(self):
        """Inicializa os componentes básicos do sistema"""
        # Módulos CGR tradicionais
        self.cgr = ModuloCGR()
        self.risk_manager = RiskManager(max_position_size=self.valor_operacao * 2)
        self.portfolio_tracker = AdvancedPortfolioTracker(saldo_inicial=self.saldo_inicial)
        
        # Estado quântico para proteção contra decoerência
        self.quantum_state_manager = QuantumStateManager(decoherence_protection=True)
        
        # Campo quântico de mercado
        self.campo_quantico = CampoQuanticoMercado(dimensao=512)
        self.analisador = AnalisadorAvancadoMercado(dimensao=512)
        
        # Coordenador multi-exchange - já inicializa com KuCoin e Kraken
        self.multi_exchange = MultiExchangeTrading()
        
        # Adiciona pares de trading
        self.pares_trading = self.multi_exchange.pares_comuns
        logger.info(f"Pares de trading: {', '.join(self.pares_trading)}")
    
    def _inicializar_quantum_trading(self):
        """Inicializa os componentes avançados de quantum trading"""
        try:
            logger.info("Inicializando componentes quantum_trading...")
            
            try:
                from quantum_trading.market_consciousness import MarketConsciousness
                self.market_consciousness = MarketConsciousness()
                logger.info("Consciência de mercado inicializada com sucesso")
            except (ImportError, NameError) as e:
                logger.error(f"Erro ao inicializar MarketConsciousness: {str(e)}")
                # Criar uma classe simulada para MarketConsciousness
                class SimulatedMarketConsciousness:
                    def get_consciousness_metrics(self):
                        return {"estado": "simulado", "nivel": 0.5, "confianca": 0.6}
                self.market_consciousness = SimulatedMarketConsciousness()
                logger.warning("Usando simulação básica para consciência de mercado")
            
            try:
                from quantum_trading.morphic_field import MorphicField
                self.morphic_field = MorphicField()
                logger.info("Campo mórfico inicializado com sucesso")
            except (ImportError, NameError) as e:
                logger.error(f"Erro ao inicializar MorphicField: {str(e)}")
                # Criar uma classe simulada para MorphicField
                class SimulatedMorphicField:
                    def get_field_metrics(self):
                        return {"campo": "simulado", "intensidade": 0.4, "coerencia": 0.7}
                self.morphic_field = SimulatedMorphicField()
                logger.warning("Usando simulação básica para campo mórfico")
            
            try:
                from quantum_trading.quantum_portfolio import QuantumPortfolio
                self.quantum_portfolio = QuantumPortfolio(initial_balance=self.saldo_inicial)
                logger.info("Portfólio quântico inicializado com sucesso")
            except (ImportError, NameError) as e:
                logger.error(f"Erro ao inicializar QuantumPortfolio: {str(e)}")
                # Criar uma classe simulada para QuantumPortfolio
                class SimulatedQuantumPortfolio:
                    def __init__(self, initial_balance=1000.0):
                        self.balance = initial_balance
                    
                    def update_portfolio(self):
                        pass
                    
                    def get_balances(self):
                        return {"USDT": self.balance}
                self.quantum_portfolio = SimulatedQuantumPortfolio(initial_balance=self.saldo_inicial)
                logger.warning("Usando simulação básica para portfólio quântico")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar quantum_trading: {str(e)}")
            logger.error(traceback.format_exc())
            self.usar_quantum_modules = False
            
    def _get_market_api_for_symbol(self, symbol):
        """Obtém a API de mercado apropriada para o símbolo dado"""
        # Por padrão, usar KuCoin como exchange primária
        exchange_name = 'kucoin'
        
        # Adaptador simples para interface com MarketAPI do quantum_trading
        class MarketAPIAdapter:
            def __init__(self, exchange_api, symbol):
                self.exchange_api = exchange_api
                self.symbol = symbol
                
            def get_ticker(self, symbol=None):
                symbol = symbol or self.symbol
                return self.exchange_api.get_ticker(symbol)
                
            def place_order(self, symbol, side, quantity, price=None, order_type="market"):
                logger.info(f"Simulando ordem: {symbol} {side} {quantity} {price} {order_type}")
                return {"orderId": f"simulated-{datetime.now().timestamp()}", "status": "filled"}
                
            def get_balance(self):
                # Simular balances para teste
                return {"USDT": 1000.0, "BTC": 0.01, "ETH": 0.1}
        
        # Obter a exchange atual para o símbolo
        exchange_api = self.multi_exchange.exchanges.get(exchange_name)
        
        # Criar e retornar o adaptador
        return MarketAPIAdapter(exchange_api, symbol)
    
    def atualizar_precos(self) -> Dict[str, float]:
        """Atualiza os preços de todos os pares de trading"""
        precos_atualizados = {}
        
        try:
            # Se estamos usando módulos quantum_trading
            if self.usar_quantum_modules:
                # Atualizar preços através dos traders quantum
                for symbol in self.pares_trading:
                    if symbol in self.traders:
                        ticker = self.traders[symbol].market_api.get_ticker(symbol)
                        if 'price' in ticker:
                            precos_atualizados[symbol] = float(ticker['price'])
                            logger.info(f"Preço atualizado para {symbol}: {precos_atualizados[symbol]}")
            
            # Caso os traders quantum não estejam disponíveis ou não retornem preços
            if not precos_atualizados:
                # Usar o multi_exchange diretamente
                for symbol in self.pares_trading:
                    ticker = self.multi_exchange.exchanges["kucoin"].get_ticker(symbol)
                    if 'price' in ticker:
                        precos_atualizados[symbol] = float(ticker['price'])
                        logger.info(f"Preço atualizado para {symbol}: {precos_atualizados[symbol]}")
            
            # Salvar preços anteriores e atualizar preços atuais
            self.precos_anteriores = self.precos.copy() if hasattr(self, 'precos_anteriores') else {}
            self.precos = precos_atualizados
            
            # Atualizar campos mórficos e quânticos com novos preços
            if self.usar_quantum_modules:
                precos_list = list(precos_atualizados.values())
                if precos_list:
                    self.market_consciousness.update_field(precos_list)
                    self.morphic_field.process_data(precos_list)
            
            return self.precos
            
        except Exception as e:
            logger.error(f"Erro ao atualizar preços: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def executar_trades(self) -> Dict[str, Any]:
        """Executa trades usando AutoTrader se disponível ou lógica tradicional"""
        resultados = {}
        
        if not self.precos:
            logger.warning("Nenhum preço disponível para executar trades")
            return {"status": "erro", "mensagem": "Nenhum preço disponível"}
        
        try:
            # Se estamos usando módulos quantum_trading
            if self.usar_quantum_modules and hasattr(self, 'traders'):
                logger.info("Executando trades com AutoTrader...")
                
                for symbol in self.pares_trading:
                    try:
                        if symbol not in self.traders or symbol not in self.precos:
                            continue
                        
                        # Obter preço atual
                        preco_atual = self.precos[symbol]
                        
                        # Atualizar trader com preço atual e obter resultado
                        resultado = self.traders[symbol].update(preco_atual)
                        
                        if resultado:
                            logger.info(f"Resultado do AutoTrader para {symbol}: {resultado}")
                            
                            # Processar resultado do trader
                            if 'action' in resultado:
                                action = resultado['action']
                                
                                if action == 'buy':
                                    logger.info(f"🟢 COMPRA recomendada para {symbol} a {preco_atual}")
                                    # Executar compra real se em modo real
                                    if self.modo_real:
                                        self._executar_compra_real(symbol, resultado.get('amount', self.valor_operacao/preco_atual))
                                
                                elif action == 'sell':
                                    logger.info(f"🔴 VENDA recomendada para {symbol} a {preco_atual}")
                                    # Executar venda real se em modo real
                                    if self.modo_real:
                                        self._executar_venda_real(symbol, resultado.get('amount', 0))
                                
                                elif action == 'hold':
                                    logger.info(f"⚪ AGUARDAR recomendado para {symbol}")
                            
                            resultados[symbol] = resultado
                    except Exception as e:
                        error_msg = f"Erro ao processar trading para {symbol}: {str(e)}"
                        if self.ignorar_erros_cgr:
                            logger.warning(error_msg)
                            logger.warning(f"Ignorando erro e continuando execução para outros pares")
                            resultados[symbol] = {"status": "erro", "mensagem": str(e)}
                        else:
                            logger.error(error_msg)
                            raise
            else:
                logger.info("AutoTrader não disponível, usando análise CGR padrão...")
                # Aqui implementaríamos a lógica tradicional de trading baseada em CGR
                for symbol in self.pares_trading:
                    try:
                        if symbol not in self.precos:
                            continue
                            
                        # Verificar se temos análise CGR para este símbolo
                        if not hasattr(self, 'resultados_analise') or not self.resultados_analise or 'analise_cgr' not in self.resultados_analise:
                            logger.warning(f"Análise CGR não disponível para {symbol}, pulando...")
                            continue
                            
                        analise_cgr = self.resultados_analise.get('analise_cgr', {}).get(symbol, {})
                        
                        # Verificar se temos dados suficientes para tomar decisão
                        if not analise_cgr:
                            logger.warning(f"Dados CGR insuficientes para {symbol}, pulando...")
                            continue
                            
                        # Lógica de trading baseada na análise CGR
                        fractal_dim = analise_cgr.get('fractal_dimension', 0)
                        arbitrage_potential = analise_cgr.get('arbitrage_potential', 0)
                        
                        # Decisão de trading simplificada
                        sinal = None
                        if fractal_dim > 1.6 and arbitrage_potential > 0.6:
                            sinal = 'compra'
                            logger.info(f"🟢 COMPRA recomendada para {symbol} baseada em análise CGR")
                            if self.modo_real:
                                valor_operacao = self.saldo_atual * self.risco_por_operacao
                                quantidade = valor_operacao / self.precos[symbol]
                                self._executar_compra_real(symbol, quantidade)
                        elif fractal_dim < 1.4 and arbitrage_potential < 0.4:
                            sinal = 'venda'
                            logger.info(f"🔴 VENDA recomendada para {symbol} baseada em análise CGR")
                            if self.modo_real:
                                self._executar_venda_real(symbol)
                        else:
                            logger.info(f"⚪ AGUARDAR recomendado para {symbol} baseado em análise CGR")
                            
                        resultados[symbol] = {
                            "acao": sinal or "aguardar",
                            "fractal_dimension": fractal_dim,
                            "arbitrage_potential": arbitrage_potential
                        }
                    except Exception as e:
                        error_msg = f"Erro ao processar trading CGR para {symbol}: {str(e)}"
                        if self.ignorar_erros_cgr:
                            logger.warning(error_msg)
                            logger.warning("Usando valores padrão para análise CGR")
                            resultados[symbol] = {"status": "erro", "mensagem": str(e)}
                        else:
                            logger.error(error_msg)
                            raise
            return resultados
            
        except Exception as e:
            logger.error(f"Erro ao executar trades: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "erro", "mensagem": str(e)}
    
    def _executar_compra_real(self, symbol: str, quantidade: float) -> Dict[str, Any]:
        """Executa uma compra real na exchange"""
        try:
            if not self.modo_real:
                logger.info(f"[SIMULAÇÃO] Compra de {quantidade} {symbol} a {self.precos[symbol]}")
                return {"status": "simulado", "symbol": symbol, "side": "buy", "quantidade": quantidade}
            
            logger.info(f"[REAL] Executando compra de {quantidade} {symbol} a {self.precos[symbol]}")
            
            # Verificar saldo disponível
            moeda_quote = symbol.split('-')[1]  # USDT em BTC-USDT
            saldo = self.portfolio.get(moeda_quote, 0)
            
            if saldo < quantidade * self.precos[symbol]:
                logger.warning(f"Saldo insuficiente para compra: {saldo} {moeda_quote}")
                return {"status": "erro", "mensagem": "Saldo insuficiente"}
            
            # Executar ordem na exchange via AutoTrader
            if symbol in self.traders:
                resultado = self.traders[symbol].execute_buy(quantidade)
                logger.info(f"Resultado da compra: {resultado}")
                return resultado
            else:
                # Fallback para execução direta via exchange
                logger.warning("AutoTrader não disponível para este símbolo")
                return {"status": "erro", "mensagem": "AutoTrader não disponível"}
                
        except Exception as e:
            logger.error(f"Erro ao executar compra: {str(e)}")
            return {"status": "erro", "mensagem": str(e)}
    
    def _executar_venda_real(self, symbol: str, quantidade: float = 0) -> Dict[str, Any]:
        """Executa uma venda real na exchange"""
        try:
            moeda_base = symbol.split('-')[0]  # BTC em BTC-USDT
            
            # Se quantidade não especificada, vender todo o saldo disponível
            if quantidade <= 0:
                quantidade = self.portfolio.get(moeda_base, 0)
            
            if quantidade <= 0:
                logger.warning(f"Nenhuma quantidade disponível para venda de {moeda_base}")
                return {"status": "erro", "mensagem": "Nenhuma quantidade disponível"}
            
            if not self.modo_real:
                logger.info(f"[SIMULAÇÃO] Venda de {quantidade} {moeda_base} a {self.precos[symbol]}")
                return {"status": "simulado", "symbol": symbol, "side": "sell", "quantidade": quantidade}
            
            logger.info(f"[REAL] Executando venda de {quantidade} {moeda_base} a {self.precos[symbol]}")
            
            # Executar ordem na exchange via AutoTrader
            if symbol in self.traders:
                resultado = self.traders[symbol].execute_sell(quantidade)
                logger.info(f"Resultado da venda: {resultado}")
                return resultado
            else:
                # Fallback para execução direta via exchange
                logger.warning("AutoTrader não disponível para este símbolo")
                return {"status": "erro", "mensagem": "AutoTrader não disponível"}
                
        except Exception as e:
            logger.error(f"Erro ao executar venda: {str(e)}")
            return {"status": "erro", "mensagem": str(e)}
    
    def atualizar_ciclo_trading_quantico(self) -> Dict[str, Any]:
        """
        Executa um ciclo completo de trading quântico, integrando todas as análises
        avançadas e executando trades conforme necessário.
        
        Returns:
            Dicionário com resultados do ciclo
        """
        resultados = {}
        
        try:
            # Registrar início do ciclo
            hora_atual = datetime.now()
            logger.info(f"\n{'='*50}")
            logger.info(f"Início do ciclo de trading quântico: {hora_atual.strftime('%H:%M:%S')}")
            
            # Atualizar preços atuais
            self.atualizar_precos()
            if not self.precos:
                return {"status": "erro", "mensagem": "Falha ao atualizar preços"}
            
            # Analisar metrics quânticas e atualizar campo mórfico se disponível
            if self.usar_quantum_modules:
                # Análise de consciência de mercado
                resultados['consciencia_mercado'] = self.market_consciousness.get_consciousness_metrics()
                logger.info(f"Consciência de mercado: {resultados['consciencia_mercado']}")
                
                # Análise de campo mórfico 
                resultados['campo_morfico'] = self.morphic_field.get_field_metrics()
                logger.info(f"Campo mórfico: {resultados['campo_morfico']}")
                
                # Atualizar portfólio quântico
                self.quantum_portfolio.update_portfolio()
                balances = self.quantum_portfolio.get_balances()
                resultados['portfolio'] = balances
                logger.info(f"Portfólio quântico: {balances}")
            else:
                # Usar análise CGR tradicional
                resultados['analise_cgr'] = {}
                for symbol in self.pares_trading:
                    if symbol in self.precos:
                        try:
                            self.cgr.adicionar_dados(self.precos[symbol])
                        except Exception as e:
                            error_msg = f"Erro ao adicionar dados CGR para {symbol}: {str(e)}"
                            if self.ignorar_erros_cgr:
                                logger.warning(error_msg)
                                logger.warning("Ignorando erro CGR e continuando execução")
                            else:
                                logger.error(error_msg)
                                raise
                        
                        try:
                            analise = self.cgr.analisar({"preco": self.precos[symbol]})
                            resultados['analise_cgr'][symbol] = analise
                            logger.info(f"Análise CGR para {symbol}: {analise}")
                        except Exception as e:
                            error_msg = f"Erro na análise CGR para {symbol}: {str(e)}"
                            if self.ignorar_erros_cgr:
                                logger.warning(error_msg)
                                logger.warning("Usando valores padrão para análise CGR")
                                resultados['analise_cgr'][symbol] = {
                                    "estado_quantico": 0.5, 
                                    "fractal_dimension": 1.5, 
                                    "pattern_count": 0, 
                                    "arbitrage_potential": 0.0
                                }
                            else:
                                logger.error(error_msg)
                                raise
            # Executar trades baseados nas análises
            resultados['trades'] = self.executar_trades()
            
            # Atualizar portfólio após trades
            self.atualizar_portfolio()
            
            # Verificar desempenho e realizar ajustes dinâmicos
            if self.usar_quantum_modules:
                for symbol, trader in self.traders.items():
                    # Ajustar parâmetros de trading com base no desempenho atual
                    trader.optimize_parameters()
                    
                    # Verificar oportunidades de arbitragem entre pares
                    trader.check_arbitrage_opportunities()
            
            # Salvar estado periodicamente
            if hora_atual.minute % 5 == 0 and hora_atual.second < 10:
                self._salvar_estado()
            
            # Registrar fim do ciclo
            logger.info(f"Fim do ciclo de trading: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*50}\n")
            
            return resultados
            
        except Exception as e:
            logger.error(f"Erro durante ciclo de trading quântico: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "erro", "mensagem": str(e)}
    
    def atualizar_portfolio(self) -> Dict[str, float]:
        """
        Atualiza o portfólio atual com base no estado das exchanges
        
        Returns:
            Portfolio atualizado
        """
        try:
            # Se estamos usando o QuantumPortfolioManager
            if self.usar_quantum_modules and hasattr(self, 'quantum_portfolio'):
                # Atualizar via quantum_portfolio
                self.quantum_portfolio.update_portfolio()
                balances = self.quantum_portfolio.get_balances()
                
                # Atualizar portfólio interno
                for moeda, valor in balances.items():
                    self.portfolio[moeda] = valor
                
                logger.info(f"Portfólio atualizado via QuantumPortfolioManager: {self.portfolio}")
                
            else:
                # Atualização tradicional
                # Implementar lógica para obter saldos direto das exchanges
                pass
                
            # Registrar valor atual do portfólio
            valor_total = self.calcular_valor_portfolio()
            logger.info(f"Valor atual do portfólio: {valor_total:.2f} USDT")
            
            return self.portfolio
            
        except Exception as e:
            logger.error(f"Erro ao atualizar portfólio: {str(e)}")
            return self.portfolio
    
    def calcular_valor_portfolio(self) -> float:
        """
        Calcula o valor total do portfólio em USDT
        
        Returns:
            Valor total do portfólio em USDT
        """
        valor_total = self.portfolio.get("USDT", 0)
        
        for symbol in self.pares_trading:
            if symbol in self.precos:
                moeda_base = symbol.split('-')[0]  # BTC em BTC-USDT
                
                # Obter quantidade disponível
                quantidade = self.portfolio.get(moeda_base, 0)
                
                # Calcular valor em USDT
                valor_total += quantidade * self.precos[symbol]
        
        return valor_total
    
    def _salvar_estado(self) -> bool:
        """
        Salva o estado atual do sistema em arquivo
        
        Returns:
            True se o estado foi salvo com sucesso
        """
        try:
            estado = {
                "timestamp": datetime.now().isoformat(),
                "portfolio": self.portfolio,
                "precos": self.precos,
                "modo_real": self.modo_real,
                "using_quantum_modules": self.usar_quantum_modules
            }
            
            if self.usar_quantum_modules:
                # Adicionar métricas quânticas
                estado["metricas_quanticas"] = {
                    "consciencia": self.market_consciousness.get_consciousness_metrics() if hasattr(self, 'market_consciousness') else {},
                    "campo_morfico": self.morphic_field.get_field_metrics() if hasattr(self, 'morphic_field') else {}
                }
            
            salvar_estado(self.arquivo_estado, estado)
            logger.info(f"Estado salvo em {self.arquivo_estado}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {str(e)}")
            return False
    
    def gerar_relatorio_final(self) -> Dict[str, Any]:
        """
        Gera um relatório final da operação
        
        Returns:
            Relatório com métricas e resultados
        """
        try:
            # Calcular valores finais
            valor_final = self.calcular_valor_portfolio()
            valor_inicial = self.saldo_inicial
            lucro = valor_final - valor_inicial
            percentual = (lucro / valor_inicial) * 100 if valor_inicial > 0 else 0
            
            # Calcular duração se ultima_atualizacao não estiver disponível
            if hasattr(self, 'ultima_atualizacao') and self.ultima_atualizacao:
                data_inicio = self.ultima_atualizacao
                duracao_minutos = (datetime.now() - self.ultima_atualizacao).total_seconds() / 60
            elif hasattr(self, 'inicio_execucao') and self.inicio_execucao:
                data_inicio = self.inicio_execucao
                duracao_minutos = (datetime.now() - self.inicio_execucao).total_seconds() / 60
            else:
                # Fallback para evitar erro
                data_inicio = datetime.now() - timedelta(minutes=5)  # assume 5 minutos como default
                duracao_minutos = 5.0
                
            # Preparar relatório
            relatorio = {
                "data_inicio": data_inicio.isoformat(),
                "data_fim": datetime.now().isoformat(),
                "duracao_minutos": duracao_minutos,
                "valor_inicial": valor_inicial,
                "valor_final": valor_final,
                "lucro_absoluto": lucro,
                "lucro_percentual": percentual,
                "portfolio_final": self.portfolio,
                "modo_real": self.modo_real,
                "usando_quantum_modules": self.usar_quantum_modules if hasattr(self, 'usar_quantum_modules') else False
            }
            
            # Adicionar métricas quânticas se disponíveis
            if self.usar_quantum_modules:
                relatorio["metricas_quanticas"] = {
                    "consciencia_mercado": self.market_consciousness.get_consciousness_metrics() if hasattr(self, 'market_consciousness') else {},
                    "campo_morfico": self.morphic_field.get_field_metrics() if hasattr(self, 'morphic_field') else {},
                }
                
                # Adicionar métricas específicas de cada trader
                relatorio["traders"] = {}
                for symbol, trader in self.traders.items():
                    # Ajustar parâmetros de trading com base no desempenho atual
                    trader.optimize_parameters()
                    
                    # Verificar oportunidades de arbitragem entre pares
                    trader.check_arbitrage_opportunities()
                    
                    relatorio["traders"][symbol] = {
                        "metricas": trader.get_performance_metrics(),
                        "posicoes": trader.get_positions()
                    }
            
            # Salvar relatório em arquivo
            arquivo_relatorio = f"dados/relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            salvar_estado(arquivo_relatorio, relatorio)
            logger.info(f"Relatório final salvo em {arquivo_relatorio}")
            
            return relatorio
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório final: {str(e)}")
            return {"status": "erro", "mensagem": str(e)}
    
    def executar_trading_real(self, duracao_minutos: int = 60) -> None:
        """
        Executa o ciclo de trading em tempo real por uma duração específica
        
        Args:
            duracao_minutos: Duração do ciclo de trading em minutos
        """
        try:
            # Verificar se estamos em modo real
            if not self.modo_real:
                logger.warning("Função chamada em modo de simulação! Recomendado executar em modo real.")
                print("ATENÇÃO: Executando em modo de SIMULAÇÃO.")
            else:
                print("INICIANDO TRADING REAL NAS EXCHANGES")
            
            # Definir tempo de execução
            inicio = datetime.now()
            fim_previsto = inicio + timedelta(minutes=duracao_minutos)
            
            logger.info(f"Iniciando ciclo de trading real por {duracao_minutos} minutos")
            logger.info(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Término previsto: {fim_previsto.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Inicializar portfolio atual
            self.atualizar_portfolio()
            self.saldo_inicial = self.calcular_valor_portfolio()
            logger.info(f"Saldo inicial: {self.saldo_inicial:.2f} USDT")
            
            # Configurar traders quânticos se disponíveis
            if self.usar_quantum_modules:
                self._configurar_traders_quanticos()
            
            # Ciclo principal de trading
            ciclos = 0
            ultima_analise_arbitragem = datetime.now() - timedelta(minutes=5)  # Forçar primeira análise
            
            print(f"Trading iniciado. Executando por {duracao_minutos} minutos até {fim_previsto.strftime('%H:%M:%S')}")
            
            while datetime.now() < fim_previsto:
                try:
                    ciclos += 1
                    hora_atual = datetime.now()
                    
                    # Imprimir cabeçalho do ciclo
                    print(f"\n{'='*50}")
                    print(f"CICLO {ciclos} | {hora_atual.strftime('%H:%M:%S')} | Restante: {(fim_previsto - hora_atual).seconds // 60} minutos")
                    
                    # Executar ciclo de trading quântico
                    resultado = self.atualizar_ciclo_trading_quantico()
                    
                    # Verificar oportunidades de arbitragem a cada 5 minutos
                    if (hora_atual - ultima_analise_arbitragem).seconds > 300:
                        logger.info("Verificando oportunidades de arbitragem entre exchanges")
                        if hasattr(self, 'multi_exchange'):
                            arbitragens = self.multi_exchange.arbitragem_oportunidades()
                            if arbitragens:
                                logger.info(f"Encontradas {len(arbitragens)} oportunidades de arbitragem")
                                print(f"Oportunidades de arbitragem detectadas: {len(arbitragens)}")
                                for arb in arbitragens[:3]:
                                    print(f"  {arb['symbol']}: Comprar em {arb['comprar_em']} ({arb['preco_compra']:.2f}) "
                                          f"e vender em {arb['vender_em']} ({arb['preco_venda']:.2f}) - Spread: {arb['spread']:.2f}%")
                        ultima_analise_arbitragem = hora_atual
                    
                    # Exibir informações do portfólio atual
                    valor_atual = self.calcular_valor_portfolio()
                    lucro_atual = valor_atual - self.saldo_inicial
                    percentual = (lucro_atual / self.saldo_inicial) * 100 if self.saldo_inicial > 0 else 0
                    
                    print(f"Portfolio atual: {valor_atual:.2f} USDT ({percentual:+.2f}%)")
                    
                    # Mostrar posições ativas
                    posicoes_ativas = []
                    for moeda in self.portfolio:
                        if moeda != "USDT" and self.portfolio[moeda] > 0:
                            symbol = f"{moeda}-USDT"
                            preco_atual = self.precos.get(symbol, 0)
                            valor = self.portfolio[moeda] * preco_atual
                            posicoes_ativas.append(f"{moeda}: {self.portfolio[moeda]:.6f} ({valor:.2f} USDT)")
                    
                    if posicoes_ativas:
                        print("Posições ativas:")
                        for pos in posicoes_ativas:
                            print(f"  {pos}")
                    else:
                        print("Nenhuma posição ativa no momento")
                    
                    # Verificar se há recomendações de trading
                    if 'trades' in resultado and resultado['trades']:
                        print("Recomendações de trading:")
                        for trade in resultado['trades']:
                            if isinstance(trade, dict) and 'status' in trade:
                                print(f"  {trade['symbol'] if 'symbol' in trade else ''}: {trade['status']} "
                                      f"({trade.get('mensagem', '')})")
                    
                    # Aguardar intervalo entre ciclos (30 segundos entre atualizações)
                    intervalo = 30
                    print(f"Aguardando {intervalo} segundos para próximo ciclo...")
                    time.sleep(intervalo)
                
                except KeyboardInterrupt:
                    logger.warning("Interrupção de teclado detectada durante ciclo")
                    print("\nInterrupção detectada! Finalizando ciclo de trading...")
                    break
                
                except Exception as e:
                    logger.error(f"Erro durante ciclo {ciclos}: {str(e)}")
                    print(f"Erro durante ciclo: {str(e)}")
                    time.sleep(10)
            
            # Trading finalizado, gerar relatório
            logger.info(f"Trading finalizado após {ciclos} ciclos")
            relatorio = self.gerar_relatorio_final()
            
            # Exibir resumo do relatório
            print("\n" + "="*50)
            print("TRADING FINALIZADO")
            print("="*50)
            print(f"Duração: {relatorio['duracao_minutos']:.2f} minutos")
            print(f"Saldo inicial: {relatorio['valor_inicial']:.2f} USDT")
            print(f"Saldo final: {relatorio['valor_final']:.2f} USDT")
            print(f"Resultado: {relatorio['lucro_absoluto']:.2f} USDT ({relatorio['lucro_percentual']:+.2f}%)")
            print("="*50)
            
            # Perguntar se deseja encerrar posições abertas
            if self.modo_real:
                posicoes_abertas = any(self.portfolio.get(moeda, 0) > 0 for moeda in self.portfolio if moeda != "USDT")
                if posicoes_abertas:
                    resposta = input("Deseja encerrar todas as posições abertas? (s/n): ").lower()
                    if resposta == 's':
                        print("Encerrando posições...")
                        for symbol in self.pares_trading:
                            moeda_base = symbol.split('-')[0]
                            if self.portfolio.get(moeda_base, 0) > 0:
                                resultado = self._executar_venda_real(symbol)
                                print(f"Venda de {moeda_base}: {resultado['status']}")
        
        except KeyboardInterrupt:
            logger.warning("Interrupção de teclado detectada durante ciclo")
            print("\nInterrupção detectada! Finalizando ciclo de trading...")
            sys.exit(0)
        
        except Exception as e:
            logger.error(f"Erro ao executar trading real: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Erro ao executar trading: {str(e)}")
        
        finally:
            self._salvar_estado()
            print("\nTrading finalizado. Estado salvo.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QUALIA Trading System - Operação em exchanges reais com CGR")
    parser.add_argument("--modo", choices=["simulacao", "real"], default="simulacao", 
                        help="Modo de operação: simulação ou real")
    parser.add_argument("--duracao", type=int, default=60, 
                        help="Duração da sessão de trading em minutos")
    parser.add_argument("--saldo", type=float, default=100.0, 
                        help="Saldo inicial para simulação (ignorado em modo real)")
    parser.add_argument("--pares", type=str, default="BTC-USDT,ETH-USDT", 
                        help="Pares de trading separados por vírgula")
    parser.add_argument("--intervalo", type=int, default=60, 
                        help="Intervalo entre ciclos de trading em segundos")
    parser.add_argument("--risco", type=float, default=0.02, 
                        help="Percentual máximo de risco por operação (0.01 = 1%)")
    args = parser.parse_args()
    
    # Configurar logging
    formato_log = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=formato_log,
        handlers=[
            logging.FileHandler("qualia_trading.log"),
            logging.StreamHandler()
        ]
    )
    
    # Criar diretório para dados se não existir
    os.makedirs("dados", exist_ok=True)
    
    # Configurar credenciais das exchanges
    modo_real = args.modo == "real"
    
    print("="*80)
    print(f"QUALIA Trading System - Modo {'REAL' if modo_real else 'SIMULAÇÃO'}")
    print("="*80)
    
    if modo_real:
        credenciais_ok = True
        
        if not os.getenv("KUCOIN_API_KEY") or not os.getenv("KUCOIN_API_SECRET") or not os.getenv("KUCOIN_API_PASSPHRASE"):
            print("AVISO: Credenciais da KuCoin não encontradas nas variáveis de ambiente")
            credenciais_ok = False
            
        if not os.getenv("KRAKEN_API_KEY") or not os.getenv("KRAKEN_API_SECRET"):
            print("AVISO: Credenciais da Kraken não encontradas nas variáveis de ambiente")
            credenciais_ok = False
            
        if not credenciais_ok:
            print("\nPara usar o modo real, configure as variáveis de ambiente:")
            print("- KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE")
            print("- KRAKEN_API_KEY, KRAKEN_API_SECRET")
            print("\nOu forneça-as quando solicitado (não recomendado para uso regular)")
            
            usar_entrada_manual = input("Deseja fornecer as credenciais manualmente? (s/N): ").lower() == "s"
            
            if usar_entrada_manual:
                import getpass
                
                if not os.getenv("KUCOIN_API_KEY"):
                    kucoin_key = input("KuCoin API Key: ")
                    os.environ["KUCOIN_API_KEY"] = kucoin_key
                    
                if not os.getenv("KUCOIN_API_SECRET"):
                    kucoin_secret = getpass.getpass("KuCoin API Secret: ")
                    os.environ["KUCOIN_API_SECRET"] = kucoin_secret
                    
                if not os.getenv("KUCOIN_API_PASSPHRASE"):
                    kucoin_passphrase = getpass.getpass("KuCoin API Passphrase: ")
                    os.environ["KUCOIN_API_PASSPHRASE"] = kucoin_passphrase
                
                if not os.getenv("KRAKEN_API_KEY"):
                    kraken_key = input("Kraken API Key: ")
                    os.environ["KRAKEN_API_KEY"] = kraken_key
                    
                if not os.getenv("KRAKEN_API_SECRET"):
                    kraken_secret = getpass.getpass("Kraken API Secret: ")
                    os.environ["KRAKEN_API_SECRET"] = kraken_secret
            else:
                print("Operação cancelada. Configure as variáveis de ambiente e tente novamente.")
                sys.exit(1)
    
    pares_trading = args.pares.split(",")
    print(f"Pares de trading: {', '.join(pares_trading)}")
    
    risk_manager = RiskManager(
        max_position_size=args.saldo * args.risco
    )
    
    risk_manager.stop_loss_percent = args.risco
    risk_manager.trailing_stop_activation = 0.02
    risk_manager.trailing_stop_percent = 0.01
    
    simulador = SimuladorTradingQuantico(
        saldo_inicial=args.saldo,
        duracao_minutos=args.duracao,
        modo_real=modo_real,
        usar_quantum_modules=True
    )
    simulador._configurar_traders_quanticos()
    
    simulador.quantum_state_manager = QuantumStateManager(decoherence_protection=True)
    
    simulador.modulo_cgr = ModuloCGR()
    
    simulador.multi_exchange = MultiExchangeTrading()
    simulador.multi_exchange.pares_comuns = pares_trading
    
    simulador.intervalo_segundos = args.intervalo
    
    simulador.risk_manager = risk_manager
    
    confirmacao = "s"
    if modo_real:
        print("\nATENÇÃO: Você está prestes a iniciar operações REAIS nas exchanges!")
        print(f"O sistema operará por {args.duracao} minutos nos pares: {', '.join(pares_trading)}")
        print(f"Risco máximo por operação: {args.risco*100:.1f}%")
        confirmacao = input("\nDigite 's' para confirmar e iniciar as operações reais: ").lower()
    
    if confirmacao == "s":
        try:
            simulador.executar_trading_real(duracao_minutos=args.duracao)
            print("\nSessão de trading concluída!")
            print("Para ver os detalhes completos, verifique o arquivo de log e relatório na pasta 'dados'")
        except KeyboardInterrupt:
            logger.warning("Interrupção de teclado detectada durante ciclo")
            print("\nInterrupção detectada! Finalizando ciclo de trading...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Erro durante ciclo: {str(e)}")
            print(f"Erro durante ciclo: {str(e)}")
            time.sleep(10)
    else:
        print("Operação cancelada pelo usuário.")
