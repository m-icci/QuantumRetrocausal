"""
Módulo de API de Mercado para o sistema de trading quântico QUALIA
Implementa interface robusta com exchanges, autenticação avançada e fallback
"""
import logging
import numpy as np
import os
import time
from datetime import datetime
import traceback
import sys
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import hmac
import hashlib
import base64
import urllib.parse
import aiohttp
import asyncio
import random
import platform

# Configuração para resolver problema do aiodns no Windows
if platform.system() == 'Windows':
    try:
        # Evitar importação duplicada
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logging.info("Política de event loop configurada para WindowsSelectorEventLoopPolicy")
    except Exception as e:
        logging.warning(f"Não foi possível configurar asyncio para Windows: {e}")

# Import exception class for error handling
class MarketAPIException(Exception):
    """Custom exception for market API errors"""
    pass

class RetryableError(MarketAPIException):
    """Exceção para erros que podem ser tentados novamente"""
    pass

logger = logging.getLogger(__name__)

class MarketAPI:
    """
    Interface robusta com APIs de exchanges de criptomoedas
    Implementa autenticação avançada, retry mechanism e fallback
    """
    
    def __init__(self, exchange_id: str = "kucoin"):
        """
        Inicializa a API de mercado com configurações avançadas
        
        Args:
            exchange_id: Identificador da exchange (kucoin, kraken)
        """
        # Carregar variáveis de ambiente
        load_dotenv(override=True)
        
        self.exchange_id = exchange_id.lower()
        self.logger = logging.getLogger(f"market_api.{exchange_id}")
        
        # Configurações de retry
        self.max_retries = 3
        self.retry_delay = 1.0  # segundos
        self.retry_backoff = 2.0  # multiplicador de delay
        
        # Cache de dados
        self._price_cache = {}
        self._balance_cache = {}
        self._last_cache_update = {}
        self._cache_lock = threading.Lock()
        
        # Configurações da exchange
        self._setup_exchange_config()
        
        # Thread pool para operações assíncronas
        self._thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Inicializar sessão aiohttp
        self._session = None
        
        # Validar credenciais
        self._validate_credentials()
            
        self.logger.info(f"MarketAPI inicializada para {exchange_id}")

    async def __aenter__(self):
        """Permite uso com 'async with' para gerenciamento automático de recursos"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Garante fechamento da sessão HTTP ao sair do contexto"""
        await self.close()
        return False  # Propaga exceções

    def _setup_exchange_config(self) -> None:
        """Configura parâmetros específicos da exchange"""
        if self.exchange_id == "kucoin":
            self.base_url = "https://api.kucoin.com"
            self.api_key = os.getenv("KUCOIN_API_KEY")
            self.api_secret = os.getenv("KUCOIN_API_SECRET")
            self.api_passphrase = os.getenv("KUCOIN_API_PASSPHRASE")
            self.api_version = "2"
            
            # Logar valores para debug
            self.logger.debug(f"KuCoin API Key: {self.api_key}")
            self.logger.debug(f"KuCoin API Secret: {'*' * len(self.api_secret) if self.api_secret else 'None'}")
            self.logger.debug(f"KuCoin API Passphrase: {'*' * len(self.api_passphrase) if self.api_passphrase else 'None'}")
            
        elif self.exchange_id == "kraken":
            self.base_url = "https://api.kraken.com"
            self.api_key = os.getenv("KRAKEN_API_KEY")
            self.api_secret = os.getenv("KRAKEN_API_SECRET")
            
            # Logar valores para debug
            self.logger.debug(f"Kraken API Key: {self.api_key}")
            self.logger.debug(f"Kraken API Secret: {'*' * len(self.api_secret) if self.api_secret else 'None'}")
            
        else:
            raise ValueError(f"Exchange não suportada: {self.exchange_id}")

    def _validate_credentials(self) -> None:
        """Valida credenciais da API e conectividade"""
        try:
            if self.exchange_id == "kucoin":
                required = ["KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSPHRASE"]
            else:  # kraken
                required = ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"]
                
            missing = []
            for var in required:
                value = os.getenv(var)
                if not value:
                    missing.append(var)
                    self.logger.error(f"Variável de ambiente {var} não encontrada")
                else:
                    self.logger.debug(f"Variável de ambiente {var} encontrada com {len(value)} caracteres")
                    
            if missing:
                raise MarketAPIException(f"Credenciais faltando: {', '.join(missing)}")
                
            # Testar conectividade
            # Nota: Removemos a chamada assíncrona aqui, pois _validate_credentials é chamado no __init__
            # e não é um método assíncrono
            self.logger.info("Credenciais validadas com sucesso")
            
        except Exception as e:
            self.logger.error(f"Falha na validação de credenciais: {str(e)}")
            raise
    
    async def initialize(self):
        """Método assíncrono para inicialização completa"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        
        # Testar conectividade com endpoints específicos por exchange
        try:
            if self.exchange_id == "kucoin":
                # KuCoin tem um endpoint específico para timestamp
                await self._request("GET", "/api/v1/timestamp")
            elif self.exchange_id == "kraken":
                # Kraken usa um endpoint público diferente para testar
                await self._request("GET", "/0/public/Time")
            else:
                # Fallback genérico
                await self._request("GET", "/")
            
            self.logger.info("Conectividade com a API testada com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao testar conectividade: {str(e)}")
            await self.close()  # Fechar sessão em caso de erro
            raise

    async def close(self):
        """Fecha a sessão HTTP e libera recursos"""
        if self._session is not None:
            try:
                await self._session.close()
                self.logger.debug("Sessão HTTP fechada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao fechar sessão HTTP: {str(e)}")
            finally:
                self._session = None
        
        # Fechar thread pool
        if hasattr(self, '_thread_pool') and self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=False)
                self.logger.debug("Thread pool encerrado")
            except Exception as e:
                self.logger.error(f"Erro ao encerrar thread pool: {str(e)}")

    def __del__(self):
        """Destrutor que tenta garantir liberação de recursos"""
        if self._session is not None:
            self.logger.warning("Sessão HTTP não foi fechada corretamente. Use 'await market_api.close()' ou 'async with' para gerenciar recursos.")
            # Não podemos chamar close() assíncrono aqui, então apenas registramos o aviso

    async def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None, 
                  auth: bool = True, retry_count: int = 0) -> Dict:
        """
        Realiza requisição HTTP para a API da exchange com retry mechanism
        
        Args:
            method: Método HTTP (GET, POST, etc)
            endpoint: Endpoint da API
            params: Parâmetros de query string
            data: Dados do corpo da requisição
            auth: Se True, adiciona autenticação
            retry_count: Contagem atual de retentativas
            
        Returns:
            Resposta da API como dicionário
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Garantir que a sessão está inicializada
        if self._session is None:
            self._session = aiohttp.ClientSession()
        
        # Adicionar autenticação se necessário
        if auth:
            if self.exchange_id == "kucoin":
                nonce = str(int(time.time() * 1000))
                
                if params:
                    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                else:
                    query_string = ''
                
                endpoint_with_query = f"{endpoint}?{query_string}" if query_string else endpoint
                
                body_str = json.dumps(data) if data else ''
                
                signature_payload = f"{nonce}{method}{endpoint_with_query}{body_str}"
                signature = base64.b64encode(
                    hmac.new(
                        self.api_secret.encode('utf-8'),
                        signature_payload.encode('utf-8'),
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
                
                headers.update({
                    "KC-API-KEY": self.api_key,
                    "KC-API-SIGN": signature,
                    "KC-API-TIMESTAMP": nonce,
                    "KC-API-PASSPHRASE": passphrase,
                    "KC-API-KEY-VERSION": self.api_version,
                    "Content-Type": "application/json"
                })
                
            elif self.exchange_id == "kraken":
                if method == "POST":
                    # Kraken utiliza formato específico para solicitações autenticadas
                    # O nonce deve ser inteiro e crescente (timestamp em milissegundos)
                    nonce = str(int(time.time() * 1000))
                    
                    # Criar o payload para a assinatura
                    post_data = (data or {}).copy()
                    post_data['nonce'] = nonce
                    
                    # A assinatura da Kraken é criada de forma especial:
                    # 1. Hash SHA256 do [nonce + dados_urlencodados]
                    # 2. Concatenar o caminho da API com o hash acima
                    # 3. Criar HMAC SHA512 utilizando a chave API secreta (decodificada de base64)
                    
                    post_data_str = urllib.parse.urlencode(post_data)
                    
                    # Gerar o hash do nonce + dados
                    sha256_hash = hashlib.sha256((str(nonce) + post_data_str).encode()).digest()
                    
                    # Concatenar o caminho da API com o hash
                    message = endpoint.encode() + sha256_hash
                    
                    try:
                        # Decodificar a chave secreta da API da base64
                        decoded_secret = base64.b64decode(self.api_secret)
                        
                        # Criar a assinatura HMAC com SHA512
                        signature = base64.b64encode(
                            hmac.new(
                                decoded_secret,
                                message,
                                hashlib.sha512
                            ).digest()
                        ).decode()
                        
                        headers.update({
                            "API-Key": self.api_key,
                            "API-Sign": signature,
                            "Content-Type": "application/x-www-form-urlencoded"
                        })
                        
                        self.logger.debug(f"Realizando request {method} para {url} (tentativa {retry_count + 1})")
                        self.logger.debug(f"Headers: {headers}")
                        self.logger.debug(f"Payload: {post_data}")
                        
                        # Para a Kraken, usar o objeto post_data diretamente (não JSON)
                        async with self._session.post(
                            url=url,
                            data=post_data,  # Enviar como form data (não JSON)
                            headers=headers,
                            timeout=30,
                            ssl=True  # Garantir conexão SSL
                        ) as response:
                            response_text = await response.text()
                            self.logger.debug(f"Resposta da Kraken: {response_text}")
                            
                            try:
                                response_data = json.loads(response_text)
                            except json.JSONDecodeError:
                                self.logger.error(f"Resposta não-JSON da Kraken: {response_text}")
                                raise MarketAPIException(f"Resposta inválida da API: {response_text}")
                            
                            # Verificar erros na resposta da Kraken
                            if 'error' in response_data and response_data['error'] and response_data['error'][0] != '':
                                error_msg = ', '.join(response_data['error'])
                                self.logger.error(f"API error: {error_msg}")
                                
                                # Determinar se o erro é retentável
                                retryable_errors = ['EAPI:Rate limit', 'EService:Unavailable', 'EService:Busy']
                                if any(e in error_msg for e in retryable_errors) and retry_count < self.max_retries:
                                    delay = self.retry_delay * (self.retry_backoff ** retry_count)
                                    self.logger.warning(f"Retryable error, waiting {delay}s before retry")
                                    await asyncio.sleep(delay)
                                    return await self._request(method, endpoint, params, data, auth, retry_count + 1)
                                
                                raise MarketAPIException(f"API error: {error_msg}")
                            
                            return response_data
                    
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if retry_count < self.max_retries:
                            delay = self.retry_delay * (self.retry_backoff ** retry_count)
                            self.logger.warning(f"Request failed: {str(e)}. Retrying in {delay}s")
                            await asyncio.sleep(delay)
                            return await self._request(method, endpoint, params, data, auth, retry_count + 1)
                        else:
                            self.logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                            raise MarketAPIException(f"Request failed: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Unexpected error: {str(e)}")
                        raise MarketAPIException(f"Unexpected error: {str(e)}")
                    
                    # Não continue para a parte geral da função
                    return {}
                else:
                    # Para Kraken GET requests (geralmente públicos)
                    headers.update({
                        "Content-Type": "application/json"
                    })
        
        try:
            self.logger.debug(f"Realizando request {method} para {url} (tentativa {retry_count + 1})")
            
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                data=json.dumps(data) if data and self.exchange_id == 'kucoin' else data,
                headers=headers,
                timeout=30
            ) as response:
                response_data = await response.json()
                
                # Verificar por erros na resposta
                if self.exchange_id == 'kucoin':
                    if 'code' in response_data and response_data['code'] != '200000':
                        error_msg = response_data.get('msg', 'Unknown error')
                        self.logger.error(f"API error: {error_msg}")
                        
                        # Determinar se o erro é retentável
                        retryable_codes = ['429000', '500000', '504000']  # Rate limit, server error, timeout
                        if response_data['code'] in retryable_codes and retry_count < self.max_retries:
                            delay = self.retry_delay * (self.retry_backoff ** retry_count)
                            self.logger.warning(f"Retryable error, waiting {delay}s before retry")
                            await asyncio.sleep(delay)
                            return await self._request(method, endpoint, params, data, auth, retry_count + 1)
                        
                        raise MarketAPIException(f"API error: {error_msg}")
                
                elif self.exchange_id == 'kraken':
                    if 'error' in response_data and response_data['error']:
                        error_msg = ', '.join(response_data['error'])
                        self.logger.error(f"API error: {error_msg}")
                        
                        # Determinar se o erro é retentável
                        retryable_errors = ['EAPI:Rate limit', 'EService:Unavailable', 'EService:Busy']
                        if any(e in error_msg for e in retryable_errors) and retry_count < self.max_retries:
                            delay = self.retry_delay * (self.retry_backoff ** retry_count)
                            self.logger.warning(f"Retryable error, waiting {delay}s before retry")
                            await asyncio.sleep(delay)
                            return await self._request(method, endpoint, params, data, auth, retry_count + 1)
                        
                        raise MarketAPIException(f"API error: {error_msg}")
                
                return response_data
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (self.retry_backoff ** retry_count)
                self.logger.warning(f"Request failed: {str(e)}. Retrying in {delay}s")
                await asyncio.sleep(delay)
                return await self._request(method, endpoint, params, data, auth, retry_count + 1)
            else:
                self.logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                raise MarketAPIException(f"Request failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise MarketAPIException(f"Unexpected error: {str(e)}")

    async def get_balance(self, currency: str = "USDT") -> float:
        """
        Obtém o saldo de uma moeda na exchange
        
        Args:
            currency: Símbolo da moeda (ex: USDT, BTC)
            
        Returns:
            Saldo total disponível (soma de todas as contas)
        """
        try:
            # Verificar cache
            cache_key = f"{self.exchange_id}:balance:{currency}"
            with self._cache_lock:
                now = time.time()
                if cache_key in self._balance_cache:
                    timestamp, balance = self._balance_cache[cache_key]
                    # Usar cache se tiver menos de 30 segundos
                    if now - timestamp < 30:
                        self.logger.debug(f"Usando saldo em cache para {currency}")
                        return balance
                    
            self.logger.info(f"Obtendo saldo de {currency} em {self.exchange_id}")
            
            if self.exchange_id == "kucoin":
                response = await self._request("GET", "/api/v1/accounts")
                
                # Registrar a resposta completa para diagnóstico
                self.logger.debug(f"Resposta completa da API: {response}")
                
                total_balance = 0.0
                
                if 'data' in response:
                    # Buscar nas diferentes contas (main, trade, margin, etc)
                    for account in response['data']:
                        if account['currency'] == currency:
                            account_type = account['type']
                            account_balance = float(account['available'])
                            
                            self.logger.info(f"Conta {account_type} tem {account_balance} {currency}")
                            total_balance += account_balance
                
                # Mostrar o total encontrado
                self.logger.info(f"Saldo total de {currency}: {total_balance}")
                
                # Armazenar no cache
                with self._cache_lock:
                    self._balance_cache[cache_key] = (time.time(), total_balance)
                    
                return total_balance
                
            elif self.exchange_id == "kraken":
                # A Kraken retorna ativos com prefixos específicos
                # Por exemplo: XXBT para BTC, ZUSD para USD, etc.
                response = await self._request("POST", "/0/private/Balance")
                
                if 'result' in response:
                    # Mapear o nome da moeda para o formato da Kraken
                    kraken_currency = self._get_kraken_asset_name(currency)
                    
                    # Buscar por várias possíveis representações do ativo
                    possible_assets = [
                        kraken_currency,            # Nome mapeado
                        currency,                   # Nome original
                        f"X{currency}",             # Formato para a maioria das criptomoedas 
                        f"Z{currency}",             # Formato para moedas fiduciárias
                        'XXBT' if currency == 'BTC' else None,  # Caso especial para BTC
                        'XETH' if currency == 'ETH' else None   # Caso especial para ETH
                    ]
                    
                    self.logger.debug(f"Buscando {currency} nas possíveis representações: {possible_assets}")
                    
                    # Verificar cada possível forma do ativo
                    for asset in possible_assets:
                        if asset and asset in response['result']:
                            balance = float(response['result'][asset])
                            self.logger.info(f"Encontrado {currency} como {asset}: {balance}")
                            
                            # Armazenar no cache
                            with self._cache_lock:
                                self._balance_cache[cache_key] = (time.time(), balance)
                                
                            return balance
                    
                    # Log detalhado para diagnóstico
                    assets_found = list(response['result'].keys())
                    if assets_found:
                        self.logger.debug(f"Ativos encontrados: {assets_found}")
                        
                    self.logger.warning(f"Moeda {currency} não encontrada em Kraken. Retornando 0.")
                    
                # Armazenar 0 no cache para evitar tentativas repetidas
                with self._cache_lock:
                    self._balance_cache[cache_key] = (time.time(), 0.0)
                    
                return 0.0
        
        except Exception as e:
            self.logger.error(f"Erro ao obter saldo de {currency}: {str(e)}")
            return 0.0

    def _get_kraken_asset_name(self, currency: str) -> str:
        """
        Converte o nome do ativo para o formato da Kraken
        
        Args:
            currency: Nome padrão do ativo
            
        Returns:
            Nome do ativo no formato da Kraken
        """
        kraken_map = {
            'BTC': 'XXBT',
            'ETH': 'XETH',
            'USDT': 'USDT',
            'USD': 'ZUSD',
            'EUR': 'ZEUR',
            'DOGE': 'XDG'
        }
        return kraken_map.get(currency, currency)

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """
        Obtém dados de ticker para um par de trading
        
        Args:
            symbol: Par de trading (ex: BTC/USDT)
            
        Returns:
            Dicionário com dados do ticker
        """
        try:
            # Verificar cache primeiro
            now = time.time()
            cache_key = f"{self.exchange_id}:{symbol}:ticker"
            
            with self._cache_lock:
                if cache_key in self._price_cache:
                    last_update, data = self._price_cache[cache_key]
                    # Usar cache se tiver menos de 10 segundos
                    if now - last_update < 10:
                        self.logger.debug(f"Usando ticker em cache para {symbol}")
                        return data
            
            # Traduzir símbolo para o formato da exchange
            symbol_formatted = self._format_symbol(symbol)
            
            if self.exchange_id == "kucoin":
                response = await self._request("GET", f"/api/v1/market/orderbook/level1?symbol={symbol_formatted}")
                
                if 'data' in response:
                    data = {
                        'price': float(response['data']['price']),
                        'bid': float(response['data']['bestBid']),
                        'ask': float(response['data']['bestAsk']),
                        'volume': float(response['data']['size'])
                    }
                    
                    # Atualizar cache
                    with self._cache_lock:
                        self._price_cache[cache_key] = (now, data)
                    
                    return data
                
            elif self.exchange_id == "kraken":
                # Verificar se o símbolo existe na Kraken
                try:
                    self.logger.debug(f"Verificando ticker para {symbol_formatted} na Kraken")
                    response = await self._request("GET", f"/0/public/Ticker?pair={symbol_formatted}")
                    
                    if 'result' in response and response['result']:
                        ticker_data = list(response['result'].values())[0]
                        
                        data = {
                            'price': float(ticker_data['c'][0]),
                            'bid': float(ticker_data['b'][0]),
                            'ask': float(ticker_data['a'][0]),
                            'volume': float(ticker_data['v'][1])
                        }
                        
                        # Atualizar cache
                        with self._cache_lock:
                            self._price_cache[cache_key] = (now, data)
                        
                        return data
                        
                    # Se não encontrou o símbolo, tentar uma alternativa
                    if 'error' in response and any('Unknown asset pair' in error for error in response['error']):
                        self.logger.warning(f"Par desconhecido: {symbol_formatted}, tentando alternativa")
                        
                        # Tentar com USD se era USDT
                        if 'USDT' in symbol_formatted:
                            alt_symbol = symbol_formatted.replace('USDT', 'USD')
                            self.logger.debug(f"Tentando par alternativo: {alt_symbol}")
                            
                            alt_response = await self._request("GET", f"/0/public/Ticker?pair={alt_symbol}")
                            if 'result' in alt_response and alt_response['result']:
                                ticker_data = list(alt_response['result'].values())[0]
                                
                                data = {
                                    'price': float(ticker_data['c'][0]),
                                    'bid': float(ticker_data['b'][0]),
                                    'ask': float(ticker_data['a'][0]),
                                    'volume': float(ticker_data['v'][1])
                                }
                                
                                # Atualizar cache
                                with self._cache_lock:
                                    self._price_cache[cache_key] = (now, data)
                                
                                return data
                except Exception as e:
                    self.logger.error(f"Erro ao obter ticker para {symbol} na Kraken: {str(e)}")
            
            # Fallback para um valor padrão ou erro
            self.logger.error(f"Não foi possível obter ticker para {symbol}")
            raise MarketAPIException(f"Não foi possível obter ticker para {symbol}")
            
        except Exception as e:
            self.logger.error(f"Erro ao obter ticker para {symbol}: {str(e)}")
            raise

    async def get_price(self, symbol: str) -> float:
        """
        Obtém o preço atual de um par de trading
        
        Args:
            symbol: Par de trading (ex: BTC/USDT)
            
        Returns:
            Preço atual
        """
        try:
            ticker = await self.get_ticker(symbol)
            return ticker['price']
        except Exception as e:
            self.logger.error(f"Erro ao obter preço para {symbol}: {str(e)}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Obtém o livro de ordens para um par de trading
        
        Args:
            symbol: Par de trading (ex: BTC/USDT)
            limit: Número de níveis a retornar
            
        Returns:
            Dicionário com livro de ordens
        """
        try:
            # Traduzir símbolo para o formato da exchange
            symbol_formatted = self._format_symbol(symbol)
            
            if self.exchange_id == "kucoin":
                response = await self._request("GET", f"/api/v1/market/orderbook/level2_20?symbol={symbol_formatted}")
                
                if 'data' in response:
                    return {
                        'asks': [[float(ask[0]), float(ask[1])] for ask in response['data']['asks']],
                        'bids': [[float(bid[0]), float(bid[1])] for bid in response['data']['bids']]
                    }
                
            elif self.exchange_id == "kraken":
                response = await self._request("GET", f"/0/public/Depth?pair={symbol_formatted}&count={limit}")
                
                if 'result' in response:
                    orderbook = list(response['result'].values())[0]
                    
                    return {
                        'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['asks']],
                        'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['bids']]
                    }
            
            raise MarketAPIException(f"Não foi possível obter orderbook para {symbol}")
            
        except Exception as e:
            self.logger.error(f"Erro ao obter orderbook para {symbol}: {str(e)}")
            raise

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[float]]:
        """
        Obtém dados de OHLCV (Open, High, Low, Close, Volume) para um par de trading
        
        Args:
            symbol: Par de trading (ex: BTC/USDT)
            timeframe: Intervalo de tempo (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Número de candles a retornar
            
        Returns:
            Lista de candles [timestamp, open, high, low, close, volume]
        """
        try:
            # Traduzir símbolo para o formato da exchange
            symbol_formatted = self._format_symbol(symbol)
            
            # Traduzir timeframe para o formato da exchange
            if self.exchange_id == "kucoin":
                # KuCoin usa: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 8hour, 12hour, 1day, 1week
                timeframe_mapping = {
                    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                    '1h': '1hour', '4h': '4hour', '12h': '12hour', '1d': '1day', '1w': '1week'
                }
                kucoin_timeframe = timeframe_mapping.get(timeframe, '1hour')
                
                response = await self._request("GET", f"/api/v1/market/candles?symbol={symbol_formatted}&type={kucoin_timeframe}&limit={limit}")
                
                if 'data' in response:
                    # KuCoin retorna dados neste formato: [timestamp, open, close, high, low, volume, turnover]
                    return [
                        [
                            int(candle[0]),  # timestamp
                            float(candle[1]),  # open
                            float(candle[3]),  # high
                            float(candle[4]),  # low
                            float(candle[2]),  # close
                            float(candle[5])   # volume
                        ] for candle in response['data']
                    ]
                
            elif self.exchange_id == "kraken":
                # Kraken usa: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
                timeframe_mapping = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
                }
                kraken_timeframe = timeframe_mapping.get(timeframe, 60)
                
                response = await self._request("GET", f"/0/public/OHLC?pair={symbol_formatted}&interval={kraken_timeframe}")
                
                if 'result' in response:
                    ohlc_data = list(response['result'].values())[0]
                    
                    # Kraken retorna dados neste formato: [timestamp, open, high, low, close, vwap, volume, count]
                    return [
                        [
                            int(candle[0]),  # timestamp
                            float(candle[1]),  # open
                            float(candle[2]),  # high
                            float(candle[3]),  # low
                            float(candle[4]),  # close
                            float(candle[6])   # volume
                        ] for candle in ohlc_data
                    ]
            
            raise MarketAPIException(f"Não foi possível obter dados OHLCV para {symbol}")
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados OHLCV para {symbol}: {str(e)}")
            raise

    def _format_symbol(self, symbol: str) -> str:
        """
        Formata um símbolo para o formato esperado pela exchange
        
        Args:
            symbol: Par de trading (ex: BTC/USDT)
            
        Returns:
            Símbolo formatado
        """
        base, quote = symbol.split('/')
        
        if self.exchange_id == "kucoin":
            return f"{base}-{quote}"
        elif self.exchange_id == "kraken":
            # Kraken usa XBT ao invés de BTC
            if base == "BTC":
                base = "XBT"
            if quote == "BTC":
                quote = "XBT"
                
            # Alguns pares têm formato especial no Kraken
            special_assets = ["XBT", "ETH", "USDT", "EUR", "USD", "GBP", "JPY"]
            
            if base in special_assets and quote in special_assets:
                return f"{base}{quote}"
            else:
                return f"{base}{quote}"
        
        return symbol  # Formato padrão
