#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import hmac
import base64
import hashlib
import json
import requests
from pprint import pprint
import logging
import urllib.parse
import asyncio
import sys
from contextlib import asynccontextmanager

# Configuração do logger
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('exchange_test')

class ExchangeBalanceTest:
    def __init__(self):
        # KuCoin
        self.kucoin_api_key = os.environ.get('KUCOIN_API_KEY', '')
        self.kucoin_secret = os.environ.get('KUCOIN_SECRET_KEY', '')
        self.kucoin_passphrase = os.environ.get('KUCOIN_PASSPHRASE', '')
        
        # Kraken
        self.kraken_api_key = os.environ.get('KRAKEN_API_KEY', '')
        self.kraken_secret = os.environ.get('KRAKEN_API_SECRET', '')
        
        # Validar credenciais
        missing_vars = []
        if not all([self.kucoin_api_key, self.kucoin_secret, self.kucoin_passphrase]):
            missing_vars.extend([var for var, val in {
                'KUCOIN_API_KEY': self.kucoin_api_key,
                'KUCOIN_SECRET_KEY': self.kucoin_secret,
                'KUCOIN_PASSPHRASE': self.kucoin_passphrase
            }.items() if not val])
            
        if not all([self.kraken_api_key, self.kraken_secret]):
            missing_vars.extend([var for var, val in {
                'KRAKEN_API_KEY': self.kraken_api_key,
                'KRAKEN_API_SECRET': self.kraken_secret
            }.items() if not val])
            
        if missing_vars:
            logger.warning(f"As seguintes variáveis de ambiente estão ausentes: {', '.join(missing_vars)}")
            logger.warning("Algumas funcionalidades podem não estar disponíveis")
        
        logger.info("Inicializando teste de saldo das exchanges")
        
        # Cache para respostas da API
        self._price_cache = {}
        self._cache_expiry = 60  # segundos

    def _generate_kucoin_signature(self, endpoint, method, params=None, data=None):
        """Gera a assinatura necessária para autenticação na API da KuCoin"""
        timestamp = str(int(time.time() * 1000))
        
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            endpoint = f"{endpoint}?{query_string}"
            
        body_str = json.dumps(data) if data else ''
        
        signature_payload = f"{timestamp}{method}{endpoint}{body_str}"
        
        signature = base64.b64encode(
            hmac.new(
                self.kucoin_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        passphrase = base64.b64encode(
            hmac.new(
                self.kucoin_secret.encode('utf-8'),
                self.kucoin_passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            "KC-API-KEY": self.kucoin_api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json"
        }
    
    def _generate_kraken_signature(self, endpoint, data):
        """Gera a assinatura necessária para autenticação na API da Kraken"""
        if not data:
            data = {}
        
        nonce = str(int(time.time() * 1000))
        data['nonce'] = nonce

        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(
            base64.b64decode(self.kraken_secret),
            message,
            hashlib.sha512
        )
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _make_kucoin_request(self, method, endpoint, params=None, data=None):
        """Realiza uma requisição à API da KuCoin com cache e retry"""
        base_url = "https://api.kucoin.com"
        
        # Verificar se é uma requisição de mercado (preço) que pode ser cacheada
        cache_key = None
        if method == 'GET' and '/market/' in endpoint and params:
            cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            now = time.time()
            
            # Verificar cache
            if cache_key in self._price_cache:
                timestamp, cached_data = self._price_cache[cache_key]
                if now - timestamp < self._cache_expiry:
                    logger.debug(f"Usando preço em cache para {cache_key}")
                    return cached_data
        
        # Se não tiver em cache, fazer requisição
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                headers = self._generate_kucoin_signature(endpoint, method, params, data)
                url = f"{base_url}{endpoint}"
                
                logger.info(f"Realizando requisição KuCoin {method} para {url} (tentativa {attempt+1})")
                response = requests.request(method, url, headers=headers, params=params, json=data, timeout=10)
                
                # Se for resposta de throttling, aguardar e tentar novamente
                if response.status_code == 429:
                    logger.warning(f"Rate limit atingido. Aguardando {retry_delay}s antes de tentar novamente.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponencial
                    continue
                
                if response.status_code != 200:
                    logger.error(f"Erro na API KuCoin: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return None
                
                result = response.json()
                
                # Armazenar em cache se aplicável
                if cache_key:
                    self._price_cache[cache_key] = (time.time(), result)
                    
                return result
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de requisição para KuCoin: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None
            except Exception as e:
                logger.error(f"Erro inesperado: {str(e)}")
                return None
    
    def _make_kraken_request(self, endpoint, data=None):
        """Realiza uma requisição à API da Kraken com retry e validação"""
        base_url = "https://api.kraken.com"
        if not data:
            data = {}
            
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                nonce = str(int(time.time() * 1000))
                data['nonce'] = nonce
                
                headers = {
                    'API-Key': self.kraken_api_key,
                    'API-Sign': self._generate_kraken_signature(endpoint, data)
                }
                
                url = f"{base_url}{endpoint}"
                logger.info(f"Realizando requisição Kraken POST para {url} (tentativa {attempt+1})")
                logger.debug(f"Headers: {headers}")
                logger.debug(f"Data: {data}")
                
                response = requests.post(url, headers=headers, data=data, timeout=15)
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response text: {response.text}")
                
                # Verificar por erros de rate limit
                if response.status_code == 429:
                    logger.warning(f"Rate limit atingido. Aguardando {retry_delay}s antes de tentar novamente.")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                    
                if response.status_code != 200:
                    logger.error(f"Erro na API Kraken: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return None
                    
                result = response.json()
                if 'error' in result and result['error']:
                    error_msg = ', '.join(result['error'])
                    logger.error(f"Erro retornado pela Kraken: {error_msg}")
                    
                    # Verificar se é erro de rate limit ou serviço indisponível
                    if any(e in error_msg for e in ['EAPI:Rate limit', 'EService:Unavailable', 'EService:Busy']) and attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                    return None
                    
                return result
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de requisição para Kraken: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None
            except Exception as e:
                logger.error(f"Erro inesperado: {str(e)}")
                return None
    
    def get_kucoin_price(self, symbol):
        """Obtém o preço atual de um par na KuCoin com cache"""
        params = {"symbol": f"{symbol}-USDT"}
        data = self._make_kucoin_request("GET", "/api/v1/market/orderbook/level1", params=params)
        if data and 'data' in data and 'price' in data['data']:
            return float(data['data']['price'])
        return None
    
    def get_kraken_price(self, symbol):
        """Obtém o preço atual de um par na Kraken com cache e tratamento de erros"""
        # Remover o sufixo .F para futuros
        base_symbol = symbol.split('.')[0]
        
        # Kraken usa nomenclatura diferente para alguns pares
        symbol_map = {
            'BTC': 'XBT',
            'DOGE': 'XDG',
            'USDT': 'USDT',  # Kraken aceita USDT diretamente
            'ETH': 'ETH',
            'SOL': 'SOL'
        }
        
        kraken_symbol = symbol_map.get(base_symbol, base_symbol)
        
        # Verificar se o par existe no cache
        cache_key = f"kraken_price:{kraken_symbol}"
        now = time.time()
        if cache_key in self._price_cache:
            timestamp, price = self._price_cache[cache_key]
            if now - timestamp < self._cache_expiry:
                logger.debug(f"Usando preço em cache para {kraken_symbol}: {price}")
                return price
        
        # Construir o par no formato da Kraken
        if kraken_symbol in ['XBT', 'ETH']:
            pair = f"X{kraken_symbol}ZUSDT"  # Usando USDT como base
        else:
            pair = f"{kraken_symbol}USDT"
            
        logger.debug(f"Consultando preço do par {pair} na Kraken")
        data = self._make_kraken_request("/0/public/Ticker", {"pair": pair})
        
        if data and 'result' in data and pair in data['result']:
            price = float(data['result'][pair]['c'][0])  # 'c' é o preço de fechamento atual
            # Armazenar no cache
            self._price_cache[cache_key] = (now, price)
            return price
        
        # Tentar com USDC como alternativa se USDT falhar
        if 'USDT' in pair:
            alt_pair = pair.replace('USDT', 'USD')
            logger.debug(f"Tentando par alternativo {alt_pair} na Kraken")
            data = self._make_kraken_request("/0/public/Ticker", {"pair": alt_pair})
            
            if data and 'result' in data and alt_pair in data['result']:
                price = float(data['result'][alt_pair]['c'][0])
                # Armazenar no cache
                self._price_cache[cache_key] = (now, price)
                return price
        
        return None
    
    def analyze_all_balances(self):
        """Analisa saldos em todas as exchanges com melhor tratamento de erros"""
        total_portfolio_value = 0.0
        results = {}
        
        # 1. KuCoin
        logger.info("\n===== ANÁLISE DE SALDOS KUCOIN =====")
        try:
            kucoin_data = self._make_kucoin_request("GET", "/api/v1/accounts")
            
            if kucoin_data and 'data' in kucoin_data:
                kucoin_total = 0.0
                accounts = kucoin_data['data']
                kucoin_balances = []
                
                for acc in accounts:
                    currency = acc.get('currency')
                    balance = float(acc.get('balance', 0))
                    account_type = acc.get('type', 'unknown')
                    
                    if balance > 0:
                        if currency == 'USDT':
                            value_usdt = balance
                        else:
                            price = self.get_kucoin_price(currency)
                            if price:
                                value_usdt = balance * price
                            else:
                                logger.warning(f"Não foi possível obter preço para {currency} na KuCoin")
                                value_usdt = 0
                                
                        logger.info(f"KuCoin - {currency} ({account_type}): {balance:.8f} ({value_usdt:.2f} USDT)")
                        kucoin_total += value_usdt
                        
                        # Armazenar para o resultado
                        kucoin_balances.append({
                            'currency': currency,
                            'balance': balance,
                            'account_type': account_type,
                            'value_usdt': value_usdt
                        })
                
                logger.info(f"Total KuCoin em USDT: {kucoin_total:.2f}")
                total_portfolio_value += kucoin_total
                
                # Armazenar no resultado
                results['kucoin'] = {
                    'balances': kucoin_balances,
                    'total_usdt': kucoin_total
                }
            else:
                logger.error("Não foi possível obter saldos da KuCoin")
                results['kucoin'] = {'error': 'Não foi possível obter saldos'}
                
        except Exception as e:
            logger.error(f"Erro ao analisar saldos da KuCoin: {str(e)}")
            results['kucoin'] = {'error': str(e)}
        
        # 2. Kraken
        logger.info("\n===== ANÁLISE DE SALDOS KRAKEN =====")
        try:
            kraken_data = self._make_kraken_request("/0/private/Balance")
            
            if kraken_data and 'result' in kraken_data:
                kraken_total = 0.0
                balances = kraken_data['result']
                kraken_balances = []
                
                for currency, balance in balances.items():
                    balance = float(balance)
                    
                    # Remover prefixos X e Z da Kraken
                    if currency.startswith('X'):
                        clean_currency = 'BTC' if currency == 'XXBT' else currency[1:]
                    elif currency.startswith('Z'):
                        clean_currency = currency[1:]
                    else:
                        clean_currency = currency
                    
                    if balance > 0:
                        if clean_currency == 'USDT':
                            value_usdt = balance
                        else:
                            price = self.get_kraken_price(clean_currency)
                            if price:
                                value_usdt = balance * price
                            else:
                                logger.warning(f"Não foi possível obter preço para {clean_currency} na Kraken")
                                value_usdt = 0
                                
                        logger.info(f"Kraken - {clean_currency}: {balance:.8f} ({value_usdt:.2f} USDT)")
                        kraken_total += value_usdt
                        
                        # Armazenar para o resultado
                        kraken_balances.append({
                            'currency': clean_currency,
                            'raw_currency': currency,
                            'balance': balance,
                            'value_usdt': value_usdt
                        })
                
                logger.info(f"Total Kraken em USDT: {kraken_total:.2f}")
                total_portfolio_value += kraken_total
                
                # Armazenar no resultado
                results['kraken'] = {
                    'balances': kraken_balances,
                    'total_usdt': kraken_total
                }
            else:
                logger.error("Não foi possível obter saldos da Kraken")
                results['kraken'] = {'error': 'Não foi possível obter saldos'}
                
        except Exception as e:
            logger.error(f"Erro ao analisar saldos da Kraken: {str(e)}")
            results['kraken'] = {'error': str(e)}
        
        # 3. Total Geral
        logger.info("\n===== VALOR TOTAL DO PORTFÓLIO =====")
        logger.info(f"Valor total em USDT: {total_portfolio_value:.6f}")
        
        results['total_portfolio'] = total_portfolio_value
        return results

@asynccontextmanager
async def run_with_timeout(timeout_seconds):
    """Context manager para executar código com timeout"""
    try:
        # Iniciar o temporizador
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        handle = loop.call_later(timeout_seconds, task.cancel)
        
        yield
    except asyncio.CancelledError:
        logger.error(f"Operação cancelada após {timeout_seconds} segundos")
        raise TimeoutError(f"Operação excedeu o tempo limite de {timeout_seconds} segundos")
    finally:
        # Cancelar o temporizador
        handle.cancel()

async def main_async():
    """Versão assíncrona da função principal"""
    try:
        logger.info("Iniciando análise de balances")
        
        # Executar com timeout para garantir que não fique preso
        async with run_with_timeout(60):  # 60 segundos de timeout
            # Executar em uma thread separada para não bloquear a loop de eventos
            loop = asyncio.get_running_loop()
            tester = ExchangeBalanceTest()
            results = await loop.run_in_executor(None, tester.analyze_all_balances)
            
            # Imprimir resultado formatado
            if results:
                logger.info("\n===== RESUMO DO PORTFÓLIO =====")
                
                # KuCoin
                if 'kucoin' in results:
                    if 'error' in results['kucoin']:
                        logger.error(f"Erro KuCoin: {results['kucoin']['error']}")
                    else:
                        logger.info(f"KuCoin: {results['kucoin']['total_usdt']:.2f} USDT")
                        
                # Kraken
                if 'kraken' in results:
                    if 'error' in results['kraken']:
                        logger.error(f"Erro Kraken: {results['kraken']['error']}")
                    else:
                        logger.info(f"Kraken: {results['kraken']['total_usdt']:.2f} USDT")
                        
                # Total
                logger.info(f"Total portfolio: {results['total_portfolio']:.2f} USDT")
                
            return results
            
    except TimeoutError as e:
        logger.error(f"Timeout: {str(e)}")
        return {'error': str(e)}
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def main():
    """Função principal que executa a análise de saldos"""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Operação interrompida pelo usuário")
        return {'error': 'Operação interrompida pelo usuário'}
    except Exception as e:
        logger.error(f"Erro na execução principal: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results and 'error' not in results else 1) 