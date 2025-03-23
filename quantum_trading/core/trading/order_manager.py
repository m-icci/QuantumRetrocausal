"""
Gerenciador de ordens de trading.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import json
import hmac
import hashlib
import base64
import time

from .trading_config import TradingConfig

class OrderManager:
    """Gerenciador de ordens de trading."""
    
    def __init__(self, config: TradingConfig):
        """
        Inicializa o gerenciador de ordens.
        
        Args:
            config: Configuração do trading.
        """
        self.config = config
        self.logger = logging.getLogger('OrderManager')
        self.orders: Dict[str, Any] = {}
        self.active_positions: Dict[str, Any] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Configurações da exchange
        self.exchange = config.exchange
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.api_passphrase = config.api_passphrase
        
        # URLs das APIs
        self.base_url = self._get_base_url()
    
    def _get_base_url(self) -> str:
        """Retorna URL base da API da exchange."""
        if self.exchange == "kucoin":
            return "https://api.kucoin.com"
        elif self.exchange == "kraken":
            return "https://api.kraken.com/0"
        else:
            raise ValueError(f"Exchange não suportada: {self.exchange}")
    
    async def start(self) -> bool:
        """
        Inicia o gerenciador de ordens.
        
        Returns:
            True se iniciado com sucesso.
        """
        try:
            # Cria sessão HTTP
            self.session = aiohttp.ClientSession()
            
            # Verifica conexão
            if not await self._check_connection():
                return False
            
            self.logger.info("Gerenciador de ordens iniciado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar gerenciador: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        Para o gerenciador de ordens.
        
        Returns:
            True se parado com sucesso.
        """
        try:
            # Fecha posições ativas
            await self._close_all_positions()
            
            # Fecha sessão HTTP
            if self.session:
                await self.session.close()
                self.session = None
            
            self.logger.info("Gerenciador de ordens parado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao parar gerenciador: {str(e)}")
            return False
    
    async def place_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executa uma ordem.
        
        Args:
            signal: Sinal de trading.
            
        Returns:
            Ordem executada ou None se erro.
        """
        try:
            # Valida sinal
            if not self._validate_signal(signal):
                return None
            
            # Prepara ordem
            order = self._prepare_order(signal)
            
            # Executa ordem
            response = await self._execute_order(order)
            
            # Processa resposta
            if response:
                self._process_order_response(response)
                return response
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao executar ordem: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancela uma ordem.
        
        Args:
            order_id: ID da ordem.
            
        Returns:
            True se cancelada com sucesso.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("cancel_order")
            params = {'order_id': order_id}
            
            # Executa requisição
            response = await self._make_request("POST", endpoint, params)
            
            if response:
                self.logger.info(f"Ordem cancelada: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao cancelar ordem: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém status de uma ordem.
        
        Args:
            order_id: ID da ordem.
            
        Returns:
            Status da ordem ou None se erro.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("order_status")
            params = {'order_id': order_id}
            
            # Executa requisição
            response = await self._make_request("GET", endpoint, params)
            
            if response:
                return self._process_order_status(response)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao obter status da ordem: {str(e)}")
            return None
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Obtém posições ativas.
        
        Returns:
            Lista de posições.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("positions")
            
            # Executa requisição
            response = await self._make_request("GET", endpoint)
            
            if response:
                return self._process_positions(response)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erro ao obter posições: {str(e)}")
            return []
    
    async def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """
        Atualiza uma posição.
        
        Args:
            position_id: ID da posição.
            updates: Atualizações.
            
        Returns:
            True se atualizada com sucesso.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("update_position")
            params = {'position_id': position_id, **updates}
            
            # Executa requisição
            response = await self._make_request("POST", endpoint, params)
            
            if response:
                self.logger.info(f"Posição atualizada: {position_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar posição: {str(e)}")
            return False
    
    async def _check_connection(self) -> bool:
        """Verifica conexão com a exchange."""
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("time")
            
            # Executa requisição
            response = await self._make_request("GET", endpoint)
            
            return response is not None
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar conexão: {str(e)}")
            return False
    
    async def _close_all_positions(self) -> None:
        """Fecha todas as posições ativas."""
        try:
            # Obtém posições
            positions = await self.get_positions()
            
            # Fecha cada posição
            for position in positions:
                await self._close_position(position['id'])
                
        except Exception as e:
            self.logger.error(f"Erro ao fechar posições: {str(e)}")
    
    async def _close_position(self, position_id: str) -> bool:
        """
        Fecha uma posição.
        
        Args:
            position_id: ID da posição.
            
        Returns:
            True se fechada com sucesso.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("close_position")
            params = {'position_id': position_id}
            
            # Executa requisição
            response = await self._make_request("POST", endpoint, params)
            
            if response:
                self.logger.info(f"Posição fechada: {position_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao fechar posição: {str(e)}")
            return False
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valida sinal de trading.
        
        Args:
            signal: Sinal de trading.
            
        Returns:
            True se válido.
        """
        try:
            # Verifica campos obrigatórios
            required_fields = ['symbol', 'side', 'type', 'price', 'size']
            if not all(field in signal for field in required_fields):
                return False
            
            # Verifica valores
            if signal['price'] <= 0 or signal['size'] <= 0:
                return False
            
            # Verifica tipo de ordem
            if signal['type'] not in ['market', 'limit']:
                return False
            
            # Verifica lado da ordem
            if signal['side'] not in ['buy', 'sell']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao validar sinal: {str(e)}")
            return False
    
    def _prepare_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara ordem para execução.
        
        Args:
            signal: Sinal de trading.
            
        Returns:
            Ordem preparada.
        """
        try:
            # Cria ordem
            order = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'type': signal['type'],
                'price': signal['price'],
                'size': signal['size'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Adiciona campos específicos da exchange
            if self.exchange == "kucoin":
                order.update({
                    'clientOid': str(int(time.time() * 1000)),
                    'stp': 'DC'  # Decrease and Cancel
                })
            elif self.exchange == "kraken":
                order.update({
                    'userref': str(int(time.time() * 1000)),
                    'validate': False
                })
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar ordem: {str(e)}")
            return {}
    
    async def _execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executa ordem na exchange.
        
        Args:
            order: Ordem a executar.
            
        Returns:
            Resposta da exchange ou None se erro.
        """
        try:
            # Prepara requisição
            endpoint = self._get_endpoint("place_order")
            
            # Executa requisição
            response = await self._make_request("POST", endpoint, order)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erro ao executar ordem: {str(e)}")
            return None
    
    def _process_order_response(self, response: Dict[str, Any]) -> None:
        """
        Processa resposta da ordem.
        
        Args:
            response: Resposta da exchange.
        """
        try:
            # Extrai dados
            order_id = response.get('orderId')
            if not order_id:
                return
            
            # Atualiza ordens
            self.orders[order_id] = {
                'id': order_id,
                'status': 'open',
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao processar resposta: {str(e)}")
    
    def _process_order_status(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa status da ordem.
        
        Args:
            response: Resposta da exchange.
            
        Returns:
            Status processado.
        """
        try:
            # Extrai dados
            order_id = response.get('orderId')
            status = response.get('status')
            
            if not order_id or not status:
                return {}
            
            # Atualiza status
            if order_id in self.orders:
                self.orders[order_id]['status'] = status
                self.orders[order_id]['last_update'] = datetime.now().isoformat()
            
            return {
                'order_id': order_id,
                'status': status,
                'response': response
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao processar status: {str(e)}")
            return {}
    
    def _process_positions(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processa posições.
        
        Args:
            response: Resposta da exchange.
            
        Returns:
            Lista de posições processadas.
        """
        try:
            # Extrai dados
            positions = response.get('positions', [])
            
            # Processa cada posição
            processed_positions = []
            for position in positions:
                position_id = position.get('id')
                if position_id:
                    self.active_positions[position_id] = position
                    processed_positions.append(position)
            
            return processed_positions
            
        except Exception as e:
            self.logger.error(f"Erro ao processar posições: {str(e)}")
            return []
    
    def _get_endpoint(self, action: str) -> str:
        """
        Retorna endpoint da API.
        
        Args:
            action: Ação da API.
            
        Returns:
            Endpoint completo.
        """
        if self.exchange == "kucoin":
            endpoints = {
                'time': '/api/v1/timestamp',
                'place_order': '/api/v1/orders',
                'cancel_order': '/api/v1/orders/{order_id}',
                'order_status': '/api/v1/orders/{order_id}',
                'positions': '/api/v1/positions',
                'update_position': '/api/v1/positions/{position_id}',
                'close_position': '/api/v1/positions/{position_id}'
            }
        elif self.exchange == "kraken":
            endpoints = {
                'time': '/public/Time',
                'place_order': '/private/AddOrder',
                'cancel_order': '/private/CancelOrder',
                'order_status': '/private/QueryOrders',
                'positions': '/private/OpenPositions',
                'update_position': '/private/EditPosition',
                'close_position': '/private/ClosePosition'
            }
        else:
            raise ValueError(f"Exchange não suportada: {self.exchange}")
        
        return endpoints.get(action, '')
    
    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Executa requisição à API.
        
        Args:
            method: Método HTTP.
            endpoint: Endpoint da API.
            params: Parâmetros da requisição.
            
        Returns:
            Resposta da API ou None se erro.
        """
        try:
            if not self.session:
                return None
            
            # Prepara URL
            url = f"{self.base_url}{endpoint}"
            
            # Prepara headers
            headers = self._prepare_headers(method, endpoint, params)
            
            # Executa requisição
            async with self.session.request(method, url, headers=headers, json=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Erro na requisição: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro na requisição: {str(e)}")
            return None
    
    def _prepare_headers(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Prepara headers da requisição.
        
        Args:
            method: Método HTTP.
            endpoint: Endpoint da API.
            params: Parâmetros da requisição.
            
        Returns:
            Headers da requisição.
        """
        try:
            # Headers básicos
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Adiciona autenticação
            if self.exchange == "kucoin":
                timestamp = str(int(time.time() * 1000))
                headers.update({
                    'KC-API-KEY': self.api_key,
                    'KC-API-TIMESTAMP': timestamp,
                    'KC-API-PASSPHRASE': self.api_passphrase,
                    'KC-API-SIGN': self._generate_kucoin_signature(method, endpoint, timestamp, params)
                })
            elif self.exchange == "kraken":
                headers.update({
                    'API-Key': self.api_key,
                    'API-Sign': self._generate_kraken_signature(method, endpoint, params)
                })
            
            return headers
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar headers: {str(e)}")
            return {}
    
    def _generate_kucoin_signature(self, method: str, endpoint: str, timestamp: str, params: Dict[str, Any] = None) -> str:
        """
        Gera assinatura para KuCoin.
        
        Args:
            method: Método HTTP.
            endpoint: Endpoint da API.
            timestamp: Timestamp da requisição.
            params: Parâmetros da requisição.
            
        Returns:
            Assinatura gerada.
        """
        try:
            # Prepara string para assinatura
            message = f"{timestamp}{method}{endpoint}"
            if params:
                message += json.dumps(params)
            
            # Gera assinatura
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar assinatura KuCoin: {str(e)}")
            return ""
    
    def _generate_kraken_signature(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> str:
        """
        Gera assinatura para Kraken.
        
        Args:
            method: Método HTTP.
            endpoint: Endpoint da API.
            params: Parâmetros da requisição.
            
        Returns:
            Assinatura gerada.
        """
        try:
            # Prepara string para assinatura
            nonce = str(int(time.time() * 1000))
            message = f"{nonce}{method}{endpoint}"
            if params:
                message += json.dumps(params)
            
            # Gera assinatura
            signature = hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha512
            ).digest()
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar assinatura Kraken: {str(e)}")
            return "" 