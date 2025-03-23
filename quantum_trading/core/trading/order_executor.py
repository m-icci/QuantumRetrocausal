"""
Executor de ordens
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from ...data.data_loader import DataLoader

logger = logging.getLogger(__name__)

class OrderExecutor:
    """Executor de ordens"""
    
    def __init__(self, config: Dict):
        """
        Inicializa executor.
        
        Args:
            config: Configuração.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        
        # Estado
        self.is_connected = False
        self.active_orders = {}
        
    async def connect(self) -> None:
        """Conecta com exchange"""
        try:
            logger.info("Conectando com exchange")
            await self.data_loader.connect()
            self.is_connected = True
            logger.info("Conectado com exchange")
            
        except Exception as e:
            logger.error(f"Erro ao conectar com exchange: {str(e)}")
            raise
            
    async def disconnect(self) -> None:
        """Desconecta da exchange"""
        try:
            logger.info("Desconectando da exchange")
            await self.data_loader.disconnect()
            self.is_connected = False
            logger.info("Desconectado da exchange")
            
        except Exception as e:
            logger.error(f"Erro ao desconectar da exchange: {str(e)}")
            raise
            
    async def execute_order(self, order: Dict) -> bool:
        """
        Executa ordem.
        
        Args:
            order: Ordem.
            
        Returns:
            True se executada.
        """
        try:
            if not self.is_connected:
                logger.error("Não conectado com exchange")
                return False
                
            # Valida ordem
            if not self._validate_order(order):
                return False
                
            # Calcula custos
            costs = await self._calculate_costs(order)
            
            # Verifica saldo
            if not await self._check_balance(order, costs):
                return False
                
            # Executa ordem
            if self.config['trading']['mode'] == 'simulated':
                success = await self._simulate_order(order)
            else:
                success = await self._execute_real_order(order)
                
            if success:
                # Registra ordem
                self.active_orders[order['id']] = {
                    'order': order,
                    'costs': costs,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Ordem executada: {order['id']}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Erro ao executar ordem: {str(e)}")
            return False
            
    async def close_position(self, position: Dict) -> bool:
        """
        Fecha posição.
        
        Args:
            position: Posição.
            
        Returns:
            True se fechada.
        """
        try:
            if not self.is_connected:
                logger.error("Não conectado com exchange")
                return False
                
            # Cria ordem de fechamento
            order = {
                'id': f"close_{position['id']}",
                'symbol': position['symbol'],
                'type': 'market',
                'side': 'sell' if position['direction'] == 'long' else 'buy',
                'size': position['size']
            }
            
            # Executa ordem
            success = await self.execute_order(order)
            
            if success:
                # Remove ordem ativa
                if position['id'] in self.active_orders:
                    del self.active_orders[position['id']]
                    
                logger.info(f"Posição fechada: {position['id']}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {str(e)}")
            return False
            
    def _validate_order(self, order: Dict) -> bool:
        """
        Valida ordem.
        
        Args:
            order: Ordem.
            
        Returns:
            True se válida.
        """
        try:
            # Verifica campos obrigatórios
            required_fields = ['id', 'symbol', 'type', 'side', 'size']
            for field in required_fields:
                if field not in order:
                    logger.error(f"Campo obrigatório ausente: {field}")
                    return False
                    
            # Verifica tipo
            valid_types = ['market', 'limit']
            if order['type'] not in valid_types:
                logger.error(f"Tipo inválido: {order['type']}")
                return False
                
            # Verifica lado
            valid_sides = ['buy', 'sell']
            if order['side'] not in valid_sides:
                logger.error(f"Lado inválido: {order['side']}")
                return False
                
            # Verifica tamanho
            min_size = self.config['scalping']['min_trade_size']
            if order['size'] < min_size:
                logger.error(f"Tamanho menor que mínimo: {order['size']} < {min_size}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar ordem: {str(e)}")
            return False
            
    async def _calculate_costs(self, order: Dict) -> float:
        """
        Calcula custos.
        
        Args:
            order: Ordem.
            
        Returns:
            Custos totais.
        """
        try:
            # Obtém preço atual
            current_price = await self.data_loader.get_current_price(order['symbol'])
            if current_price is None:
                return 0
                
            # Calcula valor
            value = current_price * order['size']
            
            # Calcula taxa
            fee_rate = self.config['scalping']['exchange_fee']
            fee = value * fee_rate
            
            # Calcula slippage
            slippage_rate = self.config['scalping']['slippage']
            slippage = value * slippage_rate
            
            # Retorna custos totais
            return fee + slippage
            
        except Exception as e:
            logger.error(f"Erro ao calcular custos: {str(e)}")
            return 0
            
    async def _check_balance(self, order: Dict, costs: float) -> bool:
        """
        Verifica saldo.
        
        Args:
            order: Ordem.
            costs: Custos.
            
        Returns:
            True se tem saldo.
        """
        try:
            # Obtém preço atual
            current_price = await self.data_loader.get_current_price(order['symbol'])
            if current_price is None:
                return False
                
            # Calcula valor necessário
            value = current_price * order['size']
            total = value + costs
            
            # Obtém saldo
            if self.config['trading']['mode'] == 'simulated':
                balance = self.config['trading']['initial_balance']
            else:
                balance = await self.data_loader.get_balance()
                
            # Verifica saldo
            if total > balance:
                logger.error(f"Saldo insuficiente: {balance} < {total}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar saldo: {str(e)}")
            return False
            
    async def _simulate_order(self, order: Dict) -> bool:
        """
        Simula execução de ordem.
        
        Args:
            order: Ordem.
            
        Returns:
            True se simulada.
        """
        try:
            # Obtém preço atual
            current_price = await self.data_loader.get_current_price(order['symbol'])
            if current_price is None:
                return False
                
            # Simula execução
            logger.info(f"Simulando ordem: {order['id']}")
            logger.info(f"Preço: {current_price}")
            logger.info(f"Tamanho: {order['size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao simular ordem: {str(e)}")
            return False
            
    async def _execute_real_order(self, order: Dict) -> bool:
        """
        Executa ordem real.
        
        Args:
            order: Ordem.
            
        Returns:
            True se executada.
        """
        try:
            # Verifica API key
            if not self.config['exchange']['api_key']:
                logger.error("API key não configurada")
                return False
                
            # Executa ordem
            logger.info(f"Executando ordem real: {order['id']}")
            
            # TODO: Implementar integração com exchange
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao executar ordem real: {str(e)}")
            return False 