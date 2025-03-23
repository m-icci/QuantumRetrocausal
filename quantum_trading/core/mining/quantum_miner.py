"""
Minerador quântico para Monero.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

from .mining_config import MiningConfig
from .mining_strategy import MiningStrategy, QuantumMiningStrategy
from .miner_integration import MinerIntegration
from .pool_manager import PoolManager

class QuantumMiner:
    """Minerador quântico para Monero."""
    
    def __init__(self, config: MiningConfig):
        """
        Inicializa o minerador quântico.
        
        Args:
            config: Configuração da mineração.
        """
        self.config = config
        self.strategy = QuantumMiningStrategy(config)
        self.miner = MinerIntegration(config)
        self.pool_manager = PoolManager()
        self.is_running = False
        self.metrics = {}
        
        # Configura logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('QuantumMiner')
    
    async def start(self) -> bool:
        """
        Inicia o minerador.
        
        Returns:
            True se iniciado com sucesso.
        """
        try:
            # Valida configuração
            if not self.config.validate():
                self.logger.error("Configuração inválida")
                return False
            
            # Inicializa estratégia
            if not self.strategy.initialize():
                self.logger.error("Falha ao inicializar estratégia")
                return False
            
            # Seleciona melhor pool
            best_pool = await self.pool_manager.select_best_pool()
            if not best_pool:
                self.logger.error("Nenhuma pool disponível")
                return False
            
            # Atualiza configuração com pool selecionada
            self.config.pool_url = best_pool['url']
            self.config.pool_port = best_pool['port']
            
            # Inicia minerador
            if not await self.miner.start():
                self.logger.error("Falha ao iniciar minerador")
                return False
            
            # Inicia mineração
            self.is_running = True
            self.logger.info("Iniciando mineração quântica")
            
            # Inicia loop principal
            asyncio.create_task(self._mining_loop())
            
            return True
        except Exception as e:
            self.logger.error(f"Erro ao iniciar minerador: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        Para o minerador.
        
        Returns:
            True se parado com sucesso.
        """
        try:
            self.is_running = False
            self.logger.info("Parando mineração")
            
            # Para minerador
            if not await self.miner.stop():
                self.logger.error("Falha ao parar minerador")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Erro ao parar minerador: {str(e)}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Retorna o status atual do minerador.
        
        Returns:
            Status do minerador.
        """
        try:
            # Obtém status do minerador
            miner_status = await self.miner.get_status()
            
            # Obtém estado da estratégia
            strategy_state = self.strategy.get_state()
            
            # Obtém métricas das pools
            pool_metrics = await self.pool_manager.get_pool_metrics()
            
            return {
                'is_running': self.is_running,
                'miner_status': miner_status,
                'strategy_state': strategy_state,
                'pool_metrics': pool_metrics,
                'metrics': self.metrics,
                'config': self.config.to_dict()
            }
        except Exception as e:
            self.logger.error(f"Erro ao obter status: {str(e)}")
            return {
                'is_running': False,
                'error': str(e)
            }
    
    async def _mining_loop(self) -> None:
        """Loop principal de mineração."""
        while self.is_running:
            try:
                # Atualiza métricas
                await self._update_metrics()
                
                # Verifica se deve trocar de pool
                if await self._should_switch_pool():
                    await self._switch_pool()
                
                # Otimiza parâmetros se necessário
                if self.strategy.should_optimize():
                    optimized_params = self.strategy.optimize()
                    self.logger.info(f"Parâmetros otimizados: {optimized_params}")
                    
                    # Aplica parâmetros otimizados
                    await self._apply_optimized_params(optimized_params)
                
                # Aguarda próximo ciclo
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de mineração: {str(e)}")
                await asyncio.sleep(5)  # Aguarda antes de tentar novamente
    
    async def _update_metrics(self) -> None:
        """Atualiza as métricas da mineração."""
        try:
            # Obtém métricas do minerador
            miner_status = await self.miner.get_status()
            
            # Obtém métricas das pools
            pool_metrics = await self.pool_manager.get_pool_metrics()
            
            # Atualiza métricas
            self.metrics.update({
                'timestamp': datetime.now().isoformat(),
                'hashrate': miner_status.get('hashrate', 0.0),
                'power_usage': miner_status.get('power_usage', 0.0),
                'temperature': miner_status.get('temperature', 0.0),
                'shares': miner_status.get('shares', 0),
                'memory_usage': miner_status.get('memory_usage', 0.0),
                'pool_metrics': pool_metrics
            })
            
            # Atualiza estratégia
            self.strategy.update_metrics(self.metrics)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar métricas: {str(e)}")
    
    async def _should_switch_pool(self) -> bool:
        """Verifica se deve trocar de pool."""
        try:
            # Obtém métricas da pool atual
            pool_metrics = await self.pool_manager.get_pool_metrics()
            current_pool = pool_metrics.get('current_pool')
            
            if not current_pool:
                return True
            
            # Verifica se pool está online
            if current_pool.get('status') != 'online':
                return True
            
            # Verifica latência
            if current_pool.get('latency', float('inf')) > 1.0:  # Mais de 1 segundo
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar troca de pool: {str(e)}")
            return False
    
    async def _switch_pool(self) -> None:
        """Troca para uma nova pool."""
        try:
            # Para minerador
            await self.miner.stop()
            
            # Seleciona nova pool
            best_pool = await self.pool_manager.select_best_pool()
            if not best_pool:
                self.logger.error("Nenhuma pool disponível")
                return
            
            # Atualiza configuração
            self.config.pool_url = best_pool['url']
            self.config.pool_port = best_pool['port']
            
            # Reinicia minerador
            if not await self.miner.start():
                self.logger.error("Falha ao reiniciar minerador")
                return
            
            self.logger.info(f"Trocado para pool: {best_pool['name']}")
            
        except Exception as e:
            self.logger.error(f"Erro ao trocar pool: {str(e)}")
    
    async def _apply_optimized_params(self, params: Dict[str, Any]) -> None:
        """Aplica parâmetros otimizados."""
        try:
            # Atualiza configuração
            for key, value in params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Reinicia minerador com novos parâmetros
            await self.miner.stop()
            await self.miner.start()
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar parâmetros otimizados: {str(e)}")
    
    def get_logs(self) -> list[str]:
        """
        Retorna os logs do minerador.
        
        Returns:
            Lista de logs.
        """
        return self.miner.get_logs() 