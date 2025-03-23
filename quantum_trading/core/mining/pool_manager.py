"""
Gerenciador de pools de mineração Monero.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import random

class PoolManager:
    """Gerenciador de pools de mineração."""
    
    def __init__(self):
        """Inicializa o gerenciador de pools."""
        self.logger = logging.getLogger('PoolManager')
        self.pools = []
        self.current_pool = None
        self.pool_metrics = {}
        
        # Carrega pools padrão
        self._load_default_pools()
    
    def _load_default_pools(self) -> None:
        """Carrega pools padrão."""
        self.pools = [
            {
                'name': 'supportxmr',
                'url': 'pool.supportxmr.com',
                'port': 3333,
                'region': 'global',
                'min_payout': 0.1,
                'fee': 0.006
            },
            {
                'name': 'xmrpool',
                'url': 'xmrpool.eu',
                'port': 3333,
                'region': 'eu',
                'min_payout': 0.1,
                'fee': 0.006
            },
            {
                'name': 'moneroocean',
                'url': 'gulf.moneroocean.stream',
                'port': 10001,
                'region': 'global',
                'min_payout': 0.1,
                'fee': 0.006
            }
        ]
    
    async def get_pool_status(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtém status de uma pool.
        
        Args:
            pool: Configuração da pool.
            
        Returns:
            Status da pool.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Tenta conectar à pool
                start_time = datetime.now()
                async with session.get(f"http://{pool['url']}:{pool['port']}/stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        return {
                            'name': pool['name'],
                            'url': pool['url'],
                            'port': pool['port'],
                            'status': 'online',
                            'hashrate': stats.get('hashrate', 0),
                            'miners': stats.get('miners', 0),
                            'last_check': datetime.now().isoformat(),
                            'latency': (datetime.now() - start_time).total_seconds()
                        }
                    else:
                        return {
                            'name': pool['name'],
                            'url': pool['url'],
                            'port': pool['port'],
                            'status': 'offline',
                            'error': f"HTTP {response.status}",
                            'last_check': datetime.now().isoformat()
                        }
        except Exception as e:
            return {
                'name': pool['name'],
                'url': pool['url'],
                'port': pool['port'],
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    async def select_best_pool(self) -> Optional[Dict[str, Any]]:
        """
        Seleciona a melhor pool disponível.
        
        Returns:
            Configuração da melhor pool.
        """
        try:
            # Obtém status de todas as pools
            pool_statuses = []
            for pool in self.pools:
                status = await self.get_pool_status(pool)
                pool_statuses.append(status)
            
            # Filtra pools online
            online_pools = [p for p in pool_statuses if p['status'] == 'online']
            if not online_pools:
                self.logger.warning("Nenhuma pool online encontrada")
                return None
            
            # Ordena por latência
            online_pools.sort(key=lambda x: x['latency'])
            
            # Seleciona pool com menor latência
            best_pool = online_pools[0]
            self.current_pool = best_pool
            
            # Atualiza métricas
            self.pool_metrics[best_pool['name']] = {
                'last_selected': datetime.now().isoformat(),
                'latency': best_pool['latency'],
                'hashrate': best_pool['hashrate'],
                'miners': best_pool['miners']
            }
            
            return best_pool
            
        except Exception as e:
            self.logger.error(f"Erro ao selecionar pool: {str(e)}")
            return None
    
    async def get_pool_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas das pools.
        
        Returns:
            Métricas das pools.
        """
        return {
            'current_pool': self.current_pool,
            'pool_metrics': self.pool_metrics,
            'total_pools': len(self.pools),
            'online_pools': len([p for p in self.pool_metrics.values() if p.get('status') == 'online'])
        }
    
    def add_pool(self, pool: Dict[str, Any]) -> bool:
        """
        Adiciona uma nova pool.
        
        Args:
            pool: Configuração da pool.
            
        Returns:
            True se adicionada com sucesso.
        """
        try:
            # Valida configuração
            required_fields = ['name', 'url', 'port']
            if not all(field in pool for field in required_fields):
                self.logger.error("Configuração de pool inválida")
                return False
            
            # Adiciona pool
            self.pools.append(pool)
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar pool: {str(e)}")
            return False
    
    def remove_pool(self, pool_name: str) -> bool:
        """
        Remove uma pool.
        
        Args:
            pool_name: Nome da pool.
            
        Returns:
            True se removida com sucesso.
        """
        try:
            # Remove pool
            self.pools = [p for p in self.pools if p['name'] != pool_name]
            
            # Remove métricas
            if pool_name in self.pool_metrics:
                del self.pool_metrics[pool_name]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao remover pool: {str(e)}")
            return False
    
    def get_pool_config(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna configuração de uma pool.
        
        Args:
            pool_name: Nome da pool.
            
        Returns:
            Configuração da pool.
        """
        for pool in self.pools:
            if pool['name'] == pool_name:
                return pool
        return None
    
    def save_pools(self, filepath: str) -> None:
        """
        Salva configurações das pools.
        
        Args:
            filepath: Caminho do arquivo.
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.pools, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro ao salvar pools: {str(e)}")
    
    def load_pools(self, filepath: str) -> None:
        """
        Carrega configurações das pools.
        
        Args:
            filepath: Caminho do arquivo.
        """
        try:
            with open(filepath, 'r') as f:
                self.pools = json.load(f)
        except Exception as e:
            self.logger.error(f"Erro ao carregar pools: {str(e)}")
            self._load_default_pools() 