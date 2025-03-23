"""
Gerenciador de Portfolio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Arquivos
        self.trades_file = os.path.join(data_dir, "trades_history.csv")
        self.portfolio_file = os.path.join(data_dir, "portfolio.json")
        
        # Carrega dados
        self.trades_history = self._load_trades()
        self.portfolio = self._load_portfolio()
        
    def _load_trades(self) -> pd.DataFrame:
        """Carrega histórico de trades"""
        try:
            if os.path.exists(self.trades_file):
                return pd.read_csv(self.trades_file)
            return pd.DataFrame(columns=[
                'timestamp', 'exchange', 'symbol', 'type', 
                'amount', 'price', 'cost', 'fee', 'profit'
            ])
        except Exception as e:
            logger.error(f"Erro carregando trades: {e}")
            return pd.DataFrame()
            
    def _load_portfolio(self) -> Dict:
        """Carrega estado do portfolio"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            return {
                'total_value_usdt': 0.0,
                'total_profit_usdt': 0.0,
                'assets': {},
                'exchanges': {
                    'kucoin': {'balance': 0.0, 'profit': 0.0},
                    'kraken': {'balance': 0.0, 'profit': 0.0}
                }
            }
        except Exception as e:
            logger.error(f"Erro carregando portfolio: {e}")
            return {}
            
    def _save_trades(self):
        """Salva histórico de trades"""
        try:
            self.trades_history.to_csv(self.trades_file, index=False)
        except Exception as e:
            logger.error(f"Erro salvando trades: {e}")
            
    def _save_portfolio(self):
        """Salva estado do portfolio"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=4)
        except Exception as e:
            logger.error(f"Erro salvando portfolio: {e}")
            
    async def update_portfolio(self, kucoin_balance: Dict, kraken_balance: Dict):
        """Atualiza estado do portfolio com dados das exchanges"""
        try:
            # Atualiza saldos
            self.portfolio['exchanges']['kucoin']['balance'] = kucoin_balance.get('total', {}).get('USDT', 0.0)
            self.portfolio['exchanges']['kraken']['balance'] = kraken_balance.get('total', {}).get('USDT', 0.0)
            
            # Atualiza total
            self.portfolio['total_value_usdt'] = sum(
                exchange['balance'] 
                for exchange in self.portfolio['exchanges'].values()
            )
            
            # Salva
            self._save_portfolio()
            
        except Exception as e:
            logger.error(f"Erro atualizando portfolio: {e}")
            
    def add_trade(self, trade: Dict):
        """Adiciona novo trade ao histórico"""
        try:
            # Adiciona trade
            self.trades_history = pd.concat([
                self.trades_history,
                pd.DataFrame([trade])
            ], ignore_index=True)
            
            # Atualiza profit da exchange
            exchange = trade['exchange']
            self.portfolio['exchanges'][exchange]['profit'] += trade['profit']
            
            # Atualiza profit total
            self.portfolio['total_profit_usdt'] = sum(
                exchange['profit']
                for exchange in self.portfolio['exchanges'].values()
            )
            
            # Salva
            self._save_trades()
            self._save_portfolio()
            
        except Exception as e:
            logger.error(f"Erro adicionando trade: {e}")
            
    def get_performance_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        try:
            if len(self.trades_history) == 0:
                return {}
                
            metrics = {
                'total_trades': len(self.trades_history),
                'profitable_trades': len(self.trades_history[self.trades_history['profit'] > 0]),
                'avg_profit': self.trades_history['profit'].mean(),
                'max_profit': self.trades_history['profit'].max(),
                'max_loss': self.trades_history['profit'].min(),
                'total_profit': self.trades_history['profit'].sum(),
                'win_rate': len(self.trades_history[self.trades_history['profit'] > 0]) / len(self.trades_history)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro calculando métricas: {e}")
            return {}
