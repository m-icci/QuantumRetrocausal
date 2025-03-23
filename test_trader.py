"""
Script para testar o scalping trader de Monero
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from quantum_trading.core.trading.trading_system import TradingSystem

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MoneroScalper:
    def __init__(self, config: Dict):
        self.config = config
        self.trading_system = None
        self.active_trades: Dict[str, Dict] = {}
        self.total_profit = 0.0
        self.trades_count = 0
        self.winning_trades = 0
        
    async def initialize(self):
        self.trading_system = TradingSystem(self.config)
        await self.trading_system.initialize()
        
    async def start(self):
        await self.trading_system.start()
        logger.info("=== Monero Scalper Iniciado ===")
        logger.info(f"Par: {self.config['trading']['symbol']}")
        logger.info(f"Timeframe: {self.config['trading']['timeframe']}")
        logger.info(f"Configuração de Scalping:")
        logger.info(f"- Lucro Mínimo: {self.config['scalping']['min_profit']*100}%")
        logger.info(f"- Perda Máxima: {self.config['scalping']['max_loss']*100}%")
        logger.info(f"- Tempo Máximo: {self.config['scalping']['max_position_time']}s")
        logger.info(f"- Volume Mínimo: {self.config['scalping']['min_volume']} XMR")
        logger.info(f"- Spread Máximo: {self.config['scalping']['max_spread']*100}%")
        logger.info("================================")
        
    async def stop(self):
        await self.trading_system.stop()
        logger.info("=== Monero Scalper Encerrado ===")
        logger.info(f"Total de Operações: {self.trades_count}")
        logger.info(f"Operações Lucrativas: {self.winning_trades}")
        logger.info(f"Taxa de Sucesso: {(self.winning_trades/self.trades_count*100 if self.trades_count > 0 else 0):.2f}%")
        logger.info(f"Lucro Total: {self.total_profit:.4f} XMR")
        logger.info("================================")
        
    async def execute_trade(self, signal: Dict) -> None:
        """Executa uma operação de scalping em Monero."""
        symbol = signal['symbol']
        movement = signal['movement']
        entry_price = 150.0  # Preço simulado de entrada em XMR
        
        # Registra a entrada
        trade_id = f"trade_{self.trades_count + 1}"
        self.active_trades[symbol] = {
            'id': trade_id,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'status': 'open',
            'movement': movement
        }
        
        logger.info(f"=== Nova Operação de Scalping ===")
        logger.info(f"ID: {trade_id}")
        logger.info(f"Par: {symbol}")
        logger.info(f"Movimento Detectado: {movement*100:.4f}%")
        logger.info(f"Preço de Entrada: {entry_price:.4f} XMR")
        logger.info(f"Volume: {self.config['scalping']['min_volume']:.4f} XMR")
        logger.info("================================")
        
    async def close_trade(self, signal: Dict) -> None:
        """Fecha uma operação de scalping em Monero."""
        symbol = signal['symbol']
        if symbol not in self.active_trades:
            return
            
        trade = self.active_trades[symbol]
        exit_price = 150.15  # Preço simulado de saída em XMR
        
        # Calcula o resultado
        profit = (exit_price - trade['entry_price']) / trade['entry_price']
        
        # Atualiza estatísticas
        self.trades_count += 1
        if profit > 0:
            self.winning_trades += 1
        self.total_profit += profit * self.config['scalping']['min_volume']
        
        # Log do resultado
        logger.info(f"=== Fechamento de Operação ===")
        logger.info(f"ID: {trade['id']}")
        logger.info(f"Par: {symbol}")
        logger.info(f"Movimento Realizado: {profit*100:.4f}%")
        logger.info(f"Preço de Entrada: {trade['entry_price']:.4f} XMR")
        logger.info(f"Preço de Saída: {exit_price:.4f} XMR")
        logger.info(f"Lucro/Prejuízo: {profit*100:.4f}%")
        logger.info(f"Volume: {self.config['scalping']['min_volume']:.4f} XMR")
        logger.info(f"Tempo na Posição: {(datetime.now() - trade['entry_time']).total_seconds():.1f}s")
        logger.info("================================")
        
        # Remove o trade ativo
        del self.active_trades[symbol]

async def main():
    # Carrega configuração
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Inicializa o scalper de Monero
    scalper = MoneroScalper(config)
    await scalper.initialize()
    await scalper.start()
    
    try:
        while True:
            # Atualiza o sistema
            await scalper.trading_system.update()
            
            # Verifica oportunidades de entrada
            entry_signal = await scalper.trading_system.check_entry(config['trading']['symbol'])
            if entry_signal:
                await scalper.execute_trade(entry_signal)
            
            # Verifica oportunidades de saída
            exit_signal = await scalper.trading_system.check_exit(config['trading']['symbol'])
            if exit_signal:
                await scalper.close_trade(exit_signal)
            
            # Aguarda 1 segundo antes da próxima iteração
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Encerrando scalper de Monero...")
    finally:
        await scalper.stop()

if __name__ == "__main__":
    asyncio.run(main()) 