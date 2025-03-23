"""
Script para testar arbitragem de BTC e ETH entre exchanges
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
import random
from typing import Dict, List
import statistics
import os
from dotenv import load_dotenv
import ccxt
import time

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoArbitrage:
    def __init__(self):
        self.active_trades = {}
        self.metrics = {}
        self.initial_balance_usdt = 1000.0  # Saldo inicial em USDT
        self.current_balance_usdt = self.initial_balance_usdt
        self.trading_pairs = ['BTC/USDT', 'ETH/USDT']  # Pares de trading
        self.exchanges = {
            'kucoin': ccxt.kucoin({
                'apiKey': os.getenv('KUCOIN_API_KEY'),
                'secret': os.getenv('KUCOIN_API_SECRET'),
                'password': os.getenv('KUCOIN_API_PASSPHRASE'),
                'enableRateLimit': True
            }),
            'kraken': ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_API_SECRET'),
                'enableRateLimit': True
            })
        }
        
        # Parâmetros de trading
        self.min_balance_usdt = 100.0  # Saldo mínimo em USDT
        self.max_position_usdt = 500.0  # Posição máxima em USDT
        self.min_profit = 0.001  # Lucro mínimo de 0.1%
        self.max_loss = 0.0005   # Perda máxima de 0.05%
        
    async def start(self):
        logger.info("=== Crypto Arbitrage Iniciado ===")
        logger.info("Configuração de Trading:")
        logger.info(f"- Pares: {', '.join(self.trading_pairs)}")
        logger.info(f"- Lucro Mínimo: {self.min_profit*100:.2f}%")
        logger.info(f"- Perda Máxima: {self.max_loss*100:.2f}%")
        logger.info("- Tempo Máximo: 300s")
        logger.info(f"- Posição Máxima: ${self.max_position_usdt:.2f} USDT")
        logger.info(f"- Saldo Inicial: ${self.initial_balance_usdt:.2f} USDT")
        logger.info(f"- Saldo Mínimo: ${self.min_balance_usdt:.2f} USDT")
        logger.info("================================")
        
        # Verifica saldos iniciais
        await self.check_balances()
        
    async def check_balances(self):
        """Verifica saldos em todas as exchanges."""
        for exchange_name, exchange in self.exchanges.items():
            try:
                balance = exchange.fetch_balance()
                usdt_balance = float(balance.get('USDT', {}).get('free', 0))
                btc_balance = float(balance.get('BTC', {}).get('free', 0))
                eth_balance = float(balance.get('ETH', {}).get('free', 0))
                
                logger.info(f"\nSaldo em {exchange_name.upper()}:")
                logger.info(f"USDT: ${usdt_balance:.2f}")
                logger.info(f"BTC: {btc_balance:.8f}")
                logger.info(f"ETH: {eth_balance:.8f}")
                
            except Exception as e:
                logger.error(f"Erro ao verificar saldo na {exchange_name}: {e}")
                
    async def check_market(self, symbol: str):
        """Obtém dados reais do mercado das exchanges."""
        try:
            market_data = {}
            
            # Obtém dados de ambas as exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Obtém ticker
                    ticker = exchange.fetch_ticker(symbol)
                    
                    # Obtém ordem book
                    orderbook = exchange.fetch_order_book(symbol)
                    
                    # Calcula spread
                    best_bid = float(orderbook['bids'][0][0])
                    best_ask = float(orderbook['asks'][0][0])
                    spread = (best_ask - best_bid) / best_bid
                    
                    market_data[exchange_name] = {
                        'price': float(ticker['last']),
                        'spread': spread,
                        'volume': float(ticker['quoteVolume']),
                        'change': float(ticker['percentage']),
                        'best_bid': best_bid,
                        'best_ask': best_ask
                    }
                    
                    logger.info(f"\n{exchange_name.upper()} - {symbol}:")
                    logger.info(f"Preço: ${market_data[exchange_name]['price']:.2f}")
                    logger.info(f"Spread: {market_data[exchange_name]['spread']*100:.4f}%")
                    logger.info(f"Volume 24h: ${market_data[exchange_name]['volume']:.2f}")
                    logger.info(f"Variação: {market_data[exchange_name]['change']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Erro ao obter dados da {exchange_name}: {e}")
                    continue
            
            # Se temos dados de ambas as exchanges, calcula a diferença de preço
            if len(market_data) == 2:
                price_diff = abs(market_data['kucoin']['price'] - market_data['kraken']['price'])
                price_diff_percent = price_diff / min(market_data['kucoin']['price'], market_data['kraken']['price'])
                logger.info(f"\nDiferença de Preço: {price_diff_percent*100:.4f}%")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado: {e}")
            return None
            
    async def execute_buy(self, exchange_name: str, symbol: str, amount_usdt: float) -> Dict:
        """Executa ordem de compra."""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Calcula quantidade baseada no preço atual
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
            amount = amount_usdt / price
            
            order = exchange.create_market_buy_order(
                symbol=symbol,
                amount=amount,
                params={'type': 'market'}
            )
            
            logger.info(f"\nOrdem de Compra Executada em {exchange_name.upper()}:")
            logger.info(f"Par: {symbol}")
            logger.info(f"Valor: ${amount_usdt:.2f} USDT")
            logger.info(f"Quantidade: {amount:.8f}")
            logger.info(f"Preço: ${float(order['price']):.2f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Erro ao executar compra na {exchange_name}: {e}")
            return None
            
    async def execute_sell(self, exchange_name: str, symbol: str, amount: float) -> Dict:
        """Executa ordem de venda."""
        try:
            exchange = self.exchanges[exchange_name]
            order = exchange.create_market_sell_order(
                symbol=symbol,
                amount=amount,
                params={'type': 'market'}
            )
            
            logger.info(f"\nOrdem de Venda Executada em {exchange_name.upper()}:")
            logger.info(f"Par: {symbol}")
            logger.info(f"Quantidade: {amount:.8f}")
            logger.info(f"Preço: ${float(order['price']):.2f}")
            logger.info(f"Valor Total: ${float(order['cost']):.2f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Erro ao executar venda na {exchange_name}: {e}")
            return None
        
    async def execute_arbitrage(self, symbol: str, market_data: Dict) -> None:
        """Executa uma operação de arbitragem."""
        if not market_data or len(market_data) < 2:
            return
            
        # Encontra a exchange com menor preço para compra
        buy_exchange = min(market_data.items(), key=lambda x: x[1]['price'])
        # Encontra a exchange com maior preço para venda
        sell_exchange = max(market_data.items(), key=lambda x: x[1]['price'])
        
        # Calcula spread entre exchanges
        spread = (sell_exchange[1]['price'] - buy_exchange[1]['price']) / buy_exchange[1]['price']
        
        # Verifica condições de mercado
        if spread < self.min_profit or buy_exchange[1]['volume'] < 10000 or sell_exchange[1]['volume'] < 10000:
            logger.info("Condições de mercado não favoráveis para arbitragem")
            return
            
        # Verifica saldos antes de executar
        await self.check_balances()
        
        # Calcula valor da operação baseado no saldo disponível
        trade_value = min(self.max_position_usdt, self.current_balance_usdt - self.min_balance_usdt)
        if trade_value < 100:  # Mínimo de $100 USDT
            logger.info("Saldo insuficiente para executar operação")
            return
        
        # Executa compra
        buy_order = await self.execute_buy(buy_exchange[0], symbol, trade_value)
        if not buy_order:
            logger.error("Falha ao executar ordem de compra")
            return
            
        # Registra a entrada
        trade_id = f"trade_{symbol}_{len(self.active_trades) + 1}"
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'entry_price': float(buy_order['price']),
            'entry_time': datetime.now(),
            'amount': float(buy_order['amount']),
            'market_data': market_data,
            'buy_exchange': buy_exchange[0],
            'sell_exchange': sell_exchange[0],
            'buy_order': buy_order
        }
        
        logger.info(f"=== Nova Operação de Arbitragem ===")
        logger.info(f"ID: {trade_id}")
        logger.info(f"Par: {symbol}")
        logger.info(f"Compra em: {buy_exchange[0].upper()} - ${float(buy_order['price']):.2f}")
        logger.info(f"Venda em: {sell_exchange[0].upper()} - ${sell_exchange[1]['price']:.2f}")
        logger.info(f"Spread entre Exchanges: {spread*100:.4f}%")
        logger.info("================================")
        
    async def close_arbitrage(self, trade_id: str) -> None:
        """Fecha uma operação de arbitragem."""
        if trade_id not in self.active_trades:
            return
            
        trade = self.active_trades[trade_id]
        current_market = await self.check_market(trade['symbol'])
        
        if not current_market or len(current_market) < 2:
            logger.error("Não foi possível obter preços atuais para fechar operação")
            return
            
        # Executa venda
        sell_order = await self.execute_sell(trade['sell_exchange'], trade['symbol'], trade['amount'])
        if not sell_order:
            logger.error("Falha ao executar ordem de venda")
            return
            
        # Calcula o resultado
        profit = (float(sell_order['price']) - trade['entry_price']) / trade['entry_price']
        profit_usdt = float(sell_order['cost']) - (trade['entry_price'] * trade['amount'])
        
        # Atualiza saldo
        self.current_balance_usdt += profit_usdt
        
        # Registra métricas
        if trade['symbol'] not in self.metrics:
            self.metrics[trade['symbol']] = {
                'trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        metrics = self.metrics[trade['symbol']]
        metrics['trades'] += 1
        if profit > 0:
            metrics['winning_trades'] += 1
        metrics['total_profit'] += profit_usdt
        metrics['best_trade'] = max(metrics['best_trade'], profit_usdt)
        metrics['worst_trade'] = min(metrics['worst_trade'], profit_usdt)
        
        # Log do resultado
        logger.info(f"=== Fechamento de Arbitragem ===")
        logger.info(f"ID: {trade_id}")
        logger.info(f"Par: {trade['symbol']}")
        logger.info(f"Movimento Realizado: {profit*100:.4f}%")
        logger.info(f"Lucro/Prejuízo: ${profit_usdt:.2f} USDT")
        logger.info(f"Preço de Entrada: ${trade['entry_price']:.2f} ({trade['buy_exchange'].upper()})")
        logger.info(f"Preço de Saída: ${float(sell_order['price']):.2f} ({trade['sell_exchange'].upper()})")
        logger.info(f"Quantidade: {trade['amount']:.8f}")
        logger.info(f"Tempo na Operação: {(datetime.now() - trade['entry_time']).total_seconds():.1f}s")
        logger.info(f"Saldo Atual: ${self.current_balance_usdt:.2f} USDT")
        logger.info("================================")
        
        # Remove o trade ativo
        del self.active_trades[trade_id]
        
    async def stop(self):
        """Encerra o arbitrador e mostra resumo de performance."""
        logger.info("=== Crypto Arbitrage Encerrado ===")
        logger.info("=== Resumo de Performance ===")
        
        for symbol in self.trading_pairs:
            if symbol in self.metrics:
                metrics = self.metrics[symbol]
                win_rate = (metrics['winning_trades'] / metrics['trades'] * 100) if metrics['trades'] > 0 else 0
                
                logger.info(f"\nPar: {symbol}")
                logger.info(f"Total de Operações: {metrics['trades']}")
                logger.info(f"Operações Ganhas: {metrics['winning_trades']}")
                logger.info(f"Taxa de Sucesso: {win_rate:.2f}%")
                logger.info(f"Lucro Total: ${metrics['total_profit']:.2f} USDT")
                logger.info(f"Melhor Trade: ${metrics['best_trade']:.2f} USDT")
                logger.info(f"Pior Trade: ${metrics['worst_trade']:.2f} USDT")
        
        total_return = ((self.current_balance_usdt - self.initial_balance_usdt) / self.initial_balance_usdt * 100)
        logger.info("\n=== Resultado Final ===")
        logger.info(f"Saldo Inicial: ${self.initial_balance_usdt:.2f} USDT")
        logger.info(f"Saldo Final: ${self.current_balance_usdt:.2f} USDT")
        logger.info(f"Retorno Total: {total_return:.2f}%")
        logger.info("================================")

async def main():
    # Inicializa o arbitrador
    arbitrage = CryptoArbitrage()
    await arbitrage.start()
    
    try:
        while True:
            for symbol in arbitrage.trading_pairs:
                # Verifica mercado
                market_data = await arbitrage.check_market(symbol)
                
                if not market_data or len(market_data) < 2:
                    await asyncio.sleep(1)
                    continue
                
                # Se não há trades ativos para este par
                active_symbol_trades = [t for t in arbitrage.active_trades.values() if t['symbol'] == symbol]
                if not active_symbol_trades:
                    # Calcula spread entre exchanges
                    spread = abs(market_data['kucoin']['price'] - market_data['kraken']['price']) / min(market_data['kucoin']['price'], market_data['kraken']['price'])
                    if spread > arbitrage.min_profit:  # Spread significativo entre exchanges
                        await arbitrage.execute_arbitrage(symbol, market_data)
                
                # Se há trades ativos, verifica fechamento
                for trade_id in list(arbitrage.active_trades.keys()):
                    trade = arbitrage.active_trades[trade_id]
                    if trade['symbol'] != symbol:
                        continue
                        
                    time_in_trade = (datetime.now() - trade['entry_time']).total_seconds()
                    
                    # Fecha se atingiu tempo máximo ou spread diminuiu significativamente
                    if time_in_trade > 300 or spread < arbitrage.min_profit/2:
                        await arbitrage.close_arbitrage(trade_id)
            
            # Aguarda 1 segundo antes da próxima iteração
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Encerrando arbitrador...")
    finally:
        await arbitrage.stop()

if __name__ == "__main__":
    asyncio.run(main()) 