from kucoin.client import Market, Trade
from config import Config
from typing import Dict, Any, Optional, List
import numpy as np
from utils.trading.quantum_trading_integrator import QuantumTradingIntegrator
from utils.core.state.quantum_state import QuantumState
import asyncio
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class KucoinQuantumIntegrator:
    """
    Enhanced KuCoin integration with quantum trading capabilities.
    Implements ICCI principles for quantum-aware market interactions.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, api_passphrase: str = None):
        self.market_client = Market(url='https://api.kucoin.com')
        if all([api_key, api_secret, api_passphrase]):
            self.trade_client = Trade(
                key=api_key,
                secret=api_secret,
                passphrase=api_passphrase,
                is_sandbox=False
            )
        else:
            self.trade_client = None
            logger.warning("Trading functionality disabled - no API credentials provided")

        self.quantum_integrator = QuantumTradingIntegrator()
        self.last_market_state: Dict[str, Any] = {}
        self.using_mock = False
        self.backtester = MockBacktester()

    async def get_enhanced_market_data(self, symbol: str = 'BTC-USDT') -> Dict[str, Any]:
        """
        Get market data enhanced with quantum metrics

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict containing market data and quantum analysis
        """
        try:
            # Get raw market data
            ticker = self.market_client.get_ticker(symbol)
            order_book = self.market_client.get_aggregated_orderv3(symbol)
            recent_trades = self.market_client.get_trade_histories(symbol)

            # Calculate market metrics
            market_data = self._calculate_market_metrics(ticker, order_book, recent_trades)

            # Process through quantum integrator
            quantum_analysis = self.quantum_integrator.process_market_state(market_data)

            # Store last state
            self.last_market_state = {
                'market_data': market_data,
                'quantum_analysis': quantum_analysis,
                'timestamp': datetime.now().timestamp()
            }

            return self.last_market_state
        except Exception as e:
            logger.error(f"Error fetching enhanced market data: {str(e)}")
            return None

    def _calculate_market_metrics(
        self,
        ticker: Dict[str, Any],
        order_book: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate advanced market metrics for quantum analysis"""

        # Extract basic metrics
        price = float(ticker['price'])
        volume = float(ticker['volume'])

        # Calculate momentum (using recent price changes)
        prices = [float(trade['price']) for trade in trades]
        momentum = self._calculate_momentum(prices)

        # Calculate volatility
        volatility = self._calculate_volatility(prices)

        # Calculate order book imbalance
        imbalance = self._calculate_order_book_imbalance(order_book)

        return {
            'price': price,
            'volume': volume,
            'momentum': momentum,
            'volatility': volatility,
            'imbalance': imbalance,
            'raw_ticker': ticker,
            'raw_order_book': order_book,
            'raw_trades': trades[:10]  # Limit to 10 recent trades
        }

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum indicator"""
        if len(prices) < 2:
            return 0.0

        # Calculate exponentially weighted price changes
        changes = np.diff(prices)
        weights = np.exp(np.linspace(-1., 0., len(changes)))
        weighted_momentum = np.sum(changes * weights) / np.sum(weights)

        # Normalize to [-1, 1]
        return np.tanh(weighted_momentum)

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0

        # Calculate rolling standard deviation
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)

        # Normalize using typical volatility range
        return np.clip(volatility / 0.02, 0, 1)  # 2% as baseline volatility

    def _calculate_order_book_imbalance(self, order_book: Dict[str, Any]) -> float:
        """Calculate order book imbalance"""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return 0.0

        # Calculate total volume at top 5 levels
        bid_volume = sum(float(bid[1]) for bid in bids[:5])
        ask_volume = sum(float(ask[1]) for ask in asks[:5])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        # Calculate imbalance ratio normalized to [-1, 1]
        return (bid_volume - ask_volume) / total_volume

    async def execute_quantum_trade(
        self,
        symbol: str,
        quantum_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute trade based on quantum signals"""
        if not self.trade_client:
            logger.error("Trading client not initialized - cannot execute trades")
            return None

        try:
            # Extract trade parameters
            side = quantum_signals['action']
            size = str(quantum_signals['position_size'])

            # Place order
            order = self.trade_client.create_market_order(
                symbol=symbol,
                side=side,
                size=size
            )

            # Place stop loss and take profit orders
            if quantum_signals.get('stop_loss'):
                self.trade_client.create_limit_order(
                    symbol=symbol,
                    side='sell' if side == 'buy' else 'buy',
                    price=str(quantum_signals['stop_loss']),
                    size=size,
                    stop='loss',
                    stopPrice=str(quantum_signals['stop_loss'])
                )

            if quantum_signals.get('take_profit'):
                self.trade_client.create_limit_order(
                    symbol=symbol,
                    side='sell' if side == 'buy' else 'buy',
                    price=str(quantum_signals['take_profit']),
                    size=size,
                    stop='entry',
                    stopPrice=str(quantum_signals['take_profit'])
                )

            return {
                'order': order,
                'quantum_signals': quantum_signals,
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            logger.error(f"Error executing quantum trade: {str(e)}")
            return None

    async def run_quantum_trading_cycle(self, symbol: str = 'BTC-USDT', interval: float = 60):
        """Run continuous quantum trading cycle"""
        while True:
            try:
                # Get enhanced market data
                market_state = await self.get_enhanced_market_data(symbol)
                if not market_state:
                    continue

                # Extract trading signals
                quantum_signals = market_state['quantum_analysis']['trading_signals']

                # Execute trade if confidence is high enough
                if quantum_signals['confidence'] > 0.75:
                    await self.execute_quantum_trade(symbol, quantum_signals)

                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in quantum trading cycle: {str(e)}")
                await asyncio.sleep(interval)

class KucoinQuantumMock:
    """Mock service que preserva coerência quântica"""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.quantum_state = QuantumState(dimension)
        self.recovery_simulator = QuantumRecoverySimulator(dimension)
        self.error_correction = QuantumErrorCorrection(dimension)
        self.parameter_optimizer = QuantumParameterOptimizer()
        self.phi = (1 + np.sqrt(5)) / 2
        self.metrics_history = []
        self.candlestick_cache = {}

    def generate_quantum_candlestick(self, timestamp: float) -> Dict[str, float]:
        """Gera candlestick preservando coerência quântica"""

        # Verifica cache
        if timestamp in self.candlestick_cache:
            return self.candlestick_cache[timestamp]

        # Estado base
        base_state = self.quantum_state.get_state()

        # Aplica evolução temporal
        evolved_state = self.recovery_simulator.run_merge_simulation(
            num_states=1,
            num_steps=10,
            decoherence_rate=0.01
        )['final_states'][0]

        # Extrai preços do estado quântico
        metrics = self.quantum_state.calculate_metrics()

        open_price = metrics['entropy'] * 1000
        close_price = metrics['coherence'] * 1000
        high_price = max(open_price, close_price) * (1 + metrics['pattern_strength'])
        low_price = min(open_price, close_price) * (1 - metrics['entanglement'])
        volume = metrics['purity'] * 10000

        # Corrige erros quânticos
        if metrics['coherence'] < 0.9:
            evolved_state = self.error_correction.correct_state(evolved_state)

        # Atualiza estado
        self.quantum_state.state = evolved_state

        # Gera candlestick
        candlestick = {
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'quantum_metrics': metrics
        }

        # Atualiza cache
        self.candlestick_cache[timestamp] = candlestick

        # Mantém histórico limitado
        if len(self.candlestick_cache) > 1000:
            oldest = min(self.candlestick_cache.keys())
            del self.candlestick_cache[oldest]

        return candlestick

    def validate_candlestick_consistency(self, candlestick: Dict[str, float]) -> bool:
        """Valida consistência do candlestick"""

        # Verifica ordem dos preços
        if not (candlestick['low'] <= candlestick['open'] <= candlestick['high'] and
                candlestick['low'] <= candlestick['close'] <= candlestick['high']):
            return False

        # Verifica coerência quântica
        if candlestick['quantum_metrics']['coherence'] < 0.7:
            return False

        # Verifica preservação de energia
        energy_preserved = abs(candlestick['open'] - candlestick['close']) <= (
            candlestick['high'] - candlestick['low']
        )
        if not energy_preserved:
            return False

        return True

    def optimize_mock_parameters(self) -> Dict[str, float]:
        """Otimiza parâmetros do mock"""
        return self.parameter_optimizer.optimize()


class DarkQuantumTrading:
    """Sistema de trading quântico com fatores ocultos"""

    def __init__(self, mock_service: KucoinQuantumMock):
        self.mock = mock_service
        self.phi = (1 + np.sqrt(5)) / 2
        self.dark_metrics = []
        self.last_optimization = None
        self.logger = logging.getLogger(__name__)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def calculate_dark_metrics(self, candlesticks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula métricas ocultas do mercado com logging aprimorado"""
        try:
            self.logger.info("Calculating dark metrics for %d candlesticks", len(candlesticks))

            # Prepara dados
            prices = np.array([c['close'] for c in candlesticks])
            volumes = np.array([c['volume'] for c in candlesticks])
            returns = np.diff(np.log(prices))

            # Log statistical properties
            self.logger.debug("Price range: %.2f - %.2f", np.min(prices), np.max(prices))
            self.logger.debug("Volume range: %.2f - %.2f", np.min(volumes), np.max(volumes))

            # Cria ativos para análise com validação
            if len(returns) < 2:
                raise ValueError("Insufficient data for dark metrics calculation")

            assets = [
                Asset(
                    name='price',
                    returns=returns,
                    volatility=np.std(returns),
                    liquidity=np.mean(volumes)
                )
            ]

            # Calcula fatores ocultos com logging
            self.logger.info("Computing hidden market factors")
            dark_risk = hidden_market_factors(assets)
            growth_potential = market_growth_potential(assets)

            # Log intermediate results
            self.logger.debug("Dark risk matrix trace: %.4f", np.trace(dark_risk))
            self.logger.debug("Growth potential: %.4f", growth_potential)

            # Otimiza portfólio considerando fatores ocultos
            weights, metrics = dark_portfolio_optimization(
                assets=assets,
                risk_tolerance=self.mock.quantum_state.calculate_metrics()['coherence']
            )

            result = {
                'dark_risk': float(np.trace(dark_risk)),
                'growth_potential': float(growth_potential),
                'portfolio_risk': metrics['portfolio_risk'],
                'dark_ratio': metrics['dark_ratio'],
                'diversification': metrics['diversification']
            }

            self.dark_metrics.append(result)
            self.logger.info("Dark metrics calculation completed successfully")

            return result

        except Exception as e:
            self.logger.error("Error calculating dark metrics: %s", str(e))
            raise

    def generate_dark_signals(self, candlestick: Dict[str, Any]) -> Dict[str, Any]:
        """Gera sinais de trading baseados em fatores ocultos"""

        if not self.dark_metrics:
            return {}

        # Obtém últimas métricas
        metrics = self.dark_metrics[-1]

        # Calcula força do sinal
        signal_strength = (
            metrics['growth_potential'] * (1 - metrics['portfolio_risk']) *
            candlestick['quantum_metrics']['coherence']
        )

        # Ajusta stops usando dark risk
        atr = self._calculate_atr(candlestick)
        dark_stop = atr * metrics['dark_ratio'] * self.phi

        return {
            'signal_strength': signal_strength,
            'dark_stop': dark_stop,
            'risk_adjustment': 1 - metrics['portfolio_risk'],
            'size_multiplier': metrics['diversification']
        }

    def _calculate_atr(self, candlestick: Dict[str, Any], period: int = 14) -> float:
        """Calcula ATR ajustado por fatores ocultos"""
        tr = max(
            candlestick['high'] - candlestick['low'],
            abs(candlestick['high'] - candlestick.get('prev_close', candlestick['open'])),
            abs(candlestick['low'] - candlestick.get('prev_close', candlestick['open']))
        )

        if not hasattr(self, '_atr'):
            self._atr = tr
        else:
            self._atr = (self._atr * (period - 1) + tr) / period

        return self._atr


class QuantumTradingBacktest:
    """Sistema de backtesting com preservação quântica e fatores ocultos"""

    def __init__(self, mock_service: KucoinQuantumMock):
        self.mock = mock_service
        self.strategy = FibonacciQuantumStrategy(mock_service.quantum_state)
        self.dark_trading = DarkQuantumTrading(mock_service)
        self.metrics = []
        self.trades = []
        self.phi = (1 + np.sqrt(5)) / 2

    async def run_backtest(self,
                          start_time: float,
                          end_time: float,
                          initial_capital: float = 10000,
                          risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """Executa backtest da estratégia"""

        # Obtém dados históricos
        candlesticks = await self.mock.get_candlestick_data(start_time, end_time)

        # Calcula métricas ocultas
        for i in range(len(candlesticks)-1):
            candlesticks[i+1]['prev_close'] = candlesticks[i]['close']
        dark_metrics = self.dark_trading.calculate_dark_metrics(candlesticks)

        # Prepara arrays para análise
        prices = np.array([c['close'] for c in candlesticks])
        highs = np.array([c['high'] for c in candlesticks])
        lows = np.array([c['low'] for c in candlesticks])

        # Analisa mercado
        self.strategy.analyze_market(prices, highs, lows)
        signals = self.strategy._generate_trading_signals()

        # Executa simulação
        capital = initial_capital
        position = None

        for i, candlestick in enumerate(candlesticks[1:], 1):
            # Gera sinais ocultos
            dark_signals = self.dark_trading.generate_dark_signals(candlestick)

            # Verifica stops
            if position:
                # Ajusta stop loss com fatores ocultos
                if 'dark_stop' in dark_signals:
                    position['stop_loss'] = min(
                        position['stop_loss'],
                        position['entry'] - dark_signals['dark_stop']
                    )

                if candlestick['low'] <= position['stop_loss']:
                    # Stop loss atingido
                    pl = (position['stop_loss'] - position['entry']) * position['size']
                    capital += pl
                    self.trades.append({
                        'entry': position['entry'],
                        'exit': position['stop_loss'],
                        'pl': pl,
                        'type': 'stop',
                        'quantum_metrics': candlestick['quantum_metrics'],
                        'dark_metrics': dark_signals
                    })
                    position = None

                elif candlestick['high'] >= position['target']:
                    # Take profit atingido
                    pl = (position['target'] - position['entry']) * position['size']
                    capital += pl
                    self.trades.append({
                        'entry': position['entry'],
                        'exit': position['target'],
                        'pl': pl,
                        'type': 'target',
                        'quantum_metrics': candlestick['quantum_metrics'],
                        'dark_metrics': dark_signals
                    })
                    position = None

            # Verifica novos sinais
            if not position:
                for entry in signals['entry_points']:
                    # Ajusta sinal com fatores ocultos
                    if dark_signals:
                        entry_strength = entry['strength'] * dark_signals['signal_strength']
                        if entry_strength < 0.5:  # Threshold mínimo
                            continue

                        # Ajusta tamanho da posição
                        size = self.strategy.get_optimal_position_size(
                            capital=capital,
                            risk_per_trade=risk_per_trade * dark_signals['risk_adjustment'],
                            current_price=entry['price']
                        ) * dark_signals['size_multiplier']

                        if candlestick['low'] <= entry['price'] <= candlestick['high']:
                            # Abre posição
                            position = {
                                'entry': entry['price'],
                                'stop_loss': entry['stop_loss'],
                                'target': entry['price'] * (1 + self.phi * entry['risk_ratio']),
                                'size': size,
                                'quantum_metrics': candlestick['quantum_metrics'],
                                'dark_metrics': dark_signals
                            }

            # Registra métricas
            self.metrics.append({
                'timestamp': candlestick['timestamp'],
                'capital': capital,
                'position': bool(position),
                'quantum_metrics': candlestick['quantum_metrics'],
                'dark_metrics': dark_signals
            })

        # Calcula métricas finais
        evaluator = PerformanceEvaluator(
            trades=self.trades,
            metrics=self.metrics,
            initial_capital=initial_capital
        )

        return {
            'final_capital': capital,
            'total_trades': len(self.trades),
            'metrics': evaluator.calculate_metrics(),
            'trade_history': self.trades,
            'quantum_correlation': self._calculate_quantum_correlation(),
            'dark_metrics': self.dark_trading.dark_metrics
        }

    def plot_results(self, save_path: Optional[str] = None):
        """Plota resultados do backtest com métricas ocultas"""
        import matplotlib.pyplot as plt

        # Configura subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))

        # Capital
        capitals = [m['capital'] for m in self.metrics]
        times = [m['timestamp'] for m in self.metrics]
        axs[0].plot(times, capitals)
        axs[0].set_title('Capital Evolution')
        axs[0].grid(True)

        # Correlação Quântica
        coherences = [m['quantum_metrics']['coherence'] for m in self.metrics]
        axs[1].scatter(coherences, capitals)
        axs[1].set_xlabel('Quantum Coherence')
        axs[1].set_ylabel('Capital')
        axs[1].set_title(f'Quantum Correlation: {self._calculate_quantum_correlation():.2f}')
        axs[1].grid(True)

        # Métricas Ocultas
        dark_metrics = self.dark_trading.dark_metrics
        if dark_metrics:
            times = range(len(dark_metrics))
            growth = [m['growth_potential'] for m in dark_metrics]
            risk = [m['portfolio_risk'] for m in dark_metrics]
            dark_ratio = [m['dark_ratio'] for m in dark_metrics]

            axs[2].plot(times, growth, label='Growth Potential')
            axs[2].plot(times, risk, label='Portfolio Risk')
            axs[2].plot(times, dark_ratio, label='Dark Ratio')
            axs[2].set_title('Dark Finance Metrics')
            axs[2].legend()
            axs[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _calculate_quantum_correlation(self) -> float:
        """Calculate correlation between capital and quantum coherence"""
        if not self.metrics:
            return 0.0
        capitals = np.array([m['capital'] for m in self.metrics])
        coherences = np.array([m['quantum_metrics']['coherence'] for m in self.metrics])
        correlation = np.corrcoef(capitals, coherences)[0, 1]
        return correlation


class EnhancedKucoinIntegrator:
    """Integrador KuCoin com fallback quântico e backtesting"""

    def __init__(self, api_key: str = None, api_secret: str = None, api_passphrase: str = None):
        # API real
        self.market_client = Market(url='https://api.kucoin.com')
        if all([api_key, api_secret, api_passphrase]):
            self.trade_client = Trade(
                key=api_key,
                secret=api_secret,
                passphrase=api_passphrase,
                is_sandbox=False
            )
        else:
            self.trade_client = None

        # Sistema de mock
        self.mock = KucoinQuantumMock()
        self.using_mock = False
        self.error_count = 0
        self.last_error_time = 0
        self.error_threshold = 3
        self.recovery_time = 300  # 5 minutos

        # Adiciona backtester
        self.backtester = QuantumTradingBacktest(self.mock)

    async def get_candlestick_data(self, symbol: str, start_time: float, end_time: float) -> List[Dict[str, float]]:
        """Obtém dados de candlestick com fallback automático"""
        try:
            if self.using_mock:
                # Verifica se podemos voltar para API real
                if time.time() - self.last_error_time > self.recovery_time:
                    self.using_mock = False
                    self.error_count = 0

            if not self.using_mock:
                # Tenta API real
                candlesticks = self.market_client.get_kline(symbol, start_time, end_time)

                # Valida dados
                for candlestick in candlesticks:
                    if not self._validate_real_candlestick(candlestick):
                        raise ValueError("Dados inconsistentes da API")

                return candlesticks

        except Exception as e:
            logger.error(f"Erro na API real: {str(e)}")
            self.error_count += 1
            self.last_error_time = time.time()

            # Ativa mock após erros consecutivos
            if self.error_count >= self.error_threshold:
                self.using_mock = True

        # Usa mock
        if self.using_mock:
            logger.info("Usando mock service")
            candlesticks = []
            current_time = start_time

            while current_time <= end_time:
                candlestick = self.mock.generate_quantum_candlestick(current_time)
                if self.mock.validate_candlestick_consistency(candlestick):
                    candlesticks.append(candlestick)
                current_time += 60  # Intervalo de 1 minuto

            return candlesticks

    def _validate_real_candlestick(self, candlestick: Dict[str, float]) -> bool:
        """Valida dados da API real"""
        try:
            # Verifica campos obrigatórios
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in candlestick for field in required_fields):
                return False

            # Converte para float
            for field in required_fields[1:]:  # Exceto timestamp
                candlestick[field] = float(candlestick[field])

            # Verifica ordem dos preços
            if not (candlestick['low'] <= candlestick['open'] <= candlestick['high'] and
                    candlestick['low'] <= candlestick['close'] <= candlestick['high']):
                return False

            # Verifica volumes positivos
            if candlestick['volume'] <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False

    def optimize_system(self):
        """Otimiza parâmetros do sistema"""
        if self.using_mock:
            self.mock.optimize_mock_parameters()

    async def test_strategy(self,
                          start_time: float,
                          end_time: float,
                          initial_capital: float = 10000,
                          risk_per_trade: float = 0.02,
                          plot_results: bool = True) -> Dict[str, Any]:
        """Testa estratégia no ambiente simulado"""

        results = await self.backtester.run_backtest(
            start_time=start_time,
            end_time=end_time,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade
        )

        if plot_results:
            self.backtester.plot_results('backtest_results.png')

        # Registra resultados no log
        logger.info(f"Backtest Results:")
        logger.info(f"Final Capital: ${results['final_capital']:.2f}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['metrics'].win_rate:.2%}")
        logger.info(f"Profit Factor: {results['metrics'].profit_factor:.2f}")
        logger.info(f"Sharpe Ratio: {results['metrics'].sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results['metrics'].max_drawdown:.2%}")
        logger.info(f"Quantum Correlation: {results['quantum_correlation']:.2f}")
        logger.info(f"Dark Metrics: {results['dark_metrics']}")

        return results


class MockBacktester:
    """Mock backtester for testing"""
    def __init__(self):
        self.metrics = []
    async def run_backtest(self, start_time: float, end_time: float, initial_capital: float = 10000, risk_per_trade: float = 0.02) -> Dict[str, Any]:
        return {
            'final_capital': initial_capital,
            'total_trades': 0,
            'metrics':PerformanceEvaluator([],[],initial_capital).calculate_metrics(),
            'trade_history': [],
            'quantum_correlation': 0.0,
            'dark_metrics': []
        }
    def plot_results(self, save_path: Optional[str] = None):
        pass

# Initialize with config
kucoin_quantum = KucoinQuantumIntegrator(
    api_key=Config.KUCOIN_API_KEY,
    api_secret=Config.KUCOIN_SECRET_KEY,
    api_passphrase=Config.KUCOIN_API_PASSPHRASE
)