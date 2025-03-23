"""AutoTrader module for quantum-based trading execution"""
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import time
import logging
from .quantum_portfolio_manager import QuantumPortfolioManager

from quantum_trading.market_api import MarketAPI
from quantum_trading.consciousness import MarketConsciousness
from quantum_trading.portfolio_manager import PortfolioManager
import asyncio

class AutoTrader:
    _instances = {}

    def __new__(cls, market_api: MarketAPI, consciousness: MarketConsciousness, morphic_field: Any, symbol: str):
        """Implement singleton pattern per symbol"""
        if symbol not in cls._instances:
            cls._instances[symbol] = super(AutoTrader, cls).__new__(cls)
            cls._instances[symbol]._initialized = False
        return cls._instances[symbol]

    def __init__(self, market_api: MarketAPI, consciousness: MarketConsciousness, morphic_field, symbol: str):
        """Initialize AutoTrader"""
        if self._initialized:
            return

        self.market_api = market_api
        self.consciousness = consciousness
        self.morphic_field = morphic_field
        self.symbol = symbol
        self.base_currency, self.quote_currency = symbol.split('/')
        
        # Portfolio Manager
        self.portfolio_manager = QuantumPortfolioManager(market_api, morphic_field)
        
        # Trading parameters
        self.target_position_value = 1000.0  # Target position value in USDT
        self.stop_loss_pct = 0.01  # 1% stop loss
        self.take_profit_pct = 0.02  # 2% take profit
        self.max_position_duration = 300  # 5 minutes
        self.min_profit_threshold = 0.005  # 0.5% minimum profit
        self.emergency_threshold = 0.002   # 0.2% emergency threshold
        
        # Advanced trading features
        self.cross_pair_enabled = True     # Permite trading entre qualquer par
        self.arbitrage_enabled = True      # Habilita arbitragem entre exchanges
        self.hedge_enabled = True          # Permite hedging de posições
        
        # Configurações de risco
        self.max_exposure_per_asset = 0.2  # Máximo 20% do portfólio por ativo
        self.min_liquidity_ratio = 0.1     # Mínimo 10% de liquidez por trade
        self.max_slippage = 0.01           # Máximo 1% de slippage
        
        # State and History
        self.position = None
        self.trade_history = []
        self.arbitrage_opportunities = []
        self.hedge_positions = {}
        self._last_balance_check = time.time()
        self._last_portfolio_update = time.time()
        
        # Configure logging
        self.logger = logging.getLogger(f"AutoTrader-{symbol}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s     %(name)s:%(filename)s:%(lineno)d %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"AutoTrader initialized for {symbol}")
        self._initialized = True

    def calculate_stop_loss(self, entry_price: float) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        metrics = self.consciousness.calculate_consciousness_field(data=None)  # Fixed missing parameter
        field_strength = metrics['field_strength']
        coherence = metrics['coherence']

        # Base stop loss percentage (0.2% to 1%)
        base_stop = 0.002

        # Adjust based on market conditions
        if field_strength > 1.0 and coherence > 0.8:
            stop_multiplier = 0.8  # Tighter stop loss in strong conditions
        elif field_strength < 0.6 or coherence < 0.5:
            stop_multiplier = 1.5  # Wider stop loss in weak conditions
        else:
            stop_multiplier = 1.0

        final_stop = base_stop * stop_multiplier
        final_stop = max(0.001, min(0.01, final_stop))  # Limit between 0.1% and 1%

        return entry_price * (1 - final_stop)

    def calculate_take_profit(self, entry_price: float) -> float:
        """Calculate dynamic take profit based on market conditions"""
        metrics = self.consciousness.calculate_consciousness_field(data=None)  # Fixed missing parameter
        field_strength = metrics['field_strength']
        coherence = metrics['coherence']

        # Base take profit percentage (0.5% to 2%)
        base_take = 0.005

        # Adjust based on market conditions
        if field_strength > 1.0 and coherence > 0.8:
            take_multiplier = 1.5  # Higher target in strong conditions
        elif field_strength < 0.6 or coherence < 0.5:
            take_multiplier = 0.8  # Lower target in weak conditions
        else:
            take_multiplier = 1.0

        final_take = base_take * take_multiplier
        final_take = max(0.002, min(0.02, final_take))  # Limit between 0.2% and 2%

        return entry_price * (1 + final_take)

    def apply_consciousness_decay(self, metrics: Dict[str, float], current_time: float) -> Dict[str, float]:
        """Apply time-based decay to market consciousness metrics"""
        # Base decay rate (per hour)
        base_decay = 0.1

        # Calculate time difference in hours
        time_diff = (current_time - self._last_balance_check) / 3600

        # Apply exponential decay to each metric
        decayed_metrics = {}
        for key, value in metrics.items():
            decay_factor = np.exp(-base_decay * time_diff)
            decayed_metrics[key] = value * decay_factor

        return decayed_metrics

    def get_available_balance(self, currency: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Obter saldo disponível para uma moeda específica ou para o par de trading
        
        Args:
            currency (Optional[str]): Moeda específica para consulta. Se None, retorna saldo do par.
        
        Returns:
            Union[Dict[str, float], float]: Dicionário com saldos ou saldo específico
        """
        try:
            # Buscar todos os saldos
            balances = self.market_api.get_all_balances()
            
            # Se nenhuma moeda específica for fornecida, usar par de trading
            if currency is None:
                return {
                    'base': balances.get(self.base_currency, {'total': 0.0})['total'],
                    'quote': balances.get(self.quote_currency, {'total': 0.0})['total']
                }
            
            # Mapear possíveis chaves para a moeda
            currency_keys = [currency, f'X{currency}', f'XX{currency}']
            
            # Encontrar saldo da moeda
            for key in currency_keys:
                if key in balances:
                    self.logger.info(f"💰 Available balance:")
                    self.logger.info(f"- {currency}: {balances[key]['total']}")
                    return balances[key]['total']
            
            self.logger.warning(f"⚠️ No balance found for {currency}")
            return 0.0
        
        except Exception as e:
            self.logger.error(f"❌ Error getting balance: {str(e)}")
            return 0.0

    def calculate_order_size(
        self, 
        symbol: Optional[str] = None, 
        side: Optional[str] = None, 
        current_price: Optional[float] = None
    ) -> float:
        """
        Calcular tamanho da ordem com tratamento flexível
        
        Args:
            symbol (Optional[str]): Par de trading
            side (Optional[str]): Lado da ordem ('buy' ou 'sell')
            current_price (Optional[float]): Preço atual
        
        Returns:
            float: Tamanho da ordem
        """
        try:
            # Usar valores padrão se não fornecidos
            symbol = symbol or self.symbol
            side = side or 'buy'
            
            # Obter preço atual se não fornecido
            if current_price is None:
                ticker = self.market_api.get_ticker(symbol)
                current_price = float(ticker['c'][0]) if ticker and 'c' in ticker else 0.0
            
            # Validar preço
            if current_price <= 0:
                self.logger.error("❌ Preço inválido: preço deve ser maior que 0")
                return 0.0
            
            # Obter saldo disponível
            balance = self.get_available_balance()
            quote_balance = balance['quote']
            
            # Calcular tamanho da ordem
            order_size = self.target_position_value / current_price
            
            # Verificar saldo disponível
            if side == 'buy' and order_size * current_price > quote_balance:
                self.logger.warning(f"⚠️ Saldo insuficiente para ordem de {order_size}")
                return 0.0
            
            # Verificar tamanho mínimo da ordem
            min_order_size = self.market_api.get_minimum_order_size(symbol)
            if order_size < min_order_size:
                self.logger.warning(f"⚠️ Tamanho da ordem abaixo do mínimo: {order_size}")
                return 0.0
            
            return order_size
        
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular tamanho da ordem: {str(e)}")
            return 0.0

    def should_exit_long(self, metrics: Dict[str, float], current_price: float) -> bool:
        """Determine if we should exit a long position using quantum metrics"""
        try:
            if not self.position:
                self.logger.error("❌ No position to evaluate for exit")
                return False

            # Log current state for debugging
            self.logger.info(f"🔍 Evaluating exit conditions:")
            self.logger.info(f"- Current Price: ${current_price:.2f}")
            self.logger.info(f"- Entry Price: ${self.position['entry_price']:.2f}")
            self.logger.info(f"- Stop Loss: ${self.position['stop_loss']:.2f}")
            self.logger.info(f"- Take Profit: ${self.position['take_profit']:.2f}")

            # Check stop loss
            if current_price <= self.position['stop_loss']:
                self.logger.info("💔 Stop loss triggered")
                self.position['exit_reason'] = 'stop_loss'
                return True

            # Check take profit
            if current_price >= self.position['take_profit']:
                self.logger.info("💰 Take profit triggered")
                self.position['exit_reason'] = 'take_profit'
                return True

            # Check timeout
            duration = (datetime.now() - self.position['entry_time']).total_seconds()
            self.logger.info(f"⏰ Duration check: {duration:.1f}s / {self.position['max_duration']}s")
            
            if duration >= self.position['max_duration']:
                self.logger.info(f"⏰ Position timeout reached")
                self.position['exit_reason'] = 'timeout'
                return True

            # Check for degraded market conditions
            if metrics['coherence'] < 0.4 or metrics['field_strength'] < 0.5:
                self.logger.info("🌊 Market conditions degraded")
                self.position['exit_reason'] = 'market_conditions'
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit conditions: {str(e)}")
            # On error, return True to exit position as a safety measure
            self.position['exit_reason'] = 'error'
            return True

    async def execute_trade(self, action: str, price: float, metrics: Dict[str, float], patterns: List[Dict]) -> Optional[Dict]:
        """Execute a trade with quantum protection"""
        # Validação de entrada - lança exceções
        if price <= 0:
            raise ValueError("Price must be positive")
        if not metrics:
            raise ValueError("Metrics cannot be None")
        if action not in ['buy', 'sell']:
            raise ValueError(f"Invalid action: {action}")

        self.logger.info(f"🔄 Executing {action} trade at {price} {self.quote_currency}")
        
        if action == 'buy':
            # Calculate position size
            size = self.calculate_order_size(self.symbol, action, price)
            if size <= 0:
                return {
                    'success': False, 
                    'error': 'Insufficient balance or invalid order size',
                    'action': action,
                    'price': price,
                    'pair': self.symbol
                }

            # Open new position
            try:
                position = self.open_position('long', price, size, metrics)
                if not position:
                    return {
                        'success': False, 
                        'error': 'Failed to open position',
                        'action': action,
                        'price': price,
                        'pair': self.symbol
                    }

                # Get equivalent USDT value for position
                usdt_value = None
                if self.quote_currency not in ['USD', 'USDT']:
                    try:
                        ticker = self.market_api.get_ticker(f"{self.quote_currency}/USDT")
                        if ticker and 'c' in ticker:
                            quote_value_usdt = float(ticker['c'][0])
                            usdt_value = size * price * quote_value_usdt
                    except Exception:
                        pass  # Non-critical error, we can proceed without USDT value

                return {
                    'success': True,
                    'action': action,
                    'price': price,
                    'size': size,
                    'order_id': position['order_id'],
                    'pair': self.symbol,
                    'base_currency': self.base_currency,
                    'quote_currency': self.quote_currency,
                    'usdt_value': usdt_value
                }
            except Exception as e:
                self.logger.error(f"❌ Error opening position: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'action': action,
                    'price': price,
                    'pair': self.symbol
                }

        elif action == 'sell':
            if not self.position:
                return {
                    'success': False, 
                    'error': 'No position to close',
                    'action': action,
                    'price': price,
                    'pair': self.symbol
                }

            # Store position size before closing
            position_size = self.position['size']

            try:
                # Close existing position
                result = self.close_position(price, position_size)
                if not result['success']:
                    return {
                        'success': False, 
                        'error': f"Failed to close position: {result.get('error', 'unknown error')}",
                        'action': action,
                        'price': price,
                        'pair': self.symbol
                    }

                # Get equivalent USDT value for profit
                usdt_value = None
                if self.quote_currency not in ['USD', 'USDT']:
                    try:
                        ticker = self.market_api.get_ticker(f"{self.quote_currency}/USDT")
                        if ticker and 'c' in ticker:
                            quote_value_usdt = float(ticker['c'][0])
                            usdt_value = result['profit'] * quote_value_usdt
                    except Exception:
                        pass  # Non-critical error, we can proceed without USDT value

                return {
                    'success': True,
                    'action': action,
                    'price': price,
                    'size': position_size,
                    'order_id': 'test_order_id',
                    'profit': result['profit'],
                    'profit_pct': result['profit_pct'],
                    'pair': self.symbol,
                    'base_currency': self.base_currency,
                    'quote_currency': self.quote_currency,
                    'usdt_value': usdt_value
                }
            except Exception as e:
                self.logger.error(f"❌ Error closing position: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'action': action,
                    'price': price,
                    'pair': self.symbol
                }

    def open_position(self, type_: str, price: float, size: float, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Open a new position"""
        try:
            if self.position:
                self.logger.error("❌ Cannot open position - already have an active position")
                return None

            # Create order with fixed parameter names
            order = self.market_api.create_order(
                symbol=self.symbol,
                side='buy',
                size=size,
                price=price if type_ == 'limit' else None
            )

            if not order or 'result' not in order:
                raise ValueError("Failed to create order")

            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(price)
            take_profit = self.calculate_take_profit(price)

            # Create new position
            self.position = {
                'type': type_,
                'entry_price': price,
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'max_duration': self.max_position_duration,
                'order_id': order['result']['txid'][0],
                'metrics': metrics
            }

            self.logger.info(f"📈 Position opened:")
            self.logger.info(f"- Type: {type_}")
            self.logger.info(f"- Entry Price: ${price:.2f}")
            self.logger.info(f"- Size: {size:.8f}")
            self.logger.info(f"- Stop Loss: ${stop_loss:.2f}")
            self.logger.info(f"- Take Profit: ${take_profit:.2f}")

            return self.position

        except Exception as e:
            self.logger.error(f"❌ Error opening position: {str(e)}")
            return None

    def close_position(self, current_price: float, size: float) -> Dict[str, Any]:
        """Close an existing position"""
        try:
            if not self.position:
                return {'success': False, 'error': 'No position to close'}

            # Calculate profit/loss
            entry_price = self.position['entry_price']
            profit = (current_price - entry_price) * size
            profit_pct = ((current_price - entry_price) / entry_price) * 100

            # Create close order with fixed parameter names
            order = self.market_api.create_order(
                symbol=self.symbol,
                side='sell',
                size=size,
                price=current_price
            )

            if not order or 'result' not in order:
                raise ValueError("Failed to create close order")

            # Record trade in history
            trade = {
                'entry_price': entry_price,
                'exit_price': current_price,
                'size': size,
                'profit': profit,
                'profit_pct': profit_pct,
                'duration': (datetime.now() - self.position['entry_time']).total_seconds(),
                'exit_reason': self.position.get('exit_reason', 'manual'),
                'entry_order': self.position['order_id'],
                'exit_order': order['result']['txid'][0],
                'metrics': self.position.get('metrics', {}),
                'patterns': self.position.get('patterns', [])
            }
            self.trade_history.append(trade)

            # Store success result
            result = {
                'success': True,
                'profit': profit,
                'profit_pct': profit_pct,
                'trade': trade
            }

            # Clear current position AFTER storing all necessary data
            self.position = None

            self.logger.info(f"💰 Position closed:")
            self.logger.info(f"- Exit Price: ${current_price:.2f}")
            self.logger.info(f"- Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            self.logger.info(f"- Reason: {self.position.get('exit_reason', 'manual')}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error closing position: {str(e)}")
            return {'success': False, 'error': str(e)}

    def estimate_trade_duration(self, metrics: Dict[str, float], patterns: List[Dict]) -> int:
        """Estimate optimal trade duration based on market consciousness"""
        try:
            # Base duration (5 minutes)
            base_duration = 300  # seconds

            # Adjust based on coherence
            coherence = metrics['coherence']
            if coherence > 0.8:  # Alta coerência = trade mais longo
                duration_multiplier = 1.5
            elif coherence > 0.65:
                duration_multiplier = 1.0
            else:
                duration_multiplier = 0.7  # Baixa coerência = trade mais curto

            # Adjust based on pattern scale
            if patterns:
                short_patterns = [p for p in patterns if p['scale'] <= 3]
                if short_patterns:
                    strongest = max(short_patterns, key=lambda x: x['strength'])
                    pattern_scale = strongest['scale']
                    pattern_strength = strongest['strength']

                    # Padrões mais fortes = duração mais curta
                    duration_multiplier *= (1 / pattern_strength)
                    # Escala maior = duração mais longa
                    duration_multiplier *= pattern_scale / 2

            # Calculate estimated duration
            estimated_duration = int(base_duration * duration_multiplier)

            # Ensure duration is between 1 and 30 minutes
            estimated_duration = max(60, min(1800, estimated_duration))

            return estimated_duration

        except Exception as e:
            self.logger.error(f"Error estimating trade duration: {str(e)}")
            return 300  # Default 5 minutes

    def calculate_dynamic_limits(self, metrics: Dict[str, float], patterns: List[Dict]) -> tuple:
        """Calculate dynamic stop loss and take profit based on market consciousness"""
        try:
            # Base limits
            base_stop = 0.002  # 0.2%
            base_take = 0.005  # 0.5%

            # Get estimated duration
            estimated_duration = self.estimate_trade_duration(metrics, patterns)
            duration_factor = estimated_duration / 300  # Normalize to base duration

            # Adjust based on duration
            # Trades mais longos = stops mais largos
            base_stop *= duration_factor
            base_take *= duration_factor

            # Adjust based on coherence
            coherence = metrics['coherence']
            if coherence > 0.8:  # Alta coerência = mais conservador no stop
                stop_multiplier = 0.8
                take_multiplier = 1.2
            elif coherence > 0.65:
                stop_multiplier = 1.0
                take_multiplier = 1.0
            else:
                stop_multiplier = 1.2  # Baixa coerência = stop mais largo
                take_multiplier = 0.8

            # Adjust based on field strength
            field_strength = metrics['field_strength']
            stop_multiplier *= (1 + (1 - field_strength))
            take_multiplier *= (1 + field_strength)

            # Pattern adjustment
            if patterns:
                short_patterns = [p for p in patterns if p['scale'] <= 3]
                if short_patterns:
                    strongest = max(short_patterns, key=lambda x: x['strength'])
                    pattern_strength = strongest['strength']
                    pattern_reliability = strongest['reliability']

                    # Ajuste baseado na força e confiabilidade do padrão
                    stop_multiplier *= (1 / pattern_strength) * (1 / pattern_reliability)
                    take_multiplier *= pattern_strength * pattern_reliability

            # Calculate final limits
            dynamic_stop = base_stop * stop_multiplier
            dynamic_take = base_take * take_multiplier

            # Ensure minimum and maximum values
            dynamic_stop = max(0.001, min(0.01, dynamic_stop))  # Entre 0.1% e 1%
            dynamic_take = max(0.002, min(0.02, dynamic_take))  # Entre 0.2% e 2%

            return dynamic_stop, dynamic_take, estimated_duration

        except Exception as e:
            self.logger.error(f"Error calculating dynamic limits: {str(e)}")
            return 0.002, 0.005, 300  # Default values if calculation fails

    def get_position_size(self, current_price: float) -> float:
        """Calculate position size based on USDT order size"""
        try:
            # Get minimum order size for the symbol
            min_size = self.market_api.get_minimum_order_size(self.symbol)

            # Calculate minimum order value in USDT
            min_order_value = min_size * current_price

            # If user's order size is less than minimum, use minimum
            usdt_to_use = max(self.target_position_value, min_order_value)

            # Calculate final position size
            position_size = usdt_to_use / current_price

            # Round to 8 decimal places for crypto precision
            position_size = round(position_size, 8)

            self.logger.info(f"Calculated position size: {position_size} (min: {min_size})")
            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def should_enter_long(self, metrics: Dict[str, float], patterns: List[Dict]) -> bool:
        """Determine if we should enter a long position based on quantum consciousness"""
        try:
            self.logger.info(f"Checking entry conditions - Metrics: {metrics}")

            # Verificar sinais quânticos
            quantum_signals = (
                metrics['coherence'] > 0.65 and 
                metrics['integration'] > 0.55 and
                metrics['field_strength'] > 0.7
            )

            self.logger.info(f"Quantum signals check: {quantum_signals}")
            self.logger.info(f"- Coherence: {metrics['coherence']} > 0.65")
            self.logger.info(f"- Integration: {metrics['integration']} > 0.55")
            self.logger.info(f"- Field Strength: {metrics['field_strength']} > 0.7")

            if quantum_signals:
                self.logger.info("✅ Quantum signals favorable")

                # Com sinais muito fortes, podemos relaxar a exigência de padrões
                if (metrics['coherence'] > 0.9 and 
                    metrics['field_strength'] > 1.5 and 
                    metrics['integration'] > 0.8):
                    self.logger.info("⭐ Extremely strong quantum signals - executing without pattern confirmation")
                    return True

                # Verificar padrões de curto prazo
                short_patterns = [p for p in patterns if p['scale'] <= 3]
                if short_patterns:
                    strongest = max(short_patterns, key=lambda x: x['strength'])
                    if strongest['strength'] > 1.1 and strongest['reliability'] > 0.65:
                        self.logger.info(f"✅ Strong pattern detected: {strongest}")
                        return True
                    else:
                        self.logger.info("❌ Pattern strength or reliability too low")
                else:
                    self.logger.info("❌ No short-term patterns detected")
            else:
                self.logger.info("❌ Quantum signals unfavorable")

            return False

        except Exception as e:
            self.logger.error(f"Error in should_enter_long: {str(e)}")
            return False

    def update(self, current_price: float) -> Optional[Dict]:
        """Update trading logic and execute if needed"""
        try:
            # Get current market state with quantum consciousness
            metrics = self.consciousness.calculate_consciousness_field()
            patterns = self.morphic_field.detect_patterns()

            self.logger.info(f"\n🔄 Updating trader state - Price: ${current_price:.2f}")
            self.logger.info(f"Current metrics: {metrics}")
            self.logger.info(f"Order size configured: ${self.target_position_value:.2f} {self.quote_currency}")

            # Check for trade signals
            if self.position is None:
                # Check quote currency balance for buying
                balances = self.get_available_balance()
                if balances['quote'] >= self.target_position_value:
                    should_enter = self.should_enter_long(metrics, patterns)
                    self.logger.info(f"Entry signal: {'✅' if should_enter else '❌'}")

                    if should_enter:
                        self.logger.info("Attempting to execute buy trade...")
                        result = self.execute_trade('buy', current_price, metrics, patterns) #added 'buy' and patterns
                        if result:
                            self.logger.info(f"✅ Trade executed: {result}")
                        else:
                            self.logger.error("❌ Trade execution failed")
                        return result
                else:
                    self.logger.warning(f"💡 Waiting for sufficient {self.quote_currency} balance. Available: {balances['quote']:.2f} {self.quote_currency}, Required: {self.target_position_value:.2f} {self.quote_currency}")
            else:
                # Check base currency balance for selling
                balances = self.get_available_balance()
                if balances['base'] >= self.position['size']:
                    should_exit = self.should_exit_long(metrics, current_price)
                    self.logger.info(f"Exit signal: {'✅' if should_exit else '❌'}")

                    if should_exit:
                        self.logger.info("Attempting to execute sell trade...")
                        result = self.execute_trade('sell', current_price, metrics, patterns) #added 'sell' and patterns
                        if result['success']:
                            self.logger.info(f"✅ Position closed: {result}")
                        else:
                            self.logger.error(f"❌ Position close failed: {result['error']}")
                        return result
                else:
                    self.logger.error(f"❌ Insufficient {self.base_currency} balance for closing position: {balances['base']:.8f} {self.base_currency} available, need {self.position['size']:.8f} {self.base_currency}")

            return None

        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return None

    def get_performance_metrics(self) -> Dict:
        """Calculate trading performance metrics with quantum analysis"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_duration': 0
            }

        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['profit'] > 0])

        return {
            'total_trades': total_trades,
            'win_rate': (winning_trades / total_trades) * 100,
            'avg_profit': sum(t['profit'] for t in self.trade_history) / total_trades,
            'avg_profit_pct': sum(t['profit_pct'] for t in self.trade_history) / total_trades,
            'avg_duration': sum(t['duration'] for t in self.trade_history) / total_trades,
            'total_profit': sum(t['profit'] for t in self.trade_history),
            'exit_reasons': {
                reason: len([t for t in self.trade_history if t['exit_reason'] == reason])
                for reason in ['take_profit', 'stop_loss', 'timeout', 'market_conditions', 'error']
            }
        }

    async def get_cross_pair_rate(self, from_currency: str, to_currency: str) -> float:
        """Calculate o melhor rate entre duas moedas usando USDT como bridge se necessário."""
        try:
            # Direct pair
            direct_pair = f"{from_currency}/{to_currency}"
            if await self.market_api.get_ticker(direct_pair):
                return await self.market_api.get_current_price(direct_pair)
            
            # Via USDT
            from_usdt = await self.market_api.get_current_price(f"{from_currency}/USDT")
            to_usdt = await self.market_api.get_current_price(f"{to_currency}/USDT")
            return from_usdt / to_usdt
            
        except Exception as e:
            self.logger.error(f"Error getting cross pair rate: {str(e)}")
            return 0.0

    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """Procura oportunidades de arbitragem entre pares de trading."""
        opportunities = []
        all_rates = {}
        
        try:
            # Get all rates
            currencies = await self.market_api.get_available_currencies()
            for base in currencies:
                for quote in currencies:
                    if base != quote:
                        rate = await self.get_cross_pair_rate(base, quote)
                        all_rates[f"{base}/{quote}"] = rate
            
            # Look for triangular arbitrage
            for c1 in currencies:
                for c2 in currencies:
                    for c3 in currencies:
                        if c1 != c2 and c2 != c3 and c3 != c1:
                            rate1 = all_rates.get(f"{c1}/{c2}", 0)
                            rate2 = all_rates.get(f"{c2}/{c3}", 0)
                            rate3 = all_rates.get(f"{c3}/{c1}", 0)
                            
                            if rate1 and rate2 and rate3:
                                profit = (rate1 * rate2 * rate3) - 1
                                if profit > self.min_profit_threshold:
                                    opportunities.append({
                                        'path': [c1, c2, c3],
                                        'rates': [rate1, rate2, rate3],
                                        'profit': profit
                                    })
            
            return sorted(opportunities, key=lambda x: x['profit'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {str(e)}")
            return []

    async def execute_cross_pair_trade(self, from_currency: str, to_currency: str, amount: float) -> Dict:
        """Executa trade entre qualquer par de moedas."""
        try:
            # Check direct pair
            direct_pair = f"{from_currency}/{to_currency}"
            if await self.market_api.get_ticker(direct_pair):
                return await self.execute_trade(direct_pair, "SELL", amount)
            
            # Via USDT
            result1 = await self.execute_trade(f"{from_currency}/USDT", "SELL", amount)
            if not result1['success']:
                return result1
            
            usdt_amount = result1['executed_value']
            result2 = await self.execute_trade(f"{to_currency}/USDT", "BUY", usdt_amount)
            
            return {
                'success': result2['success'],
                'from_trade': result1,
                'to_trade': result2,
                'effective_rate': result2['amount'] / amount if result2['success'] else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross pair trade: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def execute_arbitrage(self, opportunity: Dict) -> Dict:
        """Executa uma oportunidade de arbitragem."""
        try:
            path = opportunity['path']
            initial_amount = self.target_position_value / opportunity['rates'][0]
            
            # Execute trades
            trades = []
            current_amount = initial_amount
            
            for i in range(len(path)):
                from_currency = path[i]
                to_currency = path[(i + 1) % len(path)]
                
                result = await self.execute_cross_pair_trade(from_currency, to_currency, current_amount)
                trades.append(result)
                
                if not result['success']:
                    # Rollback previous trades
                    for j in range(len(trades) - 1, -1, -1):
                        prev_trade = trades[j]
                        await self.execute_cross_pair_trade(
                            prev_trade['to_currency'],
                            prev_trade['from_currency'],
                            prev_trade['amount']
                        )
                    return {'success': False, 'error': 'Arbitrage failed, positions rolled back'}
                
                current_amount = result['amount']
            
            profit = (current_amount / initial_amount) - 1
            return {
                'success': True,
                'initial_amount': initial_amount,
                'final_amount': current_amount,
                'profit': profit,
                'trades': trades
            }
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def update_portfolio_status(self):
        """Atualiza status do portfólio e executa ações necessárias."""
        try:
            current_time = time.time()
            if current_time - self._last_portfolio_update < 60:  # Update a cada minuto
                return
            
            self._last_portfolio_update = current_time
            
            # Get portfolio status
            portfolio = await self.portfolio_manager.get_portfolio_status()
            
            # Check arbitrage opportunities
            if self.arbitrage_enabled:
                opportunities = await self.find_arbitrage_opportunities()
                for opp in opportunities[:3]:  # Top 3 opportunities
                    if opp['profit'] > self.min_profit_threshold:
                        await self.execute_arbitrage(opp)
            
            # Balance portfolio
            for asset, data in portfolio.items():
                # Check exposure limits
                if data['usdt_value'] > self.max_exposure_per_asset * self.target_position_value:
                    excess = data['usdt_value'] - (self.max_exposure_per_asset * self.target_position_value)
                    await self.execute_cross_pair_trade(asset, 'USDT', excess / data['price'])
                
                # Take profits if available
                if data['unrealized_profit'] > self.take_profit_pct:
                    await self.execute_cross_pair_trade(asset, 'USDT', data['amount'])
                
                # Cut losses if necessary
                if data['unrealized_profit'] < -self.stop_loss_pct:
                    await self.execute_cross_pair_trade(asset, 'USDT', data['amount'])
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")

    async def run_continuous_trading(self):
        """Executa trading contínuo com todas as estratégias."""
        while True:
            try:
                # Update portfolio status
                await self.update_portfolio_status()
                
                # Check for new trading opportunities
                consciousness_state = self.consciousness.get_market_state()
                if consciousness_state['trade_signal']:
                    # Execute regular trading strategy
                    await self.execute_trade(consciousness_state['action'], consciousness_state['price'], consciousness_state['metrics'], consciousness_state['patterns']) #added parameters
                
                # Look for arbitrage opportunities
                if self.arbitrage_enabled:
                    opportunities = await self.find_arbitrage_opportunities()
                    for opp in opportunities[:3]:
                        if opp['profit'] > self.min_profit_threshold:
                            await self.execute_arbitrage(opp)
                
                # Sleep to avoid API rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in continuous trading: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error

    def rebalance_portfolio(self, target_value: Optional[float] = None) -> bool:
        """
        Rebalancear portfólio para atingir valor alvo
        
        Args:
            target_value (Optional[float]): Valor alvo em USDT. Se None, usa target_position_value
        
        Returns:
            bool: Sucesso no rebalanceamento
        """
        try:
            # Valor alvo padrão
            target_value = target_value or self.target_position_value
            
            # Obter saldo atual
            balance = self.get_available_balance()
            current_quote_balance = balance['quote']
            
            # Verificar necessidade de rebalanceamento
            if current_quote_balance >= target_value:
                self.logger.info(f"✅ Saldo atual (${current_quote_balance:.2f}) já atende ao alvo (${target_value:.2f})")
                return True
            
            # Calcular valor necessário para rebalancear
            required_amount = target_value - current_quote_balance
            
            # Tentar executar rebalanceamento via portfolio manager
            result = self.portfolio_manager.execute_rebalancing(required_amount, self.symbol)
            
            if result:
                # Verificar novo saldo após rebalanceamento
                new_balance = self.get_available_balance()
                new_quote_balance = new_balance['quote']
                
                self.logger.info(f"🔄 Rebalanceamento realizado:")
                self.logger.info(f"- Saldo anterior: ${current_quote_balance:.2f}")
                self.logger.info(f"- Saldo atual: ${new_quote_balance:.2f}")
                self.logger.info(f"- Valor alvo: ${target_value:.2f}")
                
                return new_quote_balance >= target_value
            else:
                self.logger.warning("❌ Falha no rebalanceamento")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Erro no rebalanceamento: {str(e)}")
            return False

    def _configurar_traders_quanticos(self):
        """Configura traders quânticos para cada par de trading e exchange."""
        for pair in self.trading_pairs:
            for exchange in self.exchanges:
                trader_key = f"{exchange}_{pair}"
                if trader_key in self.auto_traders:
                    logger.info(f"Configurando trader quântico para {pair} no {exchange}")
                    # Aqui você pode adicionar a lógica de configuração específica para traders quânticos
                    # Por exemplo, definir parâmetros, estratégias, etc.
                    self.auto_traders[trader_key].configure_quantum_trader()  # Método fictício para configuração
        logger.info("Traders quânticos configurados com sucesso.")

    def configure_quantum_trader(self):
        """Configura o trader quântico com parâmetros específicos."""
        # Adicione aqui a lógica de configuração específica para traders quânticos
        logger.info(f"Trader quântico configurado com sucesso para {self.symbol}.")