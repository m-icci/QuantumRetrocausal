import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

class QuantumPortfolioManager:
    def __init__(self, market_api, quantum_field, logger=None):
        self.market_api = market_api
        self.quantum_field = quantum_field
        self.logger = logger or logging.getLogger(__name__)
        self.last_rebalance = datetime.now()
        self.min_rebalance_interval = timedelta(minutes=5)
        self.emergency_profit_threshold = 0.002  # 0.2%
        self.normal_profit_threshold = 0.005     # 0.5%

    def calculate_asset_scores(self, symbol: str) -> Tuple[float, float]:
        """Calculate profit and holding scores for an asset using quantum metrics."""
        try:
            # Get quantum metrics
            field_strength = self.quantum_field.get_field_strength(symbol)
            coherence = self.quantum_field.get_coherence(symbol)
            momentum = self.quantum_field.get_momentum(symbol)
            
            # Calculate profit score (0-1)
            profit_score = (field_strength * 0.4 + 
                          momentum * 0.4 + 
                          coherence * 0.2)
            
            # Calculate holding score (0-1)
            holding_score = (coherence * 0.5 + 
                           field_strength * 0.3 + 
                           (1 - abs(momentum)) * 0.2)
            
            return profit_score, holding_score
            
        except Exception as e:
            self.logger.error(f"Error calculating scores for {symbol}: {str(e)}")
            return 0.0, 1.0  # Conservative default
            
    def get_portfolio_status(self) -> Dict[str, Dict]:
        """Get current portfolio status with quantum-enhanced metrics."""
        try:
            portfolio = self.market_api.get_portfolio()
            enhanced_portfolio = {}
            
            for symbol, data in portfolio.items():
                if data['free_amount'] <= 0:
                    continue
                    
                profit_score, holding_score = self.calculate_asset_scores(symbol)
                current_price = self.market_api.get_current_price(f"{symbol}/USDT")
                usdt_value = data['free_amount'] * current_price
                unrealized_profit = data.get('unrealized_profit_pct', 0)
                
                enhanced_portfolio[symbol] = {
                    'amount': data['free_amount'],
                    'usdt_value': usdt_value,
                    'profit_score': profit_score,
                    'holding_score': holding_score,
                    'sell_priority': profit_score - holding_score,
                    'unrealized_profit': unrealized_profit
                }
                
            return enhanced_portfolio
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {str(e)}")
            return {}

    def find_best_assets_to_sell(self, required_usdt: float) -> List[Dict]:
        """Find the best assets to sell to get the required USDT amount."""
        try:
            portfolio = self.get_portfolio_status()
            candidates = []
            
            # Sort by sell priority (profit_score - holding_score)
            sorted_assets = sorted(
                portfolio.items(),
                key=lambda x: x[1]['sell_priority'],
                reverse=True
            )
            
            # First try to find single asset with good profit
            for symbol, data in sorted_assets:
                if (data['usdt_value'] >= required_usdt and 
                    data['unrealized_profit'] >= self.normal_profit_threshold):
                    return [{
                        'symbol': symbol,
                        'amount': required_usdt / data['usdt_value'] * data['amount'],
                        'estimated_usdt': required_usdt,
                        'profit': data['unrealized_profit']
                    }]
            
            # If no single asset found, try combining assets with lower profit threshold
            accumulated_usdt = 0
            for symbol, data in sorted_assets:
                if data['unrealized_profit'] >= self.emergency_profit_threshold:
                    amount_to_sell = min(
                        data['amount'],
                        (required_usdt - accumulated_usdt) / data['usdt_value'] * data['amount']
                    )
                    usdt_value = amount_to_sell * data['usdt_value'] / data['amount']
                    
                    candidates.append({
                        'symbol': symbol,
                        'amount': amount_to_sell,
                        'estimated_usdt': usdt_value,
                        'profit': data['unrealized_profit']
                    })
                    
                    accumulated_usdt += usdt_value
                    if accumulated_usdt >= required_usdt:
                        break
            
            return candidates if accumulated_usdt >= required_usdt else []
            
        except Exception as e:
            self.logger.error(f"Error finding assets to sell: {str(e)}")
            return []

    async def execute_rebalancing(self, required_usdt: float, target_symbol: str) -> bool:
        """Execute portfolio rebalancing to get required USDT for target trade."""
        try:
            # Check rebalancing interval
            if datetime.now() - self.last_rebalance < self.min_rebalance_interval:
                self.logger.info("⏳ Rebalancing too frequent, skipping...")
                return False
                
            self.logger.info(f"🔄 Starting portfolio rebalancing for {target_symbol}")
            self.logger.info(f"Required value: ${required_usdt:.2f} USDT")
            
            # Find best assets to sell
            assets_to_sell = self.find_best_assets_to_sell(required_usdt)
            if not assets_to_sell:
                self.logger.warning("❌ No suitable assets found for selling")
                return False
                
            # Execute sells
            total_usdt = 0
            for asset in assets_to_sell:
                try:
                    result = await self.market_api.create_order(
                        symbol=f"{asset['symbol']}/USDT",
                        side='SELL',
                        amount=asset['amount']
                    )
                    
                    if result['success']:
                        total_usdt += result['executed_value']
                        self.logger.info(
                            f"✅ Sold {asset['amount']} {asset['symbol']} "
                            f"for {result['executed_value']:.2f} USDT "
                            f"(Profit: {asset['profit']*100:.2f}%)"
                        )
                    else:
                        self.logger.error(f"❌ Failed to sell {asset['symbol']}")
                        
                except Exception as e:
                    self.logger.error(f"Error selling {asset['symbol']}: {str(e)}")
                    continue
            
            self.last_rebalance = datetime.now()
            return total_usdt >= required_usdt
            
        except Exception as e:
            self.logger.error(f"Error executing rebalancing: {str(e)}")
            return False
