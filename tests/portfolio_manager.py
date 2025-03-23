"""Portfolio Manager for optimizing trading operations"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time

class PortfolioManager:
    def __init__(self, market_api, consciousness, morphic_field):
        """Initialize Portfolio Manager"""
        self.market_api = market_api
        self.consciousness = consciousness
        self.morphic_field = morphic_field
        self.min_profit_threshold = 0.005  # 0.5% mínimo de lucro
        self.emergency_threshold = 0.002    # 0.2% aceito em emergência
        self.rebalance_interval = 300       # 5 minutos
        self._last_rebalance = 0
        
        # Configure logging
        self.logger = logging.getLogger("PortfolioManager")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s     %(name)s:%(filename)s:%(lineno)d %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status with all balances in USDT"""
        try:
            balances = self.market_api.get_all_balances()
            portfolio = {}
            total_value_usdt = 0.0
            error_count = 0
            
            for currency, balance in balances.items():
                if balance['total'] > 0:
                    # Get USDT price
                    if currency == 'USDT':
                        price_usdt = 1.0
                        value_usdt = balance['total']
                        portfolio[currency] = {
                            'amount': balance['total'],
                            'price_usdt': price_usdt,
                            'value_usdt': value_usdt
                        }
                        total_value_usdt += value_usdt
                    else:
                        try:
                            ticker = self.market_api.get_ticker(f"{currency}/USDT")
                            price_usdt = float(ticker['c'][0]) if ticker and 'c' in ticker else 0.0
                            value_usdt = balance['total'] * price_usdt
                            portfolio[currency] = {
                                'amount': balance['total'],
                                'price_usdt': price_usdt,
                                'value_usdt': value_usdt
                            }
                            total_value_usdt += value_usdt
                        except:
                            error_count += 1
                            self.logger.error(f"❌ Error getting price for {currency}")
            
            # If all non-USDT currencies failed, return only USDT balance
            if error_count > 0 and error_count == len([c for c in balances if c != 'USDT' and balances[c]['total'] > 0]):
                portfolio = {}
                total_value_usdt = 0.0
                if 'USDT' in balances and balances['USDT']['total'] > 0:
                    portfolio['USDT'] = {
                        'amount': balances['USDT']['total'],
                        'price_usdt': 1.0,
                        'value_usdt': balances['USDT']['total']
                    }
                    total_value_usdt = balances['USDT']['total']
            
            return {
                'assets': portfolio,
                'total_value_usdt': total_value_usdt
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting portfolio status: {str(e)}")
            return {'assets': {}, 'total_value_usdt': 0.0}
    
    def analyze_profit_opportunities(self) -> List[Dict]:
        """Analyze all assets for profit opportunities"""
        try:
            portfolio = self.get_portfolio_status()
            opportunities = []
            
            for currency, data in portfolio['assets'].items():
                if currency == 'USDT':
                    continue
                    
                # Get market metrics for this currency
                symbol = f"{currency}/USDT"
                metrics = self.consciousness.get_market_metrics(symbol)
                field = self.morphic_field.get_field_metrics(symbol)
                
                # Calculate profit potential
                momentum = metrics.get('momentum', 0)
                trend_strength = metrics.get('trend_strength', 0)
                field_strength = field.get('field_strength', 0)
                
                # Profit score (0-1)
                profit_score = (momentum + trend_strength + field_strength) / 3
                
                # Calculate holding score based on quantum metrics
                holding_score = self.calculate_holding_score(metrics, field)

                # Calculate profit potential (0-1)
                profit_potential = max(0, min(1, profit_score - holding_score))
                
                opportunities.append({
                    'symbol': symbol,
                    'currency': currency,
                    'amount': data['amount'],
                    'current_value_usdt': data['value_usdt'],
                    'profit_score': profit_score,
                    'holding_score': holding_score,
                    'profit_potential': profit_potential,
                    'sell_priority': profit_score - holding_score
                })
            
            # Sort by sell priority (higher = better to sell)
            return sorted(opportunities, key=lambda x: x['sell_priority'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing opportunities: {str(e)}")
            return []
    
    def calculate_holding_score(self, metrics: Dict, field: Dict) -> float:
        """Calculate how valuable it is to hold this asset"""
        try:
            coherence = metrics.get('coherence', 0)
            field_stability = field.get('stability', 0)
            trend_reliability = metrics.get('trend_reliability', 0)
            
            # Higher score = better to hold
            holding_score = (coherence + field_stability + trend_reliability) / 3
            
            return holding_score
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating holding score: {str(e)}")
            return 0.0
    
    def find_best_sell_candidate(self, required_usdt: float) -> Optional[Dict]:
        """Find the best asset to sell to get required USDT"""
        try:
            opportunities = self.analyze_profit_opportunities()
            
            for opp in opportunities:
                if opp['current_value_usdt'] >= required_usdt:
                    # Check if profit is acceptable
                    if opp['profit_score'] > self.min_profit_threshold:
                        return {
                            'symbol': opp['symbol'],
                            'currency': opp['currency'],
                            'amount': opp['amount'],
                            'current_value_usdt': opp['current_value_usdt'],
                            'profit_score': opp['profit_score']
                        }
            
            # If no single asset has enough value, try combining assets
            total_available = 0.0
            combined_assets = []
            
            for opp in opportunities:
                if opp['profit_score'] > self.emergency_threshold:
                    combined_assets.append({
                        'symbol': opp['symbol'],
                        'currency': opp['currency'],
                        'amount': opp['amount'],
                        'current_value_usdt': opp['current_value_usdt'],
                        'profit_score': opp['profit_score']
                    })
                    total_available += opp['current_value_usdt']
                    
                    if total_available >= required_usdt and len(combined_assets) > 1:
                        return {
                            'combined': True,
                            'assets': combined_assets,
                            'total_value_usdt': total_available
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error finding sell candidate: {str(e)}")
            return None
    
    def execute_rebalancing(self, required_usdt: float, target_symbol: str) -> bool:
        """Execute portfolio rebalancing to get required USDT"""
        try:
            current_time = time.time()
            if current_time - self._last_rebalance < self.rebalance_interval:
                self.logger.info("⏳ Waiting for rebalance cooldown...")
                return False
                
            self.logger.info(f"🔄 Starting portfolio rebalancing for {target_symbol}")
            self.logger.info(f"Required value: ${required_usdt:.2f} USDT")
            
            candidate = self.find_best_sell_candidate(required_usdt)
            if not candidate:
                self.logger.error("❌ No suitable assets found for selling")
                return False
                
            if 'combined' in candidate:
                success = self.execute_combined_sell(candidate['assets'], required_usdt)
            else:
                success = self.execute_single_sell(candidate, required_usdt)
                
            if success:
                self._last_rebalance = current_time
                self.logger.info("✅ Portfolio rebalancing completed successfully")
                return True
            else:
                self.logger.error("❌ Portfolio rebalancing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in portfolio rebalancing: {str(e)}")
            return False
    
    def execute_single_sell(self, asset: Dict, required_usdt: float) -> bool:
        """Execute sell order for a single asset"""
        try:
            currency = asset['currency']
            symbol = f"{currency}/USDT"
            
            # Get current price
            ticker = self.market_api.get_ticker(symbol)
            if not ticker or 'c' not in ticker:
                return False
                
            current_price = float(ticker['c'][0])
            amount_to_sell = min(asset['amount'], required_usdt / current_price)
            
            # Create sell order
            order = self.market_api.create_order(
                symbol=symbol,
                type='market',
                side='sell',
                amount=amount_to_sell
            )
            
            if order and 'result' in order:
                self.logger.info(f"💰 Sold {amount_to_sell:.8f} {currency} for ~${required_usdt:.2f} USDT")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing single sell: {str(e)}")
            return False
    
    def execute_combined_sell(self, assets: List[Dict], required_usdt: float) -> bool:
        """Execute sell orders for multiple assets"""
        try:
            remaining_usdt = required_usdt
            success = True
            
            for asset in assets:
                if remaining_usdt <= 0:
                    break
                    
                # Calculate portion to sell
                portion_usdt = min(asset['current_value_usdt'], remaining_usdt)
                if self.execute_single_sell(asset, portion_usdt):
                    remaining_usdt -= portion_usdt
                else:
                    success = False
                    
            return success and remaining_usdt <= 0
            
        except Exception as e:
            self.logger.error(f"❌ Error executing combined sell: {str(e)}")
            return False

    def calculate_total_value(self) -> float:
        """
        Calculate the total portfolio value in USDT.
        
        Returns:
            float: Total portfolio value in USDT
        """
        try:
            portfolio_status = self.get_portfolio_status()
            return portfolio_status.get('total_value_usdt', 0.0)
        except Exception as e:
            self.logger.error(f"Error calculating total portfolio value: {e}")
            return 0.0
