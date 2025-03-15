import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Advanced risk management system with dynamic position sizing and stop-loss
    """
    def __init__(self, 
                 max_position_size: float = 0.1,  # Max 10% of portfolio per position
                 max_total_risk: float = 0.2,     # Max 20% total portfolio at risk
                 base_stop_loss: float = 0.02,    # 2% base stop loss
                 trailing_stop: float = 0.01):     # 1% trailing stop
        self.max_position_size = max_position_size
        self.max_total_risk = max_total_risk
        self.base_stop_loss = base_stop_loss
        self.trailing_stop = trailing_stop
        
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics: Dict[str, float] = {
            'total_risk': 0.0,
            'portfolio_var': 0.0,
            'sharpe_ratio': 0.0
        }
        self.position_history: List[Dict[str, Any]] = []
        
    def calculate_position_size(self, 
                              symbol: str, 
                              portfolio_value: float,
                              quantum_metrics: Dict[str, float],
                              cgr_metrics: Dict[str, Any]) -> float:
        """Calculate optimal position size based on multiple factors"""
        try:
            # Base position size
            base_size = portfolio_value * self.max_position_size
            
            # Adjust based on quantum coherence
            coherence = quantum_metrics.get('coherence', 0.5)
            coherence_factor = np.clip(coherence, 0.1, 1.0)
            
            # Adjust based on CGR fractal dimension
            fractal_dim = cgr_metrics.get('fractal_dimension', 1.0)
            fractal_factor = np.clip(2 - fractal_dim, 0.1, 1.0)
            
            # Current portfolio risk
            current_risk = sum(pos['risk_amount'] for pos in self.positions.values())
            risk_remaining = self.max_total_risk - current_risk
            
            if risk_remaining <= 0:
                return 0.0
                
            # Calculate final position size
            position_size = base_size * coherence_factor * fractal_factor
            
            # Ensure we don't exceed risk limits
            position_size = min(position_size, portfolio_value * risk_remaining)
            
            return float(position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def set_stop_loss(self, 
                      symbol: str, 
                      entry_price: float,
                      position_size: float,
                      quantum_metrics: Dict[str, float]) -> Dict[str, float]:
        """Set dynamic stop loss levels based on market conditions"""
        try:
            # Adjust stop loss based on quantum potential
            potential = quantum_metrics.get('potential', 0.5)
            adjusted_stop = self.base_stop_loss * (1 + (1 - potential))
            
            stop_price = entry_price * (1 - adjusted_stop)
            trailing_price = entry_price * (1 - self.trailing_stop)
            
            risk_amount = position_size * adjusted_stop
            
            return {
                'stop_price': float(stop_price),
                'trailing_price': float(trailing_price),
                'risk_amount': float(risk_amount)
            }
            
        except Exception as e:
            logger.error(f"Error setting stop loss: {str(e)}")
            return {
                'stop_price': entry_price * 0.98,  # Default 2% stop
                'trailing_price': entry_price * 0.99,
                'risk_amount': position_size * 0.02
            }
            
    def update_position(self,
                       symbol: str,
                       current_price: float,
                       quantum_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update position status and stops"""
        if symbol not in self.positions:
            return {}
            
        try:
            position = self.positions[symbol]
            
            # Update trailing stop if price moved in our favor
            if current_price > position['entry_price']:
                new_trailing = current_price * (1 - self.trailing_stop)
                if new_trailing > position['trailing_price']:
                    position['trailing_price'] = new_trailing
                    
            # Check if stops are hit
            stop_hit = current_price <= position['stop_price']
            trailing_hit = current_price <= position['trailing_price']
            
            if stop_hit or trailing_hit:
                # Record position result
                result = {
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_size': position['size'],
                    'pnl': (current_price - position['entry_price']) * position['size'],
                    'exit_reason': 'stop_loss' if stop_hit else 'trailing_stop',
                    'timestamp': datetime.now().isoformat()
                }
                self.position_history.append(result)
                
                # Remove position
                del self.positions[symbol]
                
                return {
                    'status': 'closed',
                    'reason': result['exit_reason'],
                    'pnl': result['pnl']
                }
                
            # Position still open
            return {
                'status': 'open',
                'current_price': current_price,
                'stop_price': position['stop_price'],
                'trailing_price': position['trailing_price']
            }
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            return {'status': 'error', 'error': str(e)}
            
    def add_position(self,
                    symbol: str,
                    entry_price: float,
                    position_size: float,
                    quantum_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Add new trading position"""
        try:
            # Set stop losses
            stops = self.set_stop_loss(symbol, entry_price, position_size, quantum_metrics)
            
            # Record position
            self.positions[symbol] = {
                'entry_price': entry_price,
                'size': position_size,
                'stop_price': stops['stop_price'],
                'trailing_price': stops['trailing_price'],
                'risk_amount': stops['risk_amount'],
                'timestamp': datetime.now().isoformat()
            }
            
            return self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error adding position: {str(e)}")
            return {}
            
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics"""
        try:
            # Calculate total risk
            total_risk = sum(pos['risk_amount'] for pos in self.positions.values())
            
            # Calculate portfolio variance using position history
            if len(self.position_history) > 0:
                returns = [pos['pnl'] / pos['position_size'] for pos in self.position_history]
                portfolio_var = np.var(returns) if len(returns) > 1 else 0
                avg_return = np.mean(returns) if returns else 0
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                sharpe = (avg_return - risk_free_rate) / np.sqrt(portfolio_var) if portfolio_var > 0 else 0
            else:
                portfolio_var = 0
                sharpe = 0
                
            self.risk_metrics = {
                'total_risk': float(total_risk),
                'portfolio_var': float(portfolio_var),
                'sharpe_ratio': float(sharpe)
            }
            
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'total_risk': 0.0,
                'portfolio_var': 0.0,
                'sharpe_ratio': 0.0
            }
