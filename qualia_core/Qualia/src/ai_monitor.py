# src/ai_monitor.py
import logging
import numpy as np
from typing import Dict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import asyncio

class AIMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = RandomForestRegressor(n_estimators=100)
        self.performance_history = []
        self.monitoring = False

    async def start_monitoring(self):
        """Start the AI monitoring system"""
        self.monitoring = True
        self.logger.info("AI monitoring system started")
        asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            await self._update_model()
            await asyncio.sleep(60)  # Update every minute

    async def _update_model(self):
        """Update the AI model with recent performance data"""
        if len(self.performance_history) > 100:
            data = pd.DataFrame(self.performance_history)
            X = data[['temperature', 'power_usage', 'hash_rate', 'difficulty']]
            y = data['efficiency']
            
            try:
                self.model.fit(X, y)
                self.logger.info("AI model updated successfully")
            except Exception as e:
                self.logger.error(f"Error updating AI model: {e}")

    def record_performance(self, metrics: Dict):
        """Record mining performance metrics"""
        self.performance_history.append(metrics)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    async def get_optimal_params(self) -> Dict:
        """Get optimal mining parameters based on AI analysis"""
        try:
            if len(self.performance_history) < 10:
                return self._get_default_params()

            recent_data = pd.DataFrame(self.performance_history[-100:])
            current_conditions = recent_data.mean().to_dict()

            # Generate parameter combinations
            param_space = self._generate_param_space(current_conditions)
            predictions = self.model.predict(param_space)
            
            best_params = param_space.iloc[predictions.argmax()]
            return {
                'threads': int(best_params['threads']),
                'batch_size': int(best_params['batch_size']),
                'intensity': float(best_params['intensity'])
            }
        except Exception as e:
            self.logger.error(f"Error getting optimal parameters: {e}")
            return self._get_default_params()

    def _generate_param_space(self, current_conditions: Dict) -> pd.DataFrame:
        """Generate parameter combinations for optimization"""
        threads = np.arange(1, 33, 2)
        batch_sizes = np.array([256, 512, 1024, 2048])
        intensities = np.linspace(0.5, 1.0, 6)
        
        param_combinations = []
        for t in threads:
            for b in batch_sizes:
                for i in intensities:
                    params = current_conditions.copy()
                    params.update({
                        'threads': t,
                        'batch_size': b,
                        'intensity': i
                    })
                    param_combinations.append(params)
        
        return pd.DataFrame(param_combinations)

    def _get_default_params(self) -> Dict:
        """Return default mining parameters"""
        return {
            'threads': 8,
            'batch_size': 1024,
            'intensity': 0.8
        }

    async def stop_monitoring(self):
        """Stop the AI monitoring system"""
        self.monitoring = False
        self.logger.info("AI monitoring system stopped")