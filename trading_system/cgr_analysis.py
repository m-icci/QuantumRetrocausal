import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
from sklearn.cluster import OPTICS

logger = logging.getLogger(__name__)

class CGRAnalyzer:
    """
    Chaos Game Representation Analyzer for market pattern detection
    """
    def __init__(self, resolution: int = 1024, memory_length: int = 100):
        self.resolution = resolution
        self.memory_length = memory_length
        self.price_history: List[float] = []
        self.cgr_points: List[Tuple[float, float]] = []
        self.pattern_memory: Dict[str, List[float]] = {}
        
    def _normalize_prices(self, prices: List[float]) -> np.ndarray:
        """Normalize price data to [0,1] range"""
        if not prices:
            return np.array([])
        prices_array = np.array(prices)
        min_price = np.min(prices_array)
        max_price = np.max(prices_array)
        if max_price == min_price:
            return np.zeros_like(prices_array)
        return (prices_array - min_price) / (max_price - min_price)
        
    def _generate_cgr_points(self, normalized_prices: np.ndarray) -> List[Tuple[float, float]]:
        """Generate CGR points from normalized prices"""
        if len(normalized_prices) < 3:
            return []
            
        points = []
        x, y = 0.5, 0.5  # Start at center
        
        for i in range(len(normalized_prices)-2):
            # Use three consecutive prices to determine next point
            p1, p2, p3 = normalized_prices[i:i+3]
            
            # Calculate angle based on price relationships
            angle = 2 * np.pi * (p2 - p1) / (p3 - p1) if p3 != p1 else 0
            
            # Update position using chaos game algorithm
            x = 0.5 * x + 0.5 * np.cos(angle)
            y = 0.5 * y + 0.5 * np.sin(angle)
            
            points.append((float(x), float(y)))
            
        return points
        
    def _detect_patterns(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Detect patterns in CGR points using OPTICS clustering"""
        if not points:
            return {"count": 0, "patterns": {}}
            
        try:
            # Convert points to numpy array
            X = np.array(points)
            
            # Run OPTICS clustering
            clustering = OPTICS(min_samples=5, metric='euclidean')
            clustering.fit(X)
            
            # Analyze clusters
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            patterns = {}
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                center = np.mean(cluster_points, axis=0)
                spread = np.std(cluster_points, axis=0)
                
                patterns[f"pattern_{i}"] = {
                    "center": (float(center[0]), float(center[1])),
                    "spread": (float(spread[0]), float(spread[1])),
                    "points": len(cluster_points)
                }
                
            return {
                "count": n_clusters,
                "patterns": patterns
            }
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            return {"count": 0, "patterns": {}}
            
    def _calculate_fractal_dimension(self, points: List[Tuple[float, float]]) -> float:
        """Calculate fractal dimension of CGR pattern"""
        if len(points) < 10:
            return 1.0
            
        try:
            # Box counting method
            X = np.array(points)
            scales = np.logspace(-3, 0, num=20)
            counts = []
            
            for scale in scales:
                # Count boxes that contain points
                scaled_x = (X[:, 0] / scale).astype(int)
                scaled_y = (X[:, 1] / scale).astype(int)
                unique_boxes = len(set(zip(scaled_x, scaled_y)))
                counts.append(unique_boxes)
                
            # Calculate dimension from log-log relationship
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]  # Negative slope gives fractal dimension
            
        except Exception as e:
            logger.error(f"Error calculating fractal dimension: {str(e)}")
            return 1.0
            
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform CGR analysis on market data
        
        Args:
            market_data: Dictionary containing market data for different symbols
            
        Returns:
            Dictionary containing CGR analysis results
        """
        try:
            # Extract prices
            prices = [data['price'] for data in market_data.values()]
            
            # Update price history
            self.price_history.extend(prices)
            if len(self.price_history) > self.memory_length:
                self.price_history = self.price_history[-self.memory_length:]
                
            # Generate CGR points
            normalized_prices = self._normalize_prices(self.price_history)
            self.cgr_points = self._generate_cgr_points(normalized_prices)
            
            # Detect patterns
            patterns = self._detect_patterns(self.cgr_points)
            
            # Calculate fractal dimension
            fractal_dim = self._calculate_fractal_dimension(self.cgr_points)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'patterns': patterns,
                'fractal_dimension': fractal_dim,
                'points': self.cgr_points,
                'num_points': len(self.cgr_points)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in CGR analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'patterns': {"count": 0, "patterns": {}},
                'fractal_dimension': 1.0,
                'points': [],
                'num_points': 0
            }
