"""
CGR (Chaos Game Representation) Analysis Module for Market Data.
Adapts the CGR technique for visualizing and analyzing market patterns.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Optional
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from .market_cosmos import MarketCosmicAnalyzer

@dataclass
class CGRConfig:
    RESOLUTION: int = 2048
    SMOOTHING_SIGMA: float = 1.0
    MIN_PATTERN_LENGTH: int = 3
    MAX_PATTERN_LENGTH: int = 10
    SIGNIFICANCE_ITERATIONS: int = 1000

class MarketCGRAnalyzer:
    """
    CGR analysis for market data, integrating with quantum analysis.
    """
    
    def __init__(self, market_data: np.ndarray, config: Optional[CGRConfig] = None):
        """
        Initialize the CGR analyzer with market data.
        
        Args:
            market_data: Array of market data (prices, volumes, etc)
            config: Optional configuration parameters
        """
        self.market_data = market_data
        self.config = config or CGRConfig()
        self.cgr_matrix = None
        self.patterns = {}
        self.cosmic_analyzer = MarketCosmicAnalyzer()
        
    def generate_cgr(self, normalize: bool = True, smooth: bool = True) -> np.ndarray:
        """
        Generate CGR matrix from market data.
        
        Args:
            normalize: Whether to normalize the CGR matrix
            smooth: Whether to apply Gaussian smoothing
            
        Returns:
            2D numpy array representing the CGR
        """
        resolution = self.config.RESOLUTION
        self.cgr_matrix = np.zeros((resolution, resolution))
        
        # Convert market data to relative changes
        changes = np.diff(self.market_data) / self.market_data[:-1]
        
        # Map market changes to CGR coordinates
        x, y = 0.5, 0.5  # Start at center
        for change in changes:
            # Map price changes to corners:
            # Strong up -> (1,1)
            # Weak up -> (1,0)
            # Weak down -> (0,1)
            # Strong down -> (0,0)
            threshold = np.std(changes)
            
            if change > threshold:
                next_x, next_y = 1, 1
            elif change > 0:
                next_x, next_y = 1, 0
            elif change > -threshold:
                next_x, next_y = 0, 1
            else:
                next_x, next_y = 0, 0
                
            # Update position
            x = 0.5 * (x + next_x)
            y = 0.5 * (y + next_y)
            
            # Map to matrix coordinates
            xi = int(x * (resolution-1))
            yi = int(y * (resolution-1))
            self.cgr_matrix[yi, xi] += 1
            
        if normalize:
            self.cgr_matrix /= np.max(self.cgr_matrix)
            
        if smooth:
            self.cgr_matrix = gaussian_filter(
                self.cgr_matrix,
                sigma=self.config.SMOOTHING_SIGMA
            )
            
        return self.cgr_matrix
        
    def detect_patterns(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect recurring patterns in the CGR using DBSCAN clustering.
        
        Returns:
            Dictionary mapping pattern types to lists of coordinates
        """
        if self.cgr_matrix is None:
            self.generate_cgr()
            
        # Find high-density regions
        points = np.where(self.cgr_matrix > np.mean(self.cgr_matrix))
        X = np.column_stack([points[0], points[1]])
        
        # Cluster points
        clustering = DBSCAN(eps=3, min_samples=5).fit(X)
        labels = clustering.labels_
        
        # Analyze patterns
        patterns = {}
        for label in set(labels):
            if label == -1:  # Noise points
                continue
                
            cluster_points = X[labels == label]
            
            # Analyze pattern characteristics
            density = np.mean([self.cgr_matrix[x,y] for x,y in cluster_points])
            size = len(cluster_points)
            
            if size > 20:  # Significant patterns
                if density > 0.8:
                    pattern_type = "strong_trend"
                elif density > 0.5:
                    pattern_type = "weak_trend"
                else:
                    pattern_type = "oscillation"
                    
                patterns[pattern_type] = patterns.get(pattern_type, []) + [
                    (x,y) for x,y in cluster_points
                ]
                
        self.patterns = patterns
        return patterns
        
    def analyze_quantum_correlations(self) -> Dict[str, float]:
        """
        Analyze correlations between CGR patterns and quantum market states.
        
        Returns:
            Dictionary of correlation metrics
        """
        if not self.patterns:
            self.detect_patterns()
            
        # Get quantum metrics from MarketCosmicAnalyzer
        quantum_potential = self.cosmic_analyzer.calculate_quantum_potential(
            self.market_data
        )
        market_entropy = self.cosmic_analyzer.calculate_market_entropy()
        
        # Calculate pattern densities
        pattern_densities = {
            ptype: len(coords) / self.config.RESOLUTION**2
            for ptype, coords in self.patterns.items()
        }
        
        # Correlate with quantum metrics
        correlations = {}
        for ptype, density in pattern_densities.items():
            qp_corr = np.corrcoef(density, quantum_potential)[0,1]
            entropy_corr = np.corrcoef(density, market_entropy)[0,1]
            
            correlations[f"{ptype}_qp_correlation"] = qp_corr
            correlations[f"{ptype}_entropy_correlation"] = entropy_corr
            
        return correlations
        
    def visualize_cgr(self, show_patterns: bool = True) -> None:
        """
        Create interactive visualization of the CGR with patterns highlighted.
        """
        if self.cgr_matrix is None:
            self.generate_cgr()
            
        fig = go.Figure()
        
        # Base CGR heatmap
        fig.add_trace(go.Heatmap(
            z=self.cgr_matrix,
            colorscale='Viridis',
            showscale=True,
            name='CGR'
        ))
        
        if show_patterns and self.patterns:
            # Add pattern overlays
            colors = {'strong_trend': 'red', 'weak_trend': 'yellow', 'oscillation': 'blue'}
            
            for ptype, coords in self.patterns.items():
                x, y = zip(*coords)
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(color=colors[ptype], size=2),
                    name=f'{ptype} pattern'
                ))
                
        fig.update_layout(
            title='Market CGR Analysis',
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            height=800
        )
        
        fig.show()
        
    def generate_trading_signals(self) -> List[Dict]:
        """
        Generate trading signals based on CGR pattern analysis.
        
        Returns:
            List of trading signal dictionaries
        """
        if not self.patterns:
            self.detect_patterns()
            
        signals = []
        
        # Analyze recent market position in CGR
        recent_changes = np.diff(self.market_data[-10:]) / self.market_data[-11:-1]
        x, y = 0.5, 0.5
        
        for change in recent_changes:
            threshold = np.std(recent_changes)
            if change > threshold:
                next_x, next_y = 1, 1
            elif change > 0:
                next_x, next_y = 1, 0
            elif change > -threshold:
                next_x, next_y = 0, 1
            else:
                next_x, next_y = 0, 0
                
            x = 0.5 * (x + next_x)
            y = 0.5 * (y + next_y)
            
        current_x = int(x * (self.config.RESOLUTION-1))
        current_y = int(y * (self.config.RESOLUTION-1))
        
        # Check proximity to known patterns
        for pattern_type, coords in self.patterns.items():
            distances = [
                euclidean((current_x, current_y), coord)
                for coord in coords
            ]
            min_distance = min(distances)
            
            if min_distance < 10:  # Close to pattern
                # Get quantum metrics
                qp = self.cosmic_analyzer.calculate_quantum_potential(
                    self.market_data[-20:]
                )
                entropy = self.cosmic_analyzer.calculate_market_entropy()
                
                signal = {
                    'pattern_type': pattern_type,
                    'confidence': 1.0 - (min_distance / 10),
                    'quantum_potential': float(qp),
                    'market_entropy': float(entropy),
                    'signal': 'buy' if pattern_type == 'strong_trend' else 'sell'
                }
                
                signals.append(signal)
                
        return signals
