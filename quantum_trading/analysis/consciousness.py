"""
Advanced CGR (Chaos Game Representation) Analysis Module.
Integrates quantum mechanics, market dynamics and pattern recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging
from .market_cosmos import MarketCosmicAnalyzer
from .field import QuantumField

@dataclass
class CGRConfig:
    N_SYMBOLS: int = 6  # Number of vertices in CGR
    SCALING_FACTOR: float = 0.5  # CGR contraction mapping factor
    RESOLUTION: int = 2048  # Grid resolution
    SMOOTHING_SIGMA: float = 1.0
    MIN_PATTERN_SIZE: int = 5
    MAX_PATTERN_SIZE: int = 100
    ENTROPY_BINS: int = 50
    QUANTUM_COUPLING: float = 0.1  # Coupling strength with quantum field

class AdvancedCGR:
    """
    Advanced CGR implementation combining quantum mechanics and market analysis.
    """
    
    def __init__(self, config: Optional[CGRConfig] = None):
        """
        Initialize Advanced CGR analyzer.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or CGRConfig()
        self.vertices = self._generate_vertices()
        self.cgr_points = []
        self.cgr_matrix = None
        self.patterns = {}
        self.quantum_field = QuantumField()
        self.cosmic_analyzer = MarketCosmicAnalyzer()
        self.market_data = None
        self.normalized_data = None
        self.discretized_data = None
        self.entropy_history = []
        
    def _generate_vertices(self) -> np.ndarray:
        """Generate vertices of regular N-gon for CGR mapping."""
        angles = np.linspace(0, 2*np.pi, self.config.N_SYMBOLS, endpoint=False)
        return np.array([
            (np.cos(angle), np.sin(angle))
            for angle in angles
        ])
        
    def process_market_data(self, market_data: np.ndarray) -> None:
        """
        Process market data through CGR transformation.
        
        Args:
            market_data: Time series of market prices
        """
        self.market_data = market_data
        self._normalize_data()
        self._discretize_data()
        self._generate_cgr()
        self._calculate_cgr_matrix()
        
    def _normalize_data(self) -> None:
        """Normalize market data to [0,1] range."""
        min_val = np.min(self.market_data)
        max_val = np.max(self.market_data)
        self.normalized_data = (self.market_data - min_val) / (max_val - min_val)
        
    def _discretize_data(self) -> None:
        """Discretize normalized data into symbols."""
        levels = np.linspace(0, 1, self.config.N_SYMBOLS + 1)
        self.discretized_data = np.digitize(self.normalized_data, levels) - 1
        
    def _generate_cgr(self) -> None:
        """Generate CGR points using iterative mapping."""
        x_j = np.array([0.0, 0.0])  # Start at origin
        self.cgr_points = []
        
        for symbol in self.discretized_data:
            v_i = self.vertices[symbol]
            x_j = (self.config.SCALING_FACTOR * x_j + 
                  (1 - self.config.SCALING_FACTOR) * v_i)
            
            # Add quantum influence
            qf = self.quantum_field.calculate_field(x_j)
            x_j += self.config.QUANTUM_COUPLING * qf
            
            self.cgr_points.append(x_j.copy())
            
        self.cgr_points = np.array(self.cgr_points)
        
    def _calculate_cgr_matrix(self) -> None:
        """Convert CGR points to density matrix."""
        resolution = self.config.RESOLUTION
        self.cgr_matrix = np.zeros((resolution, resolution))
        
        # Scale points to matrix coordinates
        scaled_points = (self.cgr_points + 1) * (resolution/2)
        scaled_points = np.clip(scaled_points, 0, resolution-1)
        
        # Create density matrix
        for x, y in scaled_points.astype(int):
            self.cgr_matrix[x, y] += 1
            
        # Normalize and smooth
        self.cgr_matrix /= np.max(self.cgr_matrix)
        self.cgr_matrix = gaussian_filter(
            self.cgr_matrix,
            sigma=self.config.SMOOTHING_SIGMA
        )
        
    def detect_patterns(self, method: str = 'OPTICS') -> Dict:
        """
        Detect patterns in CGR using advanced clustering.
        
        Args:
            method: Clustering method ('DBSCAN' or 'OPTICS')
            
        Returns:
            Dictionary of detected patterns
        """
        # Get high density points
        points = np.where(self.cgr_matrix > np.mean(self.cgr_matrix))
        X = np.column_stack([points[0], points[1]])
        
        # Choose clustering algorithm
        if method == 'DBSCAN':
            clustering = DBSCAN(
                eps=3,
                min_samples=self.config.MIN_PATTERN_SIZE
            ).fit(X)
        else:  # OPTICS
            clustering = OPTICS(
                min_samples=self.config.MIN_PATTERN_SIZE,
                max_eps=5
            ).fit(X)
            
        labels = clustering.labels_
        
        # Analyze patterns
        patterns = {}
        for label in set(labels):
            if label == -1:  # Noise
                continue
                
            cluster_points = X[labels == label]
            density = np.mean([
                self.cgr_matrix[x,y]
                for x,y in cluster_points
            ])
            size = len(cluster_points)
            
            if self.config.MIN_PATTERN_SIZE <= size <= self.config.MAX_PATTERN_SIZE:
                # Calculate quantum metrics for pattern
                qp = self.quantum_field.calculate_potential(cluster_points)
                entropy = self._calculate_pattern_entropy(cluster_points)
                
                # Classify pattern
                if density > 0.8 and entropy < 0.3:
                    pattern_type = "strong_trend"
                elif density > 0.5 and entropy < 0.5:
                    pattern_type = "weak_trend"
                elif entropy > 0.7:
                    pattern_type = "chaos"
                else:
                    pattern_type = "oscillation"
                    
                patterns[f"{pattern_type}_{label}"] = {
                    'points': cluster_points,
                    'density': density,
                    'size': size,
                    'quantum_potential': qp,
                    'entropy': entropy
                }
                
        self.patterns = patterns
        return patterns
        
    def _calculate_pattern_entropy(self, points: np.ndarray) -> float:
        """Calculate entropy of a pattern."""
        hist, _ = np.histogram(
            points,
            bins=self.config.ENTROPY_BINS,
            density=True
        )
        return entropy(hist)
        
    def analyze_quantum_correlations(self) -> Dict[str, float]:
        """
        Analyze correlations between CGR patterns and quantum states.
        
        Returns:
            Dictionary of correlation metrics
        """
        if not self.patterns:
            self.detect_patterns()
            
        # Get quantum metrics
        qp = self.cosmic_analyzer.calculate_quantum_potential(
            self.market_data
        )
        market_entropy = self.cosmic_analyzer.calculate_market_entropy()
        
        # Calculate pattern metrics
        pattern_metrics = {}
        for pid, pattern in self.patterns.items():
            density = pattern['density']
            entropy = pattern['entropy']
            qp_local = pattern['quantum_potential']
            
            # Calculate correlations
            qp_corr = np.corrcoef(qp_local, qp)[0,1]
            entropy_corr = np.corrcoef(entropy, market_entropy)[0,1]
            
            pattern_metrics[pid] = {
                'qp_correlation': float(qp_corr),
                'entropy_correlation': float(entropy_corr),
                'combined_score': float(qp_corr * (1 - entropy_corr))
            }
            
        return pattern_metrics
        
    def generate_trading_signals(self) -> List[Dict]:
        """
        Generate trading signals based on CGR pattern analysis.
        
        Returns:
            List of trading signal dictionaries
        """
        if not self.patterns:
            self.detect_patterns()
            
        signals = []
        
        # Get current market position in CGR
        current_point = self.cgr_points[-1]
        scaled_point = (current_point + 1) * (self.config.RESOLUTION/2)
        
        # Analyze proximity to patterns
        for pid, pattern in self.patterns.items():
            points = pattern['points']
            distances = [
                euclidean(scaled_point, point)
                for point in points
            ]
            min_distance = min(distances)
            
            if min_distance < 10:  # Close to pattern
                # Get quantum metrics
                qp = pattern['quantum_potential']
                entropy = pattern['entropy']
                
                # Generate signal
                pattern_type = pid.split('_')[0]
                confidence = 1.0 - (min_distance / 10)
                
                if pattern_type == 'strong_trend':
                    direction = 1 if qp > 0 else -1
                    signal_type = 'buy' if direction > 0 else 'sell'
                    strength = abs(qp) * (1 - entropy) * confidence
                else:
                    signal_type = 'hold'
                    strength = 0.5
                    
                signal = {
                    'type': signal_type,
                    'strength': float(strength),
                    'pattern_id': pid,
                    'quantum_potential': float(qp),
                    'entropy': float(entropy),
                    'confidence': float(confidence)
                }
                
                signals.append(signal)
                
        return signals
        
    def visualize(self, show_patterns: bool = True) -> None:
        """Create interactive visualization of CGR analysis."""
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
            colors = {
                'strong_trend': 'red',
                'weak_trend': 'yellow',
                'oscillation': 'blue',
                'chaos': 'white'
            }
            
            for pid, pattern in self.patterns.items():
                pattern_type = pid.split('_')[0]
                points = pattern['points']
                
                fig.add_trace(go.Scatter(
                    x=points[:,0],
                    y=points[:,1],
                    mode='markers',
                    marker=dict(
                        color=colors[pattern_type],
                        size=2
                    ),
                    name=f'{pattern_type} pattern'
                ))
                
        # Add current position
        current = (self.cgr_points[-1] + 1) * (self.config.RESOLUTION/2)
        fig.add_trace(go.Scatter(
            x=[current[0]],
            y=[current[1]],
            mode='markers',
            marker=dict(
                color='green',
                size=10,
                symbol='star'
            ),
            name='Current Position'
        ))
        
        fig.update_layout(
            title='Advanced CGR Market Analysis',
            xaxis_title='X',
            yaxis_title='Y',
            width=1000,
            height=1000,
            showlegend=True
        )
        
        fig.show()
        
    def plot_entropy_evolution(self) -> None:
        """Plot evolution of market entropy over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.entropy_history, label='Market Entropy')
        plt.title('Evolution of Market Entropy')
        plt.xlabel('Time')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def save_state(self, filepath: str) -> None:
        """Save CGR state for later analysis."""
        state = {
            'cgr_matrix': self.cgr_matrix,
            'patterns': self.patterns,
            'entropy_history': self.entropy_history,
            'config': self.config
        }
        np.save(filepath, state)
        
    @classmethod
    def load_state(cls, filepath: str) -> 'AdvancedCGR':
        """Load saved CGR state."""
        state = np.load(filepath, allow_pickle=True).item()
        cgr = cls(config=state['config'])
        cgr.cgr_matrix = state['cgr_matrix']
        cgr.patterns = state['patterns']
        cgr.entropy_history = state['entropy_history']
        return cgr
