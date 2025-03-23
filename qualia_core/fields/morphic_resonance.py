"""
Morphic Resonance Module for QUALIA Trading System
Implements advanced quantum-morphic field analysis with adaptive memory.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import hilbert
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class MorphicMetrics:
    """Metrics for morphic field analysis"""
    resonance_level: float = 0.0
    field_strength: float = 0.0
    coherence_quality: float = 0.0
    pattern_stability: float = 0.0
    phase_alignment: float = 0.0
    quantum_correlation: float = 0.0
    emergence_factor: float = 0.0

class AdaptiveFieldMemory:
    """
    Adaptive memory system for morphic field patterns
    
    Features:
    1. Pattern recognition and storage
    2. Similarity-based retrieval
    3. Dynamic pattern evolution
    4. Non-local correlation tracking
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize adaptive memory
        
        Args:
            capacity: Maximum number of patterns to store
        """
        self.capacity = capacity
        self.patterns: List[Dict[str, Any]] = []
        self.correlation_matrix = np.eye(1)
        self.pattern_strengths = np.array([])
    
    def add_pattern(self, pattern: Dict[str, Any], strength: float):
        """
        Add new pattern to memory
        
        Args:
            pattern: Market pattern to store
            strength: Pattern strength/importance
        """
        if len(self.patterns) >= self.capacity:
            # Remove weakest pattern if at capacity
            weakest_idx = np.argmin(self.pattern_strengths)
            self.patterns.pop(weakest_idx)
            self.pattern_strengths = np.delete(self.pattern_strengths, weakest_idx)
            
            # Update correlation matrix
            self.correlation_matrix = np.delete(self.correlation_matrix, weakest_idx, axis=0)
            self.correlation_matrix = np.delete(self.correlation_matrix, weakest_idx, axis=1)
        
        # Add new pattern
        self.patterns.append(pattern)
        self.pattern_strengths = np.append(self.pattern_strengths, strength)
        
        # Update correlation matrix
        new_size = len(self.patterns)
        new_matrix = np.zeros((new_size, new_size))
        new_matrix[:-1, :-1] = self.correlation_matrix
        
        # Calculate correlations with new pattern
        for i in range(new_size - 1):
            correlation = self._calculate_pattern_correlation(
                self.patterns[i], pattern
            )
            new_matrix[i, -1] = correlation
            new_matrix[-1, i] = correlation
        
        new_matrix[-1, -1] = 1.0
        self.correlation_matrix = new_matrix
    
    def find_similar_patterns(
        self, 
        target: Dict[str, Any], 
        threshold: float = 0.7
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find patterns similar to target
        
        Args:
            target: Pattern to match
            threshold: Minimum similarity threshold
            
        Returns:
            List of (pattern, similarity) tuples
        """
        if not self.patterns:
            return []
        
        similarities = []
        for pattern in self.patterns:
            similarity = self._calculate_pattern_similarity(pattern, target)
            if similarity >= threshold:
                similarities.append((pattern, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _calculate_pattern_correlation(
        self, 
        pattern1: Dict[str, Any], 
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate correlation between two patterns"""
        try:
            # Extract key features
            features1 = self._extract_pattern_features(pattern1)
            features2 = self._extract_pattern_features(pattern2)
            
            # Calculate correlation
            correlation = np.corrcoef(features1, features2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating pattern correlation: {e}")
            return 0.0
    
    def _calculate_pattern_similarity(
        self, 
        pattern1: Dict[str, Any], 
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two patterns"""
        try:
            # Extract features
            features1 = self._extract_pattern_features(pattern1)
            features2 = self._extract_pattern_features(pattern2)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(features1 - features2)
            
            # Convert to similarity score
            similarity = 1 / (1 + distance)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _extract_pattern_features(self, pattern: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from pattern"""
        features = []
        
        # Price features
        if 'prices' in pattern:
            prices = np.array(pattern['prices'])
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.median(prices),
                stats.skew(prices),
                stats.kurtosis(prices)
            ])
        
        # Volume features
        if 'volumes' in pattern:
            volumes = np.array(pattern['volumes'])
            features.extend([
                np.mean(volumes),
                np.std(volumes)
            ])
        
        # Time features
        if 'timestamps' in pattern:
            timestamps = np.array(pattern['timestamps'])
            features.extend([
                np.mean(np.diff(timestamps)),
                np.std(np.diff(timestamps))
            ])
        
        return np.array(features)

class MorphicFieldAnalyzer:
    """
    Advanced morphic field analysis system
    
    Features:
    1. Multi-dimensional field analysis
    2. Adaptive pattern recognition
    3. Quantum-morphic correlation
    4. Non-local resonance detection
    """
    
    def __init__(self, field_dimension: int = 64):
        """
        Initialize field analyzer
        
        Args:
            field_dimension: Dimension of morphic field
        """
        self.field_dimension = field_dimension
        self.memory = AdaptiveFieldMemory()
        
        # Field parameters
        self.coherence_threshold = 0.3
        self.resonance_threshold = 0.4
        self.phase_coupling = 0.2
        
        # Initialize quantum field
        self._initialize_quantum_field()
    
    def _initialize_quantum_field(self):
        """Initialize quantum morphic field"""
        # Create base field
        self.quantum_field = np.zeros((self.field_dimension, self.field_dimension), 
                                    dtype=np.complex128)
        
        # Add small identity component for stability
        self.quantum_field += 1e-10 * np.eye(self.field_dimension)
        
        # Add quantum noise
        self.quantum_field += np.random.normal(0, 0.01, self.quantum_field.shape) + \
                            1j * np.random.normal(0, 0.01, self.quantum_field.shape)
        
        # Normalize
        self.quantum_field /= np.trace(self.quantum_field)
    
    def analyze_market_data(
        self, 
        market_data: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[MorphicMetrics, Dict[str, Any]]:
        """
        Analyze market data through morphic field
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Tuple of (metrics, analysis results)
        """
        # Convert data to numpy if needed
        if isinstance(market_data, pd.DataFrame):
            data = market_data['close'].values
        else:
            data = market_data
        
        # Ensure data is finite
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate basic metrics
        metrics = self._calculate_field_metrics(data)
        
        # Extract patterns
        patterns = self._extract_market_patterns(data)
        
        # Update memory with new patterns
        for pattern in patterns:
            self.memory.add_pattern(pattern, metrics.resonance_level)
        
        # Find similar historical patterns
        similar_patterns = self.memory.find_similar_patterns(patterns[-1])
        
        # Prepare analysis results
        analysis = {
            'patterns': patterns,
            'similar_patterns': similar_patterns,
            'field_state': self._get_field_state()
        }
        
        return metrics, analysis
    
    def _calculate_field_metrics(self, data: np.ndarray) -> MorphicMetrics:
        """Calculate morphic field metrics"""
        metrics = MorphicMetrics()
        
        try:
            # Calculate Hilbert transform for phase analysis
            analytic_signal = hilbert(data)
            phase = np.angle(analytic_signal)
            amplitude = np.abs(analytic_signal)
            
            # Calculate field metrics
            metrics.resonance_level = float(np.mean(amplitude))
            metrics.field_strength = float(np.std(amplitude))
            metrics.coherence_quality = float(np.abs(np.mean(np.exp(1j * phase))))
            metrics.pattern_stability = float(np.mean(np.diff(phase) ** 2))
            metrics.phase_alignment = float(np.mean(np.cos(phase)))
            
            # Calculate quantum correlation
            density_matrix = np.outer(data, np.conj(data))
            density_matrix /= np.trace(density_matrix)
            metrics.quantum_correlation = float(np.abs(np.trace(
                np.dot(density_matrix, self.quantum_field)
            )))
            
            # Calculate emergence factor
            metrics.emergence_factor = float(stats.entropy(np.abs(amplitude)))
            
        except Exception as e:
            logger.error(f"Error calculating field metrics: {e}")
        
        return metrics
    
    def _extract_market_patterns(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract patterns from market data"""
        patterns = []
        
        try:
            # Use rolling windows to extract patterns
            window_size = min(50, len(data))
            for i in range(0, len(data) - window_size + 1, window_size // 2):
                window = data[i:i + window_size]
                
                pattern = {
                    'prices': window.tolist(),
                    'timestamps': list(range(i, i + window_size)),
                    'mean': float(np.mean(window)),
                    'std': float(np.std(window)),
                    'skew': float(stats.skew(window)),
                    'kurtosis': float(stats.kurtosis(window))
                }
                
                patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error extracting market patterns: {e}")
        
        return patterns
    
    def _get_field_state(self) -> Dict[str, float]:
        """Get current state of quantum field"""
        return {
            'energy': float(np.abs(np.trace(self.quantum_field))),
            'coherence': float(np.abs(np.mean(np.diag(self.quantum_field)))),
            'entanglement': float(np.abs(np.trace(
                np.dot(self.quantum_field, self.quantum_field)
            )))
        }
