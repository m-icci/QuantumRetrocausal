#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrocausal Analysis

Implements analysis of quantum field patterns for detection of retrocausal signals,
where future states influence present trading decisions through quantum non-locality
and backward causation effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats, signal
import networkx as nx
from collections import deque
from scipy.stats import entropy
from scipy.signal import correlate

logger = logging.getLogger(__name__)

class RetrocausalAnalyzer:
    """
    Analisa padrões retrocausais em estados quânticos
    """
    
    def __init__(self, coherence_threshold: float = 0.45):
        self.coherence_threshold = coherence_threshold
        self.history = []
        self.max_history = 1000
        self.temporal_window = 10
        
    async def analyze(self, current_state: np.ndarray, state_history: List[np.ndarray]) -> Dict[str, float]:
        """
        Analisa padrões retrocausais entre estados
        
        Args:
            current_state: Estado quântico atual
            state_history: Histórico de estados
            
        Returns:
            Dict com métricas retrocausais
        """
        try:
            # Verifica dados de entrada
            if len(state_history) < 2:
                logger.warning("Insufficient history for retrocausal analysis")
                return self._empty_metrics()
                
            # Normaliza estados
            current = self._normalize_state(current_state)
            history = [self._normalize_state(s) for s in state_history[-self.temporal_window:]]
            
            # Calcula métricas básicas
            metrics = {
                'resonance': self._calculate_resonance(current, history),
                'temporal_correlation': self._calculate_temporal_correlation(current, history),
                'causal_entropy': self._calculate_causal_entropy(current, history),
                'quantum_memory': self._calculate_quantum_memory(current, history)
            }
            
            # Calcula métricas avançadas
            metrics.update({
                'retrocausal_strength': self._calculate_retrocausal_strength(current, history),
                'temporal_entanglement': self._calculate_temporal_entanglement(current, history),
                'causal_symmetry': self._calculate_causal_symmetry(current, history),
                'temporal_coherence': self._calculate_temporal_coherence(current, history)
            })
            
            # Atualiza histórico
            self.history.append(metrics)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in retrocausal analysis: {str(e)}")
            return self._empty_metrics()
            
    def _empty_metrics(self) -> Dict[str, float]:
        """Retorna métricas vazias"""
        return {
            'resonance': 0.0,
            'temporal_correlation': 0.0,
            'causal_entropy': 0.0,
            'quantum_memory': 0.0,
            'retrocausal_strength': 0.0,
            'temporal_entanglement': 0.0,
            'causal_symmetry': 0.0,
            'temporal_coherence': 0.0
        }
        
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normaliza estado quântico"""
        try:
            norm = np.linalg.norm(state)
            if norm > 0:
                return state / norm
            return state
        except Exception as e:
            logger.error(f"Error normalizing state: {str(e)}")
            return state
            
    def _calculate_resonance(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula ressonância entre estados"""
        try:
            # Usa produto interno como medida de ressonância
            resonances = [np.abs(np.vdot(current, h)) for h in history]
            return float(np.mean(resonances))
        except Exception as e:
            logger.error(f"Error calculating resonance: {str(e)}")
            return 0.0
            
    def _calculate_temporal_correlation(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula correlação temporal"""
        try:
            # Usa correlação cruzada
            correlations = [np.max(np.abs(correlate(current, h))) for h in history]
            return float(np.mean(correlations))
        except Exception as e:
            logger.error(f"Error calculating temporal correlation: {str(e)}")
            return 0.0
            
    def _calculate_causal_entropy(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula entropia causal"""
        try:
            # Usa entropia relativa
            probs_current = np.abs(current) ** 2
            entropies = []
            for h in history:
                probs_hist = np.abs(h) ** 2
                # Remove zeros
                mask = (probs_current > 0) & (probs_hist > 0)
                if np.any(mask):
                    entropies.append(entropy(probs_current[mask], probs_hist[mask]))
            return float(np.mean(entropies)) if entropies else 0.0
        except Exception as e:
            logger.error(f"Error calculating causal entropy: {str(e)}")
            return 0.0
            
    def _calculate_quantum_memory(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula memória quântica"""
        try:
            # Usa sobreposição com histórico
            memory = 0.0
            for i, h in enumerate(history):
                weight = np.exp(-i/self.temporal_window)  # Peso temporal
                memory += weight * np.abs(np.vdot(current, h))
            return float(memory / len(history))
        except Exception as e:
            logger.error(f"Error calculating quantum memory: {str(e)}")
            return 0.0
            
    def _calculate_retrocausal_strength(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula força retrocausal"""
        try:
            # Usa correlação com estados futuros previstos
            strength = 0.0
            for i in range(len(history)-1):
                predicted = history[i] + (history[i+1] - history[i])  # Previsão linear
                strength += np.abs(np.vdot(current, predicted))
            return float(strength / (len(history)-1)) if len(history) > 1 else 0.0
        except Exception as e:
            logger.error(f"Error calculating retrocausal strength: {str(e)}")
            return 0.0
            
    def _calculate_temporal_entanglement(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula emaranhamento temporal"""
        try:
            # Usa matriz densidade temporal
            states = [current] + history
            density = np.zeros((len(states), len(states)), dtype=complex)
            for i, s1 in enumerate(states):
                for j, s2 in enumerate(states):
                    density[i,j] = np.vdot(s1, s2)
            # Entropia de von Neumann
            eigenvalues = np.linalg.eigvalsh(density)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
        except Exception as e:
            logger.error(f"Error calculating temporal entanglement: {str(e)}")
            return 0.0
            
    def _calculate_causal_symmetry(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula simetria causal"""
        try:
            # Compara correlações forward e backward
            forward = []
            backward = []
            for i in range(len(history)-1):
                forward.append(np.abs(np.vdot(history[i], history[i+1])))
                backward.append(np.abs(np.vdot(history[i+1], history[i])))
            if not forward:
                return 0.0
            # Simetria como razão entre correlações
            return float(np.mean(forward) / np.mean(backward))
        except Exception as e:
            logger.error(f"Error calculating causal symmetry: {str(e)}")
            return 0.0
            
    def _calculate_temporal_coherence(self, current: np.ndarray, history: List[np.ndarray]) -> float:
        """Calcula coerência temporal"""
        try:
            # Usa matriz densidade temporal
            states = [current] + history
            coherence = 0.0
            for i, s1 in enumerate(states):
                for j, s2 in enumerate(states):
                    if i != j:
                        coherence += np.abs(np.vdot(s1, s2))
            return float(coherence / (len(states) * (len(states)-1))) if len(states) > 1 else 0.0
        except Exception as e:
            logger.error(f"Error calculating temporal coherence: {str(e)}")
            return 0.0

    def add_data_point(self, 
                     timestamp: datetime,
                     price_data: Dict[str, float],
                     field_metrics: Dict[str, Any]) -> None:
        """
        Adds a new data point to the analyzer.
        
        Args:
            timestamp: Timestamp of the data point
            price_data: Dictionary with price data (open, high, low, close, volume)
            field_metrics: Metrics from quantum field (energy, coherence, etc.)
        """
        # Store data in history
        data_point = {
            'timestamp': timestamp,
            **price_data
        }
        self.price_history.append(data_point)
        
        field_point = {
            'timestamp': timestamp,
            **field_metrics
        }
        self.field_metrics_history.append(field_point)
        
        # Update metrics for each time scale
        for scale in self.time_scales:
            if len(self.price_history) >= 2:
                # Calculate metrics for this scale
                scale_metrics = self._calculate_scale_metrics(scale)
                
                if scale_metrics:
                    if len(self.metrics_by_scale[scale]) >= self.lookback_window:
                        self.metrics_by_scale[scale].pop(0)
                    self.metrics_by_scale[scale].append(scale_metrics)
    
    def _calculate_scale_metrics(self, time_scale: int) -> Dict[str, float]:
        """
        Calculates metrics for a specific time scale.
        
        Args:
            time_scale: Time scale in minutes
            
        Returns:
            Dictionary with metrics for this scale
        """
        if len(self.price_history) < 2 or len(self.field_metrics_history) < 2:
            return {}
        
        # Filter data for this time scale
        current_time = self.price_history[-1]['timestamp']
        scale_start = current_time - timedelta(minutes=time_scale)
        
        # Get price data within this scale
        scale_prices = [p['close'] for p in self.price_history 
                      if p['timestamp'] >= scale_start]
        
        # Get field metrics within this scale
        scale_field = [f for f in self.field_metrics_history 
                      if f['timestamp'] >= scale_start]
        
        if not scale_prices or not scale_field:
            return {}
        
        # Calculate price metrics
        price_return = (scale_prices[-1] / scale_prices[0] - 1) if len(scale_prices) > 1 else 0
        price_volatility = np.std(np.diff(scale_prices) / scale_prices[:-1]) if len(scale_prices) > 2 else 0
        
        # Calculate field metrics
        avg_energy = np.mean([f.get('energy', 0) for f in scale_field])
        avg_coherence = np.mean([f.get('coherence', 0) for f in scale_field])
        avg_entanglement = np.mean([f.get('entanglement', 0) for f in scale_field])
        
        # Phase coherence between price and field
        if len(scale_prices) > 2 and len(scale_field) > 2:
            # Normalize and interpolate to same length if needed
            norm_prices = (np.array(scale_prices) - np.mean(scale_prices)) / np.std(scale_prices) if np.std(scale_prices) > 0 else np.zeros(len(scale_prices))
            field_energies = np.array([f.get('energy', 0) for f in scale_field])
            norm_energies = (field_energies - np.mean(field_energies)) / np.std(field_energies) if np.std(field_energies) > 0 else np.zeros(len(field_energies))
            
            # Ensure same length
            min_len = min(len(norm_prices), len(norm_energies))
            if min_len > 1:
                norm_prices = norm_prices[-min_len:]
                norm_energies = norm_energies[-min_len:]
                
                # Calculate phase coherence
                hilbert_price = signal.hilbert(norm_prices)
                hilbert_field = signal.hilbert(norm_energies)
                
                phase_price = np.unwrap(np.angle(hilbert_price))
                phase_field = np.unwrap(np.angle(hilbert_field))
                
                # Phase difference
                phase_diff = phase_price - phase_field
                phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
            else:
                phase_coherence = 0
        else:
            phase_coherence = 0
        
        return {
            'timestamp': current_time,
            'time_scale': time_scale,
            'price_return': price_return,
            'price_volatility': price_volatility,
            'avg_energy': avg_energy,
            'avg_coherence': avg_coherence,
            'avg_entanglement': avg_entanglement,
            'phase_coherence': phase_coherence
        }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Performs retrocausal analysis on stored data.
        
        Returns:
            Dictionary with analysis results
        """
        # Ensure we have enough data
        if not all(len(self.metrics_by_scale[scale]) >= self.causality_window for scale in self.time_scales):
            return {
                'signal_detected': False,
                'confidence': 0.0,
                'direction': 0,
                'time_scale': 0,
                'explanation': "Insufficient data for analysis"
            }
        
        # Detect temporal anomalies across scales
        anomalies = self._detect_temporal_anomalies()
        
        # Test for Granger causality
        causality_results = self._test_causality()
        
        # Analyze synchronicity patterns
        sync_patterns = self._analyze_synchronicity()
        
        # Look for phi-recursion patterns
        phi_patterns = self._detect_phi_recursion()
        
        # Combine signals
        retrocausal_signal = self._combine_signals(anomalies, causality_results, sync_patterns, phi_patterns)
        
        # Update causal graph
        self._update_causal_graph(retrocausal_signal)
        
        # Track retrocausal events if signal is strong enough
        if retrocausal_signal['confidence'] >= self.confidence_threshold:
            self.retrocausal_events.append({
                'timestamp': datetime.now(),
                'signal': retrocausal_signal.copy(),
                'anomalies': anomalies,
                'causality': causality_results,
                'synchronicity': sync_patterns,
                'phi_recursion': phi_patterns
            })
        
        return retrocausal_signal
    
    def _detect_temporal_anomalies(self) -> Dict[str, Any]:
        """
        Detects temporal anomalies in the data.
        
        Returns:
            Dictionary with detected anomalies
        """
        anomalies = {}
        
        for scale in self.time_scales:
            if len(self.metrics_by_scale[scale]) < self.lookback_window:
                continue
                
            scale_metrics = self.metrics_by_scale[scale]
            
            # Get key metrics
            coherence_values = [m.get('avg_coherence', 0) for m in scale_metrics]
            entanglement_values = [m.get('avg_entanglement', 0) for m in scale_metrics]
            phase_coherence = [m.get('phase_coherence', 0) for m in scale_metrics]
            price_returns = [m.get('price_return', 0) for m in scale_metrics]
            
            # Calculate Z-scores
            z_coherence = stats.zscore(coherence_values) if len(coherence_values) > 1 else np.zeros(len(coherence_values))
            z_entanglement = stats.zscore(entanglement_values) if len(entanglement_values) > 1 else np.zeros(len(entanglement_values))
            z_phase = stats.zscore(phase_coherence) if len(phase_coherence) > 1 else np.zeros(len(phase_coherence))
            
            # Look for significant deviations
            recent_z_coherence = z_coherence[-5:].mean() if len(z_coherence) >= 5 else 0
            recent_z_entanglement = z_entanglement[-5:].mean() if len(z_entanglement) >= 5 else 0
            recent_z_phase = z_phase[-5:].mean() if len(z_phase) >= 5 else 0
            
            # Check for pattern breaks
            price_pattern = self._identify_price_pattern(price_returns[-20:]) if len(price_returns) >= 20 else "unknown"
            coherence_pattern = self._identify_metric_pattern(coherence_values[-20:]) if len(coherence_values) >= 20 else "unknown"
            
            # Calculate temporal asymmetry (difference between forward and backward predictability)
            temporal_asymmetry = self._calculate_temporal_asymmetry(price_returns, coherence_values)
            
            anomalies[scale] = {
                'z_coherence': recent_z_coherence,
                'z_entanglement': recent_z_entanglement,
                'z_phase': recent_z_phase,
                'price_pattern': price_pattern,
                'coherence_pattern': coherence_pattern,
                'temporal_asymmetry': temporal_asymmetry,
                'is_anomalous': abs(recent_z_coherence) > 1.5 or abs(recent_z_entanglement) > 1.5 or abs(recent_z_phase) > 1.5,
                'anomaly_score': max(abs(recent_z_coherence), abs(recent_z_entanglement), abs(recent_z_phase))
            }
        
        # Find maximum anomaly
        max_scale = max(anomalies.keys(), key=lambda k: anomalies[k]['anomaly_score']) if anomalies else 0
        max_anomaly = anomalies.get(max_scale, {'anomaly_score': 0})
        
        return {
            'detected': any(a['is_anomalous'] for a in anomalies.values()),
            'max_scale': max_scale,
            'max_score': max_anomaly['anomaly_score'],
            'detail': anomalies
        }
    
    def _identify_price_pattern(self, values: List[float]) -> str:
        """
        Identifies price pattern from a sequence of values.
        
        Args:
            values: List of price values
            
        Returns:
            String describing the pattern
        """
        if not values or len(values) < 3:
            return "unknown"
        
        # Calculate differences
        diffs = np.diff(values)
        
        # Check for trend
        if np.all(diffs > 0):
            return "strong_uptrend"
        elif np.all(diffs < 0):
            return "strong_downtrend"
        elif np.mean(diffs) > 0:
            return "weak_uptrend"
        elif np.mean(diffs) < 0:
            return "weak_downtrend"
        
        # Check for oscillations
        sign_changes = np.sum(np.diff(np.signbit(diffs)))
        if sign_changes > len(diffs) * 0.6:
            return "oscillating"
        
        # Check for convergence/divergence
        abs_diffs = np.abs(diffs)
        if abs_diffs[0] > abs_diffs[-1] * 2:
            return "converging"
        elif abs_diffs[-1] > abs_diffs[0] * 2:
            return "diverging"
        
        return "complex"
    
    def _identify_metric_pattern(self, values: List[float]) -> str:
        """
        Identifies pattern in quantum metrics.
        
        Args:
            values: List of metric values
            
        Returns:
            String describing the pattern
        """
        # Delegate to price pattern detection for now
        return self._identify_price_pattern(values)
    
    def _calculate_temporal_asymmetry(self, price_returns: List[float], coherence_values: List[float]) -> float:
        """
        Calculates temporal asymmetry between forward and backward predictions.
        
        Args:
            price_returns: List of price returns
            coherence_values: List of coherence values
            
        Returns:
            Temporal asymmetry score (-1 to 1)
        """
        if len(price_returns) < 5 or len(coherence_values) < 5:
            return 0.0
        
        # Forward prediction (coherence -> future returns)
        forward_corr = self._delayed_correlation(coherence_values[:-1], price_returns[1:])
        
        # Backward "prediction" (returns -> past coherence)
        backward_corr = self._delayed_correlation(price_returns[:-1], coherence_values[1:])
        
        # Calculate asymmetry
        if abs(forward_corr) < 0.01 and abs(backward_corr) < 0.01:
            return 0.0
            
        return (forward_corr - backward_corr) / max(abs(forward_corr), abs(backward_corr))
    
    def _delayed_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculates correlation between two time series.
        
        Args:
            x: First time series
            y: Second time series
            
        Returns:
            Correlation coefficient
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
    
    def _test_causality(self) -> Dict[str, Any]:
        """
        Tests for Granger causality between quantum metrics and price.
        
        Returns:
            Dictionary with causality test results
        """
        causality_results = {}
        
        for scale in self.time_scales:
            if len(self.metrics_by_scale[scale]) < self.causality_window:
                continue
                
            scale_metrics = self.metrics_by_scale[scale][-self.causality_window:]
            
            # Extract time series
            prices = [m.get('price_return', 0) for m in scale_metrics]
            energies = [m.get('avg_energy', 0) for m in scale_metrics]
            coherences = [m.get('avg_coherence', 0) for m in scale_metrics]
            
            # Calculate future-to-present correlations (adjusted time lags)
            fp_corrs = []
            pp_corrs = []
            
            # Test different lags
            for lag in range(1, min(5, len(prices) // 3)):
                # Future-to-present: future prices with current quantum metrics
                fp_corr = self._lagged_correlation(prices[lag:], coherences[:-lag])
                fp_corrs.append(fp_corr)
                
                # Present-to-present: current prices with current quantum metrics
                pp_corr = self._lagged_correlation(prices[:-lag], coherences[:-lag])
                pp_corrs.append(pp_corr)
            
            # Calculate retrocausality score
            avg_fp_corr = np.mean(fp_corrs) if fp_corrs else 0
            avg_pp_corr = np.mean(pp_corrs) if pp_corrs else 0
            
            retrocausal_score = avg_fp_corr - avg_pp_corr
            
            causality_results[scale] = {
                'future_present_corr': avg_fp_corr,
                'present_present_corr': avg_pp_corr,
                'retrocausal_score': retrocausal_score,
                'is_significant': abs(retrocausal_score) > 0.15
            }
        
        # Find maximum retrocausal effect
        max_scale = max(causality_results.keys(), 
                       key=lambda k: abs(causality_results[k]['retrocausal_score'])) if causality_results else 0
        
        max_result = causality_results.get(max_scale, {'retrocausal_score': 0})
        
        return {
            'detected': any(r['is_significant'] for r in causality_results.values()),
            'max_scale': max_scale,
            'max_score': max_result['retrocausal_score'],
            'detail': causality_results
        }
    
    def _lagged_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculates correlation between two time series.
        
        Args:
            x: First time series
            y: Second time series
            
        Returns:
            Correlation coefficient
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
    
    def _analyze_synchronicity(self) -> Dict[str, Any]:
        """
        Analyzes synchronicity patterns across time scales.
        
        Returns:
            Dictionary with synchronicity analysis
        """
        if len(self.time_scales) < 2:
            return {'detected': False}
            
        sync_scores = []
        scale_pairs = []
        
        # Compare patterns across time scales
        for i, scale1 in enumerate(self.time_scales[:-1]):
            for scale2 in self.time_scales[i+1:]:
                if (len(self.metrics_by_scale[scale1]) < self.lookback_window or 
                    len(self.metrics_by_scale[scale2]) < self.lookback_window):
                    continue
                    
                # Get metrics for both scales
                metrics1 = self.metrics_by_scale[scale1][-self.lookback_window:]
                metrics2 = self.metrics_by_scale[scale2][-self.lookback_window:]
                
                # Get coherence and entanglement
                coherence1 = [m.get('avg_coherence', 0) for m in metrics1]
                coherence2 = [m.get('avg_coherence', 0) for m in metrics2]
                
                entanglement1 = [m.get('avg_entanglement', 0) for m in metrics1]
                entanglement2 = [m.get('avg_entanglement', 0) for m in metrics2]
                
                # Normalize to same length if needed (using nearest interpolation)
                min_len = min(len(coherence1), len(coherence2))
                if min_len < 5:
                    continue
                
                # Calculate synchronicity score
                sync_score_coherence = self._calculate_synchronicity(coherence1[-min_len:], coherence2[-min_len:])
                sync_score_entanglement = self._calculate_synchronicity(entanglement1[-min_len:], entanglement2[-min_len:])
                
                sync_score = (sync_score_coherence + sync_score_entanglement) / 2
                
                sync_scores.append(sync_score)
                scale_pairs.append((scale1, scale2))
        
        # Find maximum synchronicity
        if sync_scores:
            max_index = np.argmax(np.abs(sync_scores))
            max_score = sync_scores[max_index]
            max_pair = scale_pairs[max_index]
            
            # Detect if significant
            is_significant = abs(max_score) > 0.5
            
            # Store pattern if significant
            if is_significant and len(self.sync_patterns) < 100:
                self.sync_patterns.append({
                    'timestamp': datetime.now(),
                    'scales': max_pair,
                    'score': max_score
                })
            
            return {
                'detected': is_significant,
                'max_score': max_score,
                'scales': max_pair,
                'direction': np.sign(max_score) if is_significant else 0
            }
        
        return {'detected': False}
    
    def _calculate_synchronicity(self, x: List[float], y: List[float]) -> float:
        """
        Calculates synchronicity between two time series.
        
        Args:
            x: First time series
            y: Second time series
            
        Returns:
            Synchronicity score (-1 to 1)
        """
        if len(x) != len(y) or len(x) < 5:
            return 0.0
        
        # Extract trends
        x_trend = np.gradient(x)
        y_trend = np.gradient(y)
        
        # Calculate dot product of normalized trends
        x_norm = np.linalg.norm(x_trend)
        y_norm = np.linalg.norm(y_trend)
        
        if x_norm < 1e-6 or y_norm < 1e-6:
            return 0.0
            
        dot_product = np.sum(x_trend * y_trend) / (x_norm * y_norm)
        
        return dot_product
    
    def _detect_phi_recursion(self) -> Dict[str, Any]:
        """
        Detects phi-recursive patterns in the data.
        
        Returns:
            Dictionary with phi recursion analysis
        """
        phi_results = {}
        phi = 1.618033988749895  # Golden ratio
        
        for scale in self.time_scales:
            if len(self.metrics_by_scale[scale]) < self.lookback_window:
                continue
                
            scale_metrics = self.metrics_by_scale[scale][-self.lookback_window:]
            
            # Get coherence values
            coherence = [m.get('avg_coherence', 0) for m in scale_metrics]
            energy = [m.get('avg_energy', 0) for m in scale_metrics]
            returns = [m.get('price_return', 0) for m in scale_metrics]
            
            # Find local extrema
            peaks, _ = signal.find_peaks(coherence)
            troughs, _ = signal.find_peaks([-x for x in coherence])
            
            extrema = sorted(list(peaks) + list(troughs))
            
            # Check for phi-recursive pattern
            phi_ratios = []
            
            for i in range(2, len(extrema)):
                interval1 = extrema[i] - extrema[i-1]
                interval2 = extrema[i-1] - extrema[i-2]
                
                if interval2 == 0:
                    continue
                    
                ratio = interval1 / interval2
                phi_diff = abs(ratio - phi)
                
                if phi_diff < 0.2:  # Within 20% of phi
                    phi_ratios.append(ratio)
            
            phi_score = len(phi_ratios) / max(1, len(extrema) - 2)
            
            # Detect price-field resonance
            resonance_score = self._calculate_resonance(returns, energy)
            
            phi_results[scale] = {
                'phi_score': phi_score,
                'phi_ratios': phi_ratios,
                'extrema_count': len(extrema),
                'resonance_score': resonance_score,
                'is_significant': phi_score > 0.3 or resonance_score > 0.5
            }
        
        # Find maximum phi recursion
        max_scale = max(phi_results.keys(), 
                       key=lambda k: max(phi_results[k]['phi_score'], phi_results[k]['resonance_score'])) if phi_results else 0
        
        max_result = phi_results.get(max_scale, {'phi_score': 0, 'resonance_score': 0})
        
        return {
            'detected': any(r['is_significant'] for r in phi_results.values()),
            'max_scale': max_scale,
            'max_phi_score': max_result['phi_score'],
            'max_resonance': max_result['resonance_score'],
            'detail': phi_results
        }
    
    def _calculate_resonance(self, returns: List[float], energy: List[float]) -> float:
        """
        Calculates resonance between price returns and quantum energy.
        
        Args:
            returns: List of price returns
            energy: List of quantum energy values
            
        Returns:
            Resonance score (0 to 1)
        """
        if len(returns) != len(energy) or len(returns) < 5:
            return 0.0
        
        try:
            # Calculate phase using Hilbert transform
            returns_analytic = signal.hilbert(returns)
            energy_analytic = signal.hilbert(energy)
            
            returns_phase = np.unwrap(np.angle(returns_analytic))
            energy_phase = np.unwrap(np.angle(energy_analytic))
            
            # Calculate phase difference
            phase_diff = np.abs(returns_phase - energy_phase) % (2 * np.pi)
            
            # Normalize to [0, 1]
            phase_sync = 1 - np.mean(phase_diff) / np.pi
            
            return max(0, min(1, phase_sync))
        except:
            return 0.0
    
    def _combine_signals(self, 
                       anomalies: Dict[str, Any], 
                       causality: Dict[str, Any],
                       synchronicity: Dict[str, Any],
                       phi_recursion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines various signals into a single retrocausal trading signal.
        
        Args:
            anomalies: Temporal anomaly results
            causality: Granger causality test results
            synchronicity: Synchronicity analysis results
            phi_recursion: Phi recursion analysis results
            
        Returns:
            Combined trading signal
        """
        # Check if any signal is detected
        signal_detected = (
            anomalies.get('detected', False) or
            causality.get('detected', False) or
            synchronicity.get('detected', False) or
            phi_recursion.get('detected', False)
        )
        
        if not signal_detected:
            return {
                'signal_detected': False,
                'confidence': 0.0,
                'direction': 0,
                'time_scale': 0,
                'explanation': "No retrocausal signal detected"
            }
        
        # Calculate confidence from component signals
        confidence_components = []
        
        if anomalies.get('detected', False):
            confidence_components.append(min(1.0, anomalies['max_score'] / 2.5))
            
        if causality.get('detected', False):
            confidence_components.append(min(1.0, abs(causality['max_score']) / 0.3))
            
        if synchronicity.get('detected', False):
            confidence_components.append(min(1.0, abs(synchronicity['max_score'])))
            
        if phi_recursion.get('detected', False):
            confidence_components.append(min(1.0, max(
                phi_recursion['max_phi_score'], 
                phi_recursion['max_resonance']
            )))
        
        # Calculate weighted confidence
        if confidence_components:
            # Higher weight to causality and synchronicity
            weights = []
            
            if anomalies.get('detected', False):
                weights.append(0.2)
                
            if causality.get('detected', False):
                weights.append(0.4)
                
            if synchronicity.get('detected', False):
                weights.append(0.3)
                
            if phi_recursion.get('detected', False):
                weights.append(0.1)
                
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            confidence = np.sum(np.array(confidence_components) * weights)
        else:
            confidence = 0.0
        
        # Determine signal direction
        direction_components = []
        
        if causality.get('detected', False):
            direction_components.append(np.sign(causality['max_score']))
            
        if synchronicity.get('detected', False):
            direction_components.append(synchronicity.get('direction', 0))
        
        # Determine time scale
        time_scales = []
        
        if anomalies.get('detected', False) and anomalies.get('max_scale', 0) > 0:
            time_scales.append(anomalies['max_scale'])
            
        if causality.get('detected', False) and causality.get('max_scale', 0) > 0:
            time_scales.append(causality['max_scale'])
        
        if time_scales:
            # Weight by confidence
            time_scale = max(time_scales, key=lambda t: t)
        else:
            time_scale = self.time_scales[0] if self.time_scales else 0
        
        # Determine overall direction
        if direction_components:
            direction = np.sign(sum(direction_components))
        else:
            direction = 0
        
        # Generate explanation
        explanation_parts = []
        
        if anomalies.get('detected', False):
            explanation_parts.append(f"temporal anomaly (score: {anomalies['max_score']:.2f})")
            
        if causality.get('detected', False):
            explanation_parts.append(f"retrocausality (score: {causality['max_score']:.2f})")
            
        if synchronicity.get('detected', False):
            explanation_parts.append(f"cross-scale synchronicity (score: {synchronicity['max_score']:.2f})")
            
        if phi_recursion.get('detected', False):
            explanation_parts.append(f"phi recursion (score: {phi_recursion['max_phi_score']:.2f})")
        
        explanation = f"Retrocausal signal from {', '.join(explanation_parts)}"
        
        return {
            'signal_detected': True,
            'confidence': confidence,
            'direction': direction,
            'time_scale': time_scale,
            'explanation': explanation,
            'components': {
                'anomalies': anomalies,
                'causality': causality,
                'synchronicity': synchronicity,
                'phi_recursion': phi_recursion
            }
        }
    
    def _update_causal_graph(self, signal: Dict[str, Any]) -> None:
        """
        Updates the causal graph based on the latest signal.
        
        Args:
            signal: Current retrocausal signal
        """
        if not signal.get('signal_detected', False):
            return
            
        # Add nodes if needed
        current_time = f"t={datetime.now().strftime('%H:%M:%S')}"
        future_time = f"t+{signal.get('time_scale', 0)}m"
        
        if current_time not in self.causal_graph:
            self.causal_graph.add_node(current_time, type="present", timestamp=datetime.now())
            
        if future_time not in self.causal_graph:
            self.causal_graph.add_node(future_time, type="future", timestamp=datetime.now())
        
        # Add edge from future to present (retrocausal link)
        self.causal_graph.add_edge(
            future_time, 
            current_time, 
            weight=signal.get('confidence', 0),
            direction=signal.get('direction', 0),
            explanation=signal.get('explanation', "")
        )
        
        # Prune old nodes (keep graph manageable)
        if len(self.causal_graph) > 50:
            # Remove oldest nodes
            now = datetime.now()
            old_nodes = [
                n for n, data in self.causal_graph.nodes(data=True)
                if 'timestamp' in data and (now - data['timestamp']).total_seconds() > 3600
            ]
            
            for node in old_nodes:
                self.causal_graph.remove_node(node)
    
    def get_signal(self) -> Dict[str, Any]:
        """
        Gets the current trading signal based on retrocausal analysis.
        
        Returns:
            Trading signal with direction and confidence
        """
        # Perform analysis
        signal = self.analyze()
        
        # Only consider it a valid signal if confidence exceeds threshold
        if signal.get('signal_detected', False) and signal.get('confidence', 0) >= self.confidence_threshold:
            return {
                'signal': True,
                'confidence': signal['confidence'],
                'direction': signal['direction'],
                'time_scale': signal['time_scale'],
                'explanation': signal['explanation']
            }
        
        return {
            'signal': False,
            'confidence': 0.0,
            'direction': 0,
            'time_scale': 0,
            'explanation': "No significant retrocausal signal detected"
        }
    
    def reset(self) -> None:
        """
        Resets the analyzer state.
        """
        self.price_history.clear()
        self.field_metrics_history.clear()
        self.retrocausal_events = []
        
        for scale in self.time_scales:
            self.metrics_by_scale[scale] = []
            
        self.causal_graph = nx.DiGraph()
        self.sync_patterns = []
        
        logger.info("Reset RetrocausalAnalyzer state") 