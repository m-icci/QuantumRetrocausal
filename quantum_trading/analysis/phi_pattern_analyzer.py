#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi Pattern Analyzer

Detects and analyzes phi-based patterns in market data, including:
- Fibonacci sequences
- Golden spiral formations
- Quantum harmonic resonances
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
import logging
from datetime import datetime

# Importação corrigida
# from quantum.core.QUALIA import PHI
# Definição local da constante
PHI = (1 + np.sqrt(5)) / 2

# Configuração de logger
logger = logging.getLogger(__name__)

class PhiPatternAnalyzer:
    def __init__(self, patterns: List[str] = None,
                min_pattern_strength: float = 0.6):
        self.patterns = patterns or ['fibonacci', 'golden_spiral', 'quantum_harmonics']
        self.min_pattern_strength = min_pattern_strength
        self.phi = PHI
        self.phi_seq = self._generate_phi_sequence(10)
        self.history = []
        
    def find_patterns(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes market data for phi-based patterns."""
        patterns = {}
        
        if 'fibonacci' in self.patterns:
            patterns['fibonacci'] = self._analyze_fibonacci(market_data)
            
        if 'golden_spiral' in self.patterns:
            patterns['golden_spiral'] = self._analyze_golden_spiral(market_data)
            
        if 'quantum_harmonics' in self.patterns:
            patterns['quantum_harmonics'] = self._analyze_quantum_harmonics(market_data)
        
        # Calculate overall pattern strength
        strengths = [p['strength'] for p in patterns.values()]
        overall_strength = np.mean(strengths) if strengths else 0.0
        
        # Calculate resonance
        resonance = self._calculate_resonance(patterns)
        
        return {
            'patterns': patterns,
            'strength': overall_strength,
            'resonance': resonance,
            'confidence': overall_strength * resonance
        }
    
    def get_signal(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Generates trading signal based on phi pattern analysis."""
        analysis = self.find_patterns(market_data)
        
        # Calculate signal confidence
        confidence = analysis['confidence']
        
        # Determine direction based on pattern analysis
        direction = self._calculate_direction(analysis)
        
        return {
            'confidence': confidence,
            'direction': direction,
            'strength': analysis['strength']
        }
    
    def _generate_phi_sequence(self, n: int) -> np.ndarray:
        """Generates a sequence of phi-based numbers."""
        sequence = [1, self.phi]
        for i in range(2, n):
            sequence.append(sequence[-1] * self.phi)
        return np.array(sequence)
    
    def _analyze_fibonacci(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes price movements for Fibonacci patterns."""
        prices = market_data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Find local extrema
        peaks, _ = signal.find_peaks(returns)
        troughs, _ = signal.find_peaks(-returns)
        
        # Calculate ratios between consecutive extrema
        ratios = []
        for i in range(1, len(peaks)):
            ratio = abs(returns[peaks[i]] / returns[peaks[i-1]])
            ratios.append(ratio)
        
        # Compare with Fibonacci ratios
        fib_ratios = [0.618, 1.618, 2.618, 4.236]
        pattern_strength = self._calculate_ratio_similarity(ratios, fib_ratios)
        
        return {
            'strength': pattern_strength,
            'ratios': ratios,
            'peaks': peaks.tolist(),
            'troughs': troughs.tolist()
        }
    
    def _analyze_golden_spiral(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes price movements for golden spiral patterns."""
        prices = market_data['close'].values
        
        # Calculate price changes at different scales
        changes = []
        for scale in [1, 2, 3, 5, 8, 13]:
            if len(prices) > scale:
                change = (prices[scale:] - prices[:-scale]) / prices[:-scale]
                changes.append(np.mean(np.abs(change)))
        
        # Compare changes with phi sequence
        if changes:
            ratios = np.diff(changes) / changes[:-1]
            pattern_strength = self._calculate_ratio_similarity(
                ratios,
                self.phi_seq[:len(ratios)]
            )
        else:
            pattern_strength = 0.0
        
        return {
            'strength': pattern_strength,
            'changes': changes,
            'spiral_ratios': ratios.tolist() if 'ratios' in locals() else []
        }
    
    def _analyze_quantum_harmonics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes quantum harmonic resonances in price movements."""
        prices = market_data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate frequency components
        freqs = np.fft.fft(returns)
        power = np.abs(freqs) ** 2
        
        # Find harmonic frequencies
        harmonics = []
        for i in range(1, min(5, len(power))):
            if power[i] > power[i-1] and power[i] > power[i+1]:
                harmonics.append({
                    'frequency': i,
                    'power': float(power[i])
                })
        
        # Calculate harmonic ratios
        ratios = []
        for i in range(1, len(harmonics)):
            ratio = harmonics[i]['frequency'] / harmonics[i-1]['frequency']
            ratios.append(ratio)
        
        # Compare with phi-based harmonics
        pattern_strength = self._calculate_ratio_similarity(ratios, self.phi_seq[:len(ratios)])
        
        return {
            'strength': pattern_strength,
            'harmonics': harmonics,
            'ratios': ratios
        }
    
    def _calculate_ratio_similarity(self, observed: List[float],
                                 expected: List[float]) -> float:
        """Calculates similarity between observed and expected ratios."""
        if not observed or not expected:
            return 0.0
        
        # Normalize to same scale
        obs_norm = np.array(observed) / np.mean(observed)
        exp_norm = np.array(expected) / np.mean(expected)
        
        # Calculate similarity using dynamic time warping
        min_len = min(len(obs_norm), len(exp_norm))
        distance = np.sum((obs_norm[:min_len] - exp_norm[:min_len]) ** 2)
        
        # Convert distance to similarity score
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)
    
    def _calculate_resonance(self, patterns: Dict[str, Dict[str, Any]]) -> float:
        """Calculates overall pattern resonance."""
        resonances = []
        
        if 'fibonacci' in patterns:
            fib_strength = patterns['fibonacci']['strength']
            resonances.append(fib_strength)
        
        if 'golden_spiral' in patterns:
            spiral_strength = patterns['golden_spiral']['strength']
            resonances.append(spiral_strength)
        
        if 'quantum_harmonics' in patterns:
            harmonic_strength = patterns['quantum_harmonics']['strength']
            resonances.append(harmonic_strength)
        
        # Calculate weighted resonance
        weights = [0.4, 0.3, 0.3]  # Adjust weights based on pattern importance
        resonance = np.average(resonances, weights=weights[:len(resonances)])
        return float(resonance)
    
    def _calculate_direction(self, analysis: Dict[str, Any]) -> float:
        """Determines trading direction based on pattern analysis."""
        patterns = analysis['patterns']
        
        # Collect directional signals from each pattern
        signals = []
        
        if 'fibonacci' in patterns:
            fib = patterns['fibonacci']
            if fib['peaks'] and fib['troughs']:
                # Compare latest peak/trough
                last_peak = max(fib['peaks'])
                last_trough = max(fib['troughs'])
                signals.append(1.0 if last_peak > last_trough else -1.0)
        
        if 'golden_spiral' in patterns:
            spiral = patterns['golden_spiral']
            if 'spiral_ratios' in spiral and spiral['spiral_ratios']:
                # Use last spiral ratio
                last_ratio = spiral['spiral_ratios'][-1]
                signals.append(1.0 if last_ratio > self.phi else -1.0)
        
        if 'quantum_harmonics' in patterns:
            harmonics = patterns['quantum_harmonics']
            if harmonics['harmonics']:
                # Use dominant harmonic
                dominant = max(harmonics['harmonics'], key=lambda x: x['power'])
                signals.append(1.0 if dominant['frequency'] > 1 else -1.0)
        
        # Combine signals
        if signals:
            direction = np.sign(np.mean(signals))
        else:
            direction = 0.0
        
        return float(direction) 