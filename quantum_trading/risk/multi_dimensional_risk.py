#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Dimensional Risk Manager

Implements advanced risk assessment across multiple dimensions:
- Quantum risk (coherence, entanglement)
- Market risk (volatility, liquidity)
- Temporal risk (time decay, seasonality)
- Cosmic risk (quantum field stability)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    is_acceptable: bool
    total_risk: float
    dimension_risks: Dict[str, float]
    reason: str = ""

class MultiDimensionalRiskManager:
    """
    Gerencia risco em múltiplas dimensões quânticas
    """
    
    def __init__(self, dimensions: int = 8, max_risk: float = 0.6):
        self.dimensions = dimensions
        self.max_risk = max_risk
        self.history = []
        self.max_history = 1000
        
        # Pesos para diferentes tipos de risco
        self.risk_weights = {
            'quantum': 0.3,
            'market': 0.3,
            'temporal': 0.2,
            'correlation': 0.2
        }
        
    async def evaluate(self, 
                      current_state: np.ndarray,
                      state_metrics: Dict[str, float],
                      retro_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Avalia risco multidimensional
        
        Args:
            current_state: Estado quântico atual
            state_metrics: Métricas do estado
            retro_metrics: Métricas retrocausais
            
        Returns:
            Dict com métricas de risco
        """
        try:
            # Calcula componentes de risco
            quantum_risk = self._calculate_quantum_risk(current_state, state_metrics)
            market_risk = self._calculate_market_risk(state_metrics)
            temporal_risk = self._calculate_temporal_risk(retro_metrics)
            correlation_risk = self._calculate_correlation_risk(state_metrics, retro_metrics)
            
            # Calcula risco total ponderado
            total_risk = (
                self.risk_weights['quantum'] * quantum_risk +
                self.risk_weights['market'] * market_risk +
                self.risk_weights['temporal'] * temporal_risk +
                self.risk_weights['correlation'] * correlation_risk
            )
            
            # Compõe métricas
            metrics = {
                'quantum_risk': quantum_risk,
                'market_risk': market_risk,
                'temporal_risk': temporal_risk,
                'correlation_risk': correlation_risk,
                'total_risk': total_risk,
                'risk_status': 'high' if total_risk > self.max_risk else 'normal'
            }
            
            # Atualiza histórico
            self.history.append(metrics)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating risk: {str(e)}")
            return self._empty_metrics()
            
    def _empty_metrics(self) -> Dict[str, float]:
        """Retorna métricas vazias"""
        return {
            'quantum_risk': 0.0,
            'market_risk': 0.0,
            'temporal_risk': 0.0,
            'correlation_risk': 0.0,
            'total_risk': 0.0,
            'risk_status': 'normal'
        }
        
    def _calculate_quantum_risk(self, state: np.ndarray, metrics: Dict[str, float]) -> float:
        """Calcula risco quântico"""
        try:
            # Usa coerência e entropia
            coherence = metrics.get('coherence', 0.0)
            entropy = metrics.get('entropy', 0.0)
            
            # Risco aumenta com entropia e diminui com coerência
            quantum_risk = (1.0 - coherence) * entropy
            
            # Normaliza para [0,1]
            return float(np.clip(quantum_risk, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating quantum risk: {str(e)}")
            return 0.0
            
    def _calculate_market_risk(self, metrics: Dict[str, float]) -> float:
        """Calcula risco de mercado"""
        try:
            # Usa complexidade e energia
            complexity = metrics.get('complexity', 0.0)
            energy = metrics.get('energy', 0.0)
            
            # Risco aumenta com complexidade e energia
            market_risk = (complexity * energy) / (self.dimensions ** 2)
            
            # Normaliza
            return float(np.clip(market_risk, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {str(e)}")
            return 0.0
            
    def _calculate_temporal_risk(self, metrics: Dict[str, float]) -> float:
        """Calcula risco temporal"""
        try:
            # Usa métricas retrocausais
            resonance = metrics.get('resonance', 0.0)
            temporal_coherence = metrics.get('temporal_coherence', 0.0)
            retrocausal_strength = metrics.get('retrocausal_strength', 0.0)
            
            # Risco diminui com ressonância e coerência temporal
            temporal_risk = 1.0 - (resonance * temporal_coherence * retrocausal_strength)
            
            # Normaliza
            return float(np.clip(temporal_risk, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating temporal risk: {str(e)}")
            return 0.0
            
    def _calculate_correlation_risk(self, state_metrics: Dict[str, float], retro_metrics: Dict[str, float]) -> float:
        """Calcula risco de correlação"""
        try:
            # Correlação entre métricas quânticas e retrocausais
            phase_alignment = state_metrics.get('phase_alignment', 0.0)
            causal_symmetry = retro_metrics.get('causal_symmetry', 0.0)
            temporal_entanglement = retro_metrics.get('temporal_entanglement', 0.0)
            
            # Risco diminui com alinhamento e simetria
            correlation_risk = 1.0 - (
                phase_alignment * 
                causal_symmetry * 
                np.exp(-temporal_entanglement)
            )
            
            # Normaliza
            return float(np.clip(correlation_risk, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.0
            
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Atualiza pesos dos componentes de risco
        
        Args:
            new_weights: Novos pesos para cada tipo de risco
        """
        try:
            # Valida soma dos pesos
            total = sum(new_weights.values())
            if not np.isclose(total, 1.0):
                logger.warning(f"Risk weights sum to {total}, normalizing")
                new_weights = {k: v/total for k, v in new_weights.items()}
                
            self.risk_weights.update(new_weights)
            logger.info(f"Risk weights updated: {self.risk_weights}")
            
        except Exception as e:
            logger.error(f"Error updating risk weights: {str(e)}")
            
    def get_risk_profile(self) -> Dict[str, Any]:
        """
        Retorna perfil de risco atual
        
        Returns:
            Dict com perfil de risco
        """
        try:
            if not self.history:
                return self._empty_metrics()
                
            # Calcula estatísticas do histórico
            metrics = {}
            for key in self.history[0].keys():
                if key != 'risk_status':
                    values = [h[key] for h in self.history]
                    metrics[key] = {
                        'current': values[-1],
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting risk profile: {str(e)}")
            return self._empty_metrics()

    def assess_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assesses risk across all dimensions."""
        risks = {}
        
        if 'quantum' in self.risk_dimensions:
            risks['quantum'] = self._assess_quantum_risk(data)
            
        if 'market' in self.risk_dimensions:
            risks['market'] = self._assess_market_risk(data)
            
        if 'temporal' in self.risk_dimensions:
            risks['temporal'] = self._assess_temporal_risk(data)
            
        if 'cosmic' in self.risk_dimensions:
            risks['cosmic'] = self._assess_cosmic_risk(data)
        
        # Calculate total risk
        total_risk = self._calculate_total_risk(risks)
        
        # Update risk history
        self.risk_history.append({
            'timestamp': pd.Timestamp.now(),
            'risks': risks,
            'total_risk': total_risk
        })
        
        return {
            'dimension_risks': risks,
            'total_risk': total_risk,
            'risk_breakdown': self._generate_risk_breakdown(risks)
        }
    
    def assess_trade_risk(self, market_data: pd.DataFrame,
                        quantum_signal: Dict[str, Any],
                        current_position: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """Assesses risk for a specific trade."""
        # Collect all relevant data
        data = {
            'market_data': market_data,
            'quantum_signal': quantum_signal,
            'current_position': current_position
        }
        
        # Get risk assessment
        risk_assessment = self.assess_risk(data)
        
        # Check if risk is acceptable
        is_acceptable = True
        reason = ""
        
        # Check individual dimension risks
        for dim, risk in risk_assessment['dimension_risks'].items():
            if risk > self.max_risk_per_dimension:
                is_acceptable = False
                reason = f"Risk in {dim} dimension too high: {risk:.2f}"
                break
        
        # Check total risk
        if is_acceptable and risk_assessment['total_risk'] > self.max_risk_per_dimension * 2:
            is_acceptable = False
            reason = f"Total risk too high: {risk_assessment['total_risk']:.2f}"
        
        return RiskAssessment(
            is_acceptable=is_acceptable,
            total_risk=risk_assessment['total_risk'],
            dimension_risks=risk_assessment['dimension_risks'],
            reason=reason
        )
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initializes risk dimension weights."""
        weights = {
            'quantum': 0.3,
            'market': 0.3,
            'temporal': 0.2,
            'cosmic': 0.2
        }
        return {dim: weights.get(dim, 0.25) for dim in self.risk_dimensions}
    
    def _assess_quantum_risk(self, data: Dict[str, Any]) -> float:
        """Assesses risk in the quantum dimension."""
        quantum_signal = data.get('quantum_signal', {})
        
        # Extract quantum metrics
        confidence = quantum_signal.get('confidence', 0.5)
        strength = quantum_signal.get('strength', 0.5)
        
        # Calculate quantum risk factors
        coherence_risk = 1.0 - confidence
        strength_risk = 1.0 - strength
        
        # Combine risk factors
        quantum_risk = (coherence_risk * 0.6 + strength_risk * 0.4)
        return float(quantum_risk)
    
    def _assess_market_risk(self, data: Dict[str, Any]) -> float:
        """Assesses risk in the market dimension."""
        market_data = data.get('market_data')
        if market_data is None:
            return 1.0
        
        # Calculate volatility risk
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std()
        volatility_risk = min(1.0, volatility * 10)  # Scale to [0,1]
        
        # Calculate liquidity risk
        volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].mean()
        liquidity_risk = 1.0 - min(1.0, volume / avg_volume)
        
        # Combine risk factors
        market_risk = (volatility_risk * 0.7 + liquidity_risk * 0.3)
        return float(market_risk)
    
    def _assess_temporal_risk(self, data: Dict[str, Any]) -> float:
        """Assesses risk in the temporal dimension."""
        market_data = data.get('market_data')
        if market_data is None:
            return 1.0
        
        # Calculate time decay risk
        current_time = pd.Timestamp.now()
        market_time = pd.Timestamp(market_data.index[-1])
        time_diff = (current_time - market_time).total_seconds()
        time_decay_risk = min(1.0, time_diff / 3600)  # Scale to 1 hour
        
        # Calculate seasonality risk
        hour = current_time.hour
        # Higher risk during market opens/closes
        seasonality_risk = 1.0 - np.cos(hour * np.pi / 12) * 0.5
        
        # Combine risk factors
        temporal_risk = (time_decay_risk * 0.6 + seasonality_risk * 0.4)
        return float(temporal_risk)
    
    def _assess_cosmic_risk(self, data: Dict[str, Any]) -> float:
        """Assesses risk in the cosmic dimension."""
        quantum_signal = data.get('quantum_signal', {})
        
        # Extract cosmic metrics
        field_stability = quantum_signal.get('field_stability', 0.5)
        resonance = quantum_signal.get('resonance', 0.5)
        
        # Calculate cosmic risk factors
        stability_risk = 1.0 - field_stability
        resonance_risk = 1.0 - resonance
        
        # Add quantum uncertainty
        uncertainty = np.random.normal(0, 0.1)
        
        # Combine risk factors
        cosmic_risk = (stability_risk * 0.5 + resonance_risk * 0.5 + uncertainty)
        return float(max(0.0, min(1.0, cosmic_risk)))
    
    def _calculate_total_risk(self, risks: Dict[str, float]) -> float:
        """Calculates total risk across all dimensions."""
        weighted_risks = []
        for dim, risk in risks.items():
            weight = self.dimension_weights.get(dim, 0.25)
            weighted_risks.append(risk * weight)
        
        total_risk = sum(weighted_risks)
        return float(total_risk)
    
    def _generate_risk_breakdown(self, risks: Dict[str, float]) -> Dict[str, Any]:
        """Generates detailed risk breakdown."""
        breakdown = {
            'dimension_contributions': {
                dim: risk * self.dimension_weights[dim]
                for dim, risk in risks.items()
            },
            'risk_levels': {
                dim: self._categorize_risk(risk)
                for dim, risk in risks.items()
            }
        }
        return breakdown
    
    def _categorize_risk(self, risk: float) -> str:
        """Categorizes risk level."""
        if risk < 0.2:
            return 'LOW'
        elif risk < 0.4:
            return 'MODERATE'
        elif risk < 0.6:
            return 'HIGH'
        else:
            return 'EXTREME' 