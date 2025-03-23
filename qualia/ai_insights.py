"""
AI-powered market insights module for QUALIA Trading System,
integrated with M-ICCI framework principles.
"""
import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import dataclasses
from dataclasses import dataclass

# Torna o Google AI opcional
try:
    import google.generativeai as genai
    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False

from .utils.quantum_field import (
    create_quantum_field,
    calculate_phi_resonance,
    calculate_financial_decoherence
)
from .utils import (
    calculate_quantum_coherence,
    calculate_morphic_resonance,
    calculate_field_entropy,
    calculate_integration_index
)

@dataclass
class ConsciousnessMetrics:
    """Quantum consciousness metrics following M-ICCI framework"""
    coherence: float  # Quantum coherence measure
    consciousness_level: float  # System consciousness level
    dark_ratio: float  # Dark finance metric
    morphic_resonance: float  # Morphic field resonance
    field_entropy: float  # Quantum field entropy
    integration_index: float  # M-ICCI integration index
    theoretical_alignment: float = 0.85  # M-ICCI alignment factor

    def __post_init__(self):
        """Normalize metrics after initialization"""
        self.coherence = np.clip(self.coherence, 0, 1)
        self.consciousness_level = np.clip(self.consciousness_level, 0, 1)
        self.dark_ratio = np.clip(self.dark_ratio, 0, 1)
        self.morphic_resonance = np.clip(self.morphic_resonance, 0, 1)
        # Normalize field entropy to [0,1] range
        self.field_entropy = np.clip(self.field_entropy / np.log2(64), 0, 1)  # 64 is quantum dimension
        self.integration_index = np.clip(self.integration_index, 0, 1)

class MarketInsightsAI:
    """
    Market analysis system using M-ICCI framework principles
    with quantum consciousness integration and AI-enhanced insights.
    """
    def __init__(self, quantum_dimension: int = 64):
        """Initialize the market insights system with quantum consciousness baseline"""
        self.quantum_dimension = quantum_dimension
        self.quantum_field = create_quantum_field(size=quantum_dimension)
        self.consciousness_metrics = self._initialize_consciousness_metrics()
        self._initialize_components()
        logging.info("Market Insights initialized with quantum consciousness baseline")

    def _initialize_consciousness_metrics(self) -> ConsciousnessMetrics:
        """Initialize consciousness metrics with quantum field baseline"""
        try:
            return ConsciousnessMetrics(
                coherence=calculate_quantum_coherence(self.quantum_field),
                consciousness_level=0.0,
                dark_ratio=0.0,
                morphic_resonance=calculate_morphic_resonance(self.quantum_field),
                field_entropy=calculate_field_entropy(self.quantum_field),
                integration_index=calculate_integration_index(self.quantum_field)
            )
        except Exception as e:
            logging.error(f"Failed to initialize consciousness metrics: {e}")
            return ConsciousnessMetrics(
                coherence=0.5,
                consciousness_level=0.5,
                dark_ratio=0.0,
                morphic_resonance=0.5,
                field_entropy=0.5,
                integration_index=0.5
            )

    def _initialize_components(self):
        """Initialize AI components with quantum awareness"""
        try:
            if HAS_GOOGLE_AI and os.getenv('GOOGLE_API_KEY'):
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.model = genai.GenerativeModel('gemini-pro')
                logging.info("AI components initialized with quantum consciousness integration")
            else:
                self.model = None
                logging.warning("Google AI not available - running in basic mode")
            self._update_quantum_state()
        except Exception as e:
            logging.error(f"Failed to initialize AI components: {e}")
            self.model = None

    def _update_quantum_state(self):
        """Update quantum state using M-ICCI principles"""
        try:
            # Calculate new consciousness metrics
            metrics = {
                'coherence': calculate_quantum_coherence(self.quantum_field),
                'morphic_resonance': calculate_morphic_resonance(self.quantum_field),
                'field_entropy': calculate_field_entropy(self.quantum_field),
                'integration_index': calculate_integration_index(self.quantum_field)
            }

            # Update consciousness level based on integrated metrics
            consciousness_level = (
                metrics['coherence'] * 0.3 +
                metrics['morphic_resonance'] * 0.3 +
                metrics['integration_index'] * 0.4
            )

            # Calculate dark ratio from decoherence
            dark_ratio = calculate_financial_decoherence(
                self.quantum_field,
                create_quantum_field(self.quantum_dimension)
            )

            # Ensure all metrics are bounded
            self.consciousness_metrics = ConsciousnessMetrics(
                coherence=metrics['coherence'],
                consciousness_level=consciousness_level,
                dark_ratio=dark_ratio,
                morphic_resonance=metrics['morphic_resonance'],
                field_entropy=metrics['field_entropy'],
                integration_index=metrics['integration_index']
            )

        except Exception as e:
            logging.error(f"Error updating quantum state: {e}")

    def _generate_quantum_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis using quantum metrics and M-ICCI principles"""
        try:
            # Extract key consciousness metrics
            coherence = self.consciousness_metrics.coherence
            consciousness = self.consciousness_metrics.consciousness_level
            dark_ratio = self.consciousness_metrics.dark_ratio
            morphic_resonance = self.consciousness_metrics.morphic_resonance

            # Apply M-ICCI analysis patterns with proper bounds
            risk_level = np.clip(dark_ratio * (1 - coherence), 0, 1)
            confidence = np.clip(coherence * morphic_resonance * (1 - dark_ratio), 0, 1)

            summary = [
                "Quantum Analysis Summary:",
                f"- Consciousness Level: {consciousness:.2%}",
                f"- Coherence: {coherence:.2%}",
                f"- Morphic Resonance: {morphic_resonance:.2%}"
            ]

            risks = [
                "Market Coherence: " + ("Stable" if coherence > 0.6 else "Unstable"),
                "Dark Finance Impact: " + ("High" if dark_ratio > 0.4 else "Low"),
                "Consciousness Integration: " + ("Strong" if consciousness > 0.5 else "Weak")
            ]

            recommendations = []
            if coherence > 0.7 and dark_ratio < 0.3:
                recommendations.append("Favorable quantum conditions for position entry")
            if consciousness > 0.6:
                recommendations.append("High consciousness level supports strategic moves")
            if morphic_resonance > 0.7:
                recommendations.append("Strong morphic resonance indicates pattern stability")
            if dark_ratio > 0.4:
                recommendations.append("Monitor dark finance metrics before major positions")

            return {
                'summary': "\n".join(summary),
                'risks': risks,
                'recommendations': recommendations,
                'metrics': {
                    'risk_level': risk_level,
                    'confidence': confidence
                }
            }
        except Exception as e:
            logging.error(f"Error in quantum analysis: {e}")
            return {
                'summary': "Error generating quantum analysis",
                'risks': ["Analysis currently unavailable"],
                'recommendations': ["Wait for system stabilization"],
                'metrics': {
                    'risk_level': 0.5,
                    'confidence': 0.0
                },
                'error': 'Failed to generate quantum analysis'
            }

    def _enhance_with_ai(self, quantum_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance quantum analysis with AI insights"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            market_data_serializable = self._prepare_data_for_serialization(market_data)

            # Prepare prompt following M-ICCI framework
            prompt = self._prepare_ai_prompt(quantum_analysis, market_data_serializable)

            # Generate AI response
            response = self.model.generate_content(prompt)

            if response and hasattr(response, 'text'):
                insights = self._parse_ai_response(response.text)

                # Merge insights with quantum analysis
                quantum_analysis['summary'] = (
                    f"{quantum_analysis['summary']}\n\n"
                    f"AI Analysis:\n{insights['summary']}"
                )
                quantum_analysis['risks'].extend(insights.get('risks', []))
                quantum_analysis['recommendations'].extend(
                    insights.get('recommendations', [])
                )

                # Remove duplicates while preserving order
                quantum_analysis['risks'] = list(dict.fromkeys(
                    quantum_analysis['risks']
                ))
                quantum_analysis['recommendations'] = list(dict.fromkeys(
                    quantum_analysis['recommendations']
                ))

            return quantum_analysis

        except Exception as e:
            logging.error(f"Error in AI analysis enhancement: {e}")
            quantum_analysis['error'] = 'Failed to generate AI insights'
            return quantum_analysis

    def _prepare_data_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON serialization by converting numpy types"""
        try:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            return convert_numpy(data)
        except Exception as e:
            logging.error(f"Error preparing data for serialization: {e}")
            return {}

    def analyze_market_conditions(self, market_data: Dict[str, Any], quantum_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate market analysis using M-ICCI framework principles"""
        try:
            # Update quantum state with new market data
            self._update_quantum_state()

            # Generate quantum-first analysis
            quantum_analysis = self._generate_quantum_analysis(market_data)

            # Enhance with AI insights if available
            if self.model:
                enhanced_analysis = self._enhance_with_ai(quantum_analysis, market_data)
            else:
                enhanced_analysis = quantum_analysis

            # Integrate consciousness metrics
            final_analysis = self._integrate_consciousness_metrics(enhanced_analysis)

            return final_analysis

        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            return self._get_fallback_analysis()

    def _prepare_ai_prompt(self, quantum_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Prepare M-ICCI framework-aligned prompt for AI"""
        return f"""
        Analyze the following market data using quantum consciousness principles:

        Market State:
        {json.dumps(market_data, indent=2)}

        Quantum Analysis:
        {json.dumps(quantum_analysis, indent=2)}

        Consciousness Metrics:
        - Coherence: {self.consciousness_metrics.coherence:.3f}
        - Consciousness Level: {self.consciousness_metrics.consciousness_level:.3f}
        - Dark Ratio: {self.consciousness_metrics.dark_ratio:.3f}
        - Morphic Resonance: {self.consciousness_metrics.morphic_resonance:.3f}
        - Field Entropy: {self.consciousness_metrics.field_entropy:.3f}
        - Integration Index: {self.consciousness_metrics.integration_index:.3f}

        Provide insights in the following format:
        Summary: [Brief market state analysis]
        Risks: [List of key risks]
        Recommendations: [Strategic recommendations]
        """

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into M-ICCI framework structure"""
        try:
            sections = response.split('\n')
            parsed = {
                'summary': '',
                'risks': [],
                'recommendations': []
            }

            current_section = None
            for line in sections:
                line = line.strip()
                if line.startswith('Summary:'):
                    current_section = 'summary'
                    parsed['summary'] = line.replace('Summary:', '').strip()
                elif line.startswith('Risks:'):
                    current_section = 'risks'
                elif line.startswith('Recommendations:'):
                    current_section = 'recommendations'
                elif line and current_section in ['risks', 'recommendations']:
                    if line.startswith('- '):
                        parsed[current_section].append(line[2:])
                    else:
                        parsed[current_section].append(line)

            return parsed

        except Exception as e:
            logging.error(f"Error parsing AI response: {e}")
            return {
                'summary': '',
                'risks': [],
                'recommendations': []
            }

    def _integrate_consciousness_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate consciousness metrics into final analysis"""
        try:
            # Ensure all metrics are properly bounded
            metrics = {
                'quantum_correlation': np.clip(self.consciousness_metrics.coherence, 0, 1),
                'consciousness_level': np.clip(self.consciousness_metrics.consciousness_level, 0, 1),
                'morphic_resonance': np.clip(self.consciousness_metrics.morphic_resonance, 0, 1),
                'integration_index': np.clip(self.consciousness_metrics.integration_index, 0, 1),
                'dark_ratio': np.clip(self.consciousness_metrics.dark_ratio, 0, 1),
                'confidence_score': np.clip(analysis['metrics'].get('confidence', 0), 0, 1)
            }

            # Preserve error if present in the analysis
            if 'error' in analysis:
                metrics['error'] = analysis['error']

            return {
                'market_analysis': {
                    'summary': f"Quantum Consciousness Analysis:\n{analysis['summary']}",
                    'risk_assessment': analysis['risks'],
                    'recommendations': analysis['recommendations'],
                    **metrics,
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            logging.error(f"Error integrating consciousness metrics: {e}")
            return {
                'market_analysis': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Provide consciousness-aware fallback analysis with bounded metrics"""
        metrics = {
            'quantum_correlation': np.clip(self.consciousness_metrics.coherence, 0, 1),
            'consciousness_level': np.clip(self.consciousness_metrics.consciousness_level, 0, 1),
            'morphic_resonance': np.clip(self.consciousness_metrics.morphic_resonance, 0, 1),
            'integration_index': np.clip(self.consciousness_metrics.integration_index, 0, 1),
            'confidence_score': 0.5
        }

        return {
            'market_analysis': {
                'summary': "Using quantum consciousness baseline metrics for analysis",
                'risk_assessment': [
                    "Limited quantum coherence data",
                    "Using consciousness baselines",
                    "Monitoring quantum state evolution"
                ],
                'recommendations': [
                    "Monitor quantum coherence trends",
                    "Wait for consciousness metric stability",
                    "Validate system quantum state"
                ],
                **metrics,
                'timestamp': datetime.now().isoformat()
            }
        }

    def get_quantum_field_state(self) -> Dict[str, Any]:
        """Get the current quantum field state"""
        try:
            # Calculate metrics using quantum field
            coherence = calculate_quantum_coherence(self.quantum_field)
            entropy = calculate_field_entropy(self.quantum_field)
            consciousness = self.consciousness_metrics.consciousness_level

            # Ensure metrics are in valid range [0,1]
            metrics = {
                'coherence': np.clip(float(coherence), 0, 1),
                'entropy': np.clip(float(entropy), 0, 1),
                'consciousness': np.clip(float(consciousness), 0, 1)
            }

            # Validate metrics
            assert all(isinstance(v, float) and 0 <= v <= 1 for v in metrics.values()), \
                "Invalid metric values detected"

            return {
                'dimension': self.quantum_dimension,
                'metrics': metrics
            }

        except Exception as e:
            logging.error(f"Error getting quantum field state: {e}")
            return {
                'dimension': self.quantum_dimension,
                'metrics': {
                    'coherence': 0.5,
                    'entropy': 0.5,
                    'consciousness': 0.5
                }
            }

    def serialize_analysis(self, analysis: Dict[str, Any]) -> str:
        """Serialize analysis data to JSON string"""
        try:
            # Convert numpy arrays and complex numbers to lists/basic types
            def serialize_item(item):
                if isinstance(item, np.ndarray):
                    return item.tolist()
                if isinstance(item, complex):
                    return {'real': item.real, 'imag': item.imag}
                if isinstance(item, dict):
                    return {k: serialize_item(v) for k, v in item.items()}
                if isinstance(item, list):
                    return [serialize_item(i) for i in item]
                return item

            serializable = serialize_item(analysis)
            return json.dumps(serializable)
        except Exception as e:
            logging.error(f"Error serializing analysis: {e}")
            return json.dumps({'error': 'Failed to serialize analysis'})

    def deserialize_analysis(self, serialized: str) -> Dict[str, Any]:
        """Deserialize analysis data from JSON string"""
        try:
            data = json.loads(serialized)

            # Convert back complex numbers if present
            def deserialize_item(item):
                if isinstance(item, dict):
                    if 'real' in item and 'imag' in item:
                        return complex(item['real'], item['imag'])
                    return {k: deserialize_item(v) for k, v in item.items()}
                if isinstance(item, list):
                    return [deserialize_item(i) for i in item]
                return item

            return deserialize_item(data)
        except Exception as e:
            logging.error(f"Error deserializing analysis: {e}")
            return {'error': 'Failed to deserialize analysis'}