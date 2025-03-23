import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List

class QuantumVisualizer:
    def __init__(self):
        self.colors = px.colors.sequential.Plasma

    def plot_price_consciousness(self, 
                               dates: np.ndarray, 
                               prices: np.ndarray, 
                               consciousness: Dict[str, float]) -> go.Figure:
        """Create price chart with consciousness metrics"""
        fig = go.Figure()

        # Price candlesticks with improved visibility
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#00b7ff', width=2)
        ))

        # Add consciousness metrics as a subplot
        metrics_text = '<br>'.join([
            f"{k.title()}: {v:.2f}" 
            for k, v in consciousness.items()
        ])

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        # Improve layout
        fig.update_layout(
            title="Market Price & Consciousness Analysis",
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            height=600,
            margin=dict(t=30, l=10, r=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )

        return fig

    def plot_morphic_field(self, 
                          scales: np.ndarray, 
                          resonance: np.ndarray,
                          time_points: np.ndarray) -> go.Figure:
        """Create morphic field visualization"""
        fig = go.Figure()
        
        # Create heatmap of resonance field
        fig.add_trace(go.Heatmap(
            z=resonance,
            x=time_points,
            y=scales,
            colorscale='Viridis',
            name='Morphic Field'
        ))
        
        fig.update_layout(
            title="Morphic Field Resonance",
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Scale",
            yaxis_type="log"
        )
        
        return fig
    
    def plot_quantum_patterns(self, 
                            patterns: List[Dict[str, float]], 
                            prices: np.ndarray) -> go.Figure:
        """Visualize detected quantum patterns"""
        fig = go.Figure()
        
        # Plot price
        fig.add_trace(go.Scatter(
            x=np.arange(len(prices)),
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color=self.colors[0])
        ))
        
        # Add pattern markers
        for pattern in patterns:
            fig.add_trace(go.Scatter(
                x=[pattern['position']],
                y=[prices[int(pattern['position'])]],
                mode='markers',
                name=f'Pattern (s={pattern["scale"]:.1f})',
                marker=dict(
                    size=10 * pattern['strength'],
                    symbol='circle',
                    color=self.colors[3]
                )
            ))
        
        fig.update_layout(
            title="Quantum Pattern Detection",
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True
        )
        
        return fig