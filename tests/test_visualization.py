"""
Testes para módulo de visualização
============================

Testes unitários para o sistema de visualização do QUALIA.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from quantum_trading.visualization import (
    QuantumVisualizer,
    MarketVisualizer,
    PerformanceVisualizer,
    DashboardGenerator
)
from quantum_trading.exceptions import VisualizationError

@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para visualização."""
    np.random.seed(42)
    timestamps = [
        datetime.now() + timedelta(minutes=i)
        for i in range(100)
    ]
    prices = np.cumsum(np.random.normal(0, 0.1, 100)) + 100
    volumes = np.random.exponential(1, 100) * 10
    quantum_states = np.random.random((100, 3))  # entropia, coerência, complexidade
    trades = [
        {"timestamp": t, "price": p, "amount": v/100, "side": "buy" if i % 2 == 0 else "sell"}
        for i, (t, p, v) in enumerate(zip(timestamps[:10], prices[:10], volumes[:10]))
    ]
    return {
        "timestamps": timestamps,
        "prices": prices,
        "volumes": volumes,
        "quantum_states": quantum_states,
        "trades": trades
    }

@pytest.fixture
def output_dir(tmp_path):
    """Fixture que cria diretório temporário para outputs."""
    return tmp_path

def test_quantum_state_visualization(sample_data, output_dir):
    """Testa visualização de estados quânticos."""
    qv = QuantumVisualizer()
    
    # Gera visualização
    fig = qv.plot_quantum_states(
        timestamps=sample_data["timestamps"],
        states=sample_data["quantum_states"]
    )
    
    # Verifica figura
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    
    # Salva figura
    output_file = output_dir / "quantum_states.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_market_visualization(sample_data, output_dir):
    """Testa visualização de dados de mercado."""
    mv = MarketVisualizer()
    
    # Gera visualização
    fig = mv.plot_market_data(
        timestamps=sample_data["timestamps"],
        prices=sample_data["prices"],
        volumes=sample_data["volumes"]
    )
    
    # Verifica figura
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # preço e volume
    
    # Salva figura
    output_file = output_dir / "market_data.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_performance_visualization(sample_data, output_dir):
    """Testa visualização de performance."""
    pv = PerformanceVisualizer()
    
    # Gera visualização
    fig = pv.plot_performance(
        trades=sample_data["trades"]
    )
    
    # Verifica figura
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    
    # Salva figura
    output_file = output_dir / "performance.png"
    fig.savefig(output_file)
    assert output_file.exists()
    
    plt.close(fig)

def test_dashboard_generation(sample_data, output_dir):
    """Testa geração de dashboard."""
    dg = DashboardGenerator()
    
    # Gera dashboard
    dashboard = dg.generate_dashboard(
        timestamps=sample_data["timestamps"],
        prices=sample_data["prices"],
        volumes=sample_data["volumes"],
        quantum_states=sample_data["quantum_states"],
        trades=sample_data["trades"]
    )
    
    # Verifica dashboard
    assert isinstance(dashboard, plt.Figure)
    assert len(dashboard.axes) >= 4  # múltiplos subplots
    
    # Salva dashboard
    output_file = output_dir / "dashboard.png"
    dashboard.savefig(output_file)
    assert output_file.exists()
    
    plt.close(dashboard)

def test_visualization_validation():
    """Testa validação de dados para visualização."""
    qv = QuantumVisualizer()
    
    # Testa dados inválidos
    with pytest.raises(VisualizationError):
        qv.plot_quantum_states(
            timestamps=None,
            states=np.random.random((100, 3))
        )
    
    with pytest.raises(VisualizationError):
        qv.plot_quantum_states(
            timestamps=[datetime.now()],
            states=np.random.random((2, 3))  # tamanho incompatível
        )

def test_custom_styling():
    """Testa personalização de estilo."""
    mv = MarketVisualizer()
    
    # Configura estilo
    style_config = {
        "figsize": (12, 8),
        "color_scheme": {
            "price": "blue",
            "volume": "gray",
            "buy": "green",
            "sell": "red"
        },
        "font_size": 12,
        "grid": True
    }
    
    mv.set_style(style_config)
    
    # Verifica configurações
    assert mv.style["figsize"] == (12, 8)
    assert mv.style["color_scheme"]["price"] == "blue"
    assert mv.style["font_size"] == 12
    assert mv.style["grid"] is True

def test_interactive_visualization(sample_data):
    """Testa geração de visualizações interativas."""
    mv = MarketVisualizer(interactive=True)
    
    # Gera visualização interativa
    fig = mv.plot_market_data(
        timestamps=sample_data["timestamps"],
        prices=sample_data["prices"],
        volumes=sample_data["volumes"]
    )
    
    # Verifica elementos interativos
    assert hasattr(fig, "canvas")
    assert len(fig.canvas.callbacks.callbacks) > 0
    
    plt.close(fig)

def test_real_time_updates(sample_data):
    """Testa atualizações em tempo real."""
    mv = MarketVisualizer(real_time=True)
    
    # Inicializa plot
    fig = mv.initialize_plot()
    
    # Atualiza com novos dados
    for i in range(10):
        mv.update_plot(
            timestamp=sample_data["timestamps"][i],
            price=sample_data["prices"][i],
            volume=sample_data["volumes"][i]
        )
        
        # Verifica atualização
        assert len(mv.price_line.get_xdata()) == i + 1
        assert len(mv.volume_line.get_xdata()) == i + 1
    
    plt.close(fig)

def test_export_formats(sample_data, output_dir):
    """Testa diferentes formatos de exportação."""
    mv = MarketVisualizer()
    
    # Gera visualização
    fig = mv.plot_market_data(
        timestamps=sample_data["timestamps"],
        prices=sample_data["prices"],
        volumes=sample_data["volumes"]
    )
    
    # Testa diferentes formatos
    formats = ["png", "pdf", "svg"]
    for fmt in formats:
        output_file = output_dir / f"market_data.{fmt}"
        fig.savefig(output_file, format=fmt)
        assert output_file.exists()
    
    plt.close(fig) 