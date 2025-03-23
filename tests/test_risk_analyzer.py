import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qualia.core.risk_analyzer import RiskAnalyzer
from qualia.core.market_data import MarketDataProvider

def test_risk_analyzer_initialization():
    """
    Testa a inicialização do RiskAnalyzer com MarketDataProvider.
    """
    try:
        # Inicializa MarketDataProvider
        market_data = MarketDataProvider(
            exchange="kucoin", 
            symbols=["BTC/USDT", "ETH/USDT"]
        )
        
        # Inicializa RiskAnalyzer
        risk_analyzer = RiskAnalyzer(
            config={
                'volatility_window': 20,
                'max_risk_score': 1.0,
                'risk_threshold': 0.7
            },
            market_data_provider=market_data
        )
        
        print("✅ RiskAnalyzer inicializado com sucesso!")
        
        # Testa método de validação
        if risk_analyzer.validate_market_data_provider():
            print("✅ Validação do MarketDataProvider concluída")
        else:
            print("❌ Falha na validação do MarketDataProvider")
    
    except Exception as e:
        print(f"❌ Erro durante inicialização: {e}")

if __name__ == "__main__":
    test_risk_analyzer_initialization()
