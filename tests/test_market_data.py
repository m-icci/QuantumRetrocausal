import sys
import os
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qualia.core.market_data import MarketDataProvider
from qualia.core.risk_analyzer import RiskAnalyzer

def test_market_data_provider():
    """
    Testa a obtenção de dados de mercado em tempo real.
    """
    try:
        # Inicializa MarketDataProvider
        market_data = MarketDataProvider(
            exchange="kucoin", 
            symbols=["BTC/USDT", "ETH/USDT"]
        )
        
        # Inicializa RiskAnalyzer
        risk_analyzer = RiskAnalyzer(market_data_provider=market_data)
        
        # Tenta obter estado de mercado
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            try:
                market_state = market_data.get_market_state(symbol)
                
                if market_state:
                    logging.info(f"✅ Dados obtidos para {symbol}")
                    logging.info(f"Estado do Mercado: {market_state}")
                    
                    # Analisa risco
                    risk_metrics = risk_analyzer.analyze(market_state)
                    logging.info(f"Métricas de Risco para {symbol}: {risk_metrics}")
                else:
                    logging.warning(f"❌ Nenhum dado obtido para {symbol}")
            
            except Exception as symbol_error:
                logging.error(f"Erro ao processar {symbol}: {symbol_error}")
        
    except Exception as e:
        logging.critical(f"❌ Erro crítico: {e}")

if __name__ == "__main__":
    test_market_data_provider()
