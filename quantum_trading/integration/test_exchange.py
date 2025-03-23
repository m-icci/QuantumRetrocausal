"""
Script para testar a conexão com exchanges.
"""

import sys
import os
import json
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    logger.error("CCXT não instalado. Por favor, instale: pip install ccxt")
    sys.exit(1)

def test_kucoin_connection():
    """Testa a conexão com a KuCoin usando as credenciais do arquivo .env"""
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Obter credenciais do ambiente
    api_key = os.getenv('KUCOIN_API_KEY')
    api_secret = os.getenv('KUCOIN_API_SECRET')
    passphrase = os.getenv('KUCOIN_PASSPHRASE')
    
    if not all([api_key, api_secret, passphrase]):
        logger.error("Credenciais incompletas. Verifique o arquivo .env")
        return False
    
    logger.info("Testando conexão com KuCoin...")
    logger.info(f"API Key: {api_key[:5]}...{api_key[-3:]}")
    
    try:
        # Criar instância da exchange
        exchange = ccxt.kucoin({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'enableRateLimit': True
        })
        
        logger.info(f"Exchange criada: {exchange}")
        
        # Verificar se está conectado obtendo saldo
        balance = exchange.fetch_balance()
        
        logger.info("Conexão com KuCoin estabelecida com sucesso!")
        logger.info(f"Moedas disponíveis: {list(balance['free'].keys())}")
        
        # Exibir saldos não-zero
        non_zero = {k: v for k, v in balance['free'].items() if v > 0}
        if non_zero:
            logger.info("Saldos não-zero:")
            for currency, amount in non_zero.items():
                logger.info(f"  {currency}: {amount}")
        else:
            logger.info("Nenhum saldo disponível na conta")
        
        # Obter mercados disponíveis
        markets = exchange.load_markets()
        logger.info(f"Total de mercados disponíveis: {len(markets)}")
        logger.info(f"Alguns símbolos: {list(markets.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao conectar com KuCoin: {e}")
        logger.error(f"Detalhes do erro: {str(e)}")
        return False

def test_kraken_connection():
    """Testa a conexão com a Kraken usando as credenciais do arquivo .env"""
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Obter credenciais do ambiente
    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')
    
    if not all([api_key, api_secret]):
        logger.error("Credenciais da Kraken incompletas. Verifique o arquivo .env")
        return False
    
    logger.info("Testando conexão com Kraken...")
    logger.info(f"API Key: {api_key[:5]}...{api_key[-3:]}")
    
    try:
        # Criar instância da exchange
        exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        logger.info(f"Exchange criada: {exchange}")
        
        # Verificar se está conectado obtendo saldo
        balance = exchange.fetch_balance()
        
        logger.info("Conexão com Kraken estabelecida com sucesso!")
        logger.info(f"Moedas disponíveis: {list(balance['free'].keys())}")
        
        # Exibir saldos não-zero
        non_zero = {k: v for k, v in balance['free'].items() if v > 0}
        if non_zero:
            logger.info("Saldos não-zero:")
            for currency, amount in non_zero.items():
                logger.info(f"  {currency}: {amount}")
        else:
            logger.info("Nenhum saldo disponível na conta")
        
        # Obter mercados disponíveis
        markets = exchange.load_markets()
        logger.info(f"Total de mercados disponíveis: {len(markets)}")
        logger.info(f"Alguns símbolos: {list(markets.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao conectar com Kraken: {e}")
        logger.error(f"Detalhes do erro: {str(e)}")
        return False

if __name__ == "__main__":
    print("===== Teste de Conexão com Exchanges =====")
    
    print("\n----- Testando KuCoin -----")
    kucoin_success = test_kucoin_connection()
    
    print("\n----- Testando Kraken -----")
    kraken_success = test_kraken_connection()
    
    print("\n===== Resultados =====")
    print(f"KuCoin: {'✅ Sucesso' if kucoin_success else '❌ Falha'}")
    print(f"Kraken: {'✅ Sucesso' if kraken_success else '❌ Falha'}")
    
    if not (kucoin_success or kraken_success):
        print("\n❌ ALERTA: Nenhuma conexão estabelecida!")
        sys.exit(1)
    else:
        print("\n✅ Pelo menos uma conexão estabelecida com sucesso!") 