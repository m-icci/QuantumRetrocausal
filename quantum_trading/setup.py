"""
Script para configuração e inicialização do ambiente do sistema quântico de trading.
"""

import os
import sys
import shutil
import platform
import logging
import subprocess
from pathlib import Path
import importlib.util

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QuantumSetup:
    """Configurador do ambiente quântico de trading."""
    
    def __init__(self):
        """Inicializa o configurador."""
        self.base_dir = Path(__file__).parent.absolute()
        self.os_name = platform.system()
        self.required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'ccxt', 'python-dotenv', 'psutil', 'aiohttp', 'argparse'
        ]
        self.config_template = """# Configurações da Exchange
EXCHANGE=binance
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here

# Configurações de Trading
SYMBOL=BTC/USDT
TIMEFRAME=1h
LEVERAGE=1
MAX_POSITIONS=3
DAILY_TRADES_LIMIT=10
DAILY_LOSS_LIMIT=100
MIN_CONFIDENCE=0.7
POSITION_SIZE=100
MIN_POSITION_SIZE=10
MAX_POSITION_SIZE=1000
STOP_LOSS=0.02
TAKE_PROFIT=0.04
RISK_PER_TRADE=0.01

# Configurações de Indicadores
RSI_PERIOD=14
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
BB_PERIOD=20
BB_STD=2
ATR_PERIOD=14
ATR_MULTIPLIER=2

# Configurações de Logging
LOG_LEVEL=INFO
"""
    
    def check_python_version(self) -> bool:
        """
        Verifica versão do Python.
        
        Returns:
            True se a versão é compatível
        """
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 7):
            logger.error(f"Versão do Python incompatível: {major}.{minor}. Requer Python 3.7+")
            return False
        
        logger.info(f"Versão do Python: {major}.{minor}")
        return True
    
    def check_dependencies(self) -> bool:
        """
        Verifica dependências.
        
        Returns:
            True se todas dependências estão instaladas
        """
        missing_packages = []
        
        for package in self.required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Dependências faltando: {', '.join(missing_packages)}")
            return False
        
        logger.info("Todas dependências instaladas")
        return True
    
    def install_dependencies(self) -> bool:
        """
        Instala dependências.
        
        Returns:
            True se instalação bem-sucedida
        """
        try:
            # Instalar dependências
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + self.required_packages
            
            logger.info(f"Instalando dependências: {' '.join(self.required_packages)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Erro ao instalar dependências: {stderr}")
                return False
            
            logger.info("Dependências instaladas com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao instalar dependências: {e}")
            return False
    
    def create_data_dirs(self) -> bool:
        """
        Cria diretórios de dados.
        
        Returns:
            True se criação bem-sucedida
        """
        try:
            # Criar diretórios
            dirs = [
                'data',
                'logs',
                'reports',
                'results',
                'models'
            ]
            
            for d in dirs:
                path = self.base_dir / d
                path.mkdir(exist_ok=True)
                logger.info(f"Diretório criado/verificado: {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")
            return False
    
    def create_config_file(self) -> bool:
        """
        Cria arquivo de configuração.
        
        Returns:
            True se criação bem-sucedida
        """
        try:
            # Verificar se arquivo já existe
            config_path = self.base_dir.parent / '.env'
            
            if config_path.exists():
                logger.info(f"Arquivo de configuração já existe: {config_path}")
                return True
            
            # Criar arquivo
            with open(config_path, 'w') as f:
                f.write(self.config_template)
            
            logger.info(f"Arquivo de configuração criado: {config_path}")
            logger.info("Edite o arquivo com suas configurações antes de executar o sistema")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar arquivo de configuração: {e}")
            return False
    
    def check_exchange_api(self) -> bool:
        """
        Verifica API da exchange.
        
        Returns:
            True se API está configurada
        """
        try:
            import ccxt
            from dotenv import load_dotenv
            
            # Carregar configuração
            load_dotenv()
            
            # Verificar configuração da API
            exchange_name = os.getenv('EXCHANGE')
            api_key = os.getenv('API_KEY')
            api_secret = os.getenv('API_SECRET')
            
            if exchange_name == 'binance' and api_key == 'your_api_key_here':
                logger.warning("API não configurada. Edite o arquivo .env com suas credenciais")
                return False
            
            # Criar objeto da exchange
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            
            # Testar API
            try:
                markets = exchange.fetch_markets()
                logger.info(f"API da exchange configurada e funcionando ({len(markets)} mercados disponíveis)")
                return True
            except Exception as e:
                logger.error(f"Erro ao testar API da exchange: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar API da exchange: {e}")
            return False
    
    def check_disk_space(self) -> bool:
        """
        Verifica espaço em disco.
        
        Returns:
            True se espaço suficiente
        """
        try:
            import shutil
            
            # Verificar espaço em disco
            total, used, free = shutil.disk_usage(self.base_dir)
            
            # Converter para GB
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            
            if free_gb < 1:
                logger.error(f"Espaço em disco insuficiente: {free_gb:.2f} GB livre")
                return False
            
            logger.info(f"Espaço em disco: {free_gb:.2f} GB livre de {total_gb:.2f} GB total")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar espaço em disco: {e}")
            return False
    
    def perform_setup(self) -> bool:
        """
        Executa setup completo.
        
        Returns:
            True se setup bem-sucedido
        """
        logger.info("Iniciando setup do sistema quântico de trading...")
        
        # Verificar versão do Python
        if not self.check_python_version():
            logger.error("Falha na verificação da versão do Python")
            return False
        
        # Verificar dependências
        if not self.check_dependencies():
            logger.info("Instalando dependências...")
            if not self.install_dependencies():
                logger.error("Falha na instalação de dependências")
                return False
        
        # Criar diretórios
        if not self.create_data_dirs():
            logger.error("Falha na criação de diretórios")
            return False
        
        # Criar arquivo de configuração
        if not self.create_config_file():
            logger.error("Falha na criação do arquivo de configuração")
            return False
        
        # Verificar espaço em disco
        if not self.check_disk_space():
            logger.warning("Alerta de espaço em disco insuficiente")
        
        # Verificar API da exchange
        api_status = self.check_exchange_api()
        if not api_status:
            logger.warning("API da exchange não configurada ou não disponível")
        
        logger.info("Setup do sistema quântico de trading concluído")
        
        if not api_status:
            logger.info("Importante: Configure suas credenciais da API no arquivo .env antes de executar o sistema")
        
        return True

def main():
    """Função principal."""
    try:
        setup = QuantumSetup()
        if setup.perform_setup():
            logger.info("Sistema pronto para uso")
            logger.info("Execute 'python -m quantum_trading.cli --help' para ver os comandos disponíveis")
        else:
            logger.error("Falha no setup do sistema")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Setup interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro fatal durante setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 