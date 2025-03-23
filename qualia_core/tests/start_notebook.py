"""
Script para iniciar o Jupyter Notebook com configuração personalizada.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_notebook():
    """Inicia o Jupyter Notebook com configuração personalizada."""
    try:
        # Obtém o diretório do script
        script_dir = Path(__file__).parent
        
        # Configura o ambiente
        os.environ['PYTHONPATH'] = str(script_dir.parent.parent)
        os.chdir(script_dir)
        
        # Inicia o Jupyter Notebook
        logger.info("Iniciando Jupyter Notebook...")
        subprocess.run([
            'jupyter',
            'notebook',
            '--config=jupyter_notebook_config.py',
            'analyze_results.ipynb'
        ])
        
    except Exception as e:
        logger.error(f"Erro ao iniciar o Jupyter Notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_notebook() 