"""
Configuração para o Jupyter Notebook de análise de resultados.
"""

import os
import sys
from pathlib import Path

# Configuração do diretório de trabalho
notebook_dir = Path(__file__).parent
os.chdir(notebook_dir)

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = notebook_dir.parent.parent
sys.path.append(str(root_dir))

# Configuração do Jupyter
c = get_config()

# Configurações gerais
c.NotebookApp.ip = 'localhost'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = True
c.NotebookApp.notebook_dir = str(notebook_dir)

# Configurações de segurança
c.NotebookApp.token = ''
c.NotebookApp.password = ''

# Configurações de memória
c.NotebookApp.max_buffer_size = 1000000000  # 1GB
c.NotebookApp.max_memory = 1000000000  # 1GB

# Configurações de display
c.NotebookApp.allow_root = True
c.NotebookApp.allow_origin = '*'

# Configurações de autosave
c.NotebookApp.autosave_interval = 60  # segundos
c.NotebookApp.checkpoint_interval = 300  # segundos

# Configurações de kernels
c.NotebookApp.kernel_spec_manager_class = 'jupyter_client.kernelspec.KernelSpecManager'
c.NotebookApp.kernel_spec_manager_kwargs = {
    'kernel_dirs': [str(notebook_dir / 'kernels')]
}

# Configurações de extensões
c.NotebookApp.nbserver_extensions = {
    'jupyter_nbextensions_configurator': True
}

# Configurações de logging
c.NotebookApp.log_level = 'INFO'
c.NotebookApp.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
c.NotebookApp.log_file = str(notebook_dir / 'jupyter.log') 