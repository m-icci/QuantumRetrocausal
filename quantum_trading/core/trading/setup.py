"""
Configuração do módulo de trading.
"""

from setuptools import setup, find_packages

setup(
    name='quantum_trading',
    version='0.1.0',
    description='Sistema de trading quântico',
    author='QUALIA',
    author_email='qualia@example.com',
    packages=find_packages(),
    install_requires=[
        'aiohttp>=3.8.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'ta-lib>=0.4.0',
        'python-dotenv>=0.19.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Office/Business :: Financial :: Investment'
    ]
) 