#!/usr/bin/env python3
"""
Setup script for the quantum_trading package.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Import package metadata
about = {}
with open(os.path.join('quantum_trading', '__init__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

setup(
    name="quantum_trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "ccxt",
        "aiohttp",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['PACKAGE_METADATA']['description'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=about['PACKAGE_METADATA']['url'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'qualia=quantum_trading.run:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Bug Reports': 'https://github.com/qualia-trading/quantum-trading/issues',
        'Source': 'https://github.com/qualia-trading/quantum-trading',
        'Documentation': 'https://qualia-trading.readthedocs.io/',
    },
) 