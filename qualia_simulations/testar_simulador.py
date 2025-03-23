#!/usr/bin/env python3
"""
Script para testar a geração de nonces usando o simulador de testnet com QUALIA.

Este script demonstra como o integrador RandomX com o campo retrocausal de QUALIA
pode gerar nonces válidos com maior eficiência, mesmo sem acesso a um nó Monero 
testnet real.

O sistema utiliza os princípios de retrocausalidade, emergência e auto-organização
para criar um processo de mineração mais inteligente e adaptativo.
"""

import time
import logging
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import Dict, List, Tuple, Any, Optional

from qualia_monero.tests.test_randomx_testnet_integration import TestRandomXTestnetIntegration
from qualia_monero.mining.randomx_integrator import RandomXRetrocausalIntegrator
from qualia_monero.mining.testnet_validator import MoneroTestnetValidator
from qualia_monero.retrocausal import RetrocausalConfig, RetrocausalField
from qualia_monero.core.qualia_config import (
    QualiaConfig, 
    RetrocausalConfig as QualiaRetrocausalConfig,
    QuantumParametersConfig, 
    RetrocausalBridge,
    QuantumCellularAutomaton,
    calculate_optimal_bit_configurations,
    adapt_bit_configurations_to_retrocausal_field
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Cores para saída formatada
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message: str):
    """Imprime um cabeçalho formatado."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")

def print_section(message: str):
    """Imprime um título de seção formatado."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{message}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-' * len(message)}{Colors.ENDC}")

def print_result(success: bool, message: str):
    """Imprime um resultado formatado."""
    if success:
        print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
    else:
        print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def test_retrocausal_vs_regular(template: Dict[str, Any], test_key: bytes) -> Dict[str, Any]:
    """
    Compara o desempenho da mineração com e sem o campo retrocausal.
    
    Args:
        template: Template de bloco para teste
        test_key: Chave RandomX para teste
        
    Returns:
        Dicionário com resultados da comparação
    """
    print_section("Comparação: Mineração Regular vs. Retrocausal QUALIA")
    
    # Carregar configurações do sistema QUALIA
    qualia_config = QualiaConfig.from_files()
    
    # Obter parâmetros quânticos e configuração retrocausal
    quantum_params = qualia_config.get_quantum_parameters()
    retrocausal_cfg = qualia_config.get_retrocausal_config()
    
    print(f"{Colors.CYAN}Usando configurações oficiais do Sistema QUALIA:{Colors.ENDC}")
    print(f"• Campo Retrocausal:")
    for key, value in retrocausal_cfg.items():
        print(f"  - {key}: {value}")
    
    # Obter configurações de bits calculadas pelos princípios do sistema
    bit_configs = calculate_optimal_bit_configurations()
    print(f"• Configurações de bits calculadas: {bit_configs}")
    
    # Criar um autômato celular quântico para dinâmica de bits
    automaton = QuantumCellularAutomaton(init_states=bit_configs)
    
    # Configurações para o teste
    num_runs = 2  # Reduzido para obter resultados mais rapidamente
    max_attempts_per_run = 10000  # Reduzido para teste mais rápido
    
    # Para simulação, ajustar dificuldade
    template['difficulty'] = 50000  # Dificuldade ajustada para testes mais rápidos
    
    results = {
        'regular': {'attempts': [], 'time': [], 'success_rate': 0},
        'retrocausal': {'attempts': [], 'time': [], 'success_rate': 0}
    }
    
    # Teste com mineração regular (sem campo retrocausal)
    print(f"\n{Colors.BOLD}Teste de Mineração Regular{Colors.ENDC}")
    regular_integrator = RandomXRetrocausalIntegrator(
        randomx_key=test_key,
        retrocausal_config=None,  # Sem campo retrocausal
        vm_pool_size=1
    )
    
    regular_successes = 0
    for i in range(num_runs):
        print(f"\nExecutando teste regular {i+1}/{num_runs}...")
        start_time = time.time()
        
        nonce, hash_result, attempts = regular_integrator.find_valid_nonce(
            template,
            max_attempts=max_attempts_per_run
        )
        
        elapsed = time.time() - start_time
        
        if nonce is not None:
            regular_successes += 1
            results['regular']['attempts'].append(attempts)
            results['regular']['time'].append(elapsed)
            rate = attempts / elapsed if elapsed > 0 else 0
            print(f"{Colors.GREEN}Nonce válido: {nonce} ({attempts:,} tentativas, {rate:.2f} H/s){Colors.ENDC}")
        else:
            results['regular']['attempts'].append(max_attempts_per_run)
            results['regular']['time'].append(elapsed)
            print(f"{Colors.RED}Nenhum nonce válido encontrado após {max_attempts_per_run:,} tentativas{Colors.ENDC}")
    
    # Calcular taxa de sucesso
    results['regular']['success_rate'] = regular_successes / num_runs
    
    # Libertar recursos
    regular_integrator.release_resources()
    
    # Teste com mineração retrocausal QUALIA
    print(f"\n{Colors.BOLD}Teste de Mineração com Campo Retrocausal QUALIA{Colors.ENDC}")
    
    # Definir coerência e fator retrocausal para o experimento
    field_coherence = 0.65
    retrocausal_factor = 0.70
    
    # Aplicar influência retrocausal ao autômato
    automaton.apply_retrocausal_field(field_coherence, retrocausal_factor)
    
    # Colapsar para obter configurações atuais
    adapted_configs = automaton.collapse_to_configuration()
    print(f"• Configurações adaptadas após influência retrocausal: {adapted_configs}")
    
    # Converter a configuração do sistema para o formato esperado pelo integrador
    retrocausal_config = {
        'dimensions': 5,
        'memory_size': adapted_configs[0] if adapted_configs else bit_configs[0],
        'coherence_threshold': retrocausal_cfg.get('coherence_threshold', 0.65),
        'temporal_window': retrocausal_cfg.get('temporal_window', 15.0),
        'future_depth': retrocausal_cfg.get('future_depth', 7),
        'influence_weight': 0.75,  # Influência do campo retrocausal
    }
    
    # Exibir configuração utilizada
    print(f"{Colors.CYAN}Configuração do campo retrocausal:{Colors.ENDC}")
    for key, value in retrocausal_config.items():
        print(f"• {key}: {value}")
    
    retrocausal_integrator = RandomXRetrocausalIntegrator(
        randomx_key=test_key,
        retrocausal_config=retrocausal_config,
        vm_pool_size=1
    )
    
    retrocausal_successes = 0
    for i in range(num_runs):
        print(f"\nExecutando teste retrocausal {i+1}/{num_runs}...")
        start_time = time.time()
        
        nonce, hash_result, attempts = retrocausal_integrator.find_valid_nonce(
            template,
            max_attempts=max_attempts_per_run
        )
        
        elapsed = time.time() - start_time
        
        if nonce is not None:
            retrocausal_successes += 1
            results['retrocausal']['attempts'].append(attempts)
            results['retrocausal']['time'].append(elapsed)
            rate = attempts / elapsed if elapsed > 0 else 0
            print(f"{Colors.GREEN}Nonce válido: {nonce} ({attempts:,} tentativas, {rate:.2f} H/s){Colors.ENDC}")
        else:
            results['retrocausal']['attempts'].append(max_attempts_per_run)
            results['retrocausal']['time'].append(elapsed)
            print(f"{Colors.RED}Nenhum nonce válido encontrado após {max_attempts_per_run:,} tentativas{Colors.ENDC}")
    
    # Calcular taxa de sucesso
    results['retrocausal']['success_rate'] = retrocausal_successes / num_runs
    
    # Libertar recursos
    retrocausal_integrator.release_resources()
    
    # Resumo da comparação
    print_section("Resultado da Comparação")
    
    # Calcular médias
    if results['regular']['attempts']:
        avg_regular_attempts = sum(results['regular']['attempts']) / len(results['regular']['attempts'])
        avg_regular_time = sum(results['regular']['time']) / len(results['regular']['time'])
        avg_regular_rate = avg_regular_attempts / avg_regular_time if avg_regular_time > 0 else 0
    else:
        avg_regular_attempts = avg_regular_time = avg_regular_rate = 0
        
    if results['retrocausal']['attempts']:
        avg_retrocausal_attempts = sum(results['retrocausal']['attempts']) / len(results['retrocausal']['attempts'])
        avg_retrocausal_time = sum(results['retrocausal']['time']) / len(results['retrocausal']['time'])
        avg_retrocausal_rate = avg_retrocausal_attempts / avg_retrocausal_time if avg_retrocausal_time > 0 else 0
    else:
        avg_retrocausal_attempts = avg_retrocausal_time = avg_retrocausal_rate = 0
    
    # Melhoria percentual
    if avg_regular_attempts > 0:
        attempts_improvement = ((avg_regular_attempts - avg_retrocausal_attempts) / avg_regular_attempts) * 100
    else:
        attempts_improvement = 0
        
    if avg_regular_rate > 0:
        rate_improvement = ((avg_retrocausal_rate - avg_regular_rate) / avg_regular_rate) * 100
    else:
        rate_improvement = 0
    
    # Exibir resultados
    print(f"\n{Colors.BOLD}Mineração Regular:{Colors.ENDC}")
    print(f"  Taxa de sucesso: {results['regular']['success_rate']*100:.1f}%")
    print(f"  Tentativas médias: {avg_regular_attempts:,.1f}")
    print(f"  Tempo médio: {avg_regular_time:.2f}s")
    print(f"  Taxa média: {avg_regular_rate:.2f} H/s")
    
    print(f"\n{Colors.BOLD}Mineração Retrocausal QUALIA:{Colors.ENDC}")
    print(f"  Taxa de sucesso: {results['retrocausal']['success_rate']*100:.1f}%")
    print(f"  Tentativas médias: {avg_retrocausal_attempts:,.1f}")
    print(f"  Tempo médio: {avg_retrocausal_time:.2f}s")
    print(f"  Taxa média: {avg_retrocausal_rate:.2f} H/s")
    
    print(f"\n{Colors.BOLD}Melhoria com Campo Retrocausal:{Colors.ENDC}")
    print(f"  Redução de tentativas: {attempts_improvement:.1f}%")
    print(f"  Aumento de eficiência: {rate_improvement:.1f}%")
    
    # Resultado final
    if rate_improvement > 0:
        print_result(True, f"O campo retrocausal QUALIA aumentou a eficiência em {rate_improvement:.1f}%")
    else:
        print_result(False, "O campo retrocausal não demonstrou melhoria neste teste")
    
    # Salvar resultados
    results_summary = {
        'regular': {
            'success_rate': results['regular']['success_rate'],
            'avg_attempts': float(avg_regular_attempts),
            'avg_time': avg_regular_time,
            'avg_rate': avg_regular_rate
        },
        'retrocausal': {
            'success_rate': results['retrocausal']['success_rate'],
            'avg_attempts': float(avg_retrocausal_attempts),
            'avg_time': avg_retrocausal_time,
            'avg_rate': avg_retrocausal_rate
        },
        'improvement': {
            'attempts': attempts_improvement,
            'rate': rate_improvement
        }
    }
    
    return results_summary

def gerar_visualizacoes(results: Dict[str, Any]):
    """
    Gera visualizações dos resultados da comparação.
    
    Args:
        results: Resultados da comparação
    """
    print_section("Gerando Visualizações")
    
    # Criar diretório para resultados
    results_dir = os.path.join(os.getcwd(), "resultados")
    os.makedirs(results_dir, exist_ok=True)
    
    # Salvar resultados em JSON
    results_file = os.path.join(results_dir, "comparacao_resultados.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados salvos em: {results_file}")
    
    # Gerar gráfico de comparação
    labels = ['Regular', 'Retrocausal QUALIA']
    success_rates = [results['regular']['success_rate'] * 100, 
                     results['retrocausal']['success_rate'] * 100]
    avg_attempts = [results['regular']['avg_attempts'], 
                    results['retrocausal']['avg_attempts']]
    avg_rates = [results['regular']['avg_rate'], 
                 results['retrocausal']['avg_rate']]
    
    # Gráfico 1: Taxa de Sucesso
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, success_rates, color=['#3498db', '#2ecc71'])
    plt.title('Taxa de Sucesso na Geração de Nonces (%)', fontsize=14)
    plt.ylabel('Taxa de Sucesso (%)')
    plt.ylim(0, 100)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    # Salvar gráfico
    success_plot = os.path.join(results_dir, "taxa_sucesso.png")
    plt.savefig(success_plot)
    print(f"Gráfico de taxa de sucesso salvo em: {success_plot}")
    
    # Gráfico 2: Tentativas Médias
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_attempts, color=['#3498db', '#2ecc71'])
    plt.title('Número Médio de Tentativas para Encontrar Nonce Válido', fontsize=14)
    plt.ylabel('Número de Tentativas')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{height:,.0f}', ha='center', va='bottom')
    
    # Salvar gráfico
    attempts_plot = os.path.join(results_dir, "tentativas_medias.png")
    plt.savefig(attempts_plot)
    print(f"Gráfico de tentativas médias salvo em: {attempts_plot}")
    
    # Gráfico 3: Taxa de Hash
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_rates, color=['#3498db', '#2ecc71'])
    plt.title('Taxa Média de Hash (H/s)', fontsize=14)
    plt.ylabel('Hashes por Segundo')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Salvar gráfico
    rates_plot = os.path.join(results_dir, "taxa_hash.png")
    plt.savefig(rates_plot)
    print(f"Gráfico de taxa de hash salvo em: {rates_plot}")
    
    print_result(True, "Visualizações geradas com sucesso")

def main():
    """Função principal para executar o teste de geração de nonces com QUALIA."""
    print_header("SISTEMA QUALIA - TESTE DE INTEGRAÇÃO COM RANDOMX")
    logger.info("Inicializando ambiente de teste com simulador de testnet")
    
    # Criar um validador simulado
    validator = TestRandomXTestnetIntegration._create_test_validator()
    print_result(True, "Validador simulado criado com sucesso")
    
    # Obter um template simulado
    template = validator.get_block_template()
    if not template or template.get('status') != 'OK':
        print_result(False, "Falha ao obter template simulado")
        return False
        
    difficulty = template.get('difficulty', 0)
    print_result(True, f"Template simulado obtido com dificuldade: {difficulty:,}")
    
    # Chave de teste para o RandomX
    test_key = b'QUALIA_TESTNET_INTEGRATION_TEST_KEY_01'
    
    # Executar testes comparativos
    results = test_retrocausal_vs_regular(template, test_key)
    
    # Gerar visualizações
    gerar_visualizacoes(results)
    
    print_header("TESTE DE INTEGRAÇÃO CONCLUÍDO")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperação interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro durante a execução do teste: {str(e)}")
        sys.exit(1)
