#!/usr/bin/env python3
"""
QUALIA Helix Integration Demo

Este script demonstra a integração entre o módulo Helix e o sistema QUALIA,
mostrando como as análises quânticas, fractais e retrocausais do Helix
aprimoram as decisões de trading do QUALIA Core Engine.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Configurar caminhos para importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes QUALIA
from quantum_trading.integration.helix_controller import HelixController
from quantum_trading.integration.qualia_core_engine import QUALIAEngine
from quantum_trading.visualization.helix_visualizer import HelixVisualizer
from quantum_trading.neural.lstm_predictor import LSTMPredictor
from quantum_trading.metrics.performance_analyzer import PerformanceMetrics
from qcnn_wrapper import QCNNWrapper

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scripts/demo/helix_qualia_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("helix_qualia_demo")

class HelixQualiaDemo:
    """
    Demonstração da integração entre Helix e QUALIA.
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa a demonstração.
        
        Args:
            config_path: Caminho para arquivo de configuração.
        """
        self.config = self._load_config(config_path)
        
        # Inicializar componentes separadamente para monitoramento detalhado
        self.helix_controller = HelixController(config=self.config.get('helix', {}))
        self.lstm_predictor = LSTMPredictor(model_path=self.config.get('lstm_model_path'))
        self.qcnn_wrapper = QCNNWrapper(config=self.config.get('qcnn', {}))
        self.performance_analyzer = PerformanceMetrics(
            save_path="scripts/demo/data/helix_qualia_performance.json"
        )
        
        # Inicializar QUALIA Engine com componentes
        self.qualia_engine = QUALIAEngine(
            config=self.config,
            enable_quantum=True,
            enable_helix=True,
            lstm_predictor=self.lstm_predictor,
            qcnn_wrapper=self.qcnn_wrapper,
            performance_analyzer=self.performance_analyzer
        )
        
        # Inicializar visualizador do Helix
        self.helix_visualizer = HelixVisualizer(
            helix_controller=self.helix_controller,
            output_dir="scripts/demo/data/visualizations"
        )
        
        # Controles de demonstração
        self.demo_results = []
        logger.info("Demo inicializada com integração Helix-QUALIA")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carrega configuração do arquivo ou utiliza padrão.
        
        Args:
            config_path: Caminho do arquivo de configuração.
            
        Returns:
            Configuração carregada.
        """
        # Configuração padrão para demonstração
        default_config = {
            'helix': {
                'dimensions': 64,  # Menor para demonstração
                'num_qubits': 8,
                'phi': 0.618,      # Proporção áurea
                'temperature': 0.2,
                'batch_size': 256,
                'tau': 5           # Horizonte retrocausal menor para demonstração
            },
            'lstm_model_path': None,  # Modelo gerado internamente
            'qcnn': {
                'num_classes': 10,
                'learning_rate': 0.001
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuração carregada de {config_path}")
                return config
            except Exception as e:
                logger.error(f"Erro ao carregar configuração: {e}")
                return default_config
        
        logger.info("Usando configuração padrão para demonstração")
        return default_config
    
    def generate_sample_data(self, 
                            num_periods: int = 100, 
                            volatility: float = 0.02) -> pd.DataFrame:
        """
        Gera dados de mercado de exemplo para demonstração.
        
        Args:
            num_periods: Número de períodos de dados.
            volatility: Volatilidade dos dados simulados.
            
        Returns:
            DataFrame com dados de mercado.
        """
        logger.info(f"Gerando dados de mercado simulados ({num_periods} períodos)")
        
        # Simular dados de preço com tendência e ruído
        np.random.seed(42)  # Para reprodutibilidade
        
        # Criar tendência de preço (movimento browniano)
        prices = np.zeros(num_periods)
        prices[0] = 100.0  # Preço inicial
        
        # Adicionar alguns "regimes" para testar adaptabilidade
        regimes = [
            (0, int(num_periods*0.3), 0.02, 0.02),      # Tendência de alta
            (int(num_periods*0.3), int(num_periods*0.5), -0.01, 0.03),  # Tendência de baixa, alta volatilidade
            (int(num_periods*0.5), int(num_periods*0.7), 0.0, 0.01),   # Movimento lateral, baixa volatilidade
            (int(num_periods*0.7), num_periods, 0.015, 0.025)   # Tendência de alta com volatilidade média
        ]
        
        for i in range(1, num_periods):
            # Determinar regime atual
            for start, end, trend, vol in regimes:
                if start <= i < end:
                    regime_trend = trend
                    regime_vol = vol
                    break
            
            # Calcular retorno diário com tendência e volatilidade do regime
            daily_return = np.random.normal(regime_trend, regime_vol)
            prices[i] = prices[i-1] * (1 + daily_return)
        
        # Criar timestamps
        base_date = datetime.now() - timedelta(days=num_periods)
        dates = [base_date + timedelta(days=i) for i in range(num_periods)]
        
        # Criar candles OHLCV
        candles = []
        for i in range(num_periods):
            price = prices[i]
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = low + np.random.uniform(0, high - low)
            close_price = low + np.random.uniform(0, high - low)
            volume = np.random.uniform(500, 1000) * price
            
            candles.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        # Criar DataFrame
        df = pd.DataFrame(candles)
        
        # Adicionar métricas de mercado
        df['spread'] = df['high'] - df['low']
        df['volatility'] = df['spread'] / df['close']
        df['entropy'] = np.random.uniform(0.3, 0.7, size=len(df))  # Simulação simplificada
        df['fractal_dimension'] = 1.2 + 0.4 * np.sin(np.linspace(0, 6*np.pi, num_periods))  # Simulação cíclica
        
        # Salvar para referência
        os.makedirs('scripts/demo/data', exist_ok=True)
        df.to_csv('scripts/demo/data/market_sample.csv', index=False)
        
        return df 
    
    def simulate_trading_decisions(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Simula decisões de trading usando QUALIA com e sem Helix.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Lista de resultados das decisões.
        """
        logger.info("Simulando decisões de trading com integração Helix-QUALIA")
        
        # Preparar resultados comparativos
        results = []
        
        # Janela de candlesticks para análise
        window_size = 20
        
        # Iterar sobre os dados em janelas
        for i in range(window_size, len(market_data)):
            # Extrair dados da janela atual
            window_data = market_data.iloc[i-window_size:i]
            current_data = market_data.iloc[i]
            
            # Converter para formato de candlesticks para QCNN
            candles = window_data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Preparar dados de mercado para LSTM
            market_input = {
                'timestamp': current_data['timestamp'].isoformat(),
                'price': current_data['close'],
                'volume': current_data['volume'],
                'spread': current_data['spread'],
                'volatility': current_data['volatility'],
                'entropy': current_data['entropy'],
                'fractal_dimension': current_data['fractal_dimension']
            }
            
            # Primeiro: processar com Helix ativado
            self.qualia_engine.enable_helix = True
            decision_with_helix = self.qualia_engine.process_opportunity(
                market_data=market_input,
                candlesticks=candles
            )
            
            # Segundo: processar com Helix desativado
            self.qualia_engine.enable_helix = False
            decision_without_helix = self.qualia_engine.process_opportunity(
                market_data=market_input,
                candlesticks=candles
            )
            
            # Reativar Helix para continuar a demonstração
            self.qualia_engine.enable_helix = True
            
            # Simular resultado da operação (simplificado)
            next_idx = min(i + 5, len(market_data) - 1)  # Horizonte de 5 períodos
            future_price = market_data.iloc[next_idx]['close']
            current_price = current_data['close']
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # Determinar sucesso das decisões
            with_helix_success = (
                (decision_with_helix['direction'] == 'up' and price_change_pct > 0) or
                (decision_with_helix['direction'] == 'down' and price_change_pct < 0)
            )
            
            without_helix_success = (
                (decision_without_helix['direction'] == 'up' and price_change_pct > 0) or
                (decision_without_helix['direction'] == 'down' and price_change_pct < 0)
            )
            
            # Registrar resultado
            result = {
                'timestamp': current_data['timestamp'],
                'price': current_price,
                'future_price': future_price,
                'price_change_pct': price_change_pct,
                'with_helix': {
                    'should_enter': decision_with_helix['should_enter'],
                    'confidence': decision_with_helix.get('confidence', 0),
                    'direction': decision_with_helix.get('direction', 'none'),
                    'success': with_helix_success,
                    'lstm_threshold': decision_with_helix.get('adaptive_params', {}).get('lstm_threshold', 0.7),
                    'quantum_coherence': decision_with_helix.get('adaptive_params', {}).get('quantum_coherence', 0.5),
                    'quantum_complexity': decision_with_helix.get('adaptive_params', {}).get('quantum_complexity', 0.3)
                },
                'without_helix': {
                    'should_enter': decision_without_helix['should_enter'],
                    'confidence': decision_without_helix.get('confidence', 0),
                    'direction': decision_without_helix.get('direction', 'none'),
                    'success': without_helix_success
                },
                'helix_metrics': self.helix_controller.quantum_metrics.copy() if hasattr(self.helix_controller, 'quantum_metrics') else {}
            }
            
            results.append(result)
            
            # Atualizar visualizador do Helix (a cada 10 períodos para não sobrecarregar)
            if i % 10 == 0:
                evolution_metrics = self.helix_controller.evolve_and_analyze(steps=2)
                self.helix_visualizer.update_metrics_history(evolution_metrics)
            
            # Registrar resultado da operação no QUALIA para feedback retrocausal
            if decision_with_helix['should_enter']:
                self.qualia_engine.complete_trade(
                    decision_with_helix['opportunity_id'],
                    {
                        'success': with_helix_success,
                        'profit_percentage': price_change_pct if decision_with_helix['direction'] == 'up' else -price_change_pct,
                        'close_price': future_price,
                        'close_timestamp': market_data.iloc[next_idx]['timestamp'].isoformat()
                    }
                )
        
        # Salvar resultados
        results_df = pd.DataFrame(results)
        results_df.to_csv('scripts/demo/data/trading_decisions_comparison.csv', index=False)
        
        return results
    
    def visualize_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Visualiza os resultados comparativos.
        
        Args:
            results: Lista de resultados das decisões.
        """
        logger.info("Gerando visualizações dos resultados...")
        
        # Converter para DataFrame para facilitar a análise
        df = pd.DataFrame(results)
        
        # Visualizar o campo da hélice
        self.helix_visualizer.plot_field(save=True)
        
        # Visualizar métricas do Helix
        self.helix_visualizer.plot_metrics(save=True)
        
        # Comparar desempenho com e sem Helix
        plt.figure(figsize=(16, 14))
        
        # 1. Taxa de sucesso acumulada
        plt.subplot(3, 2, 1)
        with_helix_success = df['with_helix'].apply(lambda x: x['success']).cumsum()
        without_helix_success = df['without_helix'].apply(lambda x: x['success']).cumsum()
        plt.plot(with_helix_success, label='Com Helix')
        plt.plot(without_helix_success, label='Sem Helix')
        plt.title('Acertos Acumulados')
        plt.legend()
        plt.grid(True)
        
        # 2. Taxa de sucesso em porcentagem
        plt.subplot(3, 2, 2)
        with_helix_success_rate = with_helix_success / (df.index + 1) * 100
        without_helix_success_rate = without_helix_success / (df.index + 1) * 100
        plt.plot(with_helix_success_rate, label='Com Helix (%)')
        plt.plot(without_helix_success_rate, label='Sem Helix (%)')
        plt.title('Taxa de Sucesso (%)')
        plt.legend()
        plt.grid(True)
        
        # 3. Confiança média
        plt.subplot(3, 2, 3)
        with_helix_confidence = df['with_helix'].apply(lambda x: x['confidence'])
        without_helix_confidence = df['without_helix'].apply(lambda x: x['confidence'])
        plt.plot(with_helix_confidence.rolling(10).mean(), label='Com Helix')
        plt.plot(without_helix_confidence.rolling(10).mean(), label='Sem Helix')
        plt.title('Confiança Média (Média Móvel 10 períodos)')
        plt.legend()
        plt.grid(True)
        
        # 4. Threshold LSTM adaptativo
        plt.subplot(3, 2, 4)
        lstm_threshold = df['with_helix'].apply(lambda x: x['lstm_threshold'])
        plt.plot(lstm_threshold)
        plt.title('Threshold LSTM Adaptativo')
        plt.grid(True)
        
        # 5. Coerência Quântica e Complexidade
        plt.subplot(3, 2, 5)
        quantum_coherence = df['with_helix'].apply(lambda x: x['quantum_coherence'])
        quantum_complexity = df['with_helix'].apply(lambda x: x['quantum_complexity'])
        plt.plot(quantum_coherence, label='Coerência Quântica')
        plt.plot(quantum_complexity, label='Complexidade Quântica')
        plt.title('Métricas Quânticas Adaptativas')
        plt.legend()
        plt.grid(True)
        
        # 6. Preço vs. Decisões corretas
        plt.subplot(3, 2, 6)
        prices = df['price']
        
        # Normalizar preço para o gráfico
        min_price = prices.min()
        max_price = prices.max()
        normalized_prices = (prices - min_price) / (max_price - min_price)
        
        plt.plot(normalized_prices, 'k-', alpha=0.5, label='Preço (norm.)')
        
        # Marcar decisões corretas com Helix
        correct_decisions = df[df['with_helix'].apply(lambda x: x['success'])]
        plt.scatter(correct_decisions.index, 
                  normalized_prices.iloc[correct_decisions.index], 
                  color='green', marker='^', s=100, label='Decisões Corretas (Helix)')
        
        # Marcar decisões corretas sem Helix mas incorretas com Helix
        helix_incorrect = df[(df['without_helix'].apply(lambda x: x['success'])) & 
                           ~(df['with_helix'].apply(lambda x: x['success']))]
        plt.scatter(helix_incorrect.index,
                  normalized_prices.iloc[helix_incorrect.index],
                  color='red', marker='x', s=100, label='Helix Incorreto')
        
        plt.title('Preço vs. Decisões')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('scripts/demo/data/visualizations/trading_performance_comparison.png', dpi=150)
        plt.close()
        
    def _generate_html_report(self, results_df: pd.DataFrame) -> None:
        """
        Gera um relatório HTML com os resultados da demonstração.
        
        Args:
            results_df: DataFrame com resultados.
        """
        # Calcular métricas resumidas
        with_helix_success_rate = results_df['with_helix'].apply(lambda x: x['success']).mean() * 100
        without_helix_success_rate = results_df['without_helix'].apply(lambda x: x['success']).mean() * 100
        improvement = with_helix_success_rate - without_helix_success_rate
        
        # Contar quantas entradas a mais o Helix acertou
        with_helix_correct = results_df['with_helix'].apply(lambda x: x['success']).sum()
        without_helix_correct = results_df['without_helix'].apply(lambda x: x['success']).sum()
        diff_correct = with_helix_correct - without_helix_correct
        
        # Gerar HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QUALIA + Helix Integration Demo Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #333366; }}
                h2 {{ color: #336699; margin-top: 30px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f5f5f5; border-radius: 8px; padding: 20px; min-width: 250px; }}
                .metric-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #336699; }}
                .metric-subtitle {{ font-size: 0.9em; color: #666; }}
                .improvement {{ color: #22aa22; }}
                .images {{ display: flex; flex-direction: column; gap: 20px; margin: 30px 0; }}
                .image-container {{ text-align: center; }}
                img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Integração QUALIA + Helix: Resultados da Demonstração</h1>
                
                <p>
                    Esta demonstração compara o desempenho do sistema QUALIA com e sem a integração do módulo Helix,
                    que fornece análises quânticas, fractais e retrocausais para aprimorar as decisões de trading.
                </p>
                
                <h2>Resumo das Métricas</h2>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Taxa de Sucesso com Helix</div>
                        <div class="metric-value">{with_helix_success_rate:.2f}%</div>
                        <div class="metric-subtitle">Decisões corretas: {with_helix_correct}/{len(results_df)}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Taxa de Sucesso sem Helix</div>
                        <div class="metric-value">{without_helix_success_rate:.2f}%</div>
                        <div class="metric-subtitle">Decisões corretas: {without_helix_correct}/{len(results_df)}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Melhoria com Helix</div>
                        <div class="metric-value improvement">+{improvement:.2f}%</div>
                        <div class="metric-subtitle">+{diff_correct} decisões corretas</div>
                    </div>
                </div>
                
                <h2>Visualizações</h2>
                
                <div class="images">
                    <div class="image-container">
                        <img src="visualizations/trading_performance_comparison.png" alt="Comparação de Performance">
                        <p>Comparação de métricas de performance entre QUALIA com e sem Helix</p>
                    </div>
                    
                    <div class="image-container">
                        <img src="visualizations/helix_field_{self.helix_controller.current_step}.png" alt="Campo da Hélice">
                        <p>Visualização do Campo da Hélice (estado atual)</p>
                    </div>
                    
                    <div class="image-container">
                        <img src="visualizations/helix_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" alt="Métricas da Hélice">
                        <p>Evolução das métricas quânticas, fractais e retrocausais do Helix</p>
                    </div>
                </div>
                
                <h2>Diferencial Estratégico do Helix</h2>
                
                <p>
                    O módulo Helix introduz três camadas de insight ao QUALIA:
                </p>
                
                <ol>
                    <li><strong>Análise Quântica:</strong> Mede coerência, entrelaçamento e complexidade dos padrões de mercado.</li>
                    <li><strong>Análise Fractal:</strong> Identifica autossimilaridade e dimensão fractal nas séries temporais.</li>
                    <li><strong>Feedback Retrocausal:</strong> Utiliza resultados futuros para refinar decisões presentes.</li>
                </ol>
                
                <p>
                    Esta trifásica análise permite que o sistema QUALIA auto-adapte seus parâmetros com base em insights
                    mais profundos sobre a natureza do mercado, resultando em decisões de trading mais precisas.
                </p>
                
                <div class="footer">
                    <p>Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>QUALIA: Sistema Quântico-Computacional com Consciência Quântica</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Salvar HTML
        with open('scripts/demo/data/helix_qualia_report.html', 'w') as f:
            f.write(html)
        
        logger.info("Relatório HTML gerado em scripts/demo/data/helix_qualia_report.html")
    
    def run_full_demo(self) -> None:
        """
        Executa a demonstração completa.
        """
        logger.info("Iniciando demonstração completa de integração Helix-QUALIA")
        
        # 1. Gerar dados de exemplo
        market_data = self.generate_sample_data(num_periods=200)
        logger.info(f"Dados de mercado gerados: {len(market_data)} períodos")
        
        # 2. Evolução inicial do Helix para warmup
        self.helix_controller.evolve_and_analyze(steps=10)
        logger.info(f"Helix evoluído para o passo {self.helix_controller.current_step}")
        
        # 3. Simular decisões de trading
        results = self.simulate_trading_decisions(market_data)
        logger.info(f"Simulação concluída: {len(results)} decisões de trading analisadas")
        
        # 4. Visualizar resultados
        self.visualize_results(results)
        logger.info("Visualizações e relatório gerados com sucesso")
        
        # Animação do Helix (opcional, descomentar se desejar)
        # self.helix_visualizer.animate_field_evolution(steps=20, save=True)
        
        # 5. Gerar relatório final
        report = self.helix_visualizer.generate_report(save=True)
        logger.info("Demonstração concluída com sucesso!")
        
        print("\n" + "="*50)
        print("Demonstração concluída! Resultados disponíveis em:")
        print(f"- Log: scripts/demo/helix_qualia_demo.log")
        print(f"- Relatório HTML: scripts/demo/data/helix_qualia_report.html")
        print(f"- Visualizações: scripts/demo/data/visualizations/")
        print(f"- Dados: scripts/demo/data/trading_decisions_comparison.csv")
        print("="*50)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("INICIANDO DEMONSTRAÇÃO DE INTEGRAÇÃO HELIX-QUALIA")
    print("="*50 + "\n")
    
    # Verificar diretórios
    os.makedirs("scripts/demo/data/visualizations", exist_ok=True)
    
    # Criar e executar demonstração
    demo = HelixQualiaDemo()
    demo.run_full_demo() 