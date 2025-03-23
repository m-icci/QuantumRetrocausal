#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Integrado de Trading Quântico QUALIA
============================================

Este script unifica os seguintes módulos em uma interface coerente:
  - QUALIA Engine (WAVE-Helix): Estratégia adaptativa que integra LSTM, QCNN e Performance Metrics
  - Sistema Integrado de Scalping Quântico: Detecção de oportunidades de scalping
  - Sistema de Arbitragem Dual-Exchange: Análise de oportunidades entre exchanges
  - Quantum Cosmological Integrator: Simulação da evolução do campo quântico
  
A integração proporciona:
  1. Interface unificada para trading real, simulado e backtesting
  2. Reutilização de componentes entre todos os modos de operação
  3. Consistência no processamento de dados e análise quântica
  4. Sistema coerente para análise de resultados
"""

import os
import sys
import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configuração de diretórios
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Importações do sistema integrado
from quantum_trading.integrated_quantum_scalping import IntegratedQuantumScalping
from quantum_trading.integration.exchange import Exchange
from quantum_trading.integration.quantum_trading_system import QuantumTradingSystem

# Importação condicional para componentes de backtest
try:
    from quantum_trading.services.state_sync_service import StateSyncService, ComponentType, StateType
    from quantum_trading.services.semantic_interpreter import SemanticInterpreter
    from quantum_trading.integration.qualia_core_engine import QUALIAEngine
except ImportError as e:
    logging.warning(f"Alguns componentes avançados não estão disponíveis: {e}")

# Gráficos e visualização
try:
    from cosmic_dance import plot_history
except ImportError:
    logging.warning("Módulo de visualização 'cosmic_dance' não disponível")

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_trading_system.log")
    ]
)
logger = logging.getLogger("QuantumTradingSystem")

class DummyExchange(Exchange):
    """Simulador de exchange para testes"""
    def __init__(self, exchange_id: str):
        super().__init__(exchange_id, maker_fee=0.0016, taker_fee=0.0024)
        self.balance = 100000.0

    def get_balance(self, asset: str) -> float:
        return self.balance

    def get_ohlcv(self, pair: str, interval: str, window: int) -> List[List[float]]:
        now = int(time.time() * 1000)
        return [[now, 100, 110, 90, 105, 1000] for _ in range(window)]

    def get_price(self, pair: str) -> float:
        return 105 + 5 * np.sin(time.time()/60)  # Simula variação de preço

    def create_order(self, pair: str, quantity: float, side: str) -> Dict[str, Any]:
        price = self.get_price(pair)
        if side == 'buy':
            self.balance += quantity * price
            return {"id": f"buy-{time.time()}", "price": price}
        else:
            self.balance -= quantity * price
            return {"id": f"sell-{time.time()}", "price": price}

class UnifiedTradingSystem:
    """
    Sistema Unificado de Trading Quântico QUALIA
    
    Provê uma interface coerente para trading real, simulado e backtesting,
    utilizando os mesmos componentes internos para todos os modos de operação.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o sistema unificado de trading
    
    Args:
        config_path: Caminho para arquivo de configuração
    """
        # Carrega configuração
        self.config = self._load_config(config_path)
        
        # Componentes do sistema
        self.integrated_scalping = None
        self.state_sync = None
        self.semantic_interpreter = None
        
        # Estado de execução
        self.running = False
        self.mode = self.config.get("system", {}).get("mode", "simulated")
        
        # Diretórios de dados e saída
        self.data_dir = Path(self.config.get("system", {}).get("data_dir", "./data"))
        self.output_dir = Path(self.config.get("system", {}).get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializa configuração de logging
        self._setup_logging()
        
        logger.info(f"Sistema Unificado inicializado em modo: {self.mode}")
    
    def _setup_logging(self):
        """Configura logging avançado"""
        logs_dir = Path(self.config.get("system", {}).get("logs_dir", "./logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Adiciona handler para arquivo específico do modo
        log_file = logs_dir / f"{self.mode}_trading.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        logger.info(f"Logs serão salvos em: {log_file}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Carrega configuração unificada do sistema
        
        Args:
            config_path: Caminho para arquivo de configuração
            
        Returns:
            Configuração mesclada com valores padrão
        """
            # Configuração padrão
        default_config = {
            "system": {
                "mode": "simulated",  # 'real', 'simulated', 'backtest', 'dual_backtest'
                "data_dir": "./data",
                "output_dir": "./output",
                "logs_dir": "./logs",
                "metrics_dir": "./metrics"
            },
            "trading": {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "initial_balance": 10000,
                "exchanges": ["kucoin", "kraken"]
            },
            "risk": {
                "max_position_size": 0.1,
                "max_daily_loss": 0.02,
                "max_drawdown": 0.05
            },
            "scalping": {
                "min_profit_threshold": 0.0005,
                "max_loss_threshold": 0.0003,
                "max_position_time": 300,
                "min_volume_threshold": 1000,
                "max_spread_threshold": 0.0002,
                "exchange_fee": 0.0004,
                "slippage": 0.0001,
                "min_trade_size": 0.001
            },
            "qualia": {
                "lstm_model_path": "models/lstm_predictor.h5",
                "quantum_layers": 8,
                "adaptive_threshold": 0.7,
                "max_exposure": 0.3,
                "enable_helix": True
            },
            "helix": {
                "dimensions": 64,
                "num_qubits": 8,
                "phi": 0.618,
                "temperature": 0.2,
                "batch_size": 256,
                "tau": 7
            },
            "cosmo": {
                "quantum_gravity_coupling": 0.1,
                "entanglement_strength": 0.5,
                "hubble_sensitivity": 0.05
            },
            "backtest": {
                "cycles": 30,
                "window_size": 50,
                "pair": "BTC/USDT",
                "exchanges": ["kraken", "kucoin"],
                "start_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                "end_date": datetime.now().strftime('%Y-%m-%d'),
                "data_files": {
                    "kraken": "KRAKEN_BTCUSDT.csv",
                    "kucoin": "KUCOIN_BTCUSDT.csv"
                }
            },
            "field_dimensions": 8,
            "coherence_threshold": 0.45,
            "resonance_threshold": 0.4,
            "buffer_size": 1000
        }
        
        # Se um arquivo de configuração foi fornecido, carrega-o e mescla com os padrões
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Mescla configurações recursivamente
                for section, values in config.items():
                    if isinstance(values, dict) and section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
            except Exception as e:
                logger.error(f"Erro ao carregar arquivo de configuração: {e}")
        
        return default_config
    
    async def initialize(self) -> bool:
        """
        Inicializa os componentes do sistema de acordo com o modo selecionado
        
        Returns:
            True se inicialização bem-sucedida, False caso contrário
        """
        try:
            # Inicializa componentes comuns a todos os modos
            if self.mode in ["dual_backtest", "backtest"]:
                # Inicializa componentes de backtest
                self.state_sync = StateSyncService(
                    storage_path=self.config.get("state_sync", {}).get("storage_path", "state_storage"),
                    enable_persistence=self.config.get("state_sync", {}).get("enable_persistence", True)
                )
                await self.state_sync.start()
                
                self.semantic_interpreter = SemanticInterpreter(
                    reports_dir=self.config.get("semantic_interpreter", {}).get("reports_dir", "reports"),
                    verbose=self.config.get("semantic_interpreter", {}).get("verbose", True)
                )
            
            # Inicializa o sistema integrado de scalping
            if self.mode == "dual_backtest":
                # Para backtest dual, só inicializamos o QUALIAEngine
                self.qualia_engine = QUALIAEngine(
                    config=self.config.get("qualia", {}),
                    enable_helix=self.config.get("qualia", {}).get("enable_helix", True),
                    helix_config=self.config.get("helix", {})
                )
            else:
                # Para todos os outros modos, inicializamos o IntegratedQuantumScalping
                self.integrated_scalping = IntegratedQuantumScalping(self.config)
            
            # Carrega dados históricos para modos de backtest
            if self.mode in ["backtest", "dual_backtest"]:
                if not await self._load_historical_data():
                    logger.error("Falha ao carregar dados históricos.")
                    return False
            
            logger.info(f"Sistema inicializado com sucesso no modo {self.mode}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema: {e}", exc_info=True)
            return False
    
    async def _load_historical_data(self) -> bool:
        """
        Carrega dados históricos para modos de backtest
        
        Returns:
            True se carregamento bem-sucedido, False caso contrário
        """
        try:
            data_files = self.config.get("backtest", {}).get("data_files", {})
            exchanges = self.config.get("backtest", {}).get("exchanges", [])
            
            if len(exchanges) < 1:
                logger.error("Configuração de backtest requer pelo menos uma exchange")
                return False
            
            self.exchange_data = {}
            
            for exchange in exchanges:
                if exchange not in data_files:
                    logger.error(f"Arquivo de dados não especificado para {exchange}")
                    return False
                
                file_path = self.data_dir / data_files[exchange]
                
                if not file_path.exists():
                    logger.error(f"Arquivo de dados não encontrado: {file_path}")
                    return False
                
                # Carrega dados
                df = pd.read_csv(file_path)
                
                # Normaliza nomes de colunas
                df.columns = [col.lower() for col in df.columns]
                
                # Converte timestamp para datetime se necessário
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                self.exchange_data[exchange] = df
                logger.info(f"Dados históricos carregados: {len(df)} candles de {exchange}")
            
            # Se estamos no modo dual_backtest, publicamos os dados no StateSyncService
            if self.mode == "dual_backtest" and self.state_sync:
                await self.state_sync.publish_state(
                    component_type=ComponentType.QUALIA_ENGINE,
                    state_type=StateType.MARKET_DATA,
                    data={
                        "exchanges": exchanges,
                        "rows": {ex: len(df) for ex, df in self.exchange_data.items()}
                    },
                    source_id="backtest"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados históricos: {e}", exc_info=True)
            return False
    
    async def run(self):
        """
        Executa o sistema no modo configurado
        
        Returns:
            Resultados da execução, formatados de acordo com o modo
        """
        self.running = True
        
        try:
            if self.mode == "real":
                return await self._run_real_trading()
            elif self.mode == "simulated":
                return await self._run_simulated_trading()
            elif self.mode == "backtest":
                return await self._run_backtest()
            elif self.mode == "dual_backtest":
                return await self._run_dual_backtest()
            else:
                logger.error(f"Modo desconhecido: {self.mode}")
                return {"error": f"Modo desconhecido: {self.mode}"}
        except KeyboardInterrupt:
            logger.info("Sistema interrompido pelo usuário")
            return {"status": "interrupted"}
        except Exception as e:
            logger.error(f"Erro na execução: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self.running = False
            await self.stop()
    
    async def _run_real_trading(self):
        """
        Executa trading real usando o sistema integrado de scalping
        
        Returns:
            Status da execução
        """
        # Verifica credenciais apenas para modo real
        if not os.getenv('EXCHANGE_API_KEY') or not os.getenv('EXCHANGE_API_SECRET'):
            logger.error("Variáveis de ambiente não configuradas para trading real")
            return {"error": "Credenciais de API não configuradas"}
            
        logger.info("Iniciando trading real!")
        
        try:
            # Executa o sistema integrado
            await self.integrated_scalping.run()
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"Erro no trading real: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _run_simulated_trading(self):
        """
        Executa trading simulado usando o sistema integrado de scalping
        
        Returns:
            Resultados da simulação
        """
        logger.info("Iniciando simulação de trading")
        
        # Configuração de simulação
        start_date = self.config.get("trading", {}).get("start_date", 
                                    (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = self.config.get("trading", {}).get("end_date", 
                                  datetime.now().strftime('%Y-%m-%d'))
        initial_balance = self.config.get("trading", {}).get("initial_balance", 10000)
        
        logger.info(f"Simulação de {start_date} até {end_date}")
        logger.info(f"Saldo inicial: ${initial_balance:,.2f}")
        
        try:
            # Executa o sistema integrado
            await self.integrated_scalping.run()
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"Erro na simulação: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _run_backtest(self):
        """
        Executa backtest usando o sistema integrado de scalping
        
        Returns:
            Resultados do backtest
        """
        logger.info("Iniciando backtest com sistema integrado de scalping")
        
        # Configuração de backtest
        start_date = self.config.get("backtest", {}).get("start_date")
        end_date = self.config.get("backtest", {}).get("end_date")
        initial_capital = self.config.get("backtest", {}).get("initial_balance", 10000.0)
        
        logger.info(f"Backtest de {start_date} até {end_date}")
        logger.info(f"Capital inicial: ${initial_capital:,.2f}")
        
        try:
            # Executa o backtest
            backtest_results = await self.integrated_scalping.run_backtest(
                market_data=self.exchange_data,
                initial_capital=initial_capital
            )
            
            # Salva resultados
            results_file = self.output_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(backtest_results, f, indent=2)
                
            logger.info(f"Resultados do backtest salvos em {results_file}")
            
            # Plota resultados se o módulo estiver disponível
            if 'plot_history' in globals():
                plot_path = self.output_dir / 'backtest_plots'
                plot_path.mkdir(exist_ok=True)
                plot_history(backtest_results, plot_path)
                logger.info(f"Gráficos do backtest salvos em {plot_path}")
            
            return backtest_results
        except Exception as e:
            logger.error(f"Erro no backtest: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _run_dual_backtest(self):
        """
        Executa backtest de arbitragem entre duas exchanges
        
        Returns:
            Resultados do backtest dual
        """
        logger.info("Iniciando backtest de arbitragem entre duas exchanges")
        
        # Configuração de backtest
        cycles = self.config.get("backtest", {}).get("cycles", 30)
        window_size = self.config.get("backtest", {}).get("window_size", 50)
        
        logger.info(f"Executando {cycles} ciclos com janela de {window_size}")
        
        try:
            # Define variáveis de ciclo
            self.current_cycle = 0
            self.results = []
            
            while self.running and self.current_cycle < cycles:
                result = await self._run_dual_backtest_cycle()
                
                # Verifica se chegamos ao fim dos dados
                if result.get("end_of_data", False):
                    logger.info("Backtest finalizado (fim dos dados históricos).")
                    break
                
                # Adiciona resultado à lista
                self.results.append(result)
                self.current_cycle += 1
            
            # Gera relatório de performance
            report = self._generate_dual_backtest_report()
            
            return {
                "summary": report,
                "results": self.results
            }
        except Exception as e:
            logger.error(f"Erro no backtest dual: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _run_dual_backtest_cycle(self) -> Dict[str, Any]:
        """
        Executa um ciclo do backtest dual
        
        Returns:
            Resultado do ciclo
        """
        # Calcula índices para o slice atual
        start_idx = self.current_cycle * self.config.get("backtest", {}).get("window_size", 50)
        end_idx = start_idx + self.config.get("backtest", {}).get("window_size", 50)
        
        exchanges = self.config.get("backtest", {}).get("exchanges", [])
        if len(exchanges) < 2:
            raise ValueError("Backtest dual requer duas exchanges")
        
        exchange_a = exchanges[0]
        exchange_b = exchanges[1]
        
        # Verifica se temos dados suficientes
        total_rows = min(len(self.exchange_data[exchange_a]), len(self.exchange_data[exchange_b]))
        if end_idx >= total_rows:
            logger.info("Fim dos dados históricos.")
            return {"end_of_data": True}
        
        # Obtém slices dos dados históricos
        slice_a = self.exchange_data[exchange_a].iloc[start_idx:end_idx]
        slice_b = self.exchange_data[exchange_b].iloc[start_idx:end_idx]
        
        # Prepara dados para o QUALIAEngine
        market_data = self._prepare_market_features(slice_a, slice_b)
        candle_data = slice_b[['open', 'high', 'low', 'close', 'volume']].values
        
        # Publica dados do ciclo atual no StateSyncService
        await self.state_sync.publish_state(
            component_type=ComponentType.QUALIA_ENGINE,
            state_type=StateType.MARKET_DATA,
            data={
                "cycle": self.current_cycle,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "exchange_a": exchange_a,
                "exchange_b": exchange_b,
                "market_data": market_data.to_dict(orient='records')[0]
            },
            source_id="backtest"
        )
        
        logger.info(f"Executando ciclo {self.current_cycle + 1}/{self.config.get('backtest', {}).get('cycles', 30)}")
        
        # Processa oportunidade com o QUALIAEngine
        pair = self.config.get("backtest", {}).get("pair", "BTC/USDT")
        result = self.qualia_engine.process_opportunity(
            market_data=market_data,
            candle_data=candle_data,
            pair=pair,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            spread=market_data['spread'].iloc[0]
        )
        
        # Adiciona informações do ciclo ao resultado
        result["cycle"] = self.current_cycle
        result["timestamp"] = datetime.now().isoformat()
        result["pair"] = pair
        
        # Gera explicação semântica
        explanation = self.semantic_interpreter.generate_explanation(
            decision_data={
                **result,
                "market_data": market_data.iloc[0].to_dict()
            }
        )
        
        # Adiciona explicação ao resultado
        result["explanation"] = explanation["rationale"]
        result["report_text"] = explanation["report_text"]
        
        # Publica resultado no StateSyncService
        await self.state_sync.publish_state(
            component_type=ComponentType.QUALIA_ENGINE,
            state_type=StateType.DECISION,
            data=result,
            source_id="backtest"
        )
        
        # Exibe resultado resumido
        should_enter = result.get("should_enter", False)
        confidence = result.get("confidence", 0.0)
        helix_coherence = result.get("helix_insights", {}).get("quantum_metrics", {}).get("coherence", "N/A")
        lstm_confidence = result.get("lstm_confidence", "N/A")
        
        logger.info(f"Decisão: {'ENTRAR' if should_enter else 'NÃO ENTRAR'} (Confiança: {confidence:.4f})")
        logger.info(f"Coerência Helix: {helix_coherence}, Confiança LSTM: {lstm_confidence}")
        logger.info(f"Explicação: {result['explanation']}")
        logger.info("---")
        
        # Finaliza o ciclo atual no QUALIAEngine
        self.qualia_engine.end_current_cycle()
        
        return result
    
    def _prepare_market_features(self, slice_a: pd.DataFrame, slice_b: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features de mercado para análise
        
        Args:
            slice_a: Slice da exchange A
            slice_b: Slice da exchange B
            
        Returns:
            DataFrame com features de mercado
        """
        # Cria DataFrame base
        features = pd.DataFrame(index=[0])
        
        # Calcula spread real entre as exchanges
        features['spread'] = abs(slice_a['close'].iloc[-1] - slice_b['close'].iloc[-1]) / slice_b['close'].iloc[-1]
        
        # Volume de cada exchange
        features['volume_a'] = slice_a['volume'].iloc[-1]
        features['volume_b'] = slice_b['volume'].iloc[-1]
        
        # Features derivadas da série temporal
        features['volatility_a'] = slice_a['high'].iloc[-10:].mean() / slice_a['low'].iloc[-10:].mean() - 1
        features['volatility_b'] = slice_b['high'].iloc[-10:].mean() / slice_b['low'].iloc[-10:].mean() - 1
        
        # Features sintéticas para exemplificar
        features['entropy'] = np.random.uniform(0.1, 0.9)
        features['fractal_dimension'] = np.random.uniform(1.1, 1.9)
        
        return features
    
    def _generate_dual_backtest_report(self) -> Dict[str, Any]:
        """
        Gera relatório final do backtest dual
        
        Returns:
            Dicionário com métricas agregadas
        """
        # Calcula métricas agregadas
        total_opportunities = sum(1 for r in self.results if r.get("should_enter", False))
        avg_confidence = np.mean([r.get("confidence", 0) for r in self.results])
        
        # Métricas do Helix
        helix_coherence = [r.get("helix_insights", {}).get("quantum_metrics", {}).get("coherence", np.nan) 
                          for r in self.results]
        helix_coherence = [h for h in helix_coherence if not np.isnan(h)]
        avg_coherence = np.mean(helix_coherence) if helix_coherence else 0
        
        # Métricas do LSTM
        lstm_confidence = [r.get("lstm_confidence", np.nan) for r in self.results]
        lstm_confidence = [l for l in lstm_confidence if not np.isnan(l)]
        avg_lstm_confidence = np.mean(lstm_confidence) if lstm_confidence else 0
        
        # Cria relatório
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cycles": len(self.results),
            "total_opportunities": total_opportunities,
            "opportunity_rate": total_opportunities / len(self.results) if self.results else 0,
            "avg_confidence": float(avg_confidence),
            "avg_helix_coherence": float(avg_coherence),
            "avg_lstm_confidence": float(avg_lstm_confidence),
            "exchange_a": self.config.get("backtest", {}).get("exchanges", [])[0],
            "exchange_b": self.config.get("backtest", {}).get("exchanges", [])[1],
            "pair": self.config.get("backtest", {}).get("pair", "BTC/USDT")
        }
        
        # Salva relatório
        report_path = self.output_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Relatório de backtest salvo em {report_path}")
        
        # Exibe resumo
        logger.info("\n=== RESUMO DO BACKTEST ===")
        logger.info(f"Total de ciclos: {len(self.results)}")
        logger.info(f"Oportunidades detectadas: {total_opportunities} ({report['opportunity_rate']:.2%})")
        logger.info(f"Confiança média: {avg_confidence:.4f}")
        logger.info(f"Coerência Helix média: {avg_coherence:.4f}")
        logger.info(f"Confiança LSTM média: {avg_lstm_confidence:.4f}")
        
        return report
    
    async def stop(self):
        """Para todos os componentes do sistema"""
        try:
            # Para o sistema integrado de scalping se estiver ativo
            if self.integrated_scalping:
                await self.integrated_scalping.stop()
            
            # Para o StateSyncService se estiver ativo
            if self.state_sync:
                await self.state_sync.stop()
                
            logger.info("Sistema parado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao parar sistema: {e}", exc_info=True)

async def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema Integrado de Trading QUALIA')
    parser.add_argument('--mode', choices=['real', 'simulated', 'backtest', 'dual_backtest'],
                      default='simulated', help='Modo de operação')
    parser.add_argument('--config', type=str, help='Arquivo de configuração')
    parser.add_argument('--start-date', type=str, help='Data inicial para backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Data final para backtesting (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, help='Capital inicial para simulação/backtesting')
    parser.add_argument('--data-dir', type=str, help='Diretório com dados históricos')
    parser.add_argument('--output-dir', type=str, help='Diretório para resultados')
    parser.add_argument('--cycles', type=int, help='Número de ciclos para dual_backtest')
    parser.add_argument('--pair', type=str, help='Par de trading (ex: BTC/USDT)')
    
    args = parser.parse_args()
    
    # Carrega configuração
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return
    else:
        config = {}
    
    # Sobrescreve configurações com argumentos da linha de comando
    if args.mode:
        if "system" not in config:
            config["system"] = {}
        config["system"]["mode"] = args.mode
        
    if args.data_dir:
        if "system" not in config:
            config["system"] = {}
        config["system"]["data_dir"] = args.data_dir
        
    if args.output_dir:
        if "system" not in config:
            config["system"] = {}
        config["system"]["output_dir"] = args.output_dir
        
    if args.start_date:
        if "backtest" not in config:
            config["backtest"] = {}
        config["backtest"]["start_date"] = args.start_date
        
    if args.end_date:
        if "backtest" not in config:
            config["backtest"] = {}
        config["backtest"]["end_date"] = args.end_date
        
    if args.initial_capital:
        if "backtest" not in config:
            config["backtest"] = {}
        config["backtest"]["initial_balance"] = args.initial_capital
        
    if args.cycles:
        if "backtest" not in config:
            config["backtest"] = {}
        config["backtest"]["cycles"] = args.cycles
        
    if args.pair:
        if "backtest" not in config:
            config["backtest"] = {}
        config["backtest"]["pair"] = args.pair
    
    # Inicializa o sistema unificado
    system = UnifiedTradingSystem(args.config)
    
    try:
        # Sobrescreve o modo se especificado na linha de comando
        if args.mode:
            system.mode = args.mode
            
        # Inicializa e executa o sistema
        if await system.initialize():
            results = await system.run()
            
            if system.mode in ["backtest", "dual_backtest"] and "summary" in results:
                logger.info(f"Resultados do backtest: {json.dumps(results['summary'], indent=2)}")
        else:
            logger.error("Falha ao inicializar o sistema")
            
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro na execução: {e}", exc_info=True)
    finally:
        await system.stop()

if __name__ == "__main__":
    # Configura política de loop de eventos para Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Executa o sistema
    asyncio.run(main()) 