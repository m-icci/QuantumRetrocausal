"""
Quantum Trading System

Sistema consolidado de trading quântico que integra:
1. Processamento holográfico de padrões
2. Análise quântica de mercado
3. Execução de ordens via KuCoin
4. Otimização de portfólio
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio

from qualia.core.quantum_trader import QuantumTrader
from qualia_core.Qualia.integration.kucoin_quantum_bridge import KuCoinQuantumBridge
from qualia_core.Qualia.trading.engine.holographic_engine import (
    HolographicTradingEngine, 
    HolographicPattern,
    TradingDecision
)
from qualia_core.Qualia.trading.processor.quantum_trading_processor import QuantumTradingProcessor
from qualia_core.Qualia.trading.optimization.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
from qualia_core.Qualia.trading.metrics.advanced_metrics import QuantumMarketMetrics
from qualia_core.Qualia.trading.risk.quantum_risk_manager import QuantumRiskManager, RiskAssessment
from qualia_core.Qualia.trading.position.quantum_position_manager import (
    QuantumPositionManager,
    PositionInfo,
    PortfolioState
)
from qualia_core.Qualia.consciousness import MICCIConsciousness
from qualia_core.Qualia.consciousness.holographic_core import (
    HolographicState
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingState:
    """Estado do sistema de trading"""
    portfolio_state: PortfolioState
    market_state: HolographicState
    quantum_coherence: float
    entanglement_metrics: Dict[str, float]
    active_patterns: List[HolographicPattern]
    last_decisions: List[TradingDecision]
    risk_assessment: RiskAssessment
    portfolio_status: PortfolioState
    performance_metrics: Dict[str, float]
    performance_history: List[Any] = field(default_factory=list)
    system_metrics: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 0.0,
        'memory_usage': 0.0,
        'latency': 0.0
    })
    timestamp: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = self.timestamp

class QuantumTradingSystem:
    """
    Sistema integrado de trading quântico

    Combina análise holográfica, processamento quântico e 
    execução de ordens em um único sistema coerente.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o sistema de trading quântico

        Args:
            config: Configuração do sistema
                - api_key: Chave API KuCoin
                - api_secret: Segredo API KuCoin
                - api_passphrase: Senha API KuCoin
                - trading_pairs: Lista de pares de trading
                - risk_tolerance: Tolerância a risco (0-1)
                - quantum_params: Parâmetros quânticos
                - holographic_params: Parâmetros holográficos
        """
        self.config = config
        self.trading_pairs = config['trading_pairs']

        # Componentes principais
        self.consciousness = MICCIConsciousness(config.get('consciousness', {}))
        self.holographic_field = HolographicField(
            dimensions=64,  # Dimensões do campo holográfico
            memory_capacity=1000  # Capacidade de memória
        )

        # Componentes de trading
        self.quantum_trader = QuantumTrader(
            n_assets=len(self.trading_pairs),
            risk_tolerance=config.get('risk_tolerance', 0.5)
        )

        self.kucoin_bridge = KuCoinQuantumBridge(config)

        self.trading_engine = HolographicTradingEngine({
            'consciousness': self.consciousness,
            'holographic_field': self.holographic_field,
            'trading_pairs': self.trading_pairs,
            **config.get('trading_engine', {})
        })

        # Gerenciamento de risco e posições
        self.risk_manager = QuantumRiskManager(config.get('risk_management', {
            'max_risk': 0.8,
            'volatility_threshold': 0.6,
            'coherence_threshold': 0.4,
            'risk_threshold': 0.7
        }))

        self.position_manager = QuantumPositionManager(config.get('position_management', {
            'initial_capital': config.get('initial_capital', 10000),
            'max_position_size': config.get('max_position_size', 1000),
            'min_position_size': config.get('min_position_size', 10),
            'position_step_size': config.get('position_step_size', 1),
            'max_positions': config.get('max_positions', 10)
        }))

        self.portfolio_optimizer = QuantumPortfolioOptimizer(
            n_assets=len(self.trading_pairs),
            risk_tolerance=config.get('risk_tolerance', 0.5),
            quantum_params=config.get('quantum_params', {})
        )

        self.processor = QuantumTradingProcessor(
            trading_pairs=self.trading_pairs,
            holographic_field=self.holographic_field
        )

        self.metrics = QuantumMarketMetrics()

        # Estado do sistema
        self.trading_state: Optional[TradingState] = None
        self.last_update: Optional[datetime] = None

    async def initialize(self):
        """Inicializa o sistema e conecta aos websockets"""
        await self.kucoin_bridge.connect()
        await self.subscribe_to_market_data()
        self._initialize_trading_state()

    def _initialize_trading_state(self):
        """Inicializa o estado do sistema de trading"""
        initial_portfolio_state = PortfolioState(
            positions={},
            total_value=self.config.get('initial_capital', 10000),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            quantum_coherence=1.0
        )

        initial_market_state = HolographicState()
        initial_market_state.coherence = 1.0

        self.trading_state = TradingState(
            portfolio_state=initial_portfolio_state,
            market_state=initial_market_state,
            quantum_coherence=1.0,
            entanglement_metrics={
                'resonance': 1.0,
                'field_strength': 1.0,
                'fidelity': 1.0,
                'entropy': 0.0
            },
            active_patterns=[],
            last_decisions=[],
            risk_assessment=RiskAssessment(risk_score=0.0, market_coherence=1.0),
            portfolio_status=initial_portfolio_state,
            performance_metrics={
                'total_return': 0.0,
                'daily_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0
            },
            timestamp=datetime.now()
        )

    async def subscribe_to_market_data(self):
        """Assina dados de mercado via websocket"""
        for pair in self.trading_pairs:
            await self.kucoin_bridge.subscribe_market_data(pair)
            
    async def process_market_data(self, market_data: Dict[str, Any]):
        """
        Processa novos dados de mercado com gerenciamento de risco
        
        1. Atualiza estado do mercado
        2. Detecta padrões holográficos
        3. Avalia risco atual
        4. Gera decisões de trading
        5. Filtra decisões baseado no risco
        6. Otimiza portfólio considerando risco
        7. Executa ordens necessárias com controle de risco
        8. Atualiza posições existentes
        """
        # Atualiza estado do mercado
        market_state = self.processor.process_market_data(market_data)
        
        # Detecta padrões holográficos
        patterns = self.trading_engine.detect_patterns(market_state)
        
        # Avalia risco atual
        risk_assessment = self.risk_manager.assess_risk(
            market_state=market_state,
            portfolio_state=self.trading_state.portfolio_state if self.trading_state else None,
            active_patterns=patterns,
            current_positions={
                symbol: info.size 
                for symbol, info in self.position_manager.positions.items()
            }
        )
        
        # Verifica se deve parar trading
        if self.risk_manager.should_stop_trading(risk_assessment):
            logger.warning("Alto risco detectado - Parando trading")
            await self.close_all_positions()
            return
        
        # Gera decisões de trading
        decisions = self.trading_engine.generate_decisions(
            market_state=market_state,
            patterns=patterns,
            coherence=risk_assessment.market_coherence
        )
        
        # Filtra decisões baseado no risco
        filtered_decisions = self._filter_decisions_by_risk(
            decisions, risk_assessment
        )
        
        # Otimiza portfólio considerando risco
        portfolio_allocation = self.portfolio_optimizer.optimize(
            market_state=market_state,
            decisions=filtered_decisions,
            current_portfolio=self.position_manager.get_portfolio_state()
        )
        
        # Executa ordens necessárias com controle de risco
        if filtered_decisions:
            await self.execute_trading_decisions(
                filtered_decisions, 
                portfolio_allocation,
                risk_assessment
            )
        
        # Atualiza posições existentes
        await self._update_positions(market_data, market_state, risk_assessment)
        
        # Atualiza estado do sistema
        self._update_trading_state(
            market_state=market_state,
            patterns=patterns,
            decisions=filtered_decisions,
            risk_assessment=risk_assessment
        )
        
    def _filter_decisions_by_risk(self,
                                decisions: List[TradingDecision],
                                risk_assessment: RiskAssessment) -> List[TradingDecision]:
        """Filtra decisões de trading baseado no risco"""
        filtered = []
        
        for decision in decisions:
            # Ajusta confiança baseado no risco
            adjusted_confidence = decision.confidence * (1 - risk_assessment.risk_score)
            
            # Filtra baseado na confiança ajustada
            if adjusted_confidence >= self.config.get('min_confidence', 0.7):
                # Ajusta tamanho baseado no risco
                adjusted_size = decision.size * (1 - risk_assessment.risk_score)
                
                filtered.append(TradingDecision(
                    symbol=decision.symbol,
                    action=decision.action,
                    size=adjusted_size,
                    confidence=adjusted_confidence,
                    quantum_coherence=decision.quantum_coherence,
                    supporting_patterns=decision.supporting_patterns,
                    timestamp=decision.timestamp,
                    metadata={
                        **decision.metadata,
                        'risk_score': risk_assessment.risk_score,
                        'original_size': decision.size
                    }
                ))
                
        return filtered
        
    async def _update_positions(self,
                              market_data: Dict[str, Any],
                              market_state: HolographicState,
                              risk_assessment: RiskAssessment):
        """Atualiza posições existentes"""
        for symbol in self.position_manager.positions.keys():
            if symbol in market_data:
                # Atualiza posição com novos dados
                current_price = float(market_data[symbol]['price'])
                
                position = self.position_manager.update_position(
                    symbol=symbol,
                    market_state=market_state,
                    current_price=current_price,
                    risk_assessment=risk_assessment
                )
                
                # Verifica se deve fechar posição
                if position and position.confidence < self.config.get('min_position_confidence', 0.3):
                    await self.close_position(symbol, current_price)
                    
    async def close_position(self, symbol: str, price: float):
        """Fecha uma posição específica"""
        pnl, position = self.position_manager.close_position(symbol, price)
        
        if position:
            # Executa ordem de fechamento
            await self.kucoin_bridge.place_sell_order(
                symbol=symbol,
                size=position.size,
                metadata={
                    'action': 'close_position',
                    'pnl': pnl,
                    'reason': 'risk_management'
                }
            )
            
    async def close_all_positions(self):
        """Fecha todas as posições abertas"""
        positions = self.position_manager.positions.copy()
        
        for symbol, position in positions.items():
            current_price = await self.kucoin_bridge.get_current_price(symbol)
            await self.close_position(symbol, current_price)
            
    def _update_trading_state(self,
                            market_state: HolographicState,
                            patterns: List[HolographicPattern],
                            decisions: List[TradingDecision],
                            risk_assessment: RiskAssessment):
        """Atualiza estado do sistema de trading"""
        # Calcula métricas de emaranhamento
        entanglement = self.metrics.calculate_entanglement_metrics(
            market_state, self.trading_state.portfolio_state if self.trading_state else None
        )
        
        # Obtém estado atual do portfólio
        portfolio_status = self.position_manager.get_portfolio_state()
        
        # Atualiza estado
        self.trading_state = TradingState(
            portfolio_state=portfolio_status.quantum_state if portfolio_status else None,
            market_state=market_state,
            quantum_coherence=risk_assessment.market_coherence,
            entanglement_metrics=entanglement,
            active_patterns=patterns,
            last_decisions=decisions,
            risk_assessment=risk_assessment,
            portfolio_status=portfolio_status,
            performance_metrics=self.calculate_performance_metrics(
                portfolio_status, risk_assessment
            ),
            performance_history=self.trading_state.performance_history if self.trading_state else [],
            system_metrics={
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'latency': 0.0
            },
            timestamp=datetime.now()
        )
        
    def calculate_performance_metrics(self,
                                   portfolio_status: PortfolioState,
                                   risk_assessment: RiskAssessment) -> Dict[str, float]:
        """Calcula métricas de performance do sistema"""
        return {
            'quantum_coherence': risk_assessment.market_coherence,
            'risk_score': risk_assessment.risk_score,
            'portfolio_value': portfolio_status.total_value if portfolio_status else 0.0,
            'realized_pnl': portfolio_status.realized_pnl if portfolio_status else 0.0,
            'unrealized_pnl': portfolio_status.unrealized_pnl if portfolio_status else 0.0,
            'position_count': len(portfolio_status.positions) if portfolio_status else 0,
            'pattern_confidence': np.mean([p.confidence for p in self.trading_state.active_patterns])
            if self.trading_state and self.trading_state.active_patterns else 0.0,
            'portfolio_coherence': portfolio_status.quantum_coherence if portfolio_status else 0.0
        }
    
    async def execute_trading_decisions(self,
                                     decisions: List[TradingDecision],
                                     portfolio_allocation: Dict[str, float],
                                     risk_assessment: RiskAssessment):
        """Executa decisões de trading via KuCoin"""
        for decision in decisions:
            if decision.confidence >= self.config.get('min_confidence', 0.7):
                # Ajusta tamanho da ordem baseado na alocação do portfólio
                size = decision.size * portfolio_allocation.get(decision.symbol, 0)
                
                if decision.action == 'buy':
                    await self.kucoin_bridge.place_buy_order(
                        symbol=decision.symbol,
                        size=size,
                        metadata={
                            'quantum_coherence': decision.quantum_coherence,
                            'confidence': decision.confidence,
                            'patterns': [p.pattern_id for p in decision.supporting_patterns],
                            'risk_score': risk_assessment.risk_score
                        }
                    )
                elif decision.action == 'sell':
                    await self.kucoin_bridge.place_sell_order(
                        symbol=decision.symbol,
                        size=size,
                        metadata={
                            'quantum_coherence': decision.quantum_coherence,
                            'confidence': decision.confidence,
                            'patterns': [p.pattern_id for p in decision.supporting_patterns],
                            'risk_score': risk_assessment.risk_score
                        }
                    )
                    
    async def run(self):
        """Executa o loop principal do sistema"""
        await self.initialize()
        
        while True:
            try:
                # Recebe dados de mercado
                market_data = await self.kucoin_bridge.get_market_data()
                
                # Processa dados e executa trading
                await self.process_market_data(market_data)
                
                # Aguarda próximo ciclo
                await asyncio.sleep(self.config.get('update_interval', 1))
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                await asyncio.sleep(5)  # Espera antes de tentar novamente
                
    def get_system_state(self) -> TradingState:
        """Retorna o estado atual do sistema"""
        if self.trading_state is None:
            raise RuntimeError("Trading system not initialized")
        return self.trading_state
        
    async def shutdown(self):
        """Desliga o sistema de forma segura"""
        await self.kucoin_bridge.disconnect()
        # Salva estado final se necessário
        
# Exemplo de uso
if __name__ == "__main__":
    logger.info("Iniciando Quantum Trading System...")
    config = {
        'api_key': 'sua_api_key',
        'api_secret': 'seu_api_secret',
        'api_passphrase': 'sua_passphrase',
        'trading_pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT'],
        'risk_tolerance': 0.6,
        'update_interval': 1,
        'min_confidence': 0.7,
        'quantum_params': {
            'n_qubits': 8,
            'shots': 1000
        },
        'holographic_params': {
            'dimensions': 64,
            'memory_capacity': 1000
        },
        'server_port': 5000, # Porta adicionada para monitoramento
        'risk_management': {
            'max_risk': 0.8,
            'volatility_threshold': 0.6,
            'coherence_threshold': 0.4,
            'risk_threshold': 0.7
        },
        'position_management': {
            'initial_capital': 10000,
            'max_position_size': 1000,
            'min_position_size': 10,
            'position_step_size': 1,
            'max_positions': 10
        }
    }

    system = QuantumTradingSystem(config)

    async def main():
        try:
            logger.info(f"Sistema iniciando na porta {config['server_port']}")
            await system.run()
        except KeyboardInterrupt:
            logger.info("Desligando sistema...")
            await system.shutdown()
        except Exception as e:
            logger.error(f"Erro fatal: {e}")
            await system.shutdown()
            raise

    asyncio.run(main())