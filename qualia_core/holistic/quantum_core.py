"""
Core quântico holístico integrado
"""
from core.consciousness.operator import QuantumConsciousnessOperator
from core.trading.order_executor import OrderExecutor
from core.dark_finance.dark_portfolio import DarkPortfolioIntegrator
from core.quantum.mining_bridge import MiningMorphicBridge
from typing import List, Any, Dict

class HolisticQuantumCore:
    def __init__(self):
        self.analyzers = []
        self.interfaces = []
        self.consciousness_operator = QuantumConsciousnessOperator()
        self.order_executor = OrderExecutor()
        self.dark_portfolio = DarkPortfolioIntegrator()
        self.mining_bridge = MiningMorphicBridge()
        
    def register_analyzer(self, analyzer: Any) -> None:
        """Registra um analisador quântico"""
        self.analyzers.append(analyzer)
        
    def register_interface(self, interface: Any) -> None:
        """Registra uma interface QUALIA"""
        self.interfaces.append(interface)
        
    async def synchronize_components(self) -> None:
        """Sincroniza todos os componentes do sistema"""
        # Sincroniza analisadores
        for analyzer in self.analyzers:
            await analyzer.initialize()
            
        # Sincroniza interfaces
        for interface in self.interfaces:
            await interface.initialize()
            
        # Sincroniza componentes principais
        await self.consciousness_operator.initialize()
        await self.order_executor.initialize()
        await self.dark_portfolio.initialize()
        
        # Integra estado inicial do trading
        trading_state = {
            'market_coherence': self.consciousness_operator.get_coherence(),
            'portfolio_health': self.dark_portfolio.get_health(),
            'prediction_confidence': self.order_executor.get_confidence()
        }
        await self.mining_bridge.integrate_trading_state(trading_state)
        
    def integrate_quantum_state(self) -> Dict[str, Any]:
        """Integra o estado quântico atual do sistema"""
        state = {}
        
        # Integra estado dos analisadores
        for analyzer in self.analyzers:
            state.update(analyzer.get_state())
            
        # Integra estado das interfaces
        for interface in self.interfaces:
            state.update(interface.get_state())
            
        # Integra estado dos componentes principais
        state.update(self.consciousness_operator.get_state())
        state.update(self.order_executor.get_state())
        state.update(self.dark_portfolio.get_state())
        
        # Adiciona métricas de integração com mineração
        state['mining_integration'] = self.mining_bridge.get_integration_state()
        state['optimization_suggestions'] = self.mining_bridge.get_optimization_suggestions()
        
        return state
        
    def analyze_consciousness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa o estado de consciência do sistema"""
        return self.consciousness_operator.analyze(state)
        
    def evolve_consciousness(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Evolui o estado de consciência do sistema"""
        return self.consciousness_operator.evolve(state)