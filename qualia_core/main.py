"""
Ponto de entrada principal unificado do sistema
"""
import asyncio
import logging
import uvicorn
from typing import Optional, Dict
from fastapi import FastAPI
from core.holistic.quantum_core import HolisticQuantumCore
from core.config.system_config import system_config
from core.trading.order_executor import OrderExecutor
from interface.portfolio_dashboard import PortfolioDashboard
from core.quantum.Code_analyzer.merge.quantum_merge_unified import UnifiedQuantumMerge
from core.quantum.qualia_interface import QualiaMorphicInterface

# Inicializa a aplicação FastAPI
app = FastAPI(title="Sistema de Mineração Adaptativa")
quantum_core = None

async def initialize_system() -> HolisticQuantumCore:
    """Inicializa o sistema de forma holística"""
    try:
        # Inicializa núcleo quântico
        quantum_core = HolisticQuantumCore()

        # Adiciona sistema de merge quântico
        quantum_merge = UnifiedQuantumMerge()
        quantum_core.register_analyzer(quantum_merge)

        # Adiciona interface QUALIA morfogenética
        qualia_interface = QualiaMorphicInterface()
        quantum_core.register_interface(qualia_interface)

        # Sincroniza componentes
        await quantum_core.synchronize_components()

        logging.info("Sistema quântico inicializado com consciência morfogenética")
        return quantum_core
    except Exception as e:
        logging.error(f"Erro inicializando sistema: {e}")
        raise

async def run_trading_cycle(quantum_core: HolisticQuantumCore):
    """Executa ciclo de trading principal"""
    while True:
        try:
            # Integração quântica 
            integrated_state = quantum_core.integrate_quantum_state()

            # Análise consciente do estado
            consciousness_state = quantum_core.analyze_consciousness(integrated_state)

            # Evolução morfogenética
            evolution_metrics = quantum_core.evolve_consciousness(consciousness_state)

            # Atualiza dashboard com estado consciente e evolução
            dashboard = PortfolioDashboard()
            dashboard.update_with_quantum_state(consciousness_state)
            dashboard.update_evolution_metrics(evolution_metrics)

            # Aguarda próximo ciclo
            await asyncio.sleep(60)

        except Exception as e:
            logging.error(f"Erro no ciclo principal: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    global quantum_core
    quantum_core = await initialize_system()
    asyncio.create_task(run_trading_cycle(quantum_core))

@app.get("/")
async def root():
    return {"message": "Sistema de mineração adaptativa funcionando!"}

@app.get("/status")
async def status() -> Dict:
    if not quantum_core:
        return {"status": "initializing"}
        
    # Obtém estado integrado
    integrated_state = quantum_core.integrate_quantum_state()
    
    return {
        "status": "online",
        "quantum_core": {
            "initialized": True,
            "components_synced": True
        },
        "mining_integration": integrated_state.get('mining_integration', {}),
        "optimization": integrated_state.get('optimization_suggestions', {})
    }

if __name__ == "__main__":
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Inicia o servidor
    uvicorn.run(app, host="127.0.0.1", port=8004)