import sys
import os
import subprocess
import asyncio

# Configurar PYTHONPATH para encontrar os módulos do projeto
os.environ['PYTHONPATH'] = os.path.abspath('.')

# Configurar política de evento loop para Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Executar o sistema em modo real
print("🚀 Executando QUALIA em modo real com integrated_quantum_scalping.py...")
command = """
import sys
import asyncio
from quantum_trading.integrated_quantum_scalping import IntegratedQuantumScalping

async def run_real():
    config = {
        "mode": "real",
        "trading": {
            "symbol": "BTC/USDT",
            "exchange": "kucoin",
            "mode": "real"
        },
        "exchange": {
            "name": "kucoin",
            "api_key": "",
            "api_secret": ""
        },
        "risk": {
            "max_position_size": 0.01,
            "max_daily_loss": 0.05,
            "max_total_loss": 0.1,
            "min_profit_threshold": 0.005,
            "max_loss_threshold": 0.01,
            "max_position_time": 3600,
            "max_open_positions": 3,
            "max_drawdown": 0.2
        },
        "cosmo": {
            "entanglement_strength": 0.7,
            "quantum_gravity": 0.5,
            "cosmic_entropy": 0.3,
            "temporal_coupling": 0.8,
            "grid_size": 64,
            "dt": 0.01,
            "hbar": 1.0,
            "mass": 1.0,
            "nonlinearity_strength": 2.0,
            "decoherence_rate": 0.05,
            "spatial_dimensions": 3,
            "initial_hubble": 70.0,
            "initial_lambda": 1.0e-35,
            "matter_density": 0.3,
            "beta_coupling": 0.2,
            "latent_amplitude": 5.0e-37,
            "oscillation_period": 10.0,
            "latent_noise_level": 0.2,
            "latent_mode": "mixed",
            "quantum_cosmo_coupling": 0.7,
            "cosmo_quantum_coupling": 0.5,
            "entropy_latent_coupling": 0.3,
            "potential_strength": 1.0,
            "quantum_gravity_coupling": 0.1,
            "latent_dimensions": 3
        },
        "qualia": {
            "enable_quantum": True,
            "enable_helix": True,
            "lstm_threshold": 0.65,
            "adaptation_rate": 0.02
        },
        "metrics_path": "metrics"
    }
    
    system = IntegratedQuantumScalping(config)
    await system.run()

if __name__ == "__main__":
    asyncio.run(run_real())
"""

# Executar o Python com os comandos
process = subprocess.run([sys.executable, "-c", command], capture_output=False)

print(f"Comando concluído com código de saída: {process.returncode}") 