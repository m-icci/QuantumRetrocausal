"""
Hardware Manager para QUALIA
Detecta e gerencia recursos de hardware (CPU/GPU) usando apenas operações bitwise nativas
"""

import platform
import psutil
import logging
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass
from core.quantum.qualia_unified import BitwiseOperators, UnifiedField, FieldType

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Perfil de hardware detectado"""
    device_type: str  # "cpu" ou "gpu"
    device_name: str
    total_memory: int
    available_memory: int
    compute_units: int
    architecture: str
    quantum_state: bytes

class HardwareManager:
    """Gerenciador de hardware para QUALIA usando apenas operações bitwise"""
    
    def __init__(self):
        self.ops = BitwiseOperators()
        self._quantum_state = self.ops.random_state(64)
        self.unified_field = UnifiedField(
            field_type=FieldType.ADAPTIVE,
            size=64,
            consciousness_factor=0.618  # Proporção áurea
        )
        self.profile = self._detect_hardware()
            
    def _detect_nvidia_gpu(self) -> Optional[Dict[str, Any]]:
        """Detecta GPU NVIDIA usando operações bitwise"""
        try:
            if platform.system() == "Windows":
                # Verifica DLLs NVIDIA
                nvidia_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), 
                               "NVIDIA Corporation", "NVSMI", "nvml.dll"),
                    os.path.join(os.environ.get("windir", "C:\\Windows"), 
                               "System32", "nvapi64.dll")
                ]
                
                for path in nvidia_paths:
                    if os.path.exists(path):
                        # Gera estado quântico para GPU
                        gpu_state = self.ops.quantum_and(
                            self._quantum_state,
                            self.ops.random_state(64)
                        )
                        return {
                            "available": True,
                            "name": "NVIDIA GPU",
                            "architecture": "CUDA",
                            "quantum_state": gpu_state
                        }
            return None
        except Exception as e:
            logger.error(f"Erro detectando GPU: {e}")
            return None
            
    def _detect_hardware(self) -> HardwareProfile:
        """Detecta hardware disponível usando apenas operações bitwise"""
        
        # Tenta detectar GPU
        gpu_info = self._detect_nvidia_gpu()
        
        # Gera estado base
        base_state = self.ops.random_state(64)
        
        if gpu_info and gpu_info["available"]:
            # Configuração GPU
            device_type = "gpu"
            device_name = gpu_info["name"]
            vm = psutil.virtual_memory()
            
            return HardwareProfile(
                device_type=device_type,
                device_name=device_name,
                total_memory=vm.total,
                available_memory=vm.available,
                compute_units=psutil.cpu_count(logical=False),
                architecture=gpu_info["architecture"],
                quantum_state=gpu_info["quantum_state"]
            )
        else:
            # Configuração CPU
            device_type = "cpu"
            device_name = platform.processor()
            vm = psutil.virtual_memory()
            
            # Estado quântico para CPU
            cpu_state = self.ops.quantum_or(
                base_state,
                self._quantum_state
            )
            
            return HardwareProfile(
                device_type=device_type,
                device_name=device_name,
                total_memory=vm.total,
                available_memory=vm.available,
                compute_units=psutil.cpu_count(logical=False),
                architecture="x86_64" if platform.machine().endswith('64') else "x86",
                quantum_state=cpu_state
            )
    
    def is_gpu_available(self) -> bool:
        """Verifica se GPU está disponível"""
        return self.profile.device_type == "gpu"
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Retorna informações de memória usando estados quânticos"""
        vm = psutil.virtual_memory()
        
        # Gera estado quântico para memória
        memory_state = self.ops.coherent_merge(
            self.profile.quantum_state,
            self.ops.random_state(64)
        )
        
        return {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "quantum_state": memory_state,
            "field_metrics": self.unified_field.metrics
        }
    
    def optimize_memory(self):
        """Otimiza uso de memória via operações bitwise"""
        optimized_state = self.ops.quantum_shift(self.profile.quantum_state)
        optimized_state = self.ops.quantum_and(
            optimized_state,
            self.ops.random_state(64)
        )
        self.profile.quantum_state = optimized_state
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Retorna perfil de hardware atual"""
        return self.profile
        
    def synchronize_quantum_state(self, state: bytes):
        """Sincroniza estado quântico do hardware"""
        self.profile.quantum_state = self.ops.coherent_merge(
            self.profile.quantum_state,
            state
        )
