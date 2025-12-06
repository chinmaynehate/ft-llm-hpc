"""
GPU Monitor using nvidia-ml-py (pynvml API)
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

# nvidia-ml-py uses the same pynvml module name
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[GPUMonitor] Warning: nvidia-ml-py not installed, using fallback")


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: int
    temperature: int


class GPUMonitor:
    """
    Monitor GPU status using nvidia-ml-py (pynvml API)
    """

    def __init__(self):
        self.initialized = False
        self.device_count = 0
        self.handles: List = []

    def initialize(self) -> bool:
        """Initialize NVML library"""
        if not PYNVML_AVAILABLE:
            return False

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.device_count)
            ]
            self.initialized = True
            return True
        except Exception as e:
            print(f"[GPUMonitor] Failed to initialize: {e}")
            return False

    def shutdown(self):
        """Shutdown NVML library"""
        if self.initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            self.initialized = False

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU"""
        if not self.initialized or gpu_id >= self.device_count:
            return self._get_fallback_info(gpu_id)

        try:
            handle = self.handles[gpu_id]

            # Name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_gpu = util.gpu
            except:
                util_gpu = 0

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0

            return GPUInfo(
                index=gpu_id,
                name=name,
                memory_total_mb=mem_info.total // (1024 * 1024),
                memory_used_mb=mem_info.used // (1024 * 1024),
                memory_free_mb=mem_info.free // (1024 * 1024),
                utilization_gpu=util_gpu,
                temperature=temp,
            )
        except Exception as e:
            print(f"[GPUMonitor] Error getting GPU {gpu_id} info: {e}")
            return self._get_fallback_info(gpu_id)

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs"""
        if not self.initialized:
            num_gpus = int(os.environ.get('NUM_GPUS', '4'))
            return [self._get_fallback_info(i) for i in range(num_gpus)]

        return [self.get_gpu_info(i) for i in range(self.device_count)]

    def _get_fallback_info(self, gpu_id: int) -> GPUInfo:
        """Fallback GPU info when pynvml is not available"""
        import random
        return GPUInfo(
            index=gpu_id,
            name="Simulated GPU",
            memory_total_mb=48000,
            memory_used_mb=random.randint(1000, 40000),
            memory_free_mb=random.randint(8000, 47000),
            utilization_gpu=random.randint(0, 100),
            temperature=random.randint(30, 80),
        )

    def get_status_for_ws(self, gpu_id: int, state: str = 'healthy') -> dict:
        """Get GPU status formatted for WebSocket message"""
        info = self.get_gpu_info(gpu_id)
        if info is None:
            return {
                "id": gpu_id,
                "state": "unknown",
                "vram_used_gb": 0.0,
                "vram_total_gb": 0.0,
            }

        return {
            "id": gpu_id,
            "state": state,
            "name": info.name,
            "vram_used_gb": round(info.memory_used_mb / 1024, 2),
            "vram_total_gb": round(info.memory_total_mb / 1024, 2),
            "utilization": info.utilization_gpu,
            "temperature": info.temperature,
        }


# Test
if __name__ == "__main__":
    monitor = GPUMonitor()
    if monitor.initialize():
        print(f"Found {monitor.device_count} GPUs:")
        for gpu in monitor.get_all_gpu_info():
            print(f"  GPU {gpu.index}: {gpu.name}")
            print(f"    Memory: {gpu.memory_used_mb} / {gpu.memory_total_mb} MB")
            print(f"    Utilization: {gpu.utilization_gpu}%")
            print(f"    Temperature: {gpu.temperature}°C")
        monitor.shutdown()
    else:
        print("GPU monitoring not available")
