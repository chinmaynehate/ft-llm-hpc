# gpu_monitor.py
"""
Simple wrapper around NVIDIA's NVML via pynvml for real GPU metrics.

Phase 1: we only expose:
- device_count
- per-GPU memory usage (GB)
- utilization (%)

If NVML is not available (e.g., on a laptop without an NVIDIA driver),
we raise GPUMonitorUnavailable so callers can fall back to fake metrics.
"""

import pynvml


class GPUMonitorUnavailable(Exception):
    pass


class GPUMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise GPUMonitorUnavailable(f"NVML init failed: {e}")

        try:
            self._device_count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            pynvml.nvmlShutdown()
            raise GPUMonitorUnavailable(f"NVML get count failed: {e}")

    @property
    def device_count(self) -> int:
        return self._device_count

    def get_device_info(self, index: int) -> dict:
        """
        Return a dict with:
          - id
          - vram_used_gb
          - vram_total_gb
          - utilization (0-100)
          - mem_utilization (0-100)
        """
        if index < 0 or index >= self._device_count:
            raise IndexError(f"GPU index {index} out of range (0..{self._device_count-1})")

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        return {
            "id": index,
            "vram_used_gb": used_gb,
            "vram_total_gb": total_gb,
            "utilization": util.gpu,        # %
            "mem_utilization": util.memory, # %
        }

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

