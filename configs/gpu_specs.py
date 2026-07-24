"""GPU specifications for LLM performance calculations."""

from dataclasses import dataclass
from typing import List


@dataclass
class GPUSpec:
    name: str
    fp16_tflops: float
    memory_gb: int
    memory_bandwidth_gbps: int


GPU_SPECS: List[GPUSpec] = [
    GPUSpec("NVIDIA L4", 30.29, 24, 300.1),
    GPUSpec("NVIDIA A2", 4.531, 16, 200.1),
    GPUSpec("H200 NVL", 835.5, 141, 4800),
    GPUSpec("H100 NVL", 835.5, 94, 3900),
    GPUSpec("L40s", 362, 48, 864),
    GPUSpec("RTX Pro 6000 BS", 126, 96, 1790),
]
