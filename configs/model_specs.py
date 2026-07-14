"""Model specifications for LLM performance calculations."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelSpec:
    name: str
    params_billion: float
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    max_context_window: int
    head_dim: Optional[int] = None  # defaults to d_model / n_heads
    sliding_window: Optional[int] = None  # None = all layers cache full sequence
    n_full_layers: int = 0  # layers with full (not sliding) attention
    global_head_dim: Optional[int] = None  # full-attn head dim; defaults to head_dim
    n_global_kv_heads: Optional[int] = None  # full-attn KV heads; defaults to n_kv_heads
    # MoE: params touched per token for latency; defaults to params_billion (dense)
    active_params_billion: Optional[float] = None

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        "Gemma-4-31B", 31, 3840, 16, 8, 48, 262144,
        head_dim=256,
        sliding_window=1024,
        n_full_layers=8,
        global_head_dim=512,
        n_global_kv_heads=1,
    ),
    ModelSpec(
        "DeepSeek-R1-Distill-Qwen-32B", 32, 5120, 40, 8, 64, 131072
),
    ModelSpec(
        "GPT-OSS-20B", 21, 2880, 64, 8, 24, 131072,
        head_dim=64,
        sliding_window=128,
        n_full_layers=12,
        active_params_billion=3.6,
    ),
    ModelSpec(
        "GPT-OSS-120B", 117, 2880, 64, 8, 36, 131072,
        head_dim=64,
        sliding_window=128,
        n_full_layers=18,
        active_params_billion=5.1,
    ),
    ModelSpec(
        "Qwen3-Coder-30B-A3B-Instruct", 30, 2048, 32, 4, 48, 262144,
        head_dim=128,
        active_params_billion=3,
    ),
    ModelSpec(
        "Qwen3-8B", 8, 4096, 32, 8, 36, 40960
    ),
    ModelSpec(
        "DeepSeek-R1-0528-Qwen3-8B", 8, 4096, 32, 8, 36, 131072
    ),
    ModelSpec("Llama-3.1-70B", 70, 8192, 64, 8, 80, 131072),
    ModelSpec("Llama-3-8B", 8, 4096, 32, 8, 32, 8192),
]

# Commented out models for reference
"""
LEGACY_MODEL_SPECS = [
    ModelSpec("Llama-3-8B", 8, 4096, 32, 8, 32, 8192),
    ModelSpec("Llama-3-70B", 70, 8192, 64, 8, 80, 8192),
    ModelSpec("Llama-2-7B", 7, 4096, 32, 32, 32, 8192),  # Uses MHA
    ModelSpec("Falcon-7B", 7, 4544, 71, 1, 32, 2048),    # Uses MQA
    ModelSpec("Falcon-40B", 40, 8192, 128, 1, 60, 2048), # Uses MQA
    ModelSpec("Falcon-180B", 180, 14848, 232, 1, 80, 2048), # Uses MQA
    ModelSpec( "Llama-3.1-8B", 8, 4096, 32, 8, 32, 131072),
    ModelSpec( "Llama-3.1-70B", 70, 8192, 64, 8, 80, 131072),
    ModelSpec( "Mistral-7B-v0.3", 7, 4096, 32, 8, 32, 32768),
    ModelSpec( "Qwen2.5-14B", 14.7, 5120, 40, 8, 48, 131072),
]
"""
