"""Model specifications for LLM performance calculations."""

from dataclasses import dataclass
from typing import Optional


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
    global_head_dim: Optional[int] = None  # defaults to head_dim
    n_global_kv_heads: Optional[int] = None  # defaults to n_kv_heads
    active_params_billion: Optional[float] = None  # MoE; defaults to params_billion
    architecture: str = "standard"  # standard | hybrid_sliding | qwen3_hybrid | mla | vlm_multimodal
    kv_shared: bool = False
    kv_share_group_size: int = 1
    kv_lora_rank: Optional[int] = None  # MLA compressed latent dim
    qk_rope_head_dim: Optional[int] = None  # MLA decoupled RoPE key dim
    # Fixed-size recurrent state (Qwen3-Next Gated DeltaNet); does not scale with context
    recurrent_num_heads: int = 16
    recurrent_head_dim: int = 128
    recurrent_state_dim: int = 128


ARCH_LABELS = {
    "standard": "Standard",
    "hybrid_sliding": "Hybrid sliding",
    "qwen3_hybrid": "Qwen3 hybrid",
    "mla": "MLA",
    "vlm_multimodal": "Multimodal",
}


def arch_label(architecture: str) -> str:
    return ARCH_LABELS.get(architecture, architecture)


MODEL_SPECS = [
    ModelSpec(
        "DeepSeek-VL2",
        params_billion=27,
        d_model=2560,
        n_heads=32,
        n_kv_heads=32,
        n_layers=30,
        max_context_window=4096,
        active_params_billion=4.5,
        architecture="vlm_multimodal",
    ),
    ModelSpec(
        "Qwen3-32B",
        params_billion=32.8,
        d_model=5120,
        n_heads=64,
        n_kv_heads=8,
        n_layers=64,
        max_context_window=40960,
        head_dim=128,
    ),
    ModelSpec(
        "DeepSeek-R1-Distill-Qwen-32B",
        params_billion=32.5,
        d_model=5120,
        n_heads=40,
        n_kv_heads=8,
        n_layers=64,
        max_context_window=131072,
    ),
    ModelSpec(
        "Mistral-3-8B",
        params_billion=8.8,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        n_layers=34,
        max_context_window=262144,
        head_dim=128,
    ),
    ModelSpec(
        "Qwen2.5-VL-7B",
        params_billion=8.3,
        d_model=3584,
        n_heads=28,
        n_kv_heads=4,
        n_layers=28,
        max_context_window=128000,
        architecture="vlm_multimodal",
    ),
    ModelSpec(
        "Qwen2.5-32B",
        params_billion=32.7,
        d_model=5120,
        n_heads=40,
        n_kv_heads=8,
        n_layers=64,
        n_full_layers=64,
        max_context_window=40960,
    ),
    ModelSpec(
        "Llama-3-8B",
        params_billion=8,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        max_context_window=8192,
    ),
    ModelSpec(
        "Qwen3.5-9B",
        params_billion=9,
        d_model=4096,
        n_heads=16,
        n_kv_heads=32,
        n_layers=32,
        max_context_window=262144,
        head_dim=128,
        n_full_layers=8,
        global_head_dim=256,
        n_global_kv_heads=4,
        architecture="hybrid_sliding",
    ),
    ModelSpec(
        "GPT-OSS-20B",
        params_billion=21,
        d_model=2880,
        n_heads=64,
        n_kv_heads=8,
        n_layers=24,
        max_context_window=131072,
        head_dim=64,
        sliding_window=128,
        n_full_layers=12,
        active_params_billion=3.6,
        architecture="hybrid_sliding",
    ),
    ModelSpec(
        "GPT-OSS-120B",
        params_billion=117,
        d_model=2880,
        n_heads=64,
        n_kv_heads=8,
        n_layers=36,
        max_context_window=131072,
        head_dim=64,
        sliding_window=128,
        n_full_layers=18,
        active_params_billion=5.1,
        architecture="hybrid_sliding",
    ),
    ModelSpec(
        "Gemma-4-31B",
        params_billion=31,
        d_model=3840,
        n_heads=16,
        n_kv_heads=8,
        n_layers=60,
        max_context_window=262144,
        head_dim=256,
        sliding_window=1024,
        n_full_layers=10,
        global_head_dim=512,
        n_global_kv_heads=1,
        kv_shared=True,
        kv_share_group_size=6,
        architecture="hybrid_sliding",
    ),
    ModelSpec(
        "Qwen3-Coder-Next",
        params_billion=80.0,
        active_params_billion=3.0,
        d_model=2048,
        n_layers=48,
        max_context_window=262144,
        n_heads=16,
        n_kv_heads=2,
        head_dim=256,
        n_full_layers=12,
        n_global_kv_heads=2,
        global_head_dim=256,
        recurrent_num_heads=16,
        recurrent_head_dim=128,
        recurrent_state_dim=128,
        architecture="qwen3_hybrid",
    ),
    # DeepSeek-V4-Flash: kv_lora_rank / qk_rope_head_dim are V3-style MLA stand-ins
    ModelSpec(
        "DeepSeek-V4-Flash",
        params_billion=284.0,
        d_model=4096,
        n_heads=64,
        n_kv_heads=1,
        n_layers=43,
        max_context_window=1048576,
        head_dim=512,
        sliding_window=128,
        n_full_layers=41,
        active_params_billion=13.0,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        architecture="mla",
    ),
]
