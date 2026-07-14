"""Performance calculation utilities for LLM inference."""

from dataclasses import dataclass
from math import floor
from typing import Union
from configs.gpu_specs import GPUSpec
from configs.model_specs import ModelSpec

BYTES_IN_GiB = 1_073_741_824

# Enterprise defaults (conservative, not peak-lab).
DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU = 3.0
DEFAULT_VRAM_UTILIZATION = 0.90  # leave ~10% allocator/driver headroom
DEFAULT_WEIGHT_BYTES = 2  # FP16/BF16
DEFAULT_KV_BYTES = 2
DEFAULT_KV_FRAGMENTATION = 1.10  # allocator / block waste vs ideal KV
DEFAULT_QUANT = "fp16"
DEFAULT_KV_QUANT = "fp16"

# Bytes per parameter (weights) / per KV element.
WEIGHT_BYTES_BY_QUANT = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "mxfp4": 0.6,  # ~70% smaller than FP16 (packed microscaling)
}
KV_BYTES_BY_QUANT = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}


def resolve_weight_bytes(quant: str) -> float:
    key = quant.lower()
    if key not in WEIGHT_BYTES_BY_QUANT:
        raise ValueError(
            f"Unknown weight quant {quant!r}; choose from {sorted(WEIGHT_BYTES_BY_QUANT)}"
        )
    return WEIGHT_BYTES_BY_QUANT[key]


def resolve_kv_bytes(kv_quant: str) -> float:
    key = kv_quant.lower()
    if key not in KV_BYTES_BY_QUANT:
        raise ValueError(
            f"Unknown KV quant {kv_quant!r}; choose from {sorted(KV_BYTES_BY_QUANT)}"
        )
    return KV_BYTES_BY_QUANT[key]


def tp_efficiency(num_gpu: int) -> float:
    """Tensor-parallel scaling efficiency (imperfect interconnect)."""
    if num_gpu <= 1:
        return 1.0
    if num_gpu == 2:
        return 0.85
    if num_gpu <= 4:
        return 0.75
    return 0.65


@dataclass
class PerformanceMetrics:
    kv_cache_tokens: int
    feasible: bool
    prefill_time_per_token: Union[float, str]
    tpot: Union[float, str]
    ttft: Union[float, str]
    e2e_latency: Union[float, str]
    throughput: Union[float, str]
    # Batch-1 decode ceiling (labeled upper bound; not used as headline)
    tpot_batch1: Union[float, str]


@dataclass
class ConcurrentCapacity:
    """Estimated concurrent requests from GPU memory + model KV (no assumed concurrency)."""
    kv_cache_window: int
    max_context_window: int
    avg_context_window: int
    concurrent_at_max_context: int
    concurrent_at_avg_context: int
    available_for_kv_gb: float


class PerformanceCalculator:
    """Calculator for LLM inference capacity and latency (enterprise-conservative)."""

    def __init__(
        self,
        num_gpu: int,
        system_overhead_gb_per_gpu: float = DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
        vram_utilization: float = DEFAULT_VRAM_UTILIZATION,
        weight_bytes: float = DEFAULT_WEIGHT_BYTES,
        kv_bytes: float = DEFAULT_KV_BYTES,
        kv_fragmentation: float = DEFAULT_KV_FRAGMENTATION,
    ):
        self.num_gpu = num_gpu
        self.system_overhead_gb_per_gpu = system_overhead_gb_per_gpu
        self.vram_utilization = vram_utilization
        self.weight_bytes = weight_bytes
        self.kv_bytes = kv_bytes
        self.kv_fragmentation = kv_fragmentation

    @staticmethod
    def _sliding_head_dim(model: ModelSpec) -> float:
        return model.head_dim if model.head_dim is not None else model.d_model / model.n_heads

    @staticmethod
    def _full_head_dim(model: ModelSpec) -> float:
        if model.global_head_dim is not None:
            return model.global_head_dim
        return PerformanceCalculator._sliding_head_dim(model)

    @staticmethod
    def _full_kv_heads(model: ModelSpec) -> int:
        return model.n_global_kv_heads if model.n_global_kv_heads is not None else model.n_kv_heads

    @staticmethod
    def _active_params_billion(model: ModelSpec) -> float:
        """Params touched per token (MoE active, else total)."""
        if model.active_params_billion is not None:
            return model.active_params_billion
        return model.params_billion

    def weight_gib(self, model: ModelSpec) -> float:
        """Resident weight memory (total params — MoE loads all experts)."""
        return model.params_billion * self.weight_bytes

    def system_overhead_gib(self) -> float:
        return self.system_overhead_gb_per_gpu * self.num_gpu

    def usable_vram_gib(self, gpu: GPUSpec) -> float:
        return self.num_gpu * gpu.memory_gb * self.vram_utilization

    def available_for_kv_gib(self, gpu: GPUSpec, model: ModelSpec) -> float:
        """VRAM left for KV after weights + system overhead + utilization cap."""
        return max(
            0.0,
            self.usable_vram_gib(gpu) - self.system_overhead_gib() - self.weight_gib(model),
        )

    def calc_kv_cache_gib_for_seq(self, model: ModelSpec, seq_len: int) -> float:
        """KV cache GiB for one sequence of length seq_len (hybrid-aware + frag)."""
        bytes_per_value = self.kv_bytes
        n_sliding = model.n_layers - model.n_full_layers
        slide_len = (
            min(seq_len, model.sliding_window)
            if model.sliding_window is not None and n_sliding > 0
            else seq_len
        )
        sliding = (
            2 * n_sliding * model.n_kv_heads * self._sliding_head_dim(model)
            * bytes_per_value * slide_len
        )
        full = (
            2 * model.n_full_layers * self._full_kv_heads(model) * self._full_head_dim(model)
            * bytes_per_value * seq_len
        )
        return (sliding + full) / BYTES_IN_GiB * self.kv_fragmentation

    def calc_kv_cache_size_per_token(self, model: ModelSpec, seq_len: int = 1) -> float:
        """Effective GiB/token at seq_len (= KV(seq_len) / seq_len)."""
        return self.calc_kv_cache_gib_for_seq(model, seq_len) / seq_len

    def calc_memory_footprint(
        self,
        model: ModelSpec,
        n_concurrent_request: int,
        context_window: int,
    ) -> float:
        """Total memory footprint in GB (weights + overhead + KV)."""
        return (
            self.weight_gib(model)
            + self.system_overhead_gib()
            + self.calc_kv_cache_gib_for_seq(model, context_window) * n_concurrent_request
        )

    def fits(
        self,
        gpu: GPUSpec,
        model: ModelSpec,
        n_concurrent_request: int,
        context_window: int,
    ) -> bool:
        """True if config fits in usable VRAM."""
        return self.calc_memory_footprint(
            model, n_concurrent_request, context_window
        ) <= self.usable_vram_gib(gpu)

    def calc_kv_cache_tokens(
        self, gpu: GPUSpec, model: ModelSpec, kv_cache_size: float
    ) -> float:
        """Max tokens that fit in remaining KV budget."""
        if kv_cache_size <= 0:
            return 0.0
        return max(0.0, self.available_for_kv_gib(gpu, model) / kv_cache_size)

    def calc_max_concurrent_requests(
        self, gpu: GPUSpec, model: ModelSpec, context_window: int
    ) -> int:
        """Max concurrent requests if each uses context_window tokens of KV.

        Always rounds down so we never claim a fractional request that would OOM.
        """
        if context_window <= 0:
            return 0
        free = self.available_for_kv_gib(gpu, model)
        if free <= 0:
            return 0
        kv_gib = self.calc_kv_cache_gib_for_seq(model, context_window)
        if kv_gib <= 0:
            return 0
        return floor(free / kv_gib)

    def calc_concurrent_capacity(
        self, gpu: GPUSpec, model: ModelSpec, avg_context_window: int
    ) -> ConcurrentCapacity:
        """Concurrent users at model max context and at assumed average context."""
        max_ctx = model.max_context_window
        available = self.available_for_kv_gib(gpu, model)
        kv_cache_window = floor(
            self.calc_kv_cache_tokens(
                gpu, model, self.calc_kv_cache_size_per_token(model, avg_context_window)
            )
        )
        return ConcurrentCapacity(
            kv_cache_window=kv_cache_window,
            max_context_window=max_ctx,
            avg_context_window=avg_context_window,
            concurrent_at_max_context=self.calc_max_concurrent_requests(
                gpu, model, max_ctx
            ),
            concurrent_at_avg_context=self.calc_max_concurrent_requests(
                gpu, model, avg_context_window
            ),
            available_for_kv_gb=available,
        )

    def _eff_gpus(self) -> float:
        return self.num_gpu * tp_efficiency(self.num_gpu)

    def calc_prefill_time_per_token_batch1(
        self, model: ModelSpec, gpu: GPUSpec
    ) -> float:
        """Prefill ms/token at batch=1 (compute roofline, TP-efficient)."""
        return (2 * self._active_params_billion(model) / self._eff_gpus()) / gpu.fp16_tflops

    def calc_tpot_batch1(self, model: ModelSpec, gpu: GPUSpec) -> float:
        """Decode TPOT ms/token at batch=1 (memory-BW roofline, TP-efficient)."""
        return (
            (2 * self._active_params_billion(model) / self._eff_gpus())
            / gpu.memory_bandwidth_gbps
            * 1000
        )

    def calc_prefill_time_per_token(
        self, model: ModelSpec, gpu: GPUSpec, n_concurrent_request: int = 1
    ) -> Union[float, str]:
        """Per-user prefill ms/token under concurrent fair-share of FLOPs."""
        batch = max(1, n_concurrent_request)
        return self.calc_prefill_time_per_token_batch1(model, gpu) * batch

    def calc_tpot(
        self, model: ModelSpec, gpu: GPUSpec, n_concurrent_request: int = 1
    ) -> Union[float, str]:
        """Per-user TPOT ms under concurrent fair-share of memory bandwidth."""
        batch = max(1, n_concurrent_request)
        return self.calc_tpot_batch1(model, gpu) * batch

    def calc_e2e_latency(
        self,
        prefill_time_per_token: float,
        tpot: float,
        prompt_size: int,
        response_size: int,
    ) -> float:
        """End-to-end latency in seconds."""
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    def calculate_metrics(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int = 1,
    ) -> PerformanceMetrics:
        """Latency/throughput for assumed concurrency; INFEASIBLE if it does not fit."""
        context_window = prompt_size + response_size
        kv_cache_size_per_token = self.calc_kv_cache_size_per_token(model, context_window)
        kv_cache_tokens = floor(
            self.calc_kv_cache_tokens(gpu, model, kv_cache_size_per_token)
        )
        feasible = self.fits(gpu, model, n_concurrent_request, context_window)
        tpot_b1 = self.calc_tpot_batch1(model, gpu)

        if not feasible:
            return PerformanceMetrics(
                kv_cache_tokens=kv_cache_tokens,
                feasible=False,
                prefill_time_per_token="INFEASIBLE",
                tpot="INFEASIBLE",
                ttft="INFEASIBLE",
                e2e_latency="INFEASIBLE",
                throughput="INFEASIBLE",
                tpot_batch1=tpot_b1,
            )

        prefill = self.calc_prefill_time_per_token(model, gpu, n_concurrent_request)
        tpot = self.calc_tpot(model, gpu, n_concurrent_request)
        ttft = prefill + tpot / 1000
        e2e = self.calc_e2e_latency(prefill, tpot, prompt_size, response_size)
        # Aggregate output tokens/sec across concurrent users (fair-share model).
        throughput = (
            n_concurrent_request * response_size / e2e if e2e > 0 else "INFEASIBLE"
        )

        return PerformanceMetrics(
            kv_cache_tokens=kv_cache_tokens,
            feasible=True,
            prefill_time_per_token=prefill,
            tpot=tpot,
            ttft=ttft,
            e2e_latency=e2e,
            throughput=throughput,
            tpot_batch1=tpot_b1,
        )


if __name__ == "__main__":
    from configs.gpu_specs import GPU_SPECS
    from configs.model_specs import MODEL_SPECS

    calc = PerformanceCalculator(1)
    dense = next(m for m in MODEL_SPECS if m.name == "DeepSeek-R1-Distill-Qwen-32B")
    gemma = next(m for m in MODEL_SPECS if m.name == "Gemma-4-31B")
    moe = next(m for m in MODEL_SPECS if m.name == "Qwen3-Coder-30B-A3B-Instruct")
    small = next(m for m in MODEL_SPECS if m.name == "Qwen3-8B")
    gpu = GPU_SPECS[0]

    # Dense GQA raw structure (before fragmentation)
    L = 4096
    d_head = dense.d_model / dense.n_heads
    raw = 2 * dense.n_layers * dense.n_kv_heads * d_head * 2 * L / BYTES_IN_GiB
    assert abs(calc.calc_kv_cache_gib_for_seq(dense, L) - raw * calc.kv_fragmentation) < 1e-12

    # Gemma: sliding caps; full layers grow past window
    W = gemma.sliding_window
    assert W is not None
    short, long = W, W * 4
    short_kv = calc.calc_kv_cache_gib_for_seq(gemma, short)
    long_kv = calc.calc_kv_cache_gib_for_seq(gemma, long)
    full_only_delta = (
        2 * gemma.n_full_layers * gemma.n_global_kv_heads * gemma.global_head_dim * 2
        * (long - short) / BYTES_IN_GiB * calc.kv_fragmentation
    )
    assert abs((long_kv - short_kv) - full_only_delta) < 1e-12

    # MoE: memory total, latency active; footprint includes overhead
    assert calc._active_params_billion(moe) == 3
    mem = calc.calc_memory_footprint(moe, 1, 1)
    assert abs(
        mem
        - (
            calc.weight_gib(moe)
            + calc.system_overhead_gib()
            + calc.calc_kv_cache_gib_for_seq(moe, 1)
        )
    ) < 1e-12
    assert abs(calc.calc_tpot_batch1(moe, gpu) - (2 * 3 / 1) / gpu.memory_bandwidth_gbps * 1000) < 1e-12

    # Available KV subtracts overhead + utilization
    gpu_big = next(g for g in GPU_SPECS if g.memory_gb >= 80)
    calc4 = PerformanceCalculator(4)
    usable = 4 * gpu_big.memory_gb * calc4.vram_utilization
    avail = calc4.available_for_kv_gib(gpu_big, dense)
    assert abs(avail - (usable - calc4.system_overhead_gib() - calc4.weight_gib(dense))) < 1e-12
    avg_ctx = 4096
    assert calc4.calc_max_concurrent_requests(gpu_big, dense, avg_ctx) == floor(
        avail / calc4.calc_kv_cache_gib_for_seq(dense, avg_ctx)
    )
    # Explicit round-down: never claim a fractional concurrent request
    assert floor(2.9) == 2

    # Fit gate + batch scales TPOT
    metrics_ok = calc4.calculate_metrics(small, gpu_big, 512, 128, n_concurrent_request=2)
    assert metrics_ok.feasible
    assert abs(metrics_ok.tpot - calc4.calc_tpot_batch1(small, gpu_big) * 2) < 1e-12
    metrics_bad = calc4.calculate_metrics(
        dense, gpu, 4096, 256, n_concurrent_request=100
    )
    assert not metrics_bad.feasible
    assert metrics_bad.tpot == "INFEASIBLE"

    # TP efficiency worsens effective GPUs
    assert tp_efficiency(1) == 1.0
    assert tp_efficiency(4) == 0.75

    # Quantization: INT4 weights half FP16; FP8 KV halves cache vs FP16
    assert resolve_weight_bytes("int4") == 0.5
    assert resolve_kv_bytes("fp8") == 1.0
    q = PerformanceCalculator(1, weight_bytes=resolve_weight_bytes("int4"))
    assert abs(q.weight_gib(small) - small.params_billion * 0.5) < 1e-12
    kv_fp16 = PerformanceCalculator(1, kv_bytes=2.0).calc_kv_cache_gib_for_seq(small, 1024)
    kv_fp8 = PerformanceCalculator(1, kv_bytes=1.0).calc_kv_cache_gib_for_seq(small, 1024)
    assert abs(kv_fp8 * 2 - kv_fp16) < 1e-12
    print("ok")
