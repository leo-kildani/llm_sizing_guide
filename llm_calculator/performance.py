"""LLM inference capacity and latency estimates."""

from dataclasses import dataclass
from math import floor
from typing import Union

from configs.gpu_specs import GPUSpec
from configs.model_specs import ModelSpec

BYTES_IN_GB = 1_000_000_000

DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU = 3.0
DEFAULT_VRAM_UTILIZATION = 0.90
DEFAULT_WEIGHT_BYTES = 2
DEFAULT_KV_BYTES = 2
DEFAULT_KV_FRAGMENTATION = 1.10
DEFAULT_QUANT = "fp16"
DEFAULT_KV_QUANT = "fp16"

BYTES_BY_QUANT = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}
WEIGHT_BYTES_BY_QUANT = BYTES_BY_QUANT
KV_BYTES_BY_QUANT = BYTES_BY_QUANT


def resolve_bytes(quant: str) -> float:
    key = quant.lower()
    if key not in BYTES_BY_QUANT:
        raise ValueError(f"Unknown quant {quant!r}; choose from {sorted(BYTES_BY_QUANT)}")
    return BYTES_BY_QUANT[key]


resolve_weight_bytes = resolve_bytes
resolve_kv_bytes = resolve_bytes


def tp_efficiency(num_gpu: int) -> float:
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
    tpot_batch1: Union[float, str]


@dataclass
class ConcurrentCapacity:
    kv_cache_window: int
    max_context_window: int
    avg_context_window: int
    concurrent_at_max_context: int
    concurrent_at_avg_context: int
    available_for_kv_gb: float


class PerformanceCalculator:
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
        if model.active_params_billion is not None:
            return model.active_params_billion
        return model.params_billion

    @staticmethod
    def _slide_len(model: ModelSpec, context_len: int) -> int:
        n_sliding = max(0, model.n_layers - model.n_full_layers)
        if model.sliding_window is not None and n_sliding > 0:
            return min(context_len, model.sliding_window)
        return context_len

    def _to_gb(self, nbytes: float) -> float:
        return (nbytes / BYTES_IN_GB) * self.kv_fragmentation

    def weight_gib(self, model: ModelSpec) -> float:
        return model.params_billion * self.weight_bytes

    def system_overhead_gib(self) -> float:
        return self.system_overhead_gb_per_gpu * self.num_gpu

    def usable_vram_gib(self, gpu: GPUSpec) -> float:
        return self.num_gpu * gpu.memory_gb * self.vram_utilization

    def available_for_kv_gib(self, gpu: GPUSpec, model: ModelSpec) -> float:
        return max(
            0.0,
            self.usable_vram_gib(gpu) - self.system_overhead_gib() - self.weight_gib(model),
        )

    def calc_fixed_state_gb(self, model: ModelSpec) -> float:
        if model.architecture != "qwen3_hybrid":
            return 0.0
        total_kv_layers = model.n_layers // 4
        deltanet_layers = model.n_layers - total_kv_layers
        return self._to_gb(
            deltanet_layers
            * model.recurrent_num_heads
            * model.recurrent_head_dim
            * model.recurrent_state_dim
            * self.kv_bytes
        )

    def calc_kv_cache_gib_for_seq(self, model: ModelSpec, context_len: int) -> float:
        b = self.kv_bytes

        if model.architecture == "qwen3_hybrid":
            total_kv_layers = model.n_layers // 4
            return self._to_gb(
                2
                * total_kv_layers
                * self._full_kv_heads(model)
                * self._full_head_dim(model)
                * b
                * context_len
            )

        if model.architecture == "mla":
            kv_lora_rank = model.kv_lora_rank if model.kv_lora_rank is not None else 512
            rope_dim = model.qk_rope_head_dim if model.qk_rope_head_dim is not None else 64
            per_layer = (kv_lora_rank + rope_dim) * b
            n_sliding = max(0, model.n_layers - model.n_full_layers)
            slide_len = self._slide_len(model, context_len)
            return self._to_gb(
                n_sliding * per_layer * slide_len
                + model.n_full_layers * per_layer * context_len
            )

        if model.architecture == "hybrid_sliding":
            n_sliding = max(0, model.n_layers - model.n_full_layers)
            slide_len = self._slide_len(model, context_len)
            sliding_bytes = (
                2
                * n_sliding
                * model.n_kv_heads
                * self._sliding_head_dim(model)
                * b
                * slide_len
            )
            effective_full = (
                model.n_full_layers / model.kv_share_group_size
                if model.kv_shared
                else model.n_full_layers
            )
            full_bytes = (
                2
                * effective_full
                * self._full_kv_heads(model)
                * self._full_head_dim(model)
                * b
                * context_len
            )
            return self._to_gb(sliding_bytes + full_bytes)

        return self._to_gb(
            2
            * model.n_layers
            * model.n_kv_heads
            * self._sliding_head_dim(model)
            * b
            * context_len
        )

    def calc_kv_cache_size_per_token(self, model: ModelSpec, context_len: int = 1) -> float:
        return self.calc_kv_cache_gib_for_seq(model, context_len) / context_len

    def per_request_gb(self, model: ModelSpec, context_window: int) -> float:
        return (
            self.calc_kv_cache_gib_for_seq(model, context_window)
            + self.calc_fixed_state_gb(model)
        )

    def calc_memory_footprint(
        self, model: ModelSpec, n_concurrent_request: int, context_window: int
    ) -> float:
        return (
            self.weight_gib(model)
            + self.system_overhead_gib()
            + n_concurrent_request * self.per_request_gb(model, context_window)
        )

    def fits(
        self,
        gpu: GPUSpec,
        model: ModelSpec,
        n_concurrent_request: int,
        context_window: Union[int, float],
    ) -> bool:
        return self.calc_memory_footprint(
            model, n_concurrent_request, context_window
        ) <= self.usable_vram_gib(gpu)

    def calc_kv_cache_tokens(
        self, gpu: GPUSpec, model: ModelSpec, kv_cache_size: float
    ) -> float:
        if kv_cache_size <= 0:
            return 0.0
        return max(0.0, self.available_for_kv_gib(gpu, model) / kv_cache_size)

    def calc_max_concurrent_requests(
        self, gpu: GPUSpec, model: ModelSpec, context_window: int
    ) -> int:
        if context_window <= 0:
            return 0
        free = self.available_for_kv_gib(gpu, model)
        if free <= 0:
            return 0
        per_request = self.per_request_gb(model, context_window)
        if per_request <= 0:
            return 0
        return floor(free / per_request)

    def calc_concurrent_capacity(
        self, gpu: GPUSpec, model: ModelSpec, avg_context_window: int
    ) -> ConcurrentCapacity:
        max_ctx = model.max_context_window
        return ConcurrentCapacity(
            kv_cache_window=floor(
                self.calc_kv_cache_tokens(
                    gpu, model, self.calc_kv_cache_size_per_token(model, avg_context_window)
                )
            ),
            max_context_window=max_ctx,
            avg_context_window=avg_context_window,
            concurrent_at_max_context=self.calc_max_concurrent_requests(gpu, model, max_ctx),
            concurrent_at_avg_context=self.calc_max_concurrent_requests(
                gpu, model, avg_context_window
            ),
            available_for_kv_gb=self.available_for_kv_gib(gpu, model),
        )

    def _eff_gpus(self) -> float:
        return self.num_gpu * tp_efficiency(self.num_gpu)

    def calc_prefill_time_per_token_batch1(self, model: ModelSpec, gpu: GPUSpec) -> float:
        # 2 * params_billion / (eff_gpus * tflops) → ms/token (1e9/1e12 cancels into ms)
        return (2 * self._active_params_billion(model) / self._eff_gpus()) / gpu.fp16_tflops

    def calc_tpot_batch1(self, model: ModelSpec, gpu: GPUSpec) -> float:
        return (
            (self.weight_bytes * self._active_params_billion(model) / self._eff_gpus())
            / gpu.memory_bandwidth_gbps
            * 1000
        )

    def calc_prefill_time_per_token(
        self, model: ModelSpec, gpu: GPUSpec, n_concurrent_request: int = 1
    ) -> float:
        return self.calc_prefill_time_per_token_batch1(model, gpu) * max(1, n_concurrent_request)

    def calc_tpot(
        self, model: ModelSpec, gpu: GPUSpec, n_concurrent_request: int = 1
    ) -> float:
        return self.calc_tpot_batch1(model, gpu) * max(1, n_concurrent_request)

    def calc_e2e_latency(
        self,
        prefill_time_per_token: float,
        tpot: float,
        prompt_size: int,
        response_size: int,
    ) -> float:
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    def calculate_metrics(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int = 1,
    ) -> PerformanceMetrics:
        context_window = prompt_size + response_size
        kv_cache_tokens = floor(
            self.calc_kv_cache_tokens(
                gpu, model, self.calc_kv_cache_size_per_token(model, context_window)
            )
        )
        tpot_b1 = self.calc_tpot_batch1(model, gpu)
        feasible = self.fits(gpu, model, n_concurrent_request, context_window)

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
        e2e = self.calc_e2e_latency(prefill, tpot, prompt_size, response_size)
        return PerformanceMetrics(
            kv_cache_tokens=kv_cache_tokens,
            feasible=True,
            prefill_time_per_token=prefill,
            tpot=tpot,
            ttft=(prompt_size * prefill + tpot) / 1000,
            e2e_latency=e2e,
            throughput=(
                n_concurrent_request * response_size / e2e if e2e > 0 else "INFEASIBLE"
            ),
            tpot_batch1=tpot_b1,
        )
