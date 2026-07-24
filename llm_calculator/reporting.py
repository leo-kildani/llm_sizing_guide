"""Format calculator results for the UI / API.

Column order matters: the UI renders keys in order and hides the trailing
"detail" columns until the user asks for them.
"""

from typing import Any, Dict, List, Tuple

from llm_calculator.performance import ConcurrentCapacity, PerformanceMetrics

# Primary keys first; trailing keys are detail columns (UI Details toggle).
_MEMORY_DETAIL = ("Arch", "Total VRAM")
_CONCURRENT_DETAIL = ("Free for KV", "Max Context Window", "Max KV Tokens")
_PERF_DETAIL = ("Prefill / Token", "TPOT @ batch 1", "Max KV Tokens")


def _fmt(val: Any, spec: str, unit: str = "") -> Any:
    if isinstance(val, float):
        return f"{val:{spec}}{unit}"
    return val


def _assert_detail_trailing(keys: List[str], detail: Tuple[str, ...]) -> None:
    detail_set = set(detail)
    primary = [k for k in keys if k not in detail_set]
    trailing = [k for k in keys if k in detail_set]
    assert keys == primary + trailing, (
        f"detail columns must trail primary: got {keys}, detail={detail}"
    )
    assert trailing == list(detail), f"detail order mismatch: {trailing} vs {detail}"


class PerformanceReporter:
    @staticmethod
    def format_memory_footprint_row(
        model_name: str,
        gpu_name: str,
        arch_label: str,
        total_vram_gb: float,
        model_memory_gb: float,
        kv_per_request_gb: float,
        kv_per_request_max_gb: float,
        free_for_kv_gb: float,
        memory_footprint: float,
        feasible: bool,
    ) -> Dict[str, Any]:
        return {
            "Model": model_name,
            "GPU": gpu_name,
            "Fits": "YES" if feasible else "NO",
            "Footprint": f"{memory_footprint:.2f} GB",
            "Weights": f"{model_memory_gb:.2f} GB",
            "Free for KV": f"{free_for_kv_gb:.2f} GB",
            "KV / Request": f"{kv_per_request_gb:.4f} GB",
            "KV @ Max Ctx": f"{kv_per_request_max_gb:.4f} GB",
            "Arch": arch_label,
            "Total VRAM": f"{total_vram_gb:.2f} GB",
        }

    @staticmethod
    def format_performance_row(
        model_name: str,
        gpu_name: str,
        metrics: PerformanceMetrics,
    ) -> Dict[str, Any]:
        return {
            "Model": model_name,
            "GPU": gpu_name,
            "Fits": "YES" if metrics.feasible else "NO",
            "TTFT": _fmt(metrics.ttft, ".2f", " s"),
            "TPOT": _fmt(metrics.tpot, ".2f", " ms"),
            "E2E": _fmt(metrics.e2e_latency, ".1f", " s"),
            "Throughput": _fmt(metrics.throughput, ".1f", " tok/s"),
            "Prefill / Token": _fmt(metrics.prefill_time_per_token, ".3f", " ms"),
            "TPOT @ batch 1": _fmt(metrics.tpot_batch1, ".2f", " ms"),
            "Max KV Tokens": f"{metrics.kv_cache_tokens:,}",
        }

    @staticmethod
    def format_concurrent_capacity_row(
        model_name: str,
        gpu_name: str,
        capacity: ConcurrentCapacity,
    ) -> Dict[str, Any]:
        return {
            "Model": model_name,
            "GPU": gpu_name,
            "Concurrent @ Workload": capacity.concurrent_at_avg_context,
            "Concurrent @ Max Ctx": capacity.concurrent_at_max_context,
            "Free for KV": f"{capacity.available_for_kv_gb:.2f} GB",
            "Max Context Window": f"{capacity.max_context_window:,}",
            "Max KV Tokens": f"{capacity.kv_cache_window:,}",
        }


if __name__ == "__main__":
    mem = PerformanceReporter.format_memory_footprint_row(
        "m", "g", "Standard", 80.0, 10.0, 0.1, 0.2, 50.0, 12.0, True
    )
    _assert_detail_trailing(list(mem), _MEMORY_DETAIL)

    cap = PerformanceReporter.format_concurrent_capacity_row(
        "m",
        "g",
        ConcurrentCapacity(
            kv_cache_window=1000,
            max_context_window=128000,
            avg_context_window=4352,
            concurrent_at_max_context=1,
            concurrent_at_avg_context=4,
            available_for_kv_gb=40.0,
        ),
    )
    _assert_detail_trailing(list(cap), _CONCURRENT_DETAIL)

    metrics = PerformanceMetrics(
        feasible=True,
        ttft=1.0,
        tpot=10.0,
        e2e_latency=2.0,
        throughput=100.0,
        prefill_time_per_token=0.1,
        tpot_batch1=5.0,
        kv_cache_tokens=1000,
    )
    perf = PerformanceReporter.format_performance_row("m", "g", metrics)
    _assert_detail_trailing(list(perf), _PERF_DETAIL)
    print("reporting column order ok")
