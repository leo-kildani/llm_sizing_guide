#!/usr/bin/env python3
"""
LLM Performance Calculator

Enterprise-conservative estimates for memory footprint, concurrent capacity,
and latency under assumed load for different GPU and model configurations.
"""

import argparse
from typing import List, Dict, Any

from configs.gpu_specs import GPU_SPECS, GPUSpec
from configs.model_specs import MODEL_SPECS, ModelSpec
from llm_calculator.performance import (
    PerformanceCalculator,
    DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
    DEFAULT_VRAM_UTILIZATION,
    DEFAULT_KV_FRAGMENTATION,
    DEFAULT_QUANT,
    DEFAULT_KV_QUANT,
    WEIGHT_BYTES_BY_QUANT,
    KV_BYTES_BY_QUANT,
    resolve_weight_bytes,
    resolve_kv_bytes,
)
from llm_calculator.reporting import PerformanceReporter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate LLM inference performance metrics'
    )
    parser.add_argument(
        '-g', '--num_gpu',
        type=int, default=1,
        help='Number of GPUs (tensor-parallel replica size)'
    )
    parser.add_argument(
        '-p', '--prompt_sz',
        type=int, default=4096,
        help='Prompt size in tokens'
    )
    parser.add_argument(
        '-r', '--response_sz',
        type=int, default=256,
        help='Response size in tokens'
    )
    parser.add_argument(
        '-c', '--n_concurrent_req',
        type=int, default=10,
        help='Assumed concurrent requests (SLA / footprint case)'
    )
    parser.add_argument(
        '--system-overhead-gb',
        type=float, default=DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
        help='System/framework overhead GB per GPU (default: %(default)s)'
    )
    parser.add_argument(
        '--vram-util',
        type=float, default=DEFAULT_VRAM_UTILIZATION,
        help='Fraction of VRAM treated as usable (default: %(default)s)'
    )
    parser.add_argument(
        '--kv-frag',
        type=float, default=DEFAULT_KV_FRAGMENTATION,
        help='KV fragmentation multiplier (default: %(default)s)'
    )
    parser.add_argument(
        '--quant',
        choices=sorted(WEIGHT_BYTES_BY_QUANT),
        default=DEFAULT_QUANT,
        help='Weight quantization (default: %(default)s)'
    )
    parser.add_argument(
        '--kv-quant',
        choices=sorted(KV_BYTES_BY_QUANT),
        default=DEFAULT_KV_QUANT,
        help='KV-cache quantization (default: %(default)s; independent of --quant)'
    )
    return parser.parse_args()


def check_memory_requirements(
    calculator: PerformanceCalculator,
    model: ModelSpec,
    gpu: GPUSpec,
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int,
) -> None:
    """Warn when assumed concurrency does not fit usable VRAM."""
    context_window = prompt_size + response_size
    if calculator.fits(gpu, model, n_concurrent_request, context_window):
        return

    max_n = calculator.calc_max_concurrent_requests(gpu, model, context_window)
    print(
        f"\n!!!! INFEASIBLE {model.name}: n_concurrent_request={n_concurrent_request} "
        f"does not fit ISL={prompt_size} OSL={response_size} on "
        f"{calculator.num_gpu}x {gpu.name}\n"
        f"Memory footprint={calculator.calc_memory_footprint(model, n_concurrent_request, context_window):.2f} GB "
        f"vs usable VRAM={calculator.usable_vram_gib(gpu):.2f} GB\n"
        f"Max concurrent at this context: {max_n}"
    )


def calculate_memory_footprint(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    gpus: List[GPUSpec],
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int,
    weight_quant: str,
    kv_quant: str,
) -> List[Dict[str, Any]]:
    """Memory footprint for each model×GPU under assumed concurrency."""
    table = []
    context_window = prompt_size + response_size

    for model in models:
        model_memory = calculator.weight_gib(model)
        kv_avg = calculator.calc_kv_cache_gib_for_seq(model, context_window)
        kv_max = calculator.calc_kv_cache_gib_for_seq(
            model, model.max_context_window
        )
        footprint = calculator.calc_memory_footprint(
            model, n_concurrent_request, context_window
        )
        for gpu in gpus:
            total_vram = calculator.num_gpu * gpu.memory_gb
            table.append(
                PerformanceReporter.format_memory_footprint_row(
                    model.name,
                    gpu.name,
                    prompt_size,
                    response_size,
                    n_concurrent_request,
                    weight_quant,
                    kv_quant,
                    total_vram,
                    model_memory,
                    kv_avg,
                    kv_max,
                    calculator.available_for_kv_gib(gpu, model),
                    footprint,
                    calculator.fits(gpu, model, n_concurrent_request, context_window),
                )
            )
    return table


def calculate_performance_metrics(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    gpus: List[GPUSpec],
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int,
) -> List[Dict[str, Any]]:
    """Latency metrics for all model×GPU combos (INFEASIBLE if not fit)."""
    table = []
    for model in models:
        for gpu in gpus:
            metrics = calculator.calculate_metrics(
                model, gpu, prompt_size, response_size, n_concurrent_request
            )
            table.append(
                PerformanceReporter.format_performance_row(
                    model.name,
                    gpu.name,
                    prompt_size,
                    response_size,
                    n_concurrent_request,
                    metrics,
                )
            )
    return table


def calculate_concurrent_capacity(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    gpus: List[GPUSpec],
    avg_context_window: int,
) -> List[Dict[str, Any]]:
    """Estimate concurrent users from model + GPU memory (no assumed -c)."""
    table = []
    for model in models:
        for gpu in gpus:
            capacity = calculator.calc_concurrent_capacity(
                gpu, model, avg_context_window
            )
            table.append(
                PerformanceReporter.format_concurrent_capacity_row(
                    model.name, gpu.name, calculator.num_gpu, capacity
                )
            )
    return table


def main() -> None:
    """Main execution function."""
    args = parse_args()
    avg_context_window = args.prompt_sz + args.response_sz
    weight_bytes = resolve_weight_bytes(args.quant)
    kv_bytes = resolve_kv_bytes(args.kv_quant)

    print(
        f" num_gpu = {args.num_gpu}, prompt_size = {args.prompt_sz} tokens, "
        f"response_size = {args.response_sz} tokens"
    )
    print(f" n_concurrent_request = {args.n_concurrent_req}")
    print(f" avg_context_window = {avg_context_window} tokens (prompt + response)")
    print(
        f" system_overhead = {args.system_overhead_gb} GB/GPU, "
        f"vram_util = {args.vram_util}, kv_frag = {args.kv_frag}"
    )
    print(
        f" weight_quant = {args.quant} ({weight_bytes} B/param), "
        f"kv_quant = {args.kv_quant} ({kv_bytes} B/elem)"
    )

    calculator = PerformanceCalculator(
        args.num_gpu,
        system_overhead_gb_per_gpu=args.system_overhead_gb,
        vram_utilization=args.vram_util,
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        kv_fragmentation=args.kv_frag,
    )
    reporter = PerformanceReporter()

    memory_footprint_table = calculate_memory_footprint(
        calculator,
        MODEL_SPECS,
        GPU_SPECS,
        args.prompt_sz,
        args.response_sz,
        args.n_concurrent_req,
        args.quant,
        args.kv_quant,
    )
    reporter.print_table(
        memory_footprint_table,
        "******************** Estimate LLM Memory Footprint ********************",
    )
    memory_csv_file = reporter.save_to_csv(
        memory_footprint_table, 'llm_memory_footprint'
    )

    for model in MODEL_SPECS:
        for gpu in GPU_SPECS:
            check_memory_requirements(
                calculator,
                model,
                gpu,
                args.prompt_sz,
                args.response_sz,
                args.n_concurrent_req,
            )

    concurrent_table = calculate_concurrent_capacity(
        calculator, MODEL_SPECS, GPU_SPECS, avg_context_window
    )
    reporter.print_table(
        concurrent_table,
        "******************** Estimate Concurrent Capacity ********************",
    )
    concurrent_csv_file = reporter.save_to_csv(
        concurrent_table, 'llm_concurrent_capacity'
    )

    performance_table = calculate_performance_metrics(
        calculator,
        MODEL_SPECS,
        GPU_SPECS,
        args.prompt_sz,
        args.response_sz,
        args.n_concurrent_req,
    )
    reporter.print_table(
        performance_table,
        "******************** Estimate LLM Capacity and Latency ********************",
    )
    perf_csv_file = reporter.save_to_csv(performance_table, 'llm_performance')

    print(
        f"\nResults saved to CSV files:\n"
        f"1. {memory_csv_file}\n"
        f"2. {concurrent_csv_file}\n"
        f"3. {perf_csv_file}"
    )


if __name__ == '__main__':
    main()
