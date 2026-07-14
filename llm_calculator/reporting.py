"""Reporting utilities for LLM performance calculations."""

import csv
from datetime import datetime
from typing import List, Dict, Any
from tabulate import tabulate

class PerformanceReporter:
    """Reporter class for generating and saving performance reports."""

    @staticmethod
    def save_to_csv(data: List[Dict[str, Any]], filename_prefix: str) -> str:
        """Save data to CSV file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{filename_prefix}_{timestamp}.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return filename

    @staticmethod
    def print_table(data: List[Dict[str, Any]], title: str = "") -> None:
        """Print data in tabulated format."""
        if title:
            print(f"\n{title}")
        print(tabulate(data, headers="keys", tablefmt='orgtbl'))

    @staticmethod
    def format_memory_footprint_row(
        model_name: str,
        gpu_name: str,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int,
        weight_quant: str,
        kv_quant: str,
        total_vram_gb: float,
        model_memory_gb: float,
        kv_per_request_avg_gb: float,
        kv_per_request_max_gb: float,
        available_for_inference_gb: float,
        memory_footprint: float,
        feasible: bool,
    ) -> Dict[str, Any]:
        """Format a row for memory footprint report."""
        return {
            'Model': model_name,
            'GPU': gpu_name,
            'Input Size (tokens)': prompt_size,
            'Output Size (tokens)': response_size,
            'Concurrent Requests': n_concurrent_request,
            'Weight Quant': weight_quant,
            'KV Quant': kv_quant,
            'Total VRAM Available': f"{total_vram_gb:.2f} GB",
            'Model Memory Required': f"{model_memory_gb:.2f} GB",
            'KV per Request @ Avg Context': f"{kv_per_request_avg_gb:.4f} GB",
            'KV per Request @ Max Context': f"{kv_per_request_max_gb:.4f} GB",
            'Available for Inference': f"{available_for_inference_gb:.2f} GB",
            'Memory Footprint': f"{memory_footprint:.2f} GB",
            'Fits': 'YES' if feasible else 'NO',
        }

    @staticmethod
    def format_performance_row(
        model_name: str,
        gpu_name: str,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int,
        metrics: 'PerformanceMetrics',
    ) -> Dict[str, Any]:
        """Format a row for performance metrics report."""
        def fmt(val, spec):
            return f"{val:{spec}}" if isinstance(val, float) else val

        return {
            'Model': model_name,
            'GPU': gpu_name,
            'Input Size (tokens)': prompt_size,
            'Output Size (tokens)': response_size,
            'Concurrent Requests': n_concurrent_request,
            'Fits': 'YES' if metrics.feasible else 'NO',
            'Max # KV Cache Tokens': str(metrics.kv_cache_tokens),
            'Prefill Time': (
                f"{metrics.prefill_time_per_token:.3f} ms"
                if isinstance(metrics.prefill_time_per_token, float)
                else metrics.prefill_time_per_token
            ),
            'TPOT (ms)': (
                f"{metrics.tpot:.3f} ms" if isinstance(metrics.tpot, float) else metrics.tpot
            ),
            'TPOT batch-1 (ceiling)': (
                f"{metrics.tpot_batch1:.3f} ms"
                if isinstance(metrics.tpot_batch1, float)
                else metrics.tpot_batch1
            ),
            'TTFT': fmt(metrics.ttft, '.3f') + (' s' if isinstance(metrics.ttft, float) else ''),
            'E2E Latency': (
                f"{metrics.e2e_latency:.1f} s"
                if isinstance(metrics.e2e_latency, float)
                else metrics.e2e_latency
            ),
            'Output Tokens Throughput': (
                f"{metrics.throughput:.2f} tokens/sec"
                if isinstance(metrics.throughput, float)
                else metrics.throughput
            ),
        }

    @staticmethod
    def format_concurrent_capacity_row(
        model_name: str,
        gpu_name: str,
        num_gpu: int,
        capacity: 'ConcurrentCapacity',
    ) -> Dict[str, Any]:
        """Format a row for model×GPU concurrent-user capacity (no assumed -c)."""
        return {
            'Model': model_name,
            'GPU': gpu_name,
            'Num GPUs': num_gpu,
            'Max Context Window': capacity.max_context_window,
            'Avg Context Window': capacity.avg_context_window,
            'Available for KV (GB)': f"{capacity.available_for_kv_gb:.2f}",
            'Max # KV Cache Tokens': capacity.kv_cache_window,
            'Concurrent @ Max Context': capacity.concurrent_at_max_context,
            'Concurrent @ Avg Context': capacity.concurrent_at_avg_context,
        }
