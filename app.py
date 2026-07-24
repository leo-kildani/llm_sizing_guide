"""Flask UI for the LLM sizing calculator."""

from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from configs.gpu_specs import GPU_SPECS, GPUSpec
from configs.model_specs import MODEL_SPECS, ModelSpec, arch_label
from llm_calculator.performance import (
    BYTES_BY_QUANT,
    DEFAULT_KV_FRAGMENTATION,
    DEFAULT_KV_QUANT,
    DEFAULT_QUANT,
    DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
    DEFAULT_VRAM_UTILIZATION,
    PerformanceCalculator,
    resolve_bytes,
)
from llm_calculator.reporting import PerformanceReporter

app = Flask(__name__)
app.json.sort_keys = False  # keep reporter column order for UI detail split

DEFAULT_NUM_GPU = 1
DEFAULT_PROMPT_SZ = 4096
DEFAULT_RESPONSE_SZ = 256
DEFAULT_N_CONCURRENT = 10


def _filter_specs(
    models: Optional[List[str]], gpus: Optional[List[str]]
) -> Tuple[List[ModelSpec], List[GPUSpec]]:
    selected_models = MODEL_SPECS
    selected_gpus = GPU_SPECS
    if models:
        wanted = set(models)
        selected_models = [m for m in MODEL_SPECS if m.name in wanted]
    if gpus:
        wanted = set(gpus)
        selected_gpus = [g for g in GPU_SPECS if g.name in wanted]
    return selected_models, selected_gpus


def _parse_calculate_body(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        num_gpu = int(data.get("num_gpu", DEFAULT_NUM_GPU))
        prompt_sz = int(data.get("prompt_sz", DEFAULT_PROMPT_SZ))
        response_sz = int(data.get("response_sz", DEFAULT_RESPONSE_SZ))
        n_concurrent = int(data.get("n_concurrent_req", DEFAULT_N_CONCURRENT))
        system_overhead_gb = float(
            data.get("system_overhead_gb", DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU)
        )
        vram_util = float(data.get("vram_util", DEFAULT_VRAM_UTILIZATION))
        kv_frag = float(data.get("kv_frag", DEFAULT_KV_FRAGMENTATION))
        quant = str(data.get("quant", DEFAULT_QUANT)).lower()
        kv_quant = str(data.get("kv_quant", DEFAULT_KV_QUANT)).lower()
    except (TypeError, ValueError) as exc:
        return {}, f"Invalid numeric field: {exc}"

    if num_gpu < 1 or prompt_sz < 1 or response_sz < 1 or n_concurrent < 1:
        return {}, "num_gpu, prompt_sz, response_sz, and n_concurrent_req must be >= 1"
    if system_overhead_gb < 0 or kv_frag <= 0:
        return {}, "system_overhead_gb must be >= 0 and kv_frag must be > 0"
    if not (0 < vram_util <= 1):
        return {}, "vram_util must be in (0, 1]"
    if quant not in BYTES_BY_QUANT:
        return {}, f"Unknown quant {quant!r}; choose from {sorted(BYTES_BY_QUANT)}"
    if kv_quant not in BYTES_BY_QUANT:
        return {}, f"Unknown kv_quant {kv_quant!r}; choose from {sorted(BYTES_BY_QUANT)}"

    gpu_names = data.get("gpu_names") or []
    model_names = data.get("model_names") or []
    if not isinstance(gpu_names, list) or not isinstance(model_names, list):
        return {}, "gpu_names and model_names must be lists"

    models, gpus = _filter_specs(model_names or None, gpu_names or None)
    if not models:
        return {}, "No matching models"
    if not gpus:
        return {}, "No matching GPUs"

    return {
        "num_gpu": num_gpu,
        "prompt_sz": prompt_sz,
        "response_sz": response_sz,
        "n_concurrent_req": n_concurrent,
        "system_overhead_gb": system_overhead_gb,
        "vram_util": vram_util,
        "kv_frag": kv_frag,
        "quant": quant,
        "kv_quant": kv_quant,
        "models": models,
        "gpus": gpus,
    }, None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/options")
def api_options():
    return jsonify(
        {
            "models": [m.name for m in MODEL_SPECS],
            "model_details": [
                {
                    "name": m.name,
                    "architecture": m.architecture,
                    "arch_label": arch_label(m.architecture),
                    "params_billion": m.params_billion,
                    "max_context_window": m.max_context_window,
                }
                for m in MODEL_SPECS
            ],
            "gpus": [g.name for g in GPU_SPECS],
            "gpu_details": [
                {"name": g.name, "memory_gb": g.memory_gb} for g in GPU_SPECS
            ],
            "weight_quants": sorted(BYTES_BY_QUANT),
            "kv_quants": sorted(BYTES_BY_QUANT),
            "defaults": {
                "num_gpu": DEFAULT_NUM_GPU,
                "prompt_sz": DEFAULT_PROMPT_SZ,
                "response_sz": DEFAULT_RESPONSE_SZ,
                "n_concurrent_req": DEFAULT_N_CONCURRENT,
                "system_overhead_gb": DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
                "vram_util": DEFAULT_VRAM_UTILIZATION,
                "kv_frag": DEFAULT_KV_FRAGMENTATION,
                "quant": DEFAULT_QUANT,
                "kv_quant": DEFAULT_KV_QUANT,
                "gpu_names": [g.name for g in GPU_SPECS],
                "model_names": [m.name for m in MODEL_SPECS],
            },
        }
    )


@app.post("/api/calculate")
def api_calculate():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body required"}), 400

    params, err = _parse_calculate_body(data)
    if err:
        return jsonify({"error": err}), 400

    calculator = PerformanceCalculator(
        params["num_gpu"],
        system_overhead_gb_per_gpu=params["system_overhead_gb"],
        vram_utilization=params["vram_util"],
        weight_bytes=resolve_bytes(params["quant"]),
        kv_bytes=resolve_bytes(params["kv_quant"]),
        kv_fragmentation=params["kv_frag"],
    )

    memory, concurrent, performance, warnings = [], [], [], []
    context_window = params["prompt_sz"] + params["response_sz"]

    for model in params["models"]:
        model_memory = calculator.weight_gib(model)
        # per_request_gb, not raw KV, so the column reconciles with the footprint for
        # architectures that also hold a fixed recurrent state.
        kv_per_request = calculator.per_request_gb(model, context_window)
        kv_max = calculator.per_request_gb(model, model.max_context_window)
        footprint = calculator.calc_memory_footprint(
            model, params["n_concurrent_req"], context_window
        )

        for gpu in params["gpus"]:
            total_vram = calculator.num_gpu * gpu.memory_gb
            fits = calculator.fits(
                gpu, model, params["n_concurrent_req"], context_window
            )

            if not fits:
                max_n = calculator.calc_max_concurrent_requests(
                    gpu, model, context_window
                )
                warnings.append(
                    f"INFEASIBLE {model.name}: n_concurrent_request={params['n_concurrent_req']} "
                    f"does not fit ISL={params['prompt_sz']} OSL={params['response_sz']} on "
                    f"{calculator.num_gpu}x {gpu.name}. "
                    f"Memory footprint={footprint:.2f} GB vs usable VRAM={calculator.usable_vram_gib(gpu):.2f} GB. "
                    f"Max concurrent at this context: {max_n}"
                )

            memory.append(
                PerformanceReporter.format_memory_footprint_row(
                    model.name,
                    gpu.name,
                    arch_label(model.architecture),
                    total_vram,
                    model_memory,
                    kv_per_request,
                    kv_max,
                    calculator.available_for_kv_gib(gpu, model),
                    footprint,
                    fits,
                )
            )
            concurrent.append(
                PerformanceReporter.format_concurrent_capacity_row(
                    model.name,
                    gpu.name,
                    calculator.calc_concurrent_capacity(gpu, model, context_window),
                )
            )
            performance.append(
                PerformanceReporter.format_performance_row(
                    model.name,
                    gpu.name,
                    calculator.calculate_metrics(
                        model,
                        gpu,
                        prompt_size=params["prompt_sz"],
                        response_size=params["response_sz"],
                        n_concurrent_request=params["n_concurrent_req"],
                    ),
                )
            )

    return jsonify(
        {
            "memory_footprint": memory,
            "concurrent_capacity": concurrent,
            "performance": performance,
            "warnings": warnings,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
