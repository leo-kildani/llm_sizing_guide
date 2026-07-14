"""Flask UI for the LLM sizing calculator."""

from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from configs.gpu_specs import GPU_SPECS, GPUSpec
from configs.model_specs import MODEL_SPECS, ModelSpec
from LLM_size_pef_calculator import (
    calculate_concurrent_capacity,
    calculate_memory_footprint,
    calculate_performance_metrics,
)
from llm_calculator.performance import (
    DEFAULT_KV_FRAGMENTATION,
    DEFAULT_KV_QUANT,
    DEFAULT_QUANT,
    DEFAULT_SYSTEM_OVERHEAD_GB_PER_GPU,
    DEFAULT_VRAM_UTILIZATION,
    KV_BYTES_BY_QUANT,
    WEIGHT_BYTES_BY_QUANT,
    PerformanceCalculator,
    resolve_kv_bytes,
    resolve_weight_bytes,
)

app = Flask(__name__)

# CLI defaults for the interactive form
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


def _infeasible_warnings(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    gpus: List[GPUSpec],
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int,
) -> List[str]:
    """Same feasibility checks as CLI check_memory_requirements, as strings."""
    warnings: List[str] = []
    context_window = prompt_size + response_size
    for model in models:
        for gpu in gpus:
            if calculator.fits(gpu, model, n_concurrent_request, context_window):
                continue
            max_n = calculator.calc_max_concurrent_requests(gpu, model, context_window)
            warnings.append(
                f"INFEASIBLE {model.name}: n_concurrent_request={n_concurrent_request} "
                f"does not fit ISL={prompt_size} OSL={response_size} on "
                f"{calculator.num_gpu}x {gpu.name}. "
                f"Memory footprint="
                f"{calculator.calc_memory_footprint(model, n_concurrent_request, context_window):.2f} GB "
                f"vs usable VRAM={calculator.usable_vram_gib(gpu):.2f} GB. "
                f"Max concurrent at this context: {max_n}"
            )
    return warnings


def _parse_calculate_body(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate request JSON; return (params, error_message)."""
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
    if quant not in WEIGHT_BYTES_BY_QUANT:
        return {}, f"Unknown quant {quant!r}; choose from {sorted(WEIGHT_BYTES_BY_QUANT)}"
    if kv_quant not in KV_BYTES_BY_QUANT:
        return {}, f"Unknown kv_quant {kv_quant!r}; choose from {sorted(KV_BYTES_BY_QUANT)}"

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
            "gpus": [g.name for g in GPU_SPECS],
            "weight_quants": sorted(WEIGHT_BYTES_BY_QUANT),
            "kv_quants": sorted(KV_BYTES_BY_QUANT),
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
        weight_bytes=resolve_weight_bytes(params["quant"]),
        kv_bytes=resolve_kv_bytes(params["kv_quant"]),
        kv_fragmentation=params["kv_frag"],
    )
    models = params["models"]
    gpus = params["gpus"]
    avg_context = params["prompt_sz"] + params["response_sz"]

    memory = calculate_memory_footprint(
        calculator,
        models,
        gpus,
        params["prompt_sz"],
        params["response_sz"],
        params["n_concurrent_req"],
        params["quant"],
        params["kv_quant"],
    )
    concurrent = calculate_concurrent_capacity(
        calculator, models, gpus, avg_context
    )
    performance = calculate_performance_metrics(
        calculator,
        models,
        gpus,
        params["prompt_sz"],
        params["response_sz"],
        params["n_concurrent_req"],
    )
    warnings = _infeasible_warnings(
        calculator,
        models,
        gpus,
        params["prompt_sz"],
        params["response_sz"],
        params["n_concurrent_req"],
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
    # Smoke: python -c "from app import app; c=app.test_client(); assert c.get('/api/options').status_code==200"
    app.run(debug=True)
