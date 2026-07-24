# LLM Sizing Guide

Estimate memory footprint, concurrent capacity, and latency for LLM inference across GPUs.

Blog: https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000/ (or `flask --app app run`).

Smoke check:

```bash
python -c "from app import app; c=app.test_client(); assert c.get('/api/options').status_code==200"
```

## Layout

| Path | Role |
|------|------|
| `app.py` | Flask UI + `/api/options`, `/api/calculate` |
| `configs/gpu_specs.py` | GPU catalog |
| `configs/model_specs.py` | Model catalog |
| `llm_calculator/performance.py` | Memory / latency math |
| `llm_calculator/reporting.py` | API row formatters |
| `static/`, `templates/` | Web UI |
