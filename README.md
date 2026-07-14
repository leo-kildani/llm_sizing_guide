# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements on different GPU architectures..
Blog: https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/

# Usage
Prerequisite: `pip install -r requirements.txt`

## Web UI
Interactive Flask UI to tweak GPUs, models, workload, and overheads and see the three result tables live:

```bash
python app.py
```

Then open http://127.0.0.1:5000/. Or: `flask --app app run`.

Smoke check (no browser):

```bash
python -c "from app import app; c=app.test_client(); assert c.get('/api/options').status_code==200"
```

## CLI
Here are the Flags and their abbreviations for the script.
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests you anticipate handling.

By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Example output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
 num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10

******************** Estimate LLM Memory Footprint ********************
| Model           |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests | KV Cache Size per Token   | Memory Footprint   |
|-----------------+-----------------------+------------------------+-----------------------+---------------------------+--------------------|
| Llama-3.1-8B    |                  4096 |                    256 |                    10 | 0.000122 GiB/token        | 21.31 GB           |
| Llama-3.1-70B   |                  4096 |                    256 |                    10 | 0.000305 GiB/token        | 153.28 GB          |
| Mistral-7B-v0.3 |                  4096 |                    256 |                    10 | 0.000122 GiB/token        | 19.31 GB           |
| Qwen2.5-14B     |                  4096 |                    256 |                    10 | 0.000183 GiB/token        | 37.37 GB           |

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU      |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Output Tokens Throughput   |
|-----------------+----------+-----------------------+------------------------+-----------------------+-------------------------+----------------+-------------+---------+---------------+----------------------------|
| Llama-3.1-8B    | L40s     |                  4096 |                    256 |                    10 |                 1441792 | 0.011 ms       | 4.630 ms    | 0.016 s | 1.2 s         | 208.05 tokens/sec          |
| Llama-3.1-8B    | H100 NVL |                  4096 |                    256 |                    10 |                 2949120 | 0.005 ms       | 1.026 ms    | 0.006 s | 0.3 s         | 907.24 tokens/sec          |
| Llama-3.1-8B    | H200 NVL |                  4096 |                    256 |                    10 |                 4489216 | 0.005 ms       | 0.833 ms    | 0.006 s | 0.2 s         | 1098.98 tokens/sec         |
| Llama-3.1-8B    | MI300X   |                  4096 |                    256 |                    10 |                 6160384 | 0.003 ms       | 0.755 ms    | 0.004 s | 0.2 s         | 1244.27 tokens/sec         |
| Llama-3.1-70B   | L40s     |                  4096 |                    256 |                    10 |                  170393 | 0.097 ms       | 40.509 ms   | 0.137 s | 10.8 s        | 23.78 tokens/sec           |
| Llama-3.1-70B   | H100 NVL |                  4096 |                    256 |                    10 |                  773324 | 0.042 ms       | 8.974 ms    | 0.051 s | 2.5 s         | 103.68 tokens/sec          |
| Llama-3.1-70B   | H200 NVL |                  4096 |                    256 |                    10 |                 1389363 | 0.042 ms       | 7.292 ms    | 0.049 s | 2.0 s         | 125.60 tokens/sec          |
| Llama-3.1-70B   | MI300X   |                  4096 |                    256 |                    10 |                 2057830 | 0.027 ms       | 6.604 ms    | 0.033 s | 1.8 s         | 142.20 tokens/sec          |
| Mistral-7B-v0.3 | L40s     |                  4096 |                    256 |                    10 |                 1458176 | 0.010 ms       | 4.051 ms    | 0.014 s | 1.1 s         | 237.78 tokens/sec          |
| Mistral-7B-v0.3 | H100 NVL |                  4096 |                    256 |                    10 |                 2965504 | 0.004 ms       | 0.897 ms    | 0.005 s | 0.2 s         | 1036.85 tokens/sec         |
| Mistral-7B-v0.3 | H200 NVL |                  4096 |                    256 |                    10 |                 4505600 | 0.004 ms       | 0.729 ms    | 0.005 s | 0.2 s         | 1255.98 tokens/sec         |
| Mistral-7B-v0.3 | MI300X   |                  4096 |                    256 |                    10 |                 6176768 | 0.003 ms       | 0.660 ms    | 0.003 s | 0.2 s         | 1422.02 tokens/sec         |
| Qwen2.5-14B     | L40s     |                  4096 |                    256 |                    10 |                  888012 | 0.020 ms       | 8.507 ms    | 0.029 s | 2.3 s         | 113.23 tokens/sec          |
| Qwen2.5-14B     | H100 NVL |                  4096 |                    256 |                    10 |                 1892898 | 0.009 ms       | 1.885 ms    | 0.011 s | 0.5 s         | 493.74 tokens/sec          |
| Qwen2.5-14B     | H200 NVL |                  4096 |                    256 |                    10 |                 2919628 | 0.009 ms       | 1.531 ms    | 0.010 s | 0.4 s         | 598.08 tokens/sec          |
| Qwen2.5-14B     | MI300X   |                  4096 |                    256 |                    10 |                 4033740 | 0.006 ms       | 1.387 ms    | 0.007 s | 0.4 s         | 677.15 tokens/sec          |