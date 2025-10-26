import requests
from openai import OpenAI
from prometheus_client.parser import text_string_to_metric_families

VLLM_BASE = "http://localhost:8000"
METRICS_URL = f"{VLLM_BASE}/metrics"

def scrape_vllm_metrics():
    text = requests.get(METRICS_URL, timeout=5).text
    out = {}
    for family in text_string_to_metric_families(text):
        name = family.name
        # pick a few common ones; add more as needed
        if name in {
            "vllm_num_requests_waiting",
            "vllm_num_requests_running",
            "vllm_gpu_cache_usage_perc",
            "vllm_cpu_cache_usage_perc",
            "vllm_request_success_count",
            "vllm_e2e_request_latency_seconds",
            "vllm_time_to_first_token_seconds",
            "vllm_time_per_output_token_seconds",
            "vllm_generation_tokens_count",
            "vllm_prompt_tokens_count",
        }:
            # Gauges & Counters
            if family.type in ("gauge", "counter"):
                # if there are labels, aggregate by label tuple
                vals = {}
                for sample in family.samples:
                    key = tuple(sorted(sample.labels.items()))
                    vals[key] = sample.value
                out[name] = vals if vals else family.samples[0].value if family.samples else None
            # Histograms (we’ll keep count & sum; buckets available in family.samples)
            if family.type == "histogram":
                total_count = None
                total_sum = None
                for s in family.samples:
                    if s.name.endswith("_count"):
                        total_count = s.value
                    elif s.name.endswith("_sum"):
                        total_sum = s.value
                out[name] = {"count": total_count, "sum": total_sum}
    return out

# --- your existing call ---
client = OpenAI(base_url=f"{VLLM_BASE}/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain LoRA in 2 sentences."}
    ],
    temperature=0.4,
    max_tokens=256,
)

print(resp.choices[0].message.content)

# Tokens per the OpenAI-compatible response:
print("usage:", resp.usage)  # prompt_tokens, completion_tokens, total_tokens

# Server-side metrics snapshot:
metrics = scrape_vllm_metrics()
print("vLLM metrics snapshot:", metrics)
