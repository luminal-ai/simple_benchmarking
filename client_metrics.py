"""
Client-side metrics collection for benchmark tool.

Handles sending requests, collecting client-perceived metrics (TTFT, E2E latency, tok/s),
and calculating aggregated statistics.
"""

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm


@dataclass
class RequestInput:
    """Input data for a single request."""
    prompt: str
    image_url: Optional[str]  # None for text-only models like llama
    request_id: int
    prompt_length: int  # Length of text prompt in characters
    image_size_bytes: int = 0  # Size of image in bytes (0 for text-only)


@dataclass
class RequestOutput:
    """Output and metrics for a single request."""
    request_id: int
    success: bool = False
    error: str = ""
    generated_text: str = ""
    ttft: float = 0.0  # Time to first token (seconds)
    e2e_latency: float = 0.0  # End-to-end latency (seconds)
    output_tokens: int = 0
    itl: List[float] = field(default_factory=list)  # Inter-token latencies
    prompt_tokens: int = 0  # Prompt tokens including image tokens
    completion_tokens: int = 0  # Completion tokens from usage


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics (client-side)."""
    # TTFT metrics (ms)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_avg_ms: float

    # E2E latency metrics (ms)
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    e2e_avg_ms: float

    # Tok/s metrics (per-request, not separated by prefill/decode)
    toks_p50: float
    toks_p95: float
    toks_p99: float
    toks_avg: float

    # ITL (inter-token latency) metrics (ms)
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_avg_ms: float


# --- Moondream Starmie tokenizer integration ---
_STARMIE = None

def _load_starmie_tokenizer():
    """
    Try to load moondream/starmie-v1 via `tokenizers` first (fast, no heavy deps).
    Fallback to `transformers` AutoTokenizer if needed.
    """
    global _STARMIE
    if _STARMIE is not None:
        return _STARMIE

    # Attempt 1: tokenizers (Rust) — tiny, fast
    try:
        from tokenizers import Tokenizer
        try:
            # recent versions support from_pretrained; otherwise we'll pull the file manually
            _STARMIE = Tokenizer.from_pretrained("moondream/starmie-v1")
            return _STARMIE
        except Exception:
            # Manual download of tokenizer.json
            from huggingface_hub import hf_hub_download
            tok_path = hf_hub_download("moondream/starmie-v1", filename="tokenizer.json")
            _STARMIE = Tokenizer.from_file(tok_path)
            return _STARMIE
    except Exception:
        pass

    # Attempt 2: transformers
    try:
        from transformers import AutoTokenizer
        _STARMIE = AutoTokenizer.from_pretrained(
            "moondream/starmie-v1",
            use_fast=True,
            trust_remote_code=True,
        )
        return _STARMIE
    except Exception as e:
        print(f"⚠️ Could not load starmie tokenizer: {e}")
        _STARMIE = None
        return None


def count_tokens_starmie(text: str) -> int:
    """
    Return token count using starmie-v1; fall back to naive word count if unavailable.
    """
    tok = _load_starmie_tokenizer()
    if tok is None:
        # Fallback — keeps the script running
        return len(text.split())

    # tokenizers.Tokenizer vs transformers tokenizer
    try:
        # Rust tokenizers
        return len(tok.encode(text).ids)
    except AttributeError:
        # transformers tokenizer
        return len(tok.encode(text, add_special_tokens=False))


def load_coco_dataset(num_samples: int, random_sample: bool = True) -> List[RequestInput]:
    """Load COCO dataset and prepare RequestInput objects."""
    try:
        from datasets import load_dataset
        import random
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading COCO-Caption2017 dataset from HuggingFace...")
    dataset = load_dataset("lmms-lab/COCO-Caption2017", split="val")
    print(f"Loaded {len(dataset)} images from COCO dataset")

    if len(dataset) > num_samples:
        if random_sample:
            indices = random.sample(range(len(dataset)), num_samples)
            sampled_dataset = dataset.select(indices)
        else:
            sampled_dataset = dataset.select(range(num_samples))
    else:
        print(f"Dataset has fewer than {num_samples} samples, using all {len(dataset)}")
        sampled_dataset = dataset

    requests = []
    for idx, example in enumerate(sampled_dataset):
        image = example.get("image")
        if image is None:
            continue
        import io
        import base64
        if image.mode in ["RGBA", "P", "LA"]:
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{img_str}"
        prompt = "Describe this image in detail."
        requests.append(RequestInput(
            prompt=prompt,
            image_url=image_data,
            request_id=idx,
            prompt_length=len(prompt),
            image_size_bytes=len(img_bytes)
        ))
    print(f"Created {len(requests)} requests from COCO dataset")
    return requests


def load_text_dataset(num_samples: int, random_sample: bool = True) -> List[RequestInput]:
    """Load text dataset for text-only models like llama."""
    try:
        from datasets import load_dataset
        import random
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading text dataset from HuggingFace...")

    # Try to load ShareGPT dataset, fallback to synthetic prompts if unavailable
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        print(f"Loaded {len(dataset)} conversations from ShareGPT dataset")

        if len(dataset) > num_samples:
            if random_sample:
                indices = random.sample(range(len(dataset)), num_samples)
                sampled_dataset = dataset.select(indices)
            else:
                sampled_dataset = dataset.select(range(num_samples))
        else:
            print(f"Dataset has fewer than {num_samples} samples, using all {len(dataset)}")
            sampled_dataset = dataset

        requests = []
        for idx, example in enumerate(sampled_dataset):
            conversations = example.get("conversations", [])
            if not conversations:
                continue
            # Get the first user message
            user_msg = next((c["value"] for c in conversations if c.get("from") == "human"), None)
            if user_msg:
                requests.append(RequestInput(
                    prompt=user_msg,
                    image_url=None,
                    request_id=idx,
                    prompt_length=len(user_msg),
                    image_size_bytes=0
                ))
    except Exception as e:
        print(f"Could not load ShareGPT dataset ({e}), using synthetic prompts instead")
        # Fallback: Generate synthetic prompts
        prompts = [
            "Explain the concept of quantum computing in simple terms.",
            "Write a short story about a robot learning to feel emotions.",
            "What are the main differences between Python and JavaScript?",
            "Describe the process of photosynthesis step by step.",
            "What are some effective strategies for time management?",
            "Explain how blockchain technology works.",
            "What are the health benefits of regular exercise?",
            "Describe the water cycle and its importance.",
            "What are the key principles of object-oriented programming?",
            "Explain the theory of relativity in layman's terms.",
            "What are the main causes of climate change?",
            "Describe the process of making bread from scratch.",
            "What are the differences between renewable and non-renewable energy?",
            "Explain how neural networks learn from data.",
            "What are the fundamental principles of economics?",
            "Describe the structure and function of DNA.",
            "What are the key differences between democracy and authoritarianism?",
            "Explain how vaccines work to protect against diseases.",
            "What are the main features of Gothic architecture?",
            "Describe the process of how movies are made from script to screen.",
        ]

        requests = []
        for idx in range(num_samples):
            prompt = prompts[idx % len(prompts)]
            requests.append(RequestInput(
                prompt=prompt,
                image_url=None,
                request_id=idx,
                prompt_length=len(prompt),
                image_size_bytes=0
            ))

    print(f"Created {len(requests)} text-only requests")
    return requests


def print_no_success_failure_messages(outputs: List[RequestOutput]) -> None:
    """Print aggregated failure reasons when no requests succeeded."""
    failures = [o for o in outputs if not o.success]
    if not failures:
        print("\nNo successful requests and no failures recorded (unexpected).")
        return

    counts = Counter((o.error or "Unknown error") for o in failures)

    print("\n" + "=" * 80)
    print(" All Requests Failed — Failure Reasons ".center(80, "="))
    print("=" * 80)
    total = len(failures)
    print(f"Total failed requests: {total}")
    for msg, cnt in counts.most_common():
        pct = (cnt / total) * 100.0
        print(f"[{cnt:>4} | {pct:5.1f}%] {msg}")

    # Show one concrete example to aid debugging
    example = failures[0]
    print("\nExample failure:")
    print(f"  request_id: {example.request_id}")
    print(f"  error     : {example.error}")
    print("=" * 80)


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    request_input: RequestInput,
    model: str,
    max_tokens: int,
    api_key: Optional[str] = None,
    pbar: Optional[tqdm] = None
) -> RequestOutput:
    """Send a single chat completion request and collect client-side metrics."""
    output = RequestOutput(request_id=request_input.request_id)

    # Build messages based on whether this is a vision or text-only request
    if request_input.image_url:
        # Vision model request (e.g., moondream)
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": request_input.image_url}},
                {"type": "text", "text": request_input.prompt}
            ]}
        ]
    else:
        # Text-only model request (e.g., llama)
        messages = [
            {"role": "user", "content": request_input.prompt}
        ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    # Add Authorization header if API key is provided
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.perf_counter()
    most_recent_timestamp = start_time
    generated_text = ""

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error = f"HTTP {response.status}: {response.reason}"
                output.success = False
                if pbar: pbar.update(1)
                return output

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk_str = chunk_bytes.decode("utf-8")
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]
                if chunk_str == "[DONE]":
                    break
                try:
                    data = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue

                # Check for usage information
                usage = data.get("usage")
                if usage:
                    output.prompt_tokens = usage.get("prompt_tokens", 0)
                    output.completion_tokens = usage.get("completion_tokens", 0)

                # Check if choices array has elements before accessing
                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    timestamp = time.perf_counter()
                    if output.ttft == 0.0:
                        output.ttft = timestamp - start_time
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
                    generated_text += content
                    output.output_tokens += 1

            output.e2e_latency = time.perf_counter() - start_time
            output.generated_text = generated_text
            # Use completion_tokens from API if available, otherwise use counted tokens
            if output.completion_tokens > 0:
                output.output_tokens = output.completion_tokens
            output.success = True

    except Exception as e:
        output.error = str(e)
        output.success = False

    if pbar:
        pbar.update(1)
    return output


async def generate_requests(requests: List[RequestInput], request_rate: float):
    """Generate requests at a specified rate using Poisson distribution."""
    for request in requests:
        yield request
        if request_rate == float('inf'):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def calculate_metrics(outputs: List[RequestOutput], duration_s: float) -> BenchmarkMetrics:
    """Calculate aggregated client-side metrics from request outputs."""
    successful = [o for o in outputs if o.success]
    if not successful:
        print_no_success_failure_messages(outputs)
        raise ValueError("No successful requests to calculate metrics from")

    ttfts = [o.ttft for o in successful if o.ttft > 0]
    e2e_latencies = [o.e2e_latency for o in successful]
    # Use total tokens (prompt + completion) from API usage for accurate tok/s including image tokens
    toks_per_sec = [
        (o.prompt_tokens + o.completion_tokens) / o.e2e_latency
        for o in successful
        if o.e2e_latency > 0 and (o.prompt_tokens + o.completion_tokens) > 0
    ]
    # Flatten all inter-token latencies from all requests
    all_itls = [latency for o in successful for latency in o.itl]

    return BenchmarkMetrics(
        ttft_p50_ms=float(np.percentile(ttfts, 50) * 1000) if ttfts else 0,
        ttft_p95_ms=float(np.percentile(ttfts, 95) * 1000) if ttfts else 0,
        ttft_p99_ms=float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
        ttft_avg_ms=float(np.mean(ttfts) * 1000) if ttfts else 0,
        e2e_p50_ms=float(np.percentile(e2e_latencies, 50) * 1000),
        e2e_p95_ms=float(np.percentile(e2e_latencies, 95) * 1000),
        e2e_p99_ms=float(np.percentile(e2e_latencies, 99) * 1000),
        e2e_avg_ms=float(np.mean(e2e_latencies) * 1000),
        toks_p50=float(np.percentile(toks_per_sec, 50)) if toks_per_sec else 0,
        toks_p95=float(np.percentile(toks_per_sec, 95)) if toks_per_sec else 0,
        toks_p99=float(np.percentile(toks_per_sec, 99)) if toks_per_sec else 0,
        toks_avg=float(np.mean(toks_per_sec)) if toks_per_sec else 0,
        itl_p50_ms=float(np.percentile(all_itls, 50) * 1000) if all_itls else 0,
        itl_p95_ms=float(np.percentile(all_itls, 95) * 1000) if all_itls else 0,
        itl_p99_ms=float(np.percentile(all_itls, 99) * 1000) if all_itls else 0,
        itl_avg_ms=float(np.mean(all_itls) * 1000) if all_itls else 0,
    )
