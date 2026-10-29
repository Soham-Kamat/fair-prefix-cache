"""Real-model hooks: calibrate prefill speed and demo KV prefix reuse.

Optional - the simulator runs without it. Needs ``pip install -e '.[model]'``.

    python -m engine.model --calibrate                 # tokens/s on this Mac
    python -m engine.model --demo                      # prefix reuse speed-up
    python -m engine.model --model gpt2 --calibrate    # smaller fallback

On Apple Silicon the PyTorch "mps" backend is used when available.
"""
from __future__ import annotations

import argparse
import time

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(model_name: str = DEFAULT_MODEL, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return tok, model, device


def _sync(device):
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def prefill_time(model, input_ids, device, past=None) -> float:
    import torch

    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        model(input_ids=input_ids, past_key_values=past, use_cache=True)
        _sync(device)
    return time.perf_counter() - t0


def calibrate(model_name: str = DEFAULT_MODEL, lengths=(256, 512, 1024, 2048), repeats: int = 3) -> float:
    """Measure prefill tokens/s. Pass the result to run_sweep --prefill-tps."""
    import torch

    tok, model, device = load(model_name)
    vocab = model.config.vocab_size
    prefill_time(model, torch.randint(0, vocab, (1, 64), device=device), device)  # warm-up
    rates = []
    for n in lengths:
        ids = torch.randint(0, vocab, (1, n), device=device)
        best = min(prefill_time(model, ids, device) for _ in range(repeats))
        rates.append(n / best)
        print(f"  {n:5d} tokens: {best * 1000:7.1f} ms  ->  {n / best:8.0f} tokens/s")
    tps = sorted(rates)[len(rates) // 2]
    print(f"{model_name} on {device}: median prefill ~{tps:.0f} tokens/s")
    return tps


def demo_prefix_reuse(model_name: str = DEFAULT_MODEL, prefix_len: int = 1500, suffix_len: int = 100) -> None:
    """Compute a shared prefix once, then serve a new suffix on top of its KV
    cache, and check the logits match a full recompute."""
    import copy

    import torch

    tok, model, device = load(model_name)
    vocab = model.config.vocab_size
    prefix = torch.randint(0, vocab, (1, prefix_len), device=device)
    suffix = torch.randint(0, vocab, (1, suffix_len), device=device)
    full = torch.cat([prefix, suffix], dim=1)
    with torch.no_grad():
        model(input_ids=prefix[:, :32])  # warm-up
        _sync(device)
        t0 = time.perf_counter()
        ref = model(input_ids=full, use_cache=False).logits[:, -1]
        _sync(device)
        t_full = time.perf_counter() - t0

        cached = model(input_ids=prefix, use_cache=True).past_key_values
        reuse_cache = copy.deepcopy(cached)       # the cache is mutated in place
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids=suffix, past_key_values=reuse_cache, use_cache=True).logits[:, -1]
        _sync(device)
        t_hit = time.perf_counter() - t0
    err = (out.float() - ref.float()).abs().max().item()
    print(f"full prefill {prefix_len + suffix_len} tokens: {t_full * 1000:.1f} ms")
    print(f"cache hit, suffix {suffix_len} tokens:       {t_hit * 1000:.1f} ms  ({t_full / t_hit:.1f}x faster)")
    print(f"max |logit diff| = {err:.2e}  (small = reuse is exact up to fp16 noise)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.model)
    if a.demo:
        demo_prefix_reuse(a.model)
    if not (a.calibrate or a.demo):
        ap.print_help()
