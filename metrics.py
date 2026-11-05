#!/usr/bin/env python3
"""Inference experiments against a running llama-server. Standard library only.

Each experiment matches a part of the workshop:
  1. Next-token probabilities      "The model predicts possibilities"
  2. One request, measured          "Fast means more than one number" (TTFT, gaps, tok/s)
  3. Prefill vs decode              "Prefill" and "Decode"
  4. Prefix cache                   "Caching can mean three different things"
  5. Batching                       "One accelerator can work on several requests"
  6. Memory                         "Quantization" and "The cache consumes memory"
"""

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

PARAGRAPH = (
    "Inference is the process of running a trained model on new input. "
    "The server turns text into tokens, processes the whole prompt in one pass called prefill, "
    "then generates one token at a time in a loop called decode. "
    "A key value cache keeps the attention state of earlier tokens so they are not computed again. "
)

BOLD, DIM, CYAN, YELLOW, PINK, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[35m", "\033[0m"


def heading(n, title, slide):
    print(f"\n{BOLD}{PINK}{n}. {title}{RESET}  {DIM}(slide: {slide}){RESET}")


def table(rows, headers):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(f"   {BOLD}{line}{RESET}")
    for r in rows:
        print("   " + "  ".join(f"{str(x):>{w}}" for x, w in zip(r, widths)))


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def chat(base, prompt, max_tokens=128, cache_prompt=True, echo=False):
    """Streams one chat request. Returns client timings plus the server's timings."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "cache_prompt": cache_prompt,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    arrivals, text, timings = [], [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    arrivals.append(time.perf_counter())
                    text.append(piece)
                    if echo:
                        print(piece, end="", flush=True)
    end = time.perf_counter()
    gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "ttft_ms": (arrivals[0] - start) * 1000 if arrivals else float("nan"),
        "total_ms": (end - start) * 1000,
        "gaps_ms": gaps,
        "text": "".join(text),
        "prompt_n": timings.get("prompt_n", 0),
        "cache_n": timings.get("cache_n", 0),
        "prefill_tps": timings.get("prompt_per_second", float("nan")),
        "decode_n": timings.get("predicted_n", 0),
        "decode_tps": timings.get("predicted_per_second", float("nan")),
    }


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def f(x, digits=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{digits}f}"


def next_token_probs(base):
    heading(1, "Next-token probabilities", "The model predicts possibilities, not a sentence")
    prompt = "The sky is"
    out = post(base + "/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 5, "temperature": 1})
    top = out["completion_probabilities"][0]["top_logprobs"]
    print(f'   "{prompt} ..."')
    for t in top:
        p = math.exp(t["logprob"])
        print(f"   {t['token']!r:>14}  {YELLOW}{'#' * max(1, round(p * 50))}{RESET} {p * 100:.1f}%")
    print(f"   {DIM}Temperature and top-p reshape this list before a token is picked.{RESET}")


def one_request(base):
    heading(2, "One request, measured", "Fast means more than one number")
    print(f"   {DIM}prompt: Explain inference in three sentences.{RESET}\n   ", end="")
    r = chat(base, "Explain inference in three sentences.", max_tokens=160, echo=True)
    print("\n")
    table(
        [
            ["time to first token (TTFT)", f(r["ttft_ms"]), "ms", "queue + prefill"],
            ["gap between tokens, median", f(pct(r["gaps_ms"], 50)), "ms", "one decode step"],
            ["gap between tokens, p95", f(pct(r["gaps_ms"], 95)), "ms", "the slow steps"],
            ["prefill speed", f(r["prefill_tps"]), "tok/s", f"{r['prompt_n']} prompt tokens"],
            ["decode speed", f(r["decode_tps"]), "tok/s", f"{r['decode_n']} generated tokens"],
            ["total time", f(r["total_ms"]), "ms", ""],
        ],
        ["metric", "value", "unit", "note"],
    )


def prefill_vs_decode(base):
    heading(3, "Prefill vs decode", "Prefill: process the prompt together. Decode: add a token, then repeat")
    rows = []
    for repeats in (1, 6, 18):
        prompt = PARAGRAPH * repeats + "\nSummarize the text above in one sentence."
        r = chat(base, prompt, max_tokens=64, cache_prompt=False)
        rows.append([r["prompt_n"], f(r["ttft_ms"]), f(r["prefill_tps"]), f(r["decode_tps"])])
    table(rows, ["prompt tokens", "TTFT ms", "prefill tok/s", "decode tok/s"])
    print(f"   {DIM}Longer prompts raise TTFT. Prefill reads many tokens per pass, so it runs far more")
    print(f"   tokens per second than decode, which produces one token per pass.{RESET}")


def prefix_cache(base):
    heading(4, "Prefix cache", "Caching can mean three different things")
    shared = "You are a tutor for an inference workshop.\n" + PARAGRAPH * 16
    rows = []
    for i, q in enumerate(("What is prefill?", "What is decode?")):
        r = chat(base, shared + "\nQuestion: " + q, max_tokens=48, cache_prompt=True)
        label = "first request" if i == 0 else "same prefix, new question"
        rows.append([label, r["cache_n"], r["prompt_n"], f(r["ttft_ms"])])
    table(rows, ["request", "reused tokens", "computed tokens", "TTFT ms"])
    print(f"   {DIM}The second request reuses the stored key/value state of the shared prefix,")
    print(f"   so it only prefills the new question.{RESET}")


def batching(base, parallel):
    heading(5, "Batching", "One accelerator can work on several requests")
    prompt = "Write a short paragraph about why GPUs are good at matrix math."
    rows = []
    for n in sorted({1, parallel}):
        results = [None] * n

        def worker(i):
            results[i] = chat(base, prompt + f" (variant {i})", max_tokens=128, cache_prompt=False)

        start = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - start
        tokens = sum(r["decode_n"] for r in results)
        rows.append(
            [
                n,
                f(statistics.mean(r["ttft_ms"] for r in results)),
                f(statistics.mean(r["decode_tps"] for r in results)),
                f(tokens / wall),
            ]
        )
    table(rows, ["requests at once", "mean TTFT ms", "tok/s per request", "total tok/s"])
    print(f"   {DIM}Latency is one person's wait. Throughput is the server's total work.{RESET}")


def memory(base, pid, backend):
    heading(6, "Memory", "Quantization: fewer bits per weight")
    meta = get(base + "/v1/models")["data"][0].get("meta", {})
    params, size = meta.get("n_params", 0), meta.get("size", 0)
    rows = [["model", meta.get("ftype", "?")], ["parameters", f"{params / 1e9:.2f} B"]]
    if params:
        rows.append(["bits per weight", f(size * 8 / params, 2)])
        rows.append(["weights on disk", f"{size / 2**20:,.0f} MiB"])
        rows.append(["same model at 16-bit", f"{params * 2 / 2**20:,.0f} MiB"])
    if pid:
        try:
            with open(f"/proc/{pid}/status") as fh:
                rss = next(int(line.split()[1]) for line in fh if line.startswith("VmRSS"))
            rows.append(["server RAM (RSS)", f"{rss / 1024:,.0f} MiB"])
        except (OSError, StopIteration):
            pass
    if backend == "cuda":
        try:
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            rows.append(["GPU memory used / total", used.splitlines()[0]])
        except (OSError, subprocess.CalledProcessError):
            pass
    table(rows, ["what", "value"])
    print(f"   {DIM}RAM above the weights holds the KV cache for every slot and the runtime buffers.{RESET}")


def main():
    try:
        run()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"server answered {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach the server: {e.reason}. Is ./run.sh running?")


def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--pid", type=int, default=0, help="llama-server process id, for RAM use")
    ap.add_argument("--backend", default="", help="cpu, cuda or metal")
    ap.add_argument("--parallel", type=int, default=4, help="requests sent at once in the batching test")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    props = get(base + "/props")
    print(f"\n{BOLD}{CYAN}Inference metrics{RESET}  {DIM}{base}  backend={a.backend or '?'}  "
          f"model={props.get('model_path', '?').rsplit('/', 1)[-1]}{RESET}")
    next_token_probs(base)
    one_request(base)
    prefill_vs_decode(base)
    prefix_cache(base)
    batching(base, a.parallel)
    memory(base, a.pid, a.backend)


if __name__ == "__main__":
    main()
