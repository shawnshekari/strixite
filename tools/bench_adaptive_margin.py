#!/usr/bin/env python3
"""
sweep_margin_slope.py:
Sweep margin_slope in [0.04, 0.06, 0.08, 0.10, 0.12]
with draft_min=4, draft_max=12, base_margin=1.5 on Strix Halo
to find the optimal escalation steepness.
"""

import sys
import time
import json
import urllib.request
import urllib.parse

PORT = 5400
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])

BASE_URL = f"http://127.0.0.1:{PORT}"

PROMPTS = [
    {
        "domain": "C++20 Network Engine",
        "system": "You are a principal systems architect specializing in ultra-low latency C++20.",
        "user": "Write a production-grade C++20 coroutine TCP echo engine using io_uring with ring buffers and state machine.",
        "max_tokens": 600
    },
    {
        "domain": "Linux Kernel Driver",
        "system": "You are a Linux kernel maintainer specializing in PCIe device drivers.",
        "user": "Implement a Linux kernel module for a PCIe network accelerator with MSI-X, DMA rings, and NAPI poll.",
        "max_tokens": 600
    },
    {
        "domain": "Kubernetes Helm Spec",
        "system": "You are a lead cloud-native infrastructure engineer.",
        "user": "Generate a complete Helm chart definition for a distributed banking processor with StatefulSet and NetworkPolicy.",
        "max_tokens": 600
    },
    {
        "domain": "OpenAPI 3.1 YAML Schema",
        "system": "You are a fintech API architect.",
        "user": "Write an exhaustive OpenAPI 3.1 YAML specification for an FX settlement service with JSON Schema schemas.",
        "max_tokens": 600
    },
    {
        "domain": "LSM-Tree Storage Engine",
        "system": "You are a database engine developer.",
        "user": "Design and implement in modern C++ an in-memory LSM tree with Leveled Compaction, Bloom filter, and SkipList.",
        "max_tokens": 600
    }
]

def set_mtp(draft: int, margin: float, escalating: bool, slope: float):
    params = {
        "draft": str(draft),
        "margin": f"{margin:.2f}",
        "escalating_margin": "1" if escalating else "0",
        "margin_slope": f"{slope:.3f}",
    }
    url = f"{BASE_URL}/set_mtp?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def run_query(task):
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["user"]}
        ],
        "temperature": 0.0,
        "max_tokens": task["max_tokens"],
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode())
    t1 = time.perf_counter()
    choice = res["choices"][0]
    content = choice["message"]["content"]
    usage = res.get("usage", {})
    comp_tokens = usage.get("completion_tokens", len(content.split()))
    gen_time = t1 - t0
    tok_per_sec = comp_tokens / gen_time if gen_time > 0 else 0
    return {
        "tokens": comp_tokens,
        "time_s": gen_time,
        "tok_per_sec": tok_per_sec
    }

def eval_config(label, draft, margin, escalating, slope):
    set_mtp(draft, margin, escalating, slope)
    tok_sum = 0
    time_sum = 0
    print(f"\n--- Testing: {label} ---")
    for idx, task in enumerate(PROMPTS):
        print(f"  [{idx+1}/5] {task['domain']:<26}: ", end="", flush=True)
        res = run_query(task)
        print(f"{res['tokens']} tok in {res['time_s']:.2f}s -> {res['tok_per_sec']:5.2f} tok/s")
        tok_sum += res["tokens"]
        time_sum += res["time_s"]
    avg_speed = tok_sum / time_sum if time_sum > 0 else 0
    print(f"  => Average Speed: {avg_speed:5.2f} tok/s")
    return {
        "label": label,
        "slope": slope,
        "avg_speed": avg_speed
    }

def main():
    print("=" * 80)
    print("SWEEPING MARGIN SLOPE IN [0.04, 0.06, 0.08, 0.10, 0.12]")
    print(f"Server Target: {BASE_URL}")
    print("=" * 80)

    # 1. Baseline
    base_res = eval_config("Baseline (Static 7, slope=0.00)", draft=7,
                           margin=1.5, escalating=False, slope=0.0)
    base_avg = base_res["avg_speed"]

    slopes = [0.04, 0.06, 0.08, 0.10, 0.12]
    results = []
    for s in slopes:
        res = eval_config(f"Escalating (slope={s:.2f})", draft=7,
                          margin=1.5, escalating=True, slope=s)
        res["speedup_vs_baseline"] = (res["avg_speed"] / base_avg - 1.0) * 100.0
        results.append(res)

    print("\n" + "=" * 80)
    print("SLOPE SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<30} | {'Avg Speed':<12} | {'vs Baseline':<15}")
    print("-" * 80)
    print(f"{base_res['label']:<30} | {base_avg:6.2f} tok/s | {'0.00% (Base)':<15}")
    best_res = base_res
    best_speed = base_avg
    for r in results:
        print(f"{r['label']:<30} | {r['avg_speed']:6.2f} tok/s | {r['speedup_vs_baseline']:>+7.2f}%")
        if r["avg_speed"] > best_speed:
            best_speed = r["avg_speed"]
            best_res = r
    print("=" * 80)
    print(f"OPTIMAL SLOPE: {best_res['label']} -> {best_res['avg_speed']:.2f} tok/s")
    print("=" * 80)

if __name__ == "__main__":
    main()
