#!/usr/bin/env python3
"""
tools/bench_kernel_optimizations.py

Reproducible benchmark suite for evaluating HIP kernel optimizations on AMD GFX1151 (Strix Halo).
Measures:
1. Decode throughput (tok/s) and per-token latency (ms/tok) across diverse workloads.
2. Time-To-First-Token (TTFT) and raw prefill throughput (tok/s) across context sizes.
3. Comparative reporting against baseline runs.

Zero third-party dependencies outside Python 3 standard library.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

BENCHMARKS = [
    {
        "name": "code_generation",
        "description": "C++20 Concurrency (Kernel decode stress)",
        "prompt": (
            "Write a production-ready C++20 concurrent lock-free bounded single-producer single-consumer (SPSC) "
            "ring buffer. Use std::atomic, alignas(hardware_destructive_interference_size), full memory order "
            "acquire/release annotations, and inline documentation for every function."
        ),
        "max_tokens": 1024,
    },
    {
        "name": "math_reasoning",
        "description": "Multi-step Combinatorics (MTP drafting & dense MLP stress)",
        "prompt": (
            "An urn contains 5 red, 7 blue, and 8 green balls (20 total). If 4 balls are drawn at random without replacement, "
            "calculate the exact probability as a simplified fraction that at least two balls are green and at least one is red. "
            "Show every single step, intermediate combination values, and final verification."
        ),
        "max_tokens": 1024,
    },
    {
        "name": "instruction_following",
        "description": "Deep Architecture Analysis (Long response decode stress)",
        "prompt": (
            "Analyze the memory subsystem of AMD RDNA 3.5 (GFX1151) architecture. Detail the L0/L1 cache hierarchy, "
            "Local Data Share (LDS) banking structure, Wave32 vs Wave64 execution modes, and how 256-bit LPDDR5X-8533 "
            "memory bandwidth impacts memory-bound LLM decoding operations."
        ),
        "max_tokens": 1024,
    },
]

def query_chat_stream(host, port, prompt, max_tokens=1024, temperature=0.0):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "thinking": {"type": "enabled"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    tokens = 0
    
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                    chunks.append(content)
            except Exception:
                pass
                
    t1 = time.perf_counter()
    total_time = t1 - t0
    gen_time = (t1 - (t0 + ttft)) if ttft else total_time
    # Overall tok/s and pure decode tok/s:
    overall_tps = tokens / total_time if total_time > 0 else 0
    decode_tps = (tokens - 1) / gen_time if (gen_time > 0 and tokens > 1) else overall_tps
    
    return {
        "tokens": tokens,
        "total_time_s": total_time,
        "ttft_s": ttft or 0.0,
        "decode_tps": decode_tps,
        "overall_tps": overall_tps,
        "ms_per_tok": (1000.0 / decode_tps) if decode_tps > 0 else 0.0,
        "content_length": sum(len(c) for c in chunks),
    }

def run_prefill_bench(host, port, context_sizes):
    print("\n--- Running Context Prefill Scaling Benchmark ---")
    print(f"{'Target Ctx':<12} | {'Actual Prompt':<14} | {'Total Time':<12} | {'Prefill Rate':<14}")
    print("-" * 62)
    results = []
    filler = (
        "AMD Strix Halo features a unified system memory architecture with high-bandwidth 256-bit LPDDR5X channels. "
        "The RDNA 3.5 graphics compute units and Zen 5 CPU cores share a single coherent physical memory space. "
    )
    for ctx in context_sizes:
        repeat = max(10, ctx // 32)
        haystack = filler * repeat
        prompt = f"Background Document:\n{haystack}\n\nQuestion: Summarize the document in one word."
        res = query_chat_stream(host, port, prompt, max_tokens=16, temperature=0.0)
        # In short output, ttft reflects prefill time
        prefill_tps = (ctx / res["ttft_s"]) if res["ttft_s"] > 0 else 0.0
        print(f"~{ctx:<11} | {ctx:<14} | {res['ttft_s']:6.2f}s       | {prefill_tps:6.1f} t/s")
        results.append({
            "context_size": ctx,
            "ttft_s": res["ttft_s"],
            "prefill_tps": prefill_tps,
            "total_time_s": res["total_time_s"]
        })
    print("-" * 62)
    return results

def run_full_suite(host, port, tag, prefill_sizes=[4000, 16000, 32000]):
    print("=" * 72)
    print("      STRIXITE GFX1151 KERNEL OPTIMIZATION BENCHMARK SUITE")
    print("=" * 72)
    print(f"Host: {host}:{port} | Tag: {tag}\n")
    
    suite_results = {}
    for b in BENCHMARKS:
        print(f"[*] Running: {b['name']} ({b['description']})...")
        r = query_chat_stream(host, port, b["prompt"], max_tokens=b["max_tokens"])
        print(f"    Generated: {r['tokens']} tokens in {r['total_time_s']:.2f}s | TTFT: {r['ttft_s']*1000:.1f}ms")
        print(f"    Decode Rate: {r['decode_tps']:.2f} tok/s ({r['ms_per_tok']:.2f} ms/tok)")
        suite_results[b["name"]] = r

    prefill_results = run_prefill_bench(host, port, prefill_sizes)
    
    out_data = {
        "tag": tag,
        "timestamp": time.time(),
        "decode_benchmarks": suite_results,
        "prefill_benchmarks": prefill_results,
    }
    
    fname = f"bench_kernel_{tag}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[+] Results recorded to {fname}")
    return out_data

def compare_runs(baseline_file, opt_file):
    with open(baseline_file) as f: b = json.load(f)
    with open(opt_file) as f: o = json.load(f)
    
    print("\n" + "=" * 78)
    print("          GFX1151 KERNEL OPTIMIZATIONS: PERFORMANCE COMPARISON")
    print("=" * 78)
    print(f"{'Workload / Test':<28} | {'Baseline':<12} | {'Optimized':<12} | {'Speedup':<14}")
    print("-" * 78)
    
    b_dec = b.get("decode_benchmarks", {})
    o_dec = o.get("decode_benchmarks", {})
    for k in b_dec:
        if k in o_dec:
            b_tps = b_dec[k]["decode_tps"]
            o_tps = o_dec[k]["decode_tps"]
            diff = ((o_tps - b_tps) / b_tps) * 100.0 if b_tps > 0 else 0.0
            print(f"{k:<28} | {b_tps:5.1f} tok/s  | {o_tps:5.1f} tok/s  | {diff:+5.2f}% ({o_tps - b_tps:+4.1f} t/s)")
            
    print("-" * 78)
    b_pref = {p["context_size"]: p["prefill_tps"] for p in b.get("prefill_benchmarks", [])}
    o_pref = {p["context_size"]: p["prefill_tps"] for p in o.get("prefill_benchmarks", [])}
    for ctx in b_pref:
        if ctx in o_pref:
            bp = b_pref[ctx]
            op = o_pref[ctx]
            diff = ((op - bp) / bp) * 100.0 if bp > 0 else 0.0
            name = f"Prefill ~{ctx//1000}k tokens"
            print(f"{name:<28} | {bp:5.0f} t/s    | {op:5.0f} t/s    | {diff:+5.2f}% ({op - bp:+4.0f} t/s)")
    print("=" * 78)

def main():
    parser = argparse.ArgumentParser(description="GFX1151 Kernel Optimization Benchmark Suite")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8731, help="Server port")
    parser.add_argument("--record", metavar="TAG", help="Run suite and record results to bench_kernel_<TAG>.json")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "OPT"), help="Compare two benchmark runs")
    parser.add_argument("--prefill-sizes", nargs="+", type=int, default=[4000, 16000, 32000], help="Context sizes for prefill test")
    args = parser.parse_args()
    
    if args.record:
        run_full_suite(args.host, args.port, args.record, args.prefill_sizes)
    elif args.compare:
        compare_runs(args.compare[0], args.compare[1])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
