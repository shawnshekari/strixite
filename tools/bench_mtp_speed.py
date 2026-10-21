#!/usr/bin/env python3
"""
tools/bench_mtp_speed.py: Pure decode speed and internal speculative telemetry benchmark.

Measures via Strixite's native /metrics endpoint:
1. Sustained generation throughput (tok/s)
2. MTP draft acceptance rate (%) and tokens yielded per forward pass
3. Speculative rollbacks and rollback distribution (0/1/2/3+)
4. Internal PLE GPU I/O stall latency (ms)
5. Skipped prefetch counter
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BENCHMARK_TASKS = [
    {
        "id": "code_spsc_queue",
        "category": "Code Generation",
        "prompt": (
            "Implement in C++20 a concurrent lock-free bounded single-producer single-consumer (SPSC) "
            "ring buffer queue. Use std::atomic, alignas(hardware_destructive_interference_size), "
            "and explain the exact memory ordering (acquire-release) used. Include a complete, compilable struct."
        ),
        "max_tokens": 1024,
    },
    {
        "id": "code_quicksort",
        "category": "Algorithms",
        "prompt": (
            "Write a complete, highly optimized in-place Introsort or Quicksort in C++20 with "
            "median-of-three pivot selection, tail recursion elimination, and cutoff to insertion sort "
            "for partitions smaller than 16 elements. Include a complete compilable example."
        ),
        "max_tokens": 1024,
    },
    {
        "id": "math_combinatorics",
        "category": "Math Reasoning",
        "prompt": (
            "Solve this step-by-step: An urn contains 5 red, 7 blue, and 8 green balls (20 total). "
            "If 4 balls are drawn at random without replacement, what is the exact probability (as a simplified fraction) "
            "that at least two balls are green and at least one ball is red? "
            "Show all cases and end with: Final Answer: X/Y"
        ),
        "max_tokens": 1024,
    },
    {
        "id": "formal_prime_proof",
        "category": "Formal Proof",
        "prompt": (
            "Prove that there are infinitely many primes of the form 4k + 3. "
            "Structure your proof clearly with contradiction, defining N = 4 * p1 * ... * pk - 1. "
            "Explain why at least one prime factor of N must be of the form 4m + 3."
        ),
        "max_tokens": 1024,
    },
    {
        "id": "system_consensus",
        "category": "System Design",
        "prompt": (
            "Explain the difference between Raft and Multi-Paxos regarding leader election, "
            "log compaction, and handling uncommitted log entries across term changes. "
            "Be precise about safety invariants."
        ),
        "max_tokens": 1024,
    },
]


def fetch_metrics(host, port):
    url = f"http://{host}:{port}/metrics"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"Warning: failed to read {url}: {e}", file=sys.stderr)
        return {}

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            key, val = parts[0], parts[1]
            try:
                metrics[key] = float(val)
            except ValueError:
                pass
    return metrics


def query_chat(host, port, prompt, max_tokens):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error querying {url}: {e}", file=sys.stderr)
        return None, 0.0
    t1 = time.time()
    return data, t1 - t0


def run_benchmark(host, port):
    results = []
    print("=" * 105)
    print(f"  STRIXITE SPECULATIVE DECODE & INTERNAL TELEMETRY BENCHMARK ({host}:{port})")
    print("=" * 105)
    print(f"{'Task ID':22} | {'Tokens':6} | {'Speed':10} | {'MTP Acc':9} | {'Yield':11} | {'Rollbacks':9} | {'PLE Stall':10}")
    print("-" * 105)

    for task in BENCHMARK_TASKS:
        tid = task["id"]
        m0 = fetch_metrics(host, port)
        data, elapsed = query_chat(host, port, task["prompt"], task["max_tokens"])
        if not data:
            continue
        m1 = fetch_metrics(host, port)

        usage = data.get("usage", {})
        comp_tok = usage.get("completion_tokens", 0)
        speed = comp_tok / elapsed if elapsed > 0 else 0.0

        # Metrics delta
        drafted = m1.get("strix_mtp_drafted_tokens_total", 0) - m0.get("strix_mtp_drafted_tokens_total", 0)
        accepted = m1.get("strix_mtp_accepted_tokens_total", 0) - m0.get("strix_mtp_accepted_tokens_total", 0)
        rollbacks = int(m1.get("strix_mtp_rollbacks_total", 0) - m0.get("strix_mtp_rollbacks_total", 0))
        r0 = int(m1.get('strix_mtp_rejections_total{accepted="0"}', 0) - m0.get('strix_mtp_rejections_total{accepted="0"}', 0))
        r1 = int(m1.get('strix_mtp_rejections_total{accepted="1"}', 0) - m0.get('strix_mtp_rejections_total{accepted="1"}', 0))
        r2 = int(m1.get('strix_mtp_rejections_total{accepted="2"}', 0) - m0.get('strix_mtp_rejections_total{accepted="2"}', 0))
        r3 = int(m1.get('strix_mtp_rejections_total{accepted="3+"}', 0) - m0.get('strix_mtp_rejections_total{accepted="3+"}', 0))
        ple_wait_ms = (m1.get("strix_ple_wait_seconds_total", 0) - m0.get("strix_ple_wait_seconds_total", 0)) * 1000.0
        ple_skipped = int(m1.get("strix_ple_prefetch_skipped_total", 0) - m0.get("strix_ple_prefetch_skipped_total", 0))

        acc_pct = (100.0 * accepted / drafted) if drafted > 0 else 0.0
        # Effective token yield per verify pass (tokens / number of trunk forwards)
        forwards = rollbacks + 1
        tok_per_fwd = comp_tok / forwards if forwards > 0 else 1.0

        row = {
            "id": tid,
            "category": task["category"],
            "completion_tokens": comp_tok,
            "elapsed_seconds": elapsed,
            "speed_tok_s": speed,
            "mtp_drafted": drafted,
            "mtp_accepted": accepted,
            "mtp_acceptance_rate": acc_pct,
            "tokens_per_forward": tok_per_fwd,
            "rollbacks": rollbacks,
            "rollback_distribution": f"{r0}/{r1}/{r2}/{r3}",
            "ple_stall_ms": ple_wait_ms,
            "ple_prefetch_skipped": ple_skipped,
        }
        results.append(row)

        print(
            f"{tid:22} | {comp_tok:6} | {speed:7.2f} t/s | {acc_pct:7.1f}% | {tok_per_fwd:7.2f} t/f | {rollbacks:9} | {ple_wait_ms:7.2f} ms"
        )

    if not results:
        return None

    avg_speed = sum(r["speed_tok_s"] for r in results) / len(results)
    avg_acc = sum(r["mtp_acceptance_rate"] for r in results) / len(results)
    total_rollbacks = sum(r["rollbacks"] for r in results)
    total_ple_stall = sum(r["ple_stall_ms"] for r in results)

    print("-" * 105)
    print(
        f"{'AVERAGE / TOTAL':22} | {'-':6} | {avg_speed:7.2f} t/s | {avg_acc:7.1f}% | {'-':11} | {total_rollbacks:9} | {total_ple_stall:7.2f} ms"
    )
    print("=" * 105)

    return {
        "average_speed_tok_s": avg_speed,
        "average_mtp_acceptance": avg_acc,
        "total_rollbacks": total_rollbacks,
        "total_ple_stall_ms": total_ple_stall,
        "tasks": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Strixite MTP Speed & Telemetry Benchmark")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8731, help="Server port")
    parser.add_argument("--output", default="bench_mtp_telemetry.json", help="Output JSON path")
    args = parser.parse_args()

    summary = run_benchmark(args.host, args.port)
    if not summary:
        print("Benchmark failed.")
        return 1

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Detailed benchmark saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
