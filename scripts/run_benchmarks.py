#!/usr/bin/env python3
"""Run real Ollama benchmarking to generate latency metrics for NUANCE-MCP.

Requires [ollama] extra to be installed.
Usage:
    python scripts/run_benchmarks.py --runs 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src/ to path so we can run directly without installing
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

try:
    from nuance_mcp import build_server
    from nuance_mcp.agent import run_agent, AgentDependencyError
except ImportError as e:
    print(f"Error importing nuance_mcp: {e}\nDid you install the [ollama] extra?", file=sys.stderr)
    sys.exit(1)


# The exact models presented in the manuscript
DEFAULT_MODELS = [
    "llama3.2:latest",
    "qwen2.5:latest",
    "qwen2.5-coder:latest",
    "phi4:latest",
    "gemma2:latest",
    "nemotron:latest",  # Note: ensure you have models pulled with these exact tags
    "mistral:latest"
]

BENCHMARK_PROMPT = (
    "Using your tools, check what capabilities the adapter supports, "
    "read the current microscope state, and tell me the High Tension."
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def benchmark_model(server, model: str, runs: int) -> list[float]:
    latencies = []
    logging.info(f"\n--- Benchmarking {model} ({runs} runs) ---")
    
    for i in range(runs):
        start_t = time.perf_counter()
        try:
            # temperature=0.0 to reduce variability
            # verbose=False to keep standard output clear
            reply = await run_agent(server, BENCHMARK_PROMPT, model=model, temperature=0.0)
            end_t = time.perf_counter()
            duration = end_t - start_t
            latencies.append(duration)
            logging.info(f"  Run {i+1:02d}/{runs}: {duration:.2f} s")
            logging.debug(f"  Agent reply: {reply[:60]}...")
        except Exception as e:
            end_t = time.perf_counter()
            duration = end_t - start_t
            logging.error(f"  Run {i+1:02d}/{runs}: FAILED after {duration:.2f} s. Error: {e}")
            # If the model is not found in Ollama, we usually want to stop testing it
            if "pull" in str(e).lower() or "not found" in str(e).lower():
                logging.error(f"  Aborting {model} because it is likely not pulled/installed locally.")
                break
            
    return latencies


async def main():
    parser = argparse.ArgumentParser(description="NUANCE-MCP Benchmark Script")
    parser.add_argument("--runs", type=int, default=15, help="Number of iterations per model (default: 15)")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS, help="List of Ollama models to benchmark")
    parser.add_argument("--output", type=str, default="scripts/benchmark_results.json", help="Path to save the JSON results")
    args = parser.parse_args()

    # The simulator doesn't need physical hardware, so we can hammer it all day
    logging.info("Starting simulator FastMCP server...")
    server = build_server("simulator")
    
    results = {}
    
    try:
        for model in args.models:
            latencies = await benchmark_model(server, model, args.runs)
            
            mean_lat = sum(latencies) / len(latencies) if latencies else None
            results[model] = {
                "latencies": latencies,
                "mean_latency": mean_lat,
                "success_count": len(latencies),
                "total_runs": args.runs
            }
            
    except AgentDependencyError as e:
        logging.error(f"Agent dependencies missing: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("\nBenchmark interrupted by user.")
        
    logging.info("\nBenchmark completed. Summary:")
    for m, d in results.items():
        mean_str = f"{d['mean_latency']:.2f} s" if d['mean_latency'] is not None else "N/A"
        logging.info(f"  {m:<20} {d['success_count']:>2d}/{d['total_runs']} successful, mean: {mean_str}")
        
    output_path = Path(REPO / args.output)
    output_path.write_text(json.dumps(results, indent=2))
    logging.info(f"\nSaved raw JSON results to {output_path}")
    logging.info("You can now update the plot script to read from this JSON.")


if __name__ == "__main__":
    asyncio.run(main())
