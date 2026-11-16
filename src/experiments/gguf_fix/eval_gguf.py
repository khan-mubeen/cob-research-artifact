"""run the frozen eval sets against a quantized GGUF in the project runtime
(llama-cpp-python 0.3.34), producing the same per-row records and summaries as
eval_finetune.py so native / merged / quantized results are directly
comparable. single-stream greedy, latency recorded per example.

usage (main .venv python, PATH must include the venv nvidia dlls):
  python src/experiments/gguf_fix/eval_gguf.py --gguf models/epoch-1/deployment/qwen-r1-gguf-fix/qwen35-r1-nomtp-matched-Q8_0.gguf --tag r1_q8_matched
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.environ["PATH"] = (str(REPO / ".venv" / "Lib" / "site-packages" / "nvidia" /
                          "cu13" / "bin" / "x86_64") + ";" + os.environ["PATH"])

from llama_cpp import Llama, __version__ as llama_cpp_version  # noqa: E402

# reuse the exact set loading and scoring used for the hf-runtime evals
from src.experiments.eval_finetune import load_eval_sets, score, summarize  # noqa: E402

MAX_NEW_TOKENS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    sets = load_eval_sets()
    for name, examples in sets.items():
        print(f"{name}: {len(examples)} examples")

    t0 = time.perf_counter()
    llama = Llama(model_path=args.gguf, n_ctx=4096,
                  n_gpu_layers=args.n_gpu_layers, verbose=False)
    load_s = time.perf_counter() - t0
    print(f"loaded {args.gguf} in {load_s:.1f}s "
          f"(llama-cpp-python {llama_cpp_version}, n_gpu_layers={args.n_gpu_layers})")

    results = {"tag": args.tag, "model": args.gguf, "runtime": f"llama-cpp-python {llama_cpp_version}",
               "load_seconds": round(load_s, 1),
               "generated_at": datetime.now().isoformat(timespec="seconds")}
    all_latencies = []
    total_tokens = 0
    for name, examples in sets.items():
        print(f"\ngenerating for {name}...")
        outputs, latencies = [], []
        started = time.perf_counter()
        for i, e in enumerate(examples):
            t1 = time.perf_counter()
            out = llama.create_chat_completion(
                messages=[{"role": "system", "content": e["system"]},
                          {"role": "user", "content": e["user"]}],
                temperature=0.0, max_tokens=MAX_NEW_TOKENS)
            latencies.append(time.perf_counter() - t1)
            total_tokens += out["usage"]["completion_tokens"]
            outputs.append(out["choices"][0]["message"]["content"])
            if (i + 1) % 100 == 0 or i + 1 == len(examples):
                print(f"  {i + 1}/{len(examples)} "
                      f"({(i + 1) / (time.perf_counter() - started):.2f} ex/s)", flush=True)
        rows = score(examples, outputs)
        results[name] = {"summary": summarize(name, rows), "rows": rows,
                         "latency_s": {"median": round(statistics.median(latencies), 3),
                                       "mean": round(statistics.mean(latencies), 3),
                                       "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
                                       "first": round(latencies[0], 3)}}
        all_latencies.extend(latencies)

    baseline_rows = results["baseline_six"]["rows"]
    correct = sum(r["model_decision"] == "NO_BID" for r in baseline_rows)
    results["baseline_six"]["no_bid_correct"] = correct
    print(f"\nbaseline six: {correct}/6 NO_BID correct")
    results["overall_latency"] = {
        "n": len(all_latencies),
        "median_s": round(statistics.median(all_latencies), 3),
        "total_completion_tokens": total_tokens,
        "tokens_per_s_overall": round(total_tokens / sum(all_latencies), 1),
    }

    out_dir = REPO / "reports" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_gguf_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
