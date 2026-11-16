"""phase 6: the token length of the rendered v5 training text, per model tokenizer

  PYTHONPATH=. python -m src.experiments.v5.token_lengths

each row is rendered exactly as src/experiments/finetune_lora.py renders it for training, the
system, user and assistant turns through the tokenizer's own chat template, and counted with
that tokenizer. per file, the 60 longest rows by character count and a seeded sample of 300 are
measured; the longest-by-characters rows are where the token maximum sits, so the reported
maximum is the maximum over those, and the sample gives the typical length

the six tokenizers are the epoch-2 family's, loaded from the local cache only, at the protocol's
pinned revisions. nothing is downloaded

writes reports/v5/phase6_token_lengths.json
"""

import heapq
import json
import random
import statistics
import sys

from config import PROJECT_ROOT

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
REPORT_PATH = PROJECT_ROOT / "reports" / "v5" / "phase6_token_lengths.json"
LONGEST_PER_FILE = 60
SAMPLE_PER_FILE = 300
SEED = 42

MODELS = {
    "qwen": {"source": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "kwargs": {}},
    "granite": {"source": "ibm-granite/granite-4.1-3b", "revision": "c0650403e44e78ec0262dab1c90914c65b196c4e", "kwargs": {}},
    "phi": {"source": str(PROJECT_ROOT / "models" / "base-checkpoints" / "hf" / "phi4mini-v2-base"), "revision": None, "kwargs": {}},
    "ministral": {"source": "mistralai/Ministral-3-3B-Instruct-2512-BF16", "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
                  "kwargs": {}},
    "gemma": {"source": "google/gemma-4-E2B-it", "revision": "179516f0c449474fdc46f08f30ead5b11e178497", "kwargs": {}},
    "smollm3": {"source": str(PROJECT_ROOT / "models" / "base-checkpoints" / "hf" / "smollm3-v2-base"), "revision": None,
                "kwargs": {"enable_thinking": False}},
}

FILES = {
    "sealed": [f"{stage}_train.jsonl" for stage in ("material", "tailoring", "logistics")],
    "reverse_auction": [f"reverse_auction/{stage}_train.jsonl" for stage in ("material", "tailoring", "logistics")],
    "supervisor": ["supervisor/selection_train.jsonl"],
}


def load_tokenizer(spec):
    from transformers import AutoProcessor, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["source"], revision=spec["revision"], local_files_only=True)
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(spec["source"], revision=spec["revision"], local_files_only=True)
    return getattr(tokenizer, "tokenizer", tokenizer)


def pick_rows(path):
    rng = random.Random(SEED)
    longest, sample, seen = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            seen += 1
            size = len(line)
            if len(longest) < LONGEST_PER_FILE:
                heapq.heappush(longest, (size, seen, line))
            elif size > longest[0][0]:
                heapq.heapreplace(longest, (size, seen, line))
            if len(sample) < SAMPLE_PER_FILE:
                sample.append(line)
            elif (j := rng.randrange(seen)) < SAMPLE_PER_FILE:
                sample[j] = line
    return [json.loads(x[2]) for x in longest], [json.loads(x) for x in sample]


def count(tokenizer, row, kwargs):
    messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["completion"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(FINETUNE_V5), help="corpus root to measure, default the accepted corpus")
    parser.add_argument("--modes", default=",".join(FILES), help="comma-separated modes to measure")
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()
    from pathlib import Path
    root = Path(args.root) if Path(args.root).is_absolute() else PROJECT_ROOT / args.root
    modes = [m for m in args.modes.split(",") if m]
    rows = {}
    for mode, names in FILES.items():
        if mode not in modes:
            continue
        longest, sample = [], []
        for name in names:
            l, s = pick_rows(root / name)
            longest += l
            sample += s
        rows[mode] = (longest, sample)
        print(f"{mode}: {len(longest)} longest and {len(sample)} sampled rows")

    result = {"method": __doc__.strip().splitlines()[0], "longest_per_file": LONGEST_PER_FILE,
              "sample_per_file": SAMPLE_PER_FILE, "models": {}}
    for name, spec in MODELS.items():
        try:
            tokenizer = load_tokenizer(spec)
        except Exception as exc:
            result["models"][name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"{name}: tokenizer unavailable, {exc}")
            continue
        per_mode = {}
        for mode, (longest, sample) in rows.items():
            long_counts = [count(tokenizer, r, spec["kwargs"]) for r in longest]
            sample_counts = sorted(count(tokenizer, r, spec["kwargs"]) for r in sample)
            per_mode[mode] = {"max_over_longest_rows": max(long_counts),
                              "sample_median": statistics.median(sample_counts),
                              "sample_mean": round(statistics.mean(sample_counts), 1),
                              "sample_p99": sample_counts[int(len(sample_counts) * 0.99) - 1]}
        result["models"][name] = per_mode
        print(f"{name}: {per_mode}")
    maxima = {mode: max(m[mode]["max_over_longest_rows"] for m in result["models"].values() if mode in m)
              for mode in rows}
    windows = (1024, 1280, 1536, 2048, 3072, 4096)
    result["max_tokens_any_model"] = maxima
    result["smallest_standard_window_that_fits"] = {mode: next(w for w in windows if w >= n) for mode, n in maxima.items()}
    result["v4_windows"] = {"stage_agents": 1024, "supervisor": 2048}
    result["root"] = str(root)
    report_path = Path(args.report) if Path(args.report).is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"max tokens {maxima}, windows {result['smallest_standard_window_that_fits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
