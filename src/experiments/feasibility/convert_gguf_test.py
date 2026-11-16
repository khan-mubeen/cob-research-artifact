"""
stage E probe: merged checkpoint -> GGUF -> quantize -> project-runtime load.

validates that the project's fine-tuned and merged checkpoint converts and runs
in the agent runtime (llama-cpp-python 0.3.34 cu132 in the main .venv); official
base-model GGUF availability does not cover this path.

steps, each recorded with commands, output tails, sizes, sha256:
  1 convert_hf_to_gguf.py (pinned clone) -> bf16 gguf
  2 llama-quantize.exe (pinned release) -> requested quant (default Q8_0)
  3 load the quantized gguf in the main venv llama_cpp and run one real
    bidding prompt, greedy, record raw output + json parse + latency

usage (run with the .venv-feas python):
  python src/experiments/feasibility/convert_gguf_test.py --merged-dir <dir> --tag <name> [--quant Q8_0]
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def _tool_root() -> Path:
    """llama.cpp / quantize tool location, configurable instead of hardcoded.
    override with LLAMA_TOOLS_ROOT; defaults to the historical local install."""
    return Path(os.getenv("LLAMA_TOOLS_ROOT", r"tools"))

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "reports" / "feasibility"
GGUF_DIR = REPO / "models" / "feasibility"

FEAS_PY = REPO / ".venv-feas" / "Scripts" / "python.exe"
MAIN_PY = REPO / ".venv" / "Scripts" / "python.exe"
CONVERT = _tool_root() / "llama.cpp" / "convert_hf_to_gguf.py"
QUANTIZE = _tool_root() / "llama-b10068" / "llama-quantize.exe"

LOAD_TEST_SNIPPET = r"""
import json, os, sys, time
os.environ['PATH'] = r'{repo}\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;' + os.environ['PATH']
from llama_cpp import Llama, __version__
recs = []
for stage in ('material', 'tailoring', 'logistics'):
    with open(rf'{repo}\data\finetune\v4\{{stage}}_train.jsonl'.format(stage=stage), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2: break
            recs.append(json.loads(line))
t0 = time.perf_counter()
llama = Llama(model_path=r'{gguf}', n_ctx=4096, n_gpu_layers=-1, verbose=False)
load_s = time.perf_counter() - t0
lat, toks, json_ok, first_raw = [], 0, 0, ''
for i, rec in enumerate(recs):
    t1 = time.perf_counter()
    out = llama.create_chat_completion(messages=[
        {{'role': 'system', 'content': rec['system']}},
        {{'role': 'user', 'content': rec['user']}}], temperature=0.0, max_tokens=300)
    lat.append(round(time.perf_counter() - t1, 2))
    toks += out['usage']['completion_tokens']
    text = out['choices'][0]['message']['content']
    json_ok += '"decision"' in text
    if i == 0: first_raw = text[:400]
steady = sorted(lat[1:])[len(lat[1:]) // 2] if len(lat) > 1 else lat[0]
print(json.dumps({{'llama_cpp_python': __version__, 'load_s': round(load_s, 1),
                   'latencies_s': lat, 'first_infer_s': lat[0], 'median_steady_s': steady,
                   'total_completion_tokens': toks,
                   'json_ok': str(json_ok) + '/' + str(len(recs)),
                   'raw_head': first_raw}}))
"""


def sha256(path, limit_gb=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log_path):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        log.write(proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-4000:] + "\n")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--quant", default="Q8_0")
    args = parser.parse_args()

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"convert_{args.tag}.log"
    result = {"tag": args.tag, "merged_dir": args.merged_dir, "quant": args.quant,
              "convert_script": "llama.cpp@571d0d540df04f25298d0e159e520d9fc62ed121",
              "quantize_bin": "llama.cpp release b10068"}

    bf16_path = GGUF_DIR / f"{args.tag}-bf16.gguf"
    quant_path = GGUF_DIR / f"{args.tag}-{args.quant}.gguf"

    proc = run([FEAS_PY, CONVERT, args.merged_dir, "--outfile", bf16_path, "--outtype", "bf16"],
               log_path)
    result["convert_rc"] = proc.returncode
    result["convert_stderr_tail"] = proc.stderr[-600:]
    if proc.returncode == 0 and bf16_path.exists():
        result["bf16_gib"] = round(bf16_path.stat().st_size / 1024**3, 2)

        proc = run([QUANTIZE, bf16_path, quant_path, args.quant], log_path)
        result["quantize_rc"] = proc.returncode
        if proc.returncode == 0 and quant_path.exists():
            result["quant_gib"] = round(quant_path.stat().st_size / 1024**3, 2)
            result["quant_sha256"] = sha256(quant_path)

            snippet = LOAD_TEST_SNIPPET.format(repo=str(REPO), gguf=str(quant_path))
            proc = run([MAIN_PY, "-c", snippet], log_path)
            result["runtime_load_rc"] = proc.returncode
            if proc.returncode == 0:
                try:
                    result["runtime_test"] = json.loads(proc.stdout.strip().splitlines()[-1])
                except Exception:
                    result["runtime_stdout_tail"] = proc.stdout[-600:]
            else:
                result["runtime_stderr_tail"] = proc.stderr[-600:]

    out_path = OUT_DIR / f"convert_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
