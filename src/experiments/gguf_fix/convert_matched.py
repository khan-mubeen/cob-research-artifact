"""version-matched GGUF conversion of the re-merged r1 checkpoint.

fix under test, from the root-cause analysis in reports/epoch-1/deployment/qwen-r1-gguf-fix/:
- converter pinned to llama.cpp e3546c79 (the exact commit vendored by the
  project runtime llama-cpp-python 0.3.34), not master
- --no-mtp, because the merged checkpoint kept mtp_num_hidden_layers=1 in
  config.json while the causal-lm merge dropped the mtp tensors; without the
  flag the converter writes block_count=33 with no blk.32.* tensors, an
  internally inconsistent file (that is what broke the feasibility artifact)

steps, all recorded: convert (bf16) -> verify block_count metadata ->
quantize Q8_0 (b10068, verified metadata-neutral) -> verify again -> load in
the MAIN .venv 0.3.34 runtime and run six real bidding prompts greedy.

usage (any python, spawns the right interpreters itself):
  python src/experiments/gguf_fix/convert_matched.py [--converter master]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _tool_root() -> Path:
    """llama.cpp / quantize tool location, configurable instead of hardcoded.
    override with LLAMA_TOOLS_ROOT; defaults to the historical local install."""
    return Path(os.getenv("LLAMA_TOOLS_ROOT", r"tools"))

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "reports" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix"
GGUF_DIR = REPO / "models" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix"
MERGED = GGUF_DIR / "qwen35-r1-merged"

FEAS_PY = REPO / ".venv-feas" / "Scripts" / "python.exe"
MAIN_PY = REPO / ".venv" / "Scripts" / "python.exe"
CONVERTERS = {
    "matched": _tool_root() / "llama.cpp-e3546c79" / "convert_hf_to_gguf.py",
    "master": _tool_root() / "llama.cpp" / "convert_hf_to_gguf.py",
}
CONVERTER_COMMITS = {
    "matched": "e3546c7948e3af463d0b401e6421d5a4c2faf565 (vendored by llama-cpp-python 0.3.34)",
    "master": "571d0d540df04f25298d0e159e520d9fc62ed121 (feasibility-era master)",
}
QUANTIZE = _tool_root() / "llama-b10068" / "llama-quantize.exe"

# only these accept --mtp / --no-mtp in the converter. matched on prefix
# because the variants carry suffixes, eg qwen3_5_text
MTP_ARCHITECTURES = ("qwen3_5", "qwen3_6", "step3_5")

BLOCK_COUNT_SNIPPET = r"""
import sys
sys.path.insert(0, r'{ggufpy}')
from gguf import GGUFReader
r = GGUFReader(r'{path}')
import json
arch = str(bytes(r.fields['general.architecture'].parts[-1]).decode())
blocks = int(r.fields[f'{{arch}}.block_count'].contents())
print(json.dumps({{
    'architecture': arch,
    'block_count': blocks,
    'tensor_count': len(r.tensors),
    'orphan_block': any(t.name.startswith(f'blk.{{blocks}}.') for t in r.tensors),
}}))
"""

LOAD_TEST_SNIPPET = r"""
import json, os, time
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log_path, env=None):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        log.write(proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-4000:] + "\n")
    return proc


def block_count_of(path, ggufpy):
    proc = subprocess.run(
        [str(FEAS_PY), "-c", BLOCK_COUNT_SNIPPET.format(ggufpy=ggufpy, path=path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[-400:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", choices=("matched", "master"), default="matched")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="conversion + metadata checks only, no GPU load test")
    # paths are overridable so a later merge can reuse this conversion and its
    # block-count check; defaults reproduce the round-one conversion unchanged
    parser.add_argument("--merged", type=Path, default=MERGED)
    parser.add_argument("--gguf-dir", type=Path, default=GGUF_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--name", default="qwen35-r1")
    args = parser.parse_args()
    merged, gguf_dir, out_dir = args.merged, args.gguf_dir, args.out_dir
    gguf_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_script = CONVERTERS[args.converter]
    ggufpy = convert_script.parent / "gguf-py"
    tag = f"{args.name}-nomtp-{args.converter}"
    log_path = out_dir / f"convert_{tag}.log"
    # --no-mtp exists to drop the multi-token-prediction block that otherwise
    # leaves block_count one past the real tensors. the converter only accepts it
    # for qwen3.5/3.6 and step3.5 and errors on anything else, so it is applied by
    # architecture rather than unconditionally
    model_type = json.loads((merged / "config.json").read_text(encoding="utf-8")).get("model_type", "")
    mtp_flags = ["--no-mtp"] if model_type.startswith(MTP_ARCHITECTURES) else []

    result = {"tag": tag, "merged_dir": str(merged), "model_type": model_type,
              "converter": CONVERTER_COMMITS[args.converter],
              "flags": mtp_flags, "quantize_bin": "llama.cpp release b10068"}

    bf16_path = gguf_dir / f"{tag}-bf16.gguf"
    quant_path = gguf_dir / f"{tag}-Q8_0.gguf"

    t0 = time.perf_counter()
    proc = run([FEAS_PY, convert_script, merged, "--outfile", bf16_path,
                "--outtype", "bf16", *mtp_flags], log_path)
    result["convert_rc"] = proc.returncode
    result["convert_seconds"] = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0 or not bf16_path.exists():
        result["convert_stderr_tail"] = proc.stderr[-800:]
    else:
        result["bf16_gib"] = round(bf16_path.stat().st_size / 1024**3, 2)
        result["bf16_meta"] = block_count_of(bf16_path, ggufpy)

        proc = run([QUANTIZE, bf16_path, quant_path, "Q8_0"], log_path)
        result["quantize_rc"] = proc.returncode
        if proc.returncode == 0 and quant_path.exists():
            result["quant_gib"] = round(quant_path.stat().st_size / 1024**3, 2)
            result["quant_sha256"] = sha256(quant_path)
            result["quant_meta"] = block_count_of(quant_path, ggufpy)

            if not args.skip_runtime:
                snippet = LOAD_TEST_SNIPPET.format(repo=str(REPO), gguf=str(quant_path))
                proc = run([MAIN_PY, "-c", snippet], log_path)
                result["runtime_load_rc"] = proc.returncode
                if proc.returncode == 0:
                    result["runtime_test"] = json.loads(proc.stdout.strip().splitlines()[-1])
                else:
                    result["runtime_stderr_tail"] = proc.stderr[-800:]

    out_path = out_dir / f"convert_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result.get("runtime_load_rc", result.get("convert_rc")) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
