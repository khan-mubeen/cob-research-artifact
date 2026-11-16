"""run the deferred GPU verification for built GGUFs

convert_matched.py does this at the end of a build, but it needs about 5 GB of
VRAM and only about 3.4 GB is free while a batch runs, so builds that happen
alongside a batch are run with --skip-runtime and verified here in a gap.

gates, all three required before a model is used for a run:
  block_count matches the model's own config num_hidden_layers
  no orphan block one past the end
  6/6 parseable json from real generations

usage:
  python -m src.experiments.dynamic_resources.verify_ggufs [--only granite,gemma]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MAIN_PY = REPO / ".venv" / "Scripts" / "python.exe"
CUDA = REPO / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"

GEN = r"""
import json, os, time
os.environ['PATH'] = r'{cuda}' + ';' + os.environ['PATH']
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
lat, ok = [], 0
for rec in recs:
    msgs = rec['messages'][:-1] if 'messages' in rec else [
        {{'role': 'system', 'content': rec['system']}}, {{'role': 'user', 'content': rec['user']}}]
    t = time.perf_counter()
    out = llama.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=512)
    lat.append(time.perf_counter() - t)
    txt = out['choices'][0]['message']['content']
    try:
        json.loads(txt[txt.index('{{'):txt.rindex('}}') + 1]); ok += 1
    except Exception:
        pass
lat_sorted = sorted(lat[1:]) or lat
print(json.dumps({{'llama_cpp_python': __version__, 'load_s': round(load_s, 1),
                  'median_steady_s': round(lat_sorted[len(lat_sorted) // 2], 2),
                  'json_ok': f'{{ok}}/{{len(recs)}}'}}))
"""


def layer_count(config_path):
    """num_hidden_layers, wherever this architecture keeps it

    multimodal wrappers such as gemma4 nest the text tower under text_config, so
    the count is not always at the top level"""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if "num_hidden_layers" in cfg:
        return cfg["num_hidden_layers"]
    for key in ("text_config", "language_model", "llm_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict) and "num_hidden_layers" in nested:
            return nested["num_hidden_layers"]
    return None


def targets(only):
    out = []
    for mode, models_root, reports_root in (
            ("sealed", REPO / "models/epoch-2/deployment", REPO / "reports/epoch-2/deployment"),
            ("auction", REPO / "models/reverse_auction/epoch-1/v4/deployment",
             REPO / "reports/reverse_auction/v4_deployment")):
        if not models_root.is_dir():
            continue
        for d in sorted(models_root.iterdir()):
            if not d.is_dir() or not d.name.endswith(f"-{mode}"):
                continue
            slug = d.name[: -len(mode) - 1]
            if only and slug not in only:
                continue
            gguf = d / f"{slug}-{mode}-nomtp-matched-Q8_0.gguf"
            if gguf.is_file():
                out.append((slug, mode, gguf, d, reports_root / d.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    results, failed = [], []
    for slug, mode, gguf, model_dir, report_dir in targets(only):
        report = report_dir / f"convert_{slug}-{mode}-nomtp-matched.json"
        data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
        if data.get("runtime_test", {}).get("json_ok") == "6/6":
            print(f"{slug} {mode}: already verified")
            continue

        merged_cfg = model_dir / "merged" / "config.json"
        want = layer_count(merged_cfg) if merged_cfg.is_file() else None
        meta = data.get("quant_meta", {})
        gates = {
            "block_count matches config": want is None or meta.get("block_count") == want,
            "no orphan block": meta.get("orphan_block") is False,
        }
        proc = subprocess.run([str(MAIN_PY), "-c", GEN.format(cuda=CUDA, repo=REPO, gguf=gguf)],
                              capture_output=True, text=True, cwd=REPO)
        rt = {}
        if proc.returncode == 0 and proc.stdout.strip():
            rt = json.loads(proc.stdout.strip().splitlines()[-1])
        gates["json 6/6"] = rt.get("json_ok") == "6/6"

        data["runtime_test"] = rt or {"error": (proc.stderr or "")[-400:]}
        data["verify_gates"] = gates
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(data, indent=2), encoding="utf-8")

        passed = all(gates.values())
        flag = "PASS" if passed else "FAIL"
        print(f"{slug} {mode}: {flag}  blocks={meta.get('block_count')} want={want} "
              f"json={rt.get('json_ok')} steady={rt.get('median_steady_s')}s")
        for k, v in gates.items():
            if not v:
                print(f"    failed gate: {k}")
        (results if passed else failed).append(f"{slug} {mode}")

    print(f"\nverified {len(results)}: {', '.join(results) or 'none'}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
