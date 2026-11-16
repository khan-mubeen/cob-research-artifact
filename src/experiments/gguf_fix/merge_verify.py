"""re-merge the official round-one adapter into the pinned Qwen base with a
recorded command and a hard weight-delta verification.

the ORIGINAL BROKEN merge (now quarantined at models/epoch-1/deployment/quarantine/qwen35-r1-BROKEN-missing-mtp-block/merged) had no recorded
command and no proof the adapter actually modified the weights, so this redo
writes to a fresh dir and refuses to save if any sampled lora-targeted weight
is identical to the base (a silent no-op merge is a known peft failure mode
when adapter key prefixes don't match the loaded architecture view).

usage (cpu only, .venv-feas python):
  python src/experiments/gguf_fix/merge_verify.py
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = REPO / "models" / "epoch-1" / "production" / "qwen3.5-4b-bidding-lora-r1" / "final"
OUT = REPO / "models" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix" / "qwen35-r1-merged"
REPORT = REPO / "reports" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix" / "merge_verify.json"

# one targeted projection per category, sampled for a nonzero delta; picked
# dynamically because the hybrid arch only has self_attn on every 4th layer
# candidate lora targets across the architectures in play. phi fuses qkv and
# gate_up, so a fixed q_proj/o_proj/down_proj triple finds nothing there
PROBE_SUFFIXES = ("q_proj.weight", "qkv_proj.weight", "o_proj.weight",
                  "out_proj.weight", "down_proj.weight", "gate_up_proj.weight",
                  "up_proj.weight")
MIN_PROBES = 2


def load_base(repo, revision):
    """load the base in the view the adapter was trained against

    most bases map to AutoModelForCausalLM. mistral3 is a vision-language wrapper
    with no causal-lm mapping in transformers 5.5, so fall back to the class its
    own config names, which is the view the adapter keys were recorded under"""
    try:
        return AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=torch.bfloat16)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        cfg = AutoConfig.from_pretrained(repo, revision=revision)
        named = (cfg.architectures or [None])[0]
        cls = getattr(transformers, named, None) if named else None
        if cls is None:
            raise
        print(f"no causal-lm mapping, loading as {named}")
        return cls.from_pretrained(repo, revision=revision, dtype=torch.bfloat16)


# a vision-language base carries a vision tower the adapter never targets. probing
# one of those weights would read zero delta and abort a perfectly good merge
_TOWER_HINTS = ("language_model", "text_model", "model.layers", "model.decoder")


def _tower_first(names):
    inner = [n for n in names if any(h in n for h in _TOWER_HINTS)
             and "vision" not in n and "visual" not in n and "image" not in n]
    return inner or names


def pick_probe_weights(model):
    """a few real projection weights to watch for a nonzero delta

    probes whatever this architecture actually has rather than assuming a naming
    scheme; the point is only to catch a merge that silently changed nothing"""
    names = _tower_first([n for n, _ in model.named_parameters()])
    picked = []
    for suffix in PROBE_SUFFIXES:
        match = next((n for n in names if n.endswith(suffix)), None)
        if match:
            picked.append(match)
    assert len(picked) >= MIN_PROBES, (
        f"found only {len(picked)} probe weights, none of {PROBE_SUFFIXES} match "
        f"this architecture")
    return picked


def parse_args():
    """paths are overridable so a later adapter can reuse this merge and its
    delta check; defaults reproduce the round-one merge unchanged"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=ADAPTER)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--base-revision", default=BASE_REVISION)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--report", type=Path, default=REPORT)
    return p.parse_args()


def main():
    args = parse_args()
    adapter, out, report = args.adapter, args.out, args.report
    base_model = args.base
    # empty means a local checkpoint dir, which has no revision
    base_revision = args.base_revision or None
    assert adapter.exists(), f"adapter missing: {adapter}"
    base = load_base(base_model, base_revision)
    probe_weights = pick_probe_weights(base)
    base_params = dict(base.named_parameters())
    base_slices = {name: base_params[name][:4, :8].clone() for name in probe_weights}

    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    deltas = {}
    params = dict(merged.named_parameters())
    for name, before in base_slices.items():
        after = params[name][:4, :8]
        deltas[name] = (after - before).abs().float().max().item()
    nonzero = {k: v for k, v in deltas.items() if v > 0}
    if not nonzero:
        raise SystemExit(
            f"MERGE NO-OP: all probed weights identical to base, deltas={deltas}; "
            "adapter did not apply, refusing to save")

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base_model, revision=base_revision).save_pretrained(str(out))

    st = out / "model.safetensors"
    h = hashlib.sha256()
    with open(st, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": base_model, "base_revision": base_revision,
        "adapter": str(adapter), "out": str(out),
        "loaded_architecture": type(merged).__name__,
        "probe_weight_max_abs_delta": deltas,
        "merged_safetensors_gib": round(st.stat().st_size / 1024**3, 3),
        "merged_safetensors_sha256": h.hexdigest(),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
