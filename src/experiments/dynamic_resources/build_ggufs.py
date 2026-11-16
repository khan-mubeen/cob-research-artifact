"""build the sealed and auction GGUFs for every model, CPU only

merge and convert are CPU and disk work and can run beside a GPU batch: measured
on a live batch, orders took 30.9/31.7/31.5 s against a 30.5 s baseline, because a
batch is window-bound and contention eats slack rather than wall time. the final
verification generation is NOT included here, it needs about 5 GB of VRAM and only
about 3.4 GB is free during a batch. run verify_ggufs.py in a gap instead.

three things vary by architecture and are handled rather than assumed:
  --no-mtp        only qwen3.5/3.6 and step3.5 accept it, the converter errors otherwise
  chat template   only qwen's stock template defaults to thinking and needs swapping
  block_count     checked against the model's own config, not a fixed 32

usage:
  python -m src.experiments.dynamic_resources.build_ggufs [--only qwen,granite] [--mode sealed|auction|both]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FEAS_PY = REPO / ".venv-feas" / "Scripts" / "python.exe"
MAIN_PY = REPO / ".venv" / "Scripts" / "python.exe"
MERGE = REPO / "src" / "experiments" / "gguf_fix" / "merge_verify.py"
CONVERT = REPO / "src" / "experiments" / "gguf_fix" / "convert_matched.py"
# the known-good non-thinking-default template, only relevant to qwen
QWEN_TEMPLATE = (REPO / "models" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix"
                 / "qwen35-r1-merged" / "chat_template.jinja")
HUB = Path.home() / ".cache" / "huggingface" / "hub"

MODELS = {
    "qwen":      ("qwen3.5-4b-bidding-lora-v2", "qwen3.5-4b-reverse-auction-lora"),
    "granite":   ("granite-4.1-3b-bidding-lora-v2", "granite-4.1-3b-reverse-auction-lora"),
    "phi":       ("phi4mini-bidding-lora-v2", "phi-reverse-auction-lora"),
    "ministral": ("ministral3-bidding-lora-v2", "ministral-reverse-auction-lora"),
    "gemma":     ("gemma4-e2b-bidding-lora-v2", "gemma-reverse-auction-lora"),
    "smollm3":   ("smollm3-bidding-lora-v2", "smollm3-reverse-auction-lora"),
}


def adapter_dir(slug, mode):
    sealed, auction = MODELS[slug]
    if mode == "sealed":
        return REPO / "models" / "epoch-2" / "production" / sealed / "final"
    return REPO / "models" / "reverse_auction" / "epoch-1" / "v4" / auction / "final"


def target(slug, mode):
    if mode == "sealed":
        return (REPO / "models" / "epoch-2" / "deployment" / f"{slug}-sealed",
                REPO / "reports" / "epoch-2" / "deployment" / f"{slug}-sealed")
    return (REPO / "models" / "reverse_auction" / "epoch-1" / "v4" / "deployment" / f"{slug}-auction",
            REPO / "reports" / "reverse_auction" / "v4_deployment" / f"{slug}-auction")


def base_of(adapter):
    """what the adapter was trained on, and a revision to pin if it is a hub repo

    several adapters record a local directory instead of a repo id: phi and
    smollm3 were trained from local snapshots because the unsloth mirror is
    broken, and two others recorded an absolute snapshot path. a local base is
    used as-is with no revision; a repo id is pinned to its single cached
    revision so a build cannot start a download mid-queue"""
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    recorded = cfg["base_model_name_or_path"]
    for candidate in (Path(recorded), REPO / recorded):
        if candidate.is_dir():
            return str(candidate.resolve()), ""
    snaps = HUB / f"models--{recorded.replace('/', '--')}" / "snapshots"
    revs = sorted(x.name for x in snaps.iterdir()) if snaps.is_dir() else []
    if len(revs) != 1:
        raise RuntimeError(f"{recorded}: not a local dir and expected one cached "
                           f"revision, found {revs}")
    return recorded, revs[0]


# save_pretrained writes tokenizer.json and drops the rest; some converters want
# the originals. phi's converter reads a sentencepiece or bpe vocab and fails
# outright without them
TOKENIZER_FILES = ("tokenizer.model", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json", "spiece.model")


def carry_tokenizer_files(base, merged):
    """copy tokenizer artifacts the merge did not re-emit, when the base is local"""
    if not base.is_dir():
        return
    copied = []
    for name in TOKENIZER_FILES:
        src, dst = base / name, merged / name
        if src.is_file() and not dst.is_file():
            dst.write_bytes(src.read_bytes())
            copied.append(name)
    if copied:
        print(f"  carried tokenizer files: {', '.join(copied)}")
    restore_tokenizer_class(base, merged)


def restore_tokenizer_class(base, merged):
    """put back the tokenizer_class the base declared

    transformers 5.5 save_pretrained rewrites this field to TokenizersBackend.
    llama.cpp's phi converter branches on it: GPT2Tokenizer takes the bpe path,
    anything else falls through to a sentencepiece path that demands a
    tokenizer.model this checkpoint does not have. same shape of trap as the
    chat template, a metadata field the converter reads being rewritten on save"""
    src, dst = base / "tokenizer_config.json", merged / "tokenizer_config.json"
    if not (src.is_file() and dst.is_file()):
        return
    want = json.loads(src.read_text(encoding="utf-8")).get("tokenizer_class")
    cfg = json.loads(dst.read_text(encoding="utf-8"))
    if want and cfg.get("tokenizer_class") != want:
        was = cfg.get("tokenizer_class")
        cfg["tokenizer_class"] = want
        dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"  tokenizer_class restored: {was} -> {want}")


def needs_template_swap(merged):
    """true when the merged template defaults to thinking with no flag passed

    llama_cpp never passes enable_thinking, so a template that only disables
    thinking when explicitly told to will leave it on, which is the 0/6 JSON
    failure. a template that does not mention thinking at all is left alone"""
    path = merged / "chat_template.jinja"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "enable_thinking" not in text:
        return False
    return "enable_thinking is defined and enable_thinking is false" in text


def run(cmd, label):
    print(f"  {label} ...", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"  {label} FAILED rc={proc.returncode}")
        print("  " + (proc.stderr or proc.stdout)[-600:].replace("\n", "\n  "))
    return proc.returncode == 0


def build(slug, mode):
    adapter = adapter_dir(slug, mode)
    if not adapter.is_dir():
        print(f"{slug} {mode}: adapter missing at {adapter}")
        return False
    gguf_dir, report_dir = target(slug, mode)
    name = f"{slug}-{mode}"
    quant = gguf_dir / f"{name}-nomtp-matched-Q8_0.gguf"
    if quant.is_file():
        print(f"{slug} {mode}: gguf already present, skipping")
        return True

    merged = gguf_dir / "merged"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    repo, revision = base_of(adapter)
    print(f"\n=== {slug} {mode} === base {repo} @ {revision[:12]}")

    if not merged.is_dir():
        if not run([FEAS_PY, MERGE, "--adapter", adapter, "--base", repo,
                    "--base-revision", revision, "--out", merged,
                    "--report", report_dir / "merge_verify.json"], "merge"):
            return False

    carry_tokenizer_files(Path(repo), merged)

    if needs_template_swap(merged):
        stock = merged / "chat_template.stock.jinja"
        if not stock.is_file():
            (merged / "chat_template.jinja").replace(stock)
            stock.read_bytes()
        (merged / "chat_template.jinja").write_bytes(QWEN_TEMPLATE.read_bytes())
        print("  chat template swapped to the non-thinking default")
    else:
        print("  chat template left alone, no thinking branch to fix")

    return run([MAIN_PY, CONVERT, "--merged", merged, "--gguf-dir", gguf_dir,
                "--out-dir", report_dir, "--name", name, "--skip-runtime"], "convert")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated slugs")
    ap.add_argument("--mode", choices=("sealed", "auction", "both"), default="both")
    args = ap.parse_args()
    slugs = [s for s in (args.only.split(",") if args.only else MODELS) if s]
    modes = ["sealed", "auction"] if args.mode == "both" else [args.mode]

    ok, failed = [], []
    for slug in slugs:
        for mode in modes:
            (ok if build(slug, mode) else failed).append(f"{slug} {mode}")
    print(f"\nbuilt {len(ok)}: {', '.join(ok)}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
