"""fetch the six base checkpoints once, at their pinned revisions, into a destination a profile owns

  PYTHONPATH=. python -m src.experiments.v5.download_models --profile lumi --plan
  PYTHONPATH=. python -m src.experiments.v5.download_models --profile lumi
  PYTHONPATH=. python -m src.experiments.v5.download_models --profile lumi --layout base --plan

the destination comes from the execution profile, never from a constant here, so the same command
fills this machine's store, LUMI's project storage or a future site's, and every later job reads the
same copy instead of downloading per run. --plan prints what would be fetched, where each model
would land, and writes nothing, which is the safe thing to run on a login node first.

--profile is required because the destination is exactly what a profile decides. --backend is not:
a download is bytes on a filesystem and has nothing to do with the accelerator.

two layouts, and cache is still the default so an existing command line does exactly what it did:

  cache   what this command has always done. the four hub models are fetched into the profile's
          hugging face cache and the two derived local copies are reported and left alone, because
          no download produces them
  base    the storage hierarchy's models/base/<model>/, one self-contained directory per
          architecture. the four hub models are materialized there at their pinned revisions; the
          two derived copies are copied there from the pad-token-fixed directories that already
          exist, and every copied file is hashed against its source rather than assumed

why two of the six are copied and not downloaded: phi and smollm3 ship pad_token == eos_token, which
unsloth rejects, so each trains from a local base whose pad token is set from an existing reserved
token. that fix lives in the small config and tokenizer files, so those are exactly the files the
copy is verified on. --derive-from names the directory holding them when it is not this machine's
models/base-checkpoints/hf/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.experiments.v5 import execution, model_registry

LAYOUT_CHOICES = ("cache", "base")
DEFAULT_LAYOUT = "cache"


def hub_cache_destination(context, spec):
    """the folder a snapshot_download of this revision lands in

    the cache lays a repo out as models--<org>--<name>/snapshots/<revision>, which is the same shape
    freeze_check tests for presence, so the plan names the real directory rather than the cache root"""
    return (Path(context.hf_hub_cache) / f"models--{spec['checkpoint'].replace('/', '--')}"
            / "snapshots" / str(spec["revision"]))


def destination(context, name, layout=DEFAULT_LAYOUT):
    """where this model's weights end up under the chosen layout, as a real path"""
    spec = model_registry.model(name)
    if layout == "base":
        return Path(model_registry.base_dir(name, context.root))
    if spec["local"]:
        return Path(spec["checkpoint"])
    return hub_cache_destination(context, spec)


def plan(context, layout=DEFAULT_LAYOUT, derive_from=None):
    """what each model needs and where it lands, before anything is fetched"""
    if layout not in LAYOUT_CHOICES:
        raise execution.ExecutionError(
            f"unknown layout {layout!r}, expected one of {', '.join(LAYOUT_CHOICES)}")
    entries = []
    for name in sorted(model_registry.MODELS):
        spec = model_registry.model(name)
        source = model_registry.base_source(name)
        target = destination(context, name, layout)
        derived_source = Path(derive_from or model_registry.LOCAL_BASE) / Path(spec["checkpoint"]).name \
            if spec["local"] else None
        if layout == "base":
            action = (f"copy {derived_source} into {target} and hash every provenance file against "
                      f"its source" if spec["local"]
                      else f"snapshot_download {source['hub_source']} @ {spec['revision']} into {target}")
        else:
            action = ("copy this directory to the profile root; it carries a pad-token fix and is "
                      "not reproducible by a plain download" if spec["local"]
                      else f"snapshot_download into {context.hf_hub_cache}")
        entries.append({
            "model": name,
            "kind": "local copy" if spec["local"] else "hub download",
            "source": source["hub_source"],
            "revision": spec["revision"],
            "destination": str(target),
            "derive_from": str(derived_source) if derived_source else None,
            "path_used_for_training": str(target) if layout == "base" else spec["checkpoint"],
            "present": target.is_dir(),
            "action": action,
        })
    return entries


def download(entry, context, allow_patterns=None, layout=DEFAULT_LAYOUT):
    """one hub snapshot, into the cache or into its own base directory

    local_dir gives the base layout a self-contained directory instead of a cache symlink farm;
    cache_dir stays the profile's either way, so a second layout does not mean a second copy of the
    bytes on a site that has both"""
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": entry["source"], "revision": entry["revision"],
              "cache_dir": str(context.hf_hub_cache), "allow_patterns": allow_patterns}
    if layout == "base":
        kwargs["local_dir"] = entry["destination"]
    return snapshot_download(**kwargs)


def copy_derived(entry):
    """materialize one pad-token-fixed local base into the base tree, then verify it file by file

    the copy is only worth anything if it is the same copy, so every provenance file is hashed on
    both sides and any mismatch is returned rather than printed and forgotten. the weight shards are
    compared by size: they are multiple gigabytes each and the fix is not in them, which is the same
    rule freeze_check's fingerprint already records"""
    from src.experiments.v5.freeze_check import local_fingerprint

    source = Path(entry["derive_from"])
    target = Path(entry["destination"])
    if not source.is_dir():
        return {"copied": False, "problems": [f"derive_from {source} is not a directory"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)

    before, after = local_fingerprint(source), local_fingerprint(target)
    problems = []
    for name, recorded in before["provenance_files"].items():
        got = after["provenance_files"].get(name)
        if got is None or got["sha256"] != recorded["sha256"]:
            problems.append({"file": name, "expected": recorded["sha256"][:16],
                             "actual": (got or {}).get("sha256", "missing")})
    for shard, recorded in before["weight_shards"].items():
        got = after["weight_shards"].get(shard)
        if got is None or got["bytes"] != recorded["bytes"]:
            problems.append({"file": shard, "expected": f"{recorded['bytes']} bytes",
                             "actual": f"{(got or {}).get('bytes', 'missing')} bytes"})
    return {"copied": True, "problems": problems, "fingerprint": after}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    execution.add_arguments(ap)
    ap.add_argument("--plan", action="store_true", help="print what would be fetched and write nothing")
    ap.add_argument("--only", default="", help="comma-separated model names, default all six")
    ap.add_argument("--layout", choices=LAYOUT_CHOICES, default=DEFAULT_LAYOUT,
                    help=f"destination layout, default {DEFAULT_LAYOUT}: the profile's hugging face "
                         f"cache, with the two derived copies reported and skipped. base materializes "
                         f"all six under <root>/models/base/<model>/")
    ap.add_argument("--derive-from", default="",
                    help="directory holding the pad-token-fixed phi and smollm3 bases, used by "
                         "--layout base; defaults to this machine's models/base-checkpoints/hf/")
    args = ap.parse_args(argv)
    if not args.profile:
        raise execution.ExecutionError(
            "download_models: --profile must be given explicitly.\n"
            f"  --profile {' | '.join(execution.PROFILE_CHOICES)}\n"
            "the destination cache is exactly what a profile decides, so it is never guessed")
    context = execution.resolve(args.backend or "cuda", args.profile, root=args.profile_root or None)

    entries = plan(context, args.layout, args.derive_from or None)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        entries = [e for e in entries if e["model"] in wanted]
    print(f"profile {context.profile.name}: HF_HOME {context.hf_home}")
    print(f"                 HF_HUB_CACHE {context.hf_hub_cache}")
    print(f"                 layout {args.layout}, base tree {context.base_dir}")
    for entry in entries:
        print(f"  {entry['model']:10s} {entry['kind']:14s} {entry['source']} @ {entry['revision']}")
        print(f"             -> {entry['destination']}{'' if entry['present'] else '   (not present yet)'}")
    if args.plan:
        print("\nplan only, nothing downloaded")
        print(json.dumps({"execution": context.as_dict(), "layout": args.layout, "entries": entries},
                         indent=2))
        return 0

    context.hf_hub_cache.mkdir(parents=True, exist_ok=True)
    fetched, skipped, failed = [], [], []
    for entry in entries:
        if entry["kind"] == "local copy":
            if args.layout != "base":
                present = Path(entry["path_used_for_training"]).is_dir()
                skipped.append({**entry, "present_here": present})
                print(f"  {entry['model']}: local copy, {'present' if present else 'MISSING'}, not downloaded")
                continue
            print(f"  {entry['model']}: copying {entry['derive_from']}", flush=True)
            outcome = copy_derived(entry)
            if outcome["problems"]:
                failed.append({**entry, "problems": outcome["problems"]})
                print(f"    VERIFY FAILED: {outcome['problems'][:3]}")
            else:
                fetched.append({**entry, "local_path": entry["destination"]})
                print(f"    into {entry['destination']}, every provenance file verified against its source")
            continue
        print(f"  {entry['model']}: fetching {entry['source']} @ {entry['revision']}", flush=True)
        path = download(entry, context, layout=args.layout)
        fetched.append({**entry, "local_path": path})
        print(f"    into {path}")
    print(f"\n{len(fetched)} materialized, {len(skipped)} local copies to carry across by hand, "
          f"{len(failed)} failed verification")
    if failed:
        print("a failed copy is not a usable base: fix the source directory and rerun", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
