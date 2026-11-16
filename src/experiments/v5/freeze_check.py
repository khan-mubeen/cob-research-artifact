"""the corpus-to-launch-config freeze gate, read-only, in three modes

  PYTHONPATH=. python -m src.experiments.v5.freeze_check --mode candidate   before bid_check/slm_check
  PYTHONPATH=. python -m src.experiments.v5.freeze_check --mode freeze      after they and any fix are committed
  PYTHONPATH=. python -m src.experiments.v5.freeze_check --remote           on LUMI, after rsync and git pull

this does not re-validate the corpus. src/experiments/v5/validate.py and the test suite already do
that, and nothing here repeats them. it checks one link that nothing else checked: that the launch
configuration actually points at the state being declared frozen. the manifest verification and the
model lookups are imported from the modules that own them rather than rewritten here.

the three modes and why they are separate:

  candidate   a look at the alignment picture BEFORE bid_check, slm_check and any fix they justify.
              writes reports/v5/freeze_check_candidate.json, which says inside itself that it is
              non-authoritative, disposable, and not what LUMI checks against. if a fix lands after
              it, it is stale by definition and must not be reused
  freeze      the authoritative record, reports/v5/freeze_check_local.json. it refuses outright on a
              dirty worktree, and the check happens BEFORE anything is written so that writing the
              report cannot be the thing that dirties the tree. the clean HEAD is captured first,
              the report is written second, and the worktree is then re-read to confirm the report
              is the only new item. the report is a generated artifact: it is not committed, because
              committing it would move HEAD and invalidate the commit it just recorded. it travels
              to LUMI alongside the gitignored corpus
  remote      the hard gate on LUMI. it reads the authoritative record only, refuses if just a
              candidate file is present, hashes files rather than running anything expensive, and
              exits non-zero naming the failing check with expected against actual

what "aligned" means here, in both local modes:
  git         the commit and whether the worktree is clean (reported in candidate, required in freeze)
  corpus      data/finetune/v5/manifest.json against the files on disk right now, via
              src/experiments/v5/manifest.verify
  pins        every model revision in model_registry.py against what the generated slurm scripts
              actually pass on their command lines, so the two cannot drift apart
  freshness   the expected script text re-rendered in memory from the current registry, backend and
              profile, compared with the files under slurm/. a stale generated script is a failure,
              never evidence of alignment
  paths       the train files each script's --dataset resolves to exist and are in the manifest, and
              the cache paths resolve under the profile root rather than a machine-specific path
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from config import PROJECT_ROOT
from src.experiments.v5 import execution, manifest as manifest_module, model_registry, results_store, slurm
from src.experiments.v5 import training_data
from src.experiments.v5.common import FINETUNE_V5, sha256_file, write_json

REPORT_DIR = PROJECT_ROOT / "reports" / "v5"
CANDIDATE_PATH = REPORT_DIR / "freeze_check_candidate.json"
LOCAL_PATH = REPORT_DIR / "freeze_check_local.json"
SLURM_ROOT = PROJECT_ROOT / "slurm"

# the small files that carry a derived copy's identity: the pad-token fix that makes phi and
# smollm3 local copies rather than plain downloads lives in exactly these
PROVENANCE_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
                    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                    "tokenizer.json", "model.safetensors.index.json")

# how the base artifacts in a record can be verified, written into the record and read back by the
# remote gate. the version is what makes this additive rather than a rewrite:
#
#   1   what every record written before the storage hierarchy carries. a hub model is verified by
#       the presence of its revision in the hugging face cache and a derived copy by its
#       fingerprint. a version 1 record is still checked exactly that way, so an older freeze
#       verifies to the same answer it always did
#   2   the same, plus a fingerprint of <root>/models/base/<model> for all six. where the frozen
#       record actually has that directory, the gate verifies the directory the job will read
#       instead of only asking whether a cache entry exists. where it does not, because the tree
#       was not materialized when the record was written, it falls back to the version 1 check for
#       that model and says so
BASE_ARTIFACT_CHECK_VERSION = 2
RENDER_ROOT = "/freeze-check-render-root"     # scripts are root-independent, so any root renders the same

BASE_MODEL_RE = re.compile(r"--base-model (\"[^\"]+\"|'[^']+'|\S+)")
REVISION_RE = re.compile(r"--revision (\S+)")
DATASET_RE = re.compile(r"--dataset (\S+)")
PARTITION_RE = re.compile(r"^#SBATCH --partition=(\S+)", re.M)
TIME_RE = re.compile(r"^#SBATCH --time=(\d+):", re.M)
GPUS_RE = re.compile(r"^#SBATCH --gpus-per-node=(\d+)", re.M)
CPUS_RE = re.compile(r"^#SBATCH --cpus-per-task=(\d+)", re.M)
ACCOUNT_RE = re.compile(r"^#SBATCH --account=(\S+)", re.M)
EPOCHS_RE = re.compile(r"--num-train-epochs (\d+)")
SUBSAMPLE_RE = re.compile(r"--subsample (\d+)")


def check(name, ok, detail="", expected=None, actual=None):
    return {"check": name, "ok": bool(ok), "detail": detail, "expected": expected, "actual": actual}


def unquote(value):
    return value[1:-1] if value[:1] in ("'", '"') and value[-1:] == value[:1] else value


# ── the pieces every mode reads ─────────────────────────────────────────────

def read_manifest():
    path = FINETUNE_V5 / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generated_sets(root=SLURM_ROOT):
    """every slurm/<backend>-<profile>/ directory that holds generated training scripts

    discovering them beats asking for --backend and --profile: whatever was generated is what would
    be launched, so that is what gets checked"""
    out = []
    if not Path(root).is_dir():
        return out
    for directory in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        backend, _, profile = directory.name.partition("-")
        scripts = sorted(directory.glob("train_*.sh"))
        if backend in execution.BACKEND_CHOICES and profile in execution.PROFILE_CHOICES and scripts:
            out.append({"directory": directory, "backend": backend, "profile": profile, "scripts": scripts})
    return out


def args_from_script(text, epochs=None, subsample=None, layout=None, run_tag=None, base_layout=None):
    """rebuild the generator arguments a script was written with

    two kinds of field, deliberately treated differently:

      scheduler choices   partition, wall time, gpus, cpus, account. these are legitimately chosen
                          per submission, so they are parsed back out of the script and re-rendered
                          as they are; reporting them as drift would be reporting a free choice
      experiment fields   epochs and subsample. these define what is trained, so they are NOT parsed
                          back. they come from what the gate expects, which means a training call
                          edited by hand renders differently and is caught as stale. a deliberate
                          choice is declared with --expect-epochs or --expect-subsample rather than
                          inferred from the very file being checked"""
    partition = PARTITION_RE.search(text)
    return SimpleNamespace(
        account=(ACCOUNT_RE.search(text).group(1) if ACCOUNT_RE.search(text) else ""),
        partition=(partition.group(1) if partition else ""),
        hours=int(TIME_RE.search(text).group(1)) if TIME_RE.search(text) else 0,
        gpus=int(GPUS_RE.search(text).group(1)) if GPUS_RE.search(text) else 0,
        cpus=int(CPUS_RE.search(text).group(1)) if CPUS_RE.search(text) else 7,
        epochs=slurm.DEFAULT_EPOCHS if epochs is None else epochs,
        subsample=slurm.DEFAULT_SUBSAMPLE if subsample is None else subsample,
        # layout fields join the experiment side, not the scheduler side: they decide where an
        # attempt's artifacts land, so they come from what the gate expects rather than being read
        # back out of the very file being checked
        layout=slurm.DEFAULT_LAYOUT if layout is None else layout,
        run_tag=slurm.DEFAULT_RUN_TAG if run_tag is None else run_tag,
        # left as None when not stated, which is exactly what slurm.py's own --base-layout does:
        # script_text then resolves it from the profile being rendered for. coercing it to the
        # registry default here would render a registry script for a site that pins the base tree
        base_layout=base_layout,
        out_dir="", profile_root=RENDER_ROOT,
    )


def local_fingerprint(directory, deep=False):
    """a derived copy's identity: its small provenance files hashed, its weight shards sized

    the weights are multiple gigabytes each and this gate has to stay fast, so they are recorded by
    size unless --deep is asked for. that is stated in the record rather than left to be assumed:
    what is hashed is exactly what makes the copy derived, the tokenizer and config files carrying
    the pad-token fix"""
    directory = Path(directory)
    if not directory.is_dir():
        return {"present": False, "path": str(directory)}
    files, weights = {}, {}
    for name in PROVENANCE_FILES:
        path = directory / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for path in sorted(directory.glob("*.safetensors")):
        entry = {"bytes": path.stat().st_size}
        if deep:
            entry["sha256"] = sha256_file(path)
        weights[path.name] = entry
    return {"present": True, "path": str(directory), "provenance_files": files, "weight_shards": weights,
            "weights_hashed": bool(deep)}


def fingerprint_problems(name, kind, frozen_fp, here, absent):
    """one recorded fingerprint against what is on disk now

    factored out because the derived copies and the base tree are checked the same way and were
    drifting apart as two copies of the same loop. an absent directory is one problem, not one per
    file it would have contained"""
    if not here.get("present"):
        return [{"model": name, "kind": kind, "expected": frozen_fp.get("path"), "actual": absent}]
    problems = []
    for filename, recorded in (frozen_fp.get("provenance_files") or {}).items():
        got = here["provenance_files"].get(filename)
        if got is None or got["sha256"] != recorded["sha256"]:
            problems.append({"model": name, "file": filename, "expected": recorded["sha256"][:16],
                             "actual": (got or {}).get("sha256", "missing")})
    for shard, recorded in (frozen_fp.get("weight_shards") or {}).items():
        got = here["weight_shards"].get(shard)
        if got is None or got["bytes"] != recorded["bytes"]:
            problems.append({"model": name, "file": shard, "expected": f"{recorded['bytes']} bytes",
                             "actual": f"{(got or {}).get('bytes', 'missing')} bytes"})
    return problems


def model_artifacts(deep=False, base_root=None):
    """what each of the six base checkpoints is, and how it can be verified later

    a hub model is identified by repository and revision; a locally derived copy cannot be, because
    no hub revision produces it, so it carries a fingerprint instead.

    every model also records its place in the base tree and a fingerprint of it. an absent
    directory is recorded as absent rather than skipped: that is the honest state today, and it is
    what tells the remote gate to fall back to the version 1 check for that model instead of
    failing on a directory nobody has created yet"""
    root = model_registry.resolve_base_root(base_root)
    out = {}
    for name in sorted(model_registry.MODELS):
        spec = model_registry.model(name)
        entry = {"kind": "local_derived_copy" if spec["local"] else "hub_revision",
                 "checkpoint": spec["checkpoint"], "revision": spec["revision"],
                 "adapter_stem": spec["adapter_stem"]}
        if spec["local"]:
            entry["upstream_source"] = spec.get("hub_source")
            entry["fingerprint"] = local_fingerprint(spec["checkpoint"], deep=deep)
        base = model_registry.base_dir(name, root)
        entry["base_dir"] = base
        entry["base_fingerprint"] = local_fingerprint(base, deep=deep)
        out[name] = entry
    return out


# ── the checks ──────────────────────────────────────────────────────────────

def git_checks(require_clean):
    commit = execution.git_commit()
    entries = execution.worktree_entries()
    # git being unable to answer is not a clean tree and must never pass as one
    unknown = entries is None
    entries = [] if unknown else entries
    dirty = bool(entries)
    checks = [check("git_commit_known", commit not in ("unknown", "git-not-available"),
                    "the commit this state is recorded at", actual=commit)]
    if require_clean:
        checks.append(check("worktree_clean", not dirty and not unknown,
                            "freeze refuses on a dirty tree, checked before anything is written",
                            expected="no uncommitted or untracked entries",
                            actual="git could not report the worktree state" if unknown
                            else (f"{len(entries)} entr(y/ies): {entries[:5]}" if dirty else "clean")))
    else:
        checks.append(check("worktree_state_recorded", True,
                            "candidate mode records the tree state, it does not require it",
                            actual=f"{len(entries)} uncommitted entr(y/ies)" if dirty else "clean"))
    return checks, {"commit": commit, "dirty": dirty, "worktree_entries": entries}


def corpus_checks():
    """the manifest against the files on disk, through the module that owns that verification"""
    stored = read_manifest()
    if stored is None:
        return [check("corpus_manifest_present", False, "data/finetune/v5/manifest.json is missing",
                      expected=str(FINETUNE_V5 / "manifest.json"), actual="missing")], {}
    problems = manifest_module.verify(stored)
    digest = results_store.manifest_sha256()
    checks = [
        check("corpus_manifest_present", True, "manifest read", actual=str(FINETUNE_V5 / "manifest.json")),
        check("corpus_files_match_manifest", not problems,
              "every listed file hashed and counted again on disk, via manifest.verify",
              expected="0 problems", actual=f"{len(problems)} problem(s): {problems[:5]}" if problems else "0 problems"),
    ]
    files = {f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "rows": f["rows"]} for f in stored["files"]}
    return checks, {"manifest_sha256": digest, "file_count": len(files), "files": files,
                    "main_corpus_rows": stored.get("main_corpus_rows")}


def expected_base_layout(context, base_layout=None):
    """which base layout this site's scripts should carry

    where a set reads its weights from is a property of THAT site, not of the gate run: lumi has a
    materialized models/base tree and pins base_layout on its profile, roihu has none and stays on
    the registry. one expectation applied to every set could only ever match one of them, so it is
    read from the same profile field the generator itself reads, which is what stops the two
    drifting apart again. an explicit expectation still wins, for declaring a deliberate change
    rather than inferring it from the very files being checked"""
    if base_layout is not None:
        return base_layout
    return context.profile.base_layout or model_registry.BASE_LAYOUT_DEFAULT


def inspect_script(text, model, mode, context, epochs=None, subsample=None, layout=None,
                   run_tag=None, base_layout=None):
    """one generated script against the registry and against a fresh render of itself

    returns (pin drift, stale, missing train files). factored out so a test can hand it a
    deliberately altered script and see the drift reported rather than only exercise the happy path"""
    spec = model_registry.model(model)
    resolved_base_layout = expected_base_layout(context, base_layout)
    drift = []
    base = BASE_MODEL_RE.search(text)
    expected_base = model_registry.training_path(model, "${COB_PROFILE_ROOT}",
                                                 layout=resolved_base_layout)
    actual_base = unquote(base.group(1)) if base else None
    if actual_base != expected_base:
        drift.append({"field": "--base-model", "expected": expected_base, "actual": actual_base})
    revision = REVISION_RE.search(text)
    actual_revision = revision.group(1) if revision else None
    expected_revision = (spec["revision"] if model_registry.pins_revision(model, resolved_base_layout)
                         else None)
    if actual_revision != expected_revision:
        drift.append({"field": "--revision", "expected": expected_revision, "actual": actual_revision})

    # the re-render has to use the SAME resolved layout as the pin check above, or a site whose
    # profile pins one would re-render against the other and every script would look stale
    rendered = slurm.script_text(
        args_from_script(text, epochs, subsample, layout, run_tag, resolved_base_layout),
        context, model, mode)
    stale = rendered != text

    missing = []
    dataset = DATASET_RE.search(text)
    if dataset:
        for train_file in training_data.train_set_paths(dataset.group(1)):
            if not Path(train_file).is_file():
                missing.append(str(train_file))
    return drift, stale, missing


def launch_checks(deep=False, epochs=None, subsample=None, layout=None, run_tag=None, base_layout=None):
    """the generated scripts against the registry, and against a fresh render of themselves"""
    sets = generated_sets()
    checks = [check("generated_scripts_present", bool(sets),
                    "a freeze with nothing to launch proves nothing",
                    expected="at least one slurm/<backend>-<profile>/ with train_*.sh",
                    actual=f"{len(sets)} generated set(s)")]
    if not sets:
        return checks, {}

    recorded = {}
    for entry in sets:
        backend, profile = entry["backend"], entry["profile"]
        context = execution.resolve(backend, profile, root=RENDER_ROOT)
        # resolved per set, from this site's own profile; see expected_base_layout()
        set_base_layout = expected_base_layout(context, base_layout)
        stale, pin_drift, dataset_missing = [], [], []
        for path in entry["scripts"]:
            text = path.read_text(encoding="utf-8")
            model, _, mode = path.stem[len("train_"):].partition("_")
            drift, is_stale, missing = inspect_script(text, model, mode, context, epochs, subsample,
                                                      layout, run_tag, set_base_layout)
            pin_drift += [{"script": path.name, **item} for item in drift]
            if is_stale:
                stale.append(path.name)
            dataset_missing += [{"script": path.name, "missing": item} for item in missing]

        job_env = context.job_environment("${COB_PROFILE_ROOT}")
        cache_rooted = all(str(value).startswith("${COB_PROFILE_ROOT}")
                           for key, value in job_env.items() if key in ("HF_HOME", "HF_HUB_CACHE"))
        label = f"{backend}-{profile}"
        checks += [
            check(f"{label}:pins_match_registry", not pin_drift,
                  "every --base-model and --revision in the scripts is the registry's",
                  expected="no drift", actual=pin_drift[:5] if pin_drift else "no drift"),
            check(f"{label}:scripts_are_freshly_rendered", not stale,
                  "each script re-rendered from the current registry and compared byte for byte",
                  expected="no stale scripts", actual=stale[:5] if stale else "no stale scripts"),
            check(f"{label}:train_files_exist", not dataset_missing,
                  "each script's --dataset resolves to train files that are on disk",
                  expected="all present", actual=dataset_missing[:5] if dataset_missing else "all present"),
            check(f"{label}:caches_resolve_under_the_profile_root", cache_rooted,
                  "HF_HOME and HF_HUB_CACHE come from the profile, not from a machine-specific path",
                  expected="${COB_PROFILE_ROOT}/...", actual=job_env),
        ]
        recorded[label] = {"backend": backend, "profile": profile,
                           "expected_epochs": slurm.DEFAULT_EPOCHS if epochs is None else epochs,
                           "expected_subsample": (slurm.DEFAULT_SUBSAMPLE if subsample is None
                                                 else subsample),
                           "expected_layout": slurm.DEFAULT_LAYOUT if layout is None else layout,
                           "expected_run_tag": slurm.DEFAULT_RUN_TAG if run_tag is None else run_tag,
                           "expected_base_layout": set_base_layout,
                           "directory": str(entry["directory"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                           "scripts": [p.name for p in entry["scripts"]],
                           "job_environment": job_env,
                           "script_sha256": {p.name: sha256_file(p) for p in entry["scripts"]}}
    return checks, recorded


def collect(require_clean, deep=False, epochs=None, subsample=None, layout=None, run_tag=None,
            base_layout=None, base_root=None):
    """every check plus the artifacts a later gate compares against"""
    checks, git_state = git_checks(require_clean)
    corpus, corpus_state = corpus_checks()
    launch, launch_state = launch_checks(deep=deep, epochs=epochs, subsample=subsample,
                                         layout=layout, run_tag=run_tag, base_layout=base_layout)
    checks += corpus + launch
    record = {
        "base_artifact_check_version": BASE_ARTIFACT_CHECK_VERSION,
        "git": git_state,
        "corpus": corpus_state,
        "launch": launch_state,
        "models": model_artifacts(deep=deep, base_root=base_root),
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return record


def print_checks(record):
    for entry in record["checks"]:
        mark = "ok  " if entry["ok"] else "FAIL"
        print(f"  {mark} {entry['check']}")
        if not entry["ok"]:
            print(f"       expected: {entry['expected']}")
            print(f"       actual:   {entry['actual']}")


# ── modes ───────────────────────────────────────────────────────────────────

CANDIDATE_LABELS = {
    "authoritative": False,
    "disposable": True,
    "what_this_is": ("a candidate look at corpus-to-launch alignment, taken BEFORE bid_check, "
                     "slm_check and any fix they justify"),
    "what_this_is_not": ("this is NOT what LUMI checks against. src/experiments/v5/freeze_check.py "
                         "--remote reads reports/v5/freeze_check_local.json and refuses this file"),
    "staleness_rule": ("if any fix lands after this file was written, it is stale and must not be "
                       "reused. delete it and rerun --mode candidate, or move on to --mode freeze"),
}


def run_candidate(deep=False, path=CANDIDATE_PATH, epochs=None, subsample=None, layout=None,
                  run_tag=None, base_layout=None, base_root=None):
    record = collect(require_clean=False, deep=deep, epochs=epochs, subsample=subsample,
                     layout=layout, run_tag=run_tag, base_layout=base_layout, base_root=base_root)
    record = {**CANDIDATE_LABELS, "mode": "candidate", **record}
    write_json(path, record)
    print(f"candidate freeze check, {sum(c['ok'] for c in record['checks'])}/{len(record['checks'])} checks ok")
    print_checks(record)
    print(f"\nnon-authoritative and disposable, written to {path}")
    print("LUMI will not read this file; run --mode freeze once bid_check, slm_check and any fix are committed")
    return 0 if record["passed"] else 1


def run_freeze(deep=False, path=LOCAL_PATH, epochs=None, subsample=None, layout=None, run_tag=None,
               base_layout=None, base_root=None):
    """the authoritative record; refuses on a dirty tree before writing anything

    the git state is read first and on its own, so a refusal costs nothing and happens long before
    any file is touched: the corpus hashing that follows is the expensive part"""
    gate, _ = git_checks(require_clean=True)
    if not all(c["ok"] for c in gate):
        print("freeze REFUSED before any work, the worktree is not in a freezable state")
        print_checks({"checks": gate})
        print("\ncommit or stash the listed entries first: a freeze record pins a commit, and a dirty")
        print("tree means the pinned commit does not describe what is on disk")
        return 1

    record = collect(require_clean=True, deep=deep, epochs=epochs, subsample=subsample,
                     layout=layout, run_tag=run_tag, base_layout=base_layout, base_root=base_root)
    if not record["passed"]:
        print(f"freeze REFUSED, {sum(not c['ok'] for c in record['checks'])} check(s) failed, nothing written")
        print_checks(record)
        dirty = [c for c in record["checks"] if c["check"] == "worktree_clean" and not c["ok"]]
        if dirty:
            print("\ncommit or stash the listed entries first: a freeze record pins a commit, and a dirty")
            print("tree means the pinned commit does not describe what is on disk")
        return 1

    clean_head = record["git"]["commit"]
    before = set(execution.worktree_entries())
    record = {
        "authoritative": True,
        "mode": "freeze",
        "what_this_is": ("the authoritative record of the state declared frozen; "
                         "src/experiments/v5/freeze_check.py --remote checks LUMI against this file"),
        "commit_recorded_before_writing": clean_head,
        "artifact_rule": ("this file is a generated artifact and is deliberately NOT committed: a commit "
                          "would move HEAD and invalidate the commit recorded above. transfer it to LUMI "
                          "alongside the gitignored corpus"),
        **record,
    }
    write_json(path, record)

    after = set(execution.worktree_entries() or [])
    try:
        relative = str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        # a record written outside the repository, which a test does and an operator may
        relative = Path(path).name
    new_entries = sorted(entry for entry in after - before)
    only_report = all(relative in entry for entry in new_entries)
    print(f"freeze check PASSED, {len(record['checks'])} checks ok, commit {clean_head}")
    print_checks(record)
    print(f"\nwritten to {path}")
    print(f"worktree after writing: {len(new_entries)} new entr(y/ies) {new_entries}")
    if only_report:
        print("the freeze report is the only new item, as intended; do not commit it")
    else:
        print("WARNING: something other than the freeze report changed the worktree, investigate before transferring")
    return 0


def run_remote(deep=False, local_path=LOCAL_PATH, candidate_path=CANDIDATE_PATH):
    """the hard gate on LUMI: compare this machine against the authoritative record"""
    if not Path(local_path).exists():
        if Path(candidate_path).exists():
            print(f"REFUSED: only a candidate record is present ({candidate_path}).", file=sys.stderr)
            print("a candidate is non-authoritative and disposable, and is never a launch gate.", file=sys.stderr)
            print("run 'freeze_check.py --mode freeze' on the local machine once bid_check, slm_check", file=sys.stderr)
            print("and any resulting fix are committed, then transfer freeze_check_local.json here.", file=sys.stderr)
            return 2
        print(f"REFUSED: no authoritative record at {local_path}", file=sys.stderr)
        return 2

    frozen = json.loads(Path(local_path).read_text(encoding="utf-8"))
    if not frozen.get("authoritative"):
        print(f"REFUSED: {local_path} is not marked authoritative, it looks like a candidate copy", file=sys.stderr)
        return 2

    checks = []
    frozen_commit = frozen.get("commit_recorded_before_writing") or frozen["git"]["commit"]
    here_commit = execution.git_commit()
    checks.append(check("git_head_matches_frozen_commit", here_commit == frozen_commit,
                        "this checkout is the commit the freeze recorded",
                        expected=frozen_commit, actual=here_commit))

    # every corpus file against the FROZEN list, file by file, not against the transferred manifest
    frozen_files = frozen["corpus"].get("files", {})
    missing, differing = [], []
    for relative, entry in sorted(frozen_files.items()):
        path = FINETUNE_V5 / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != entry["sha256"]:
            differing.append(relative)
    checks.append(check("corpus_files_match_frozen_manifest", not missing and not differing,
                        f"{len(frozen_files)} file(s) hashed against the frozen record",
                        expected="every file present and identical",
                        actual={"missing": missing[:5], "sha256_differs": differing[:5]}
                        if (missing or differing) else "all identical"))

    # base-model artifacts, each verified the way its kind and the record's version allow
    version = frozen.get("base_artifact_check_version", 1)
    model_problems, verified_from_base_tree = [], []
    for name, entry in sorted(frozen.get("models", {}).items()):
        frozen_base = entry.get("base_fingerprint") or {}
        if version >= 2 and frozen_base.get("present"):
            # the record has the directory the job will actually read, so check that directory
            here = local_fingerprint(entry.get("base_dir") or frozen_base.get("path"), deep=deep)
            model_problems += fingerprint_problems(name, "base_dir", frozen_base, here,
                                                   absent="base directory not present")
            verified_from_base_tree.append(name)
        elif entry["kind"] == "hub_revision":
            found = hub_revision_present(entry["checkpoint"], entry["revision"])
            if not found["present"]:
                model_problems.append({"model": name, "kind": entry["kind"], "expected": found["expected"],
                                       "actual": "not found in the hugging face cache"})
        else:
            here = local_fingerprint(model_registry.model(name)["checkpoint"], deep=deep)
            model_problems += fingerprint_problems(name, entry["kind"], entry.get("fingerprint", {}),
                                                   here, absent="derived copy not present")
    detail = "hub models by repository and revision, derived copies by fingerprint"
    if verified_from_base_tree:
        detail += f"; {len(verified_from_base_tree)} verified from the base tree ({', '.join(verified_from_base_tree)})"
    checks.append(check("base_model_artifacts_match_frozen_provenance", not model_problems,
                        f"[base artifact check version {version}] {detail}",
                        expected="no problems", actual=model_problems[:5] if model_problems else "no problems"))

    # results_store's own live resolution, the values a new record would carry, against the frozen ones
    live = results_store.provenance("freeze-gate", "check", 0, None, backend=None, profile=None,
                                    hardware={"skipped": "not needed for the gate"})
    checks.append(check("results_store_live_commit_matches", live["git_commit"] == frozen_commit,
                        "the commit a new result record would carry",
                        expected=frozen_commit, actual=live["git_commit"]))
    checks.append(check("results_store_live_manifest_matches",
                        live["corpus_manifest_sha256"] == frozen["corpus"].get("manifest_sha256"),
                        "the corpus hash a new result record would carry",
                        expected=frozen["corpus"].get("manifest_sha256"),
                        actual=live["corpus_manifest_sha256"]))

    passed = all(c["ok"] for c in checks)
    print(f"remote freeze gate against {local_path}")
    print(f"frozen commit {frozen_commit}, {len(frozen_files)} corpus file(s) checked")
    print_checks({"checks": checks})
    if passed:
        print("\ngate PASSED, this machine matches the frozen state")
        return 0
    print("\ngate FAILED, nothing may be submitted until every check above is ok", file=sys.stderr)
    return 1


def hub_revision_present(repo_id, revision, cache_dir=None):
    """is this exact revision in the hugging face cache

    the cache lays a repo out as models--<org>--<name>/snapshots/<revision>, so presence is a
    directory test rather than a download"""
    root = Path(cache_dir) if cache_dir else Path(
        execution.resolve("cuda", "local").hf_hub_cache)
    folder = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / str(revision)
    return {"present": folder.is_dir(), "expected": str(folder)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("candidate", "freeze"), default=None)
    ap.add_argument("--remote", action="store_true", help="the LUMI gate, reads the authoritative record only")
    ap.add_argument("--record", default="",
                    help="with --remote, the authoritative record to check against; defaults to "
                         "reports/v5/freeze_check_local.json")
    ap.add_argument("--deep", action="store_true",
                    help="also hash the multi-gigabyte weight shards; off by default so the gate stays fast")
    ap.add_argument("--expect-epochs", type=int, default=None,
                    help=f"the epoch count the generated scripts should train for, default {slurm.DEFAULT_EPOCHS}; "
                         "declare a deliberate change here rather than editing a script by hand")
    ap.add_argument("--expect-subsample", type=int, default=None,
                    help=f"the --subsample the generated scripts should carry; defaults to "
                         f"{slurm.DEFAULT_SUBSAMPLE}, the production draw, exactly as --expect-epochs "
                         f"defaults to the generator's epoch count. 0 means the full train split")
    ap.add_argument("--expect-layout", choices=slurm.LAYOUT_CHOICES, default=slurm.DEFAULT_LAYOUT,
                    help=f"the training output hierarchy the generated scripts should carry, default "
                         f"{slurm.DEFAULT_LAYOUT}; declare a deliberate change here rather than "
                         f"editing a script by hand")
    ap.add_argument("--expect-run-tag", default=slurm.DEFAULT_RUN_TAG,
                    help=f"the attempt segment of the run label the scripts should carry, default "
                         f"{slurm.DEFAULT_RUN_TAG}")
    ap.add_argument("--expect-base-layout", choices=model_registry.BASE_LAYOUT_CHOICES, default=None,
                    help="where the scripts' --base-model should point; defaults to each generated "
                         "set's own profile base_layout, else registry. give it only to declare a "
                         "deliberate change, and note it then applies to every set at once")
    ap.add_argument("--base-root", default="",
                    help="root whose models/base/<model> directories are fingerprinted into the "
                         "record; defaults to COB_PROFILE_ROOT, else this checkout")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.remote and args.mode:
        raise SystemExit("--remote and --mode are different gates, pass one of them")
    if args.remote:
        return run_remote(deep=args.deep, local_path=Path(args.record) if args.record else LOCAL_PATH)
    shared = {"deep": args.deep, "epochs": args.expect_epochs, "subsample": args.expect_subsample,
              "layout": args.expect_layout, "run_tag": args.expect_run_tag,
              "base_layout": args.expect_base_layout, "base_root": args.base_root or None}
    if args.mode == "candidate":
        return run_candidate(**shared)
    if args.mode == "freeze":
        return run_freeze(**shared)
    raise SystemExit("pass --mode candidate, --mode freeze, or --remote")


if __name__ == "__main__":
    sys.exit(main())
