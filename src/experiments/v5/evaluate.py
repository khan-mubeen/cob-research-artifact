"""v5 checkpoint evaluation on the full validation or test split, the only input v5 epoch selection reads

  PYTHONPATH=. COB_DATASET_VERSION=v5 python src/experiments/eval_finetune.py \
      --mode sealed --split validation --run qwen-sealed --epoch 1 --adapter <dir>
  PYTHONPATH=. python src/experiments/compare_epochs.py --arch qwen \
      --epoch-1 <validation result> --epoch-2 <validation result> --epoch-3 <validation result>
  PYTHONPATH=. COB_DATASET_VERSION=v5 python src/experiments/eval_finetune.py \
      --mode sealed --split test --run qwen-sealed --epoch <chosen> --adapter <chosen dir>

eval_finetune.py dispatches here when COB_DATASET_VERSION resolves to v5. results land in
reports/v5/selection/<mode>/<model>/ (epoch_selection.run_dir)

  validation  every row of the mode's three validation files (the supervisor has one), scored and
              written as validation_epoch<n>.json (validation_base.json for the base arm). a run's
              epoch is evaluated once; an existing result is never overwritten
  test        refused unless the run's selection.json exists and chose this exact adapter and epoch,
              and refused if the run already has a test result, so the test split is read once
  smoke       --smoke-rows N scores only the first N rows per file, to prove the path on this machine;
              the result is marked smoke and epoch selection refuses it

scoring reuses the v4 metric definitions rather than restating them: eval_finetune.score and
summarize for sealed rows, eval_reverse_auction.score and summarize for reverse-auction rows. the
supervisor prompt is in USD, so its candidate table is parsed here with a USD pattern and judged by
the same objective eval_supervisor.py uses
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

from config import PROJECT_ROOT
from src.experiments.v5 import epoch_selection, model_registry, training_data

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
SPLITS = ("validation", "test")
STAGES = ("material", "tailoring", "logistics")
MODE_FILES = {
    "sealed": lambda split: [f"{stage}_{split}.jsonl" for stage in STAGES],
    "reverse_auction": lambda split: [f"reverse_auction/{stage}_{split}.jsonl" for stage in STAGES],
    "supervisor": lambda split: [f"supervisor/selection_{split}.jsonl"],
}
MODE_DATASET = {"sealed": "v5_sealed", "reverse_auction": "v5_reverse_auction", "supervisor": "v5_supervisor"}
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

USD_CANDIDATE = re.compile(
    r"candidate (\d+): "
    r"material\(cost_usd=([\d.]+), days=([\d.]+), defect_p=([\d.]+)\), "
    r"manufacturing\(cost_usd=([\d.]+), days=([\d.]+), defect_p=([\d.]+)\), "
    r"logistics\(cost_usd=([\d.]+), days=([\d.]+), damage_p=([\d.]+)\), "
    r"total_cost_usd=([\d.]+), total_completion_days=([\d.]+)")
USD_BUDGET = re.compile(r"budget \(USD\): ([\d.]+)")
DAYS = re.compile(r"days remaining until deadline: ([\d.]+)")
CAP = re.compile(r"max acceptable defect or damage probability: ([\d.]+)")


def load_rows(mode, split, smoke_rows=0, root=FINETUNE_V5):
    rows = []
    for relative in MODE_FILES[mode](split):
        with open(root / relative, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if smoke_rows and i >= smoke_rows:
                    break
                rows.append(json.loads(line))
    return rows


def identity(row):
    return "/".join(str(row.get(k)) for k in ("order_id", "agent_id", "stage", "category", "round_number"))


def parse_supervisor(user):
    candidates = []
    for m in USD_CANDIDATE.finditer(user):
        g = m.groups()
        candidates.append({"index": int(g[0]), "material_defect_probability": float(g[3]),
                           "manufacturer_defect_probability": float(g[6]), "logistics_damage_probability": float(g[9]),
                           "total_cost": float(g[10]), "total_days": float(g[11])})
    order = {"budget": float(USD_BUDGET.search(user).group(1)), "days_until_deadline": float(DAYS.search(user).group(1)),
             "max_defect_rate": float(CAP.search(user).group(1))}
    return candidates, order


def priority_terms(cand, order, strategy):
    """(primary, secondary) for one candidate, compared as the rule compares them

    cost and days at the two decimals the prompt prints, so a candidate that reads as equal is
    equal here. the rule's last term, the agent-id tuple, is deliberately absent: the prompt does
    not print agent ids, so nothing judged from the prompt can reach it"""
    from src.experiments.eval_supervisor import objective
    cost, days = round(cand["total_cost"], 2), round(cand["total_days"], 2)
    if strategy == "urgent":
        return days, cost
    if strategy == "flexible":
        return cost, days
    return objective(cand, order, strategy), (cost, days)


def miss_kind(selected, truth, order, strategy):
    """why a selection differs from the label, judged on what the prompt shows"""
    primary, secondary = priority_terms(selected, order, strategy)
    best_primary, best_secondary = priority_terms(truth, order, strategy)
    if primary != best_primary:
        return "primary_miss"
    if secondary != best_secondary:
        return "secondary_error"
    return "indistinguishable_tie"


def pct(count, total):
    return round(count / total * 100, 2) if total else None


def score_supervisor(rows, outputs):
    """the supervisor metric set: what was chosen, how far off it was, and why it differed

    the labels this scores are full-enumeration optima, so regret and the two deviations measure
    distance from the best chain that existed for the order, not from the best of a shortlist.
    json parsing and a usable index are reported apart, and a difference that the prompt cannot
    resolve, an exact tie the label settles by agent id, is counted as its own outcome rather than
    as a model error"""
    from src.common.slm import _extract_json
    from src.experiments.eval_supervisor import objective
    n_parsed = n_valid = n_correct = n_infeasible = 0
    kinds, by_strategy = Counter(), defaultdict(Counter)
    regrets, cost_devs, days_devs, records = [], [], [], []
    for row, raw in zip(rows, outputs):
        truth_index = json.loads(row["completion"])["selected_candidate"]
        candidates, order = parse_supervisor(row["user"])
        strategy = row["strategy"]
        parsed = _extract_json(raw)
        parsed_ok = isinstance(parsed, dict)
        index = parsed.get("selected_candidate") if parsed_ok else None
        valid = isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(candidates)
        n_parsed += parsed_ok
        by_strategy[strategy]["n"] += 1
        regret = cost_dev = days_dev = kind = None
        feasible = None
        if valid:
            n_valid += 1
            truth, chosen = candidates[truth_index], candidates[index]
            best = objective(truth, order, strategy)
            regret = (objective(chosen, order, strategy) - best) / best if best > 0 else 0.0
            regrets.append(regret)
            cost_dev = (chosen["total_cost"] - truth["total_cost"]) / truth["total_cost"] * 100 \
                if truth["total_cost"] else 0.0
            days_dev = (chosen["total_days"] - truth["total_days"]) / truth["total_days"] * 100 \
                if truth["total_days"] else 0.0
            cost_devs.append(cost_dev)
            days_devs.append(days_dev)
            feasible = (chosen["total_cost"] <= order["budget"]
                        and chosen["total_days"] <= order["days_until_deadline"])
            n_infeasible += not feasible
            if index != truth_index:
                kind = miss_kind(chosen, truth, order, strategy)
                kinds[kind] += 1
                by_strategy[strategy][kind] += 1
        correct = bool(valid and index == truth_index)
        n_correct += correct
        by_strategy[strategy]["correct"] += correct
        records.append({"order_id": row["order_id"], "strategy": strategy, "truth": truth_index, "predicted": index,
                        "json_parsed": parsed_ok, "valid_json": valid, "correct": correct, "regret": regret,
                        "cost_dev_pct": cost_dev, "days_dev_pct": days_dev, "feasible": feasible,
                        "miss_kind": kind, "raw": raw[:200]})
    n = len(rows)
    summary = {
        "n": n,
        "json_parse_pct": pct(n_parsed, n),
        # json_valid_pct keeps its established meaning, parseable AND in range, and is the epoch
        # selection key; usable_index_pct names the same number for a reader
        "json_valid_pct": pct(n_valid, n),
        "usable_index_pct": pct(n_valid, n),
        "invalid_or_out_of_range_pct": pct(n - n_valid, n),
        "selection_accuracy_pct": pct(n_correct, n),
        "mean_regret_pct": round(statistics.mean(regrets) * 100, 4) if regrets else None,
        "mean_cost_dev_pct": round(statistics.mean(cost_devs), 4) if cost_devs else None,
        "mean_abs_cost_dev_pct": round(statistics.mean(abs(d) for d in cost_devs), 4) if cost_devs else None,
        "mean_days_dev_pct": round(statistics.mean(days_devs), 4) if days_devs else None,
        "mean_abs_days_dev_pct": round(statistics.mean(abs(d) for d in days_devs), 4) if days_devs else None,
        "feasibility_violation_pct": pct(n_infeasible, n_valid),
        "primary_miss_pct": pct(kinds["primary_miss"], n_valid),
        "secondary_error_pct": pct(kinds["secondary_error"], n_valid),
        "indistinguishable_tie_pct": pct(kinds["indistinguishable_tie"], n_valid),
        "by_strategy": {s: {"n": c["n"], "accuracy_pct": pct(c["correct"], c["n"]),
                            "primary_miss": c["primary_miss"], "secondary_error": c["secondary_error"],
                            "indistinguishable_tie": c["indistinguishable_tie"]}
                        for s, c in sorted(by_strategy.items())},
    }
    return summary, records


SPLIT_METADATA = "split/metadata.json"
PIN_KEY = "test_row_ids_sha256"


def pinned_test_digests(root=FINETUNE_V5):
    """the frozen authority: the recorded test digests from split/metadata.json

    that file is itself listed in manifest.json, so it is checked against the manifest before its
    contents are trusted. that check is what lets a later failure name the artifact that moved: an
    edited pin fails here, as metadata integrity, instead of being reported as corpus-row drift"""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    listed = {f["path"]: f for f in manifest["files"]}
    entry = listed.get(SPLIT_METADATA)
    if entry is None:
        raise epoch_selection.SelectionError(f"{SPLIT_METADATA} is not listed in the corpus manifest")
    path = root / SPLIT_METADATA
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise epoch_selection.SelectionError(
            f"{SPLIT_METADATA} does not match the corpus manifest\n"
            f"  manifest records: {entry['sha256']}\n"
            f"  file on disk is:  {digest}\n"
            f"  the split metadata carries the pinned test digests, so it is not trusted while it "
            f"disagrees with the manifest. this is a metadata integrity failure, not corpus-row drift")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    pins = metadata.get(PIN_KEY)
    if not pins:
        raise epoch_selection.SelectionError(
            f"{SPLIT_METADATA} has no {PIN_KEY}: the pinned digest is the authority a test read is "
            f"checked against, and it is never derived on the fly to fill the gap. pin it once from "
            f"the corpus, record it, and rebuild the manifest")
    return pins


def canonical_test_digest(mode, root=FINETUNE_V5):
    """the test split's row-identity digest: the pinned value, cross-checked against the live corpus

    the order matters and is deliberate:
      1 derive the digest live from the actual corpus rows
      2 check those corpus files against manifest.json
      3 read the pinned value from split/metadata.json, itself manifest-checked
      4 compare, and refuse when they disagree

    the pinned metadata value is the frozen authority. live derivation is the cross-check that
    proves the pin still matches reality, which is the direction that can go stale and get caught.
    a disagreement is never resolved by regenerating the pin from the live rows: that would let
    whatever changed quietly become the new truth, which is the whole failure this pin exists to
    prevent"""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    listed = {f["path"]: f for f in manifest["files"]}
    files = []
    for relative in MODE_FILES[mode]("test"):
        entry = listed.get(relative)
        if entry is None:
            raise epoch_selection.SelectionError(f"{relative} is not listed in the corpus manifest")
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise epoch_selection.SelectionError(
                f"{relative} does not match the manifest ({digest[:16]} against {entry['sha256'][:16]}); "
                f"the frozen corpus changed under the test gate")
        files.append({"path": relative, "sha256": entry["sha256"], "rows": entry["rows"]})
    identities = [identity(r) for r in load_rows(mode, "test", root=root)]
    live = epoch_selection.row_ids_digest(identities)

    pinned = pinned_test_digests(root).get(mode)
    if not pinned:
        raise epoch_selection.SelectionError(
            f"{SPLIT_METADATA} pins no test digest for mode {mode!r}; the pinned value is the authority "
            f"and is not filled in from the live corpus")
    if live != pinned:
        raise epoch_selection.SelectionError(
            f"the {mode} test rows do not match the pinned digest in {SPLIT_METADATA}\n"
            f"  pinned (the frozen authority): {pinned}\n"
            f"  derived live from the corpus:  {live} over {len(identities)} rows\n"
            f"  one of the two artifacts lost integrity: either the corpus rows drifted or the pinned\n"
            f"  metadata was modified. nothing is regenerated here. run the manifest verification and\n"
            f"  freeze_check to see which file stopped matching its recorded hash")
    return {"mode": mode, "rows": len(identities), "row_ids_sha256": pinned,
            "pinned": pinned, "derived_live": live, "authority": SPLIT_METADATA, "files": files}


def verify_test_digest(mode, identities, root=FINETUNE_V5):
    """each test arm checks its rows against the canonical value, never against the other arm"""
    canonical = canonical_test_digest(mode, root)
    actual = epoch_selection.row_ids_digest(identities)
    if actual != canonical["row_ids_sha256"]:
        raise epoch_selection.SelectionError(
            f"test rows are not the canonical {mode} test split: expected {canonical['row_ids_sha256']} "
            f"over {canonical['rows']} rows, got {actual} over {len(identities)} rows")
    return canonical


def is_base(args):
    """a base-model arm: no adapter, epoch 0

    the two are kept in step deliberately. an adapter with epoch 0, or a base run claiming epoch 2,
    would produce a record that reads as one thing and means another"""
    return not args.adapter or args.epoch == 0


def guard(args, root=epoch_selection.RESULT_ROOT):
    """what this evaluation is allowed to write; raises SelectionError otherwise"""
    run = epoch_selection.run_dir(args.run, root)
    if is_base(args):
        if args.adapter or args.epoch != 0:
            raise epoch_selection.SelectionError(
                "a base-model arm is --adapter '' with --epoch 0; got "
                f"adapter {args.adapter!r} and epoch {args.epoch}")
        if args.split == "test":
            # the base arm gets its own one-shot test read, separate from the tuned checkpoint's:
            # two independent things, two authorizations, neither able to spend the other's slot
            if args.smoke_rows:
                raise epoch_selection.SelectionError("the test split is never smoke-evaluated")
            authorization = epoch_selection.authorize_base_test(args.run, root)
            return epoch_selection.base_test_result_path(args.run, root), authorization
        out = epoch_selection.validation_path(args.run, 0, bool(args.smoke_rows), root)
        if out.exists():
            raise epoch_selection.SelectionError(f"{out} already exists; the base arm is written once")
        return out, None
    if args.split == "test":
        if args.smoke_rows:
            raise epoch_selection.SelectionError("the test split is never smoke-evaluated")
        authorization = epoch_selection.authorize_test(args.run, args.adapter, args.epoch, root)
        return epoch_selection.test_result_path(args.run, root), authorization
    if args.epoch not in epoch_selection.EPOCHS:
        raise epoch_selection.SelectionError(f"epoch must be one of {epoch_selection.EPOCHS}")
    out = epoch_selection.validation_path(args.run, args.epoch, bool(args.smoke_rows), root)
    if out.exists():
        raise epoch_selection.SelectionError(f"{out} already exists; an epoch's validation result is written once")
    return out, None


def generate_and_score(args, model, tokenizer, rows):
    """(summary, scored rows, stop-rule record): generation with the v5 stop rule, then the mode's scorer"""
    from src.experiments.v5 import stop_rule
    model_name, _ = stop_rule.cell_of(args.run, getattr(args, "model_name", ""))
    stops, stop_info = stop_rule.stop_ids(model, tokenizer, model_name, args.mode)
    if args.mode == "sealed":
        from src.experiments.eval_finetune import generate_all, score, summarize
        examples = [{"id": f"{r['order_id']}/{r['agent_id']}", "stage": r["stage"], "system": r["system"], "user": r["user"],
                     "label": json.loads(r["completion"])} for r in rows]
        scored = score(examples, generate_all(model, tokenizer, examples, stop_ids=stops))
        summary = summarize(f"{args.split} sealed", scored)
    elif args.mode == "reverse_auction":
        from src.experiments.eval_reverse_auction import generate_all, prompt_priority, score, summarize
        for r in rows:
            r["priority"] = prompt_priority(r["user"])
        scored = score(rows, generate_all(model, tokenizer, rows, stop_ids=stops))
        summary = summarize(args.tag or args.run, args.adapter, scored)
    else:
        from src.experiments.eval_finetune import generate_all
        examples = [{"system": r["system"], "user": r["user"]} for r in rows]
        summary, scored = score_supervisor(rows, generate_all(model, tokenizer, examples, stop_ids=stops))
    return summary, scored, stop_info


def resolve_checkpoint(args):
    """(checkpoint, revision) for this arm

    a trained arm loads its adapter directory and pins nothing; the base arm loads the registry's
    pinned revision, so the untrained number is taken against the same weights the training runs
    start from.

    --base-layout profile moves only the base arm, to <root>/models/base/<model>. a directory
    carries no hub revision, so the pin goes with it and the record then names the directory that
    was actually read. the tuned arm is untouched: its adapter string is passed through exactly as
    given, because epoch_selection compares that string and does not normalize it"""
    if not is_base(args):
        return args.adapter, None
    if args.base_model:
        return args.base_model, None
    if args.model_name:
        spec = model_registry.model(args.model_name)
        if getattr(args, "base_layout", model_registry.BASE_LAYOUT_DEFAULT) == "profile":
            root = model_registry.resolve_base_root(getattr(args, "base_root", "") or None)
            return model_registry.base_dir(args.model_name, root), None
        return spec["checkpoint"], (None if spec["local"] else spec["revision"])
    return BASE_MODEL, BASE_REVISION


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=epoch_selection.MODES)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--run", required=True, help="one model and mode, e.g. qwen-sealed")
    ap.add_argument("--epoch", required=True, type=int,
                    help="1, 2 or 3 for a trained checkpoint, or 0 for the base-model arm")
    ap.add_argument("--adapter", default="",
                    help="the epoch's lora adapter directory; empty with --epoch 0 evaluates the "
                         "untrained base model, mirroring the v4 scripts' --adapter ''")
    ap.add_argument("--base-model", default="",
                    help="hf checkpoint for the base arm; defaults to the model registry's entry "
                         "for --model-name, else Qwen/Qwen3.5-4B")
    ap.add_argument("--base-layout", choices=model_registry.BASE_LAYOUT_CHOICES,
                    default=model_registry.BASE_LAYOUT_DEFAULT,
                    help=f"where the base arm reads its weights, default "
                         f"{model_registry.BASE_LAYOUT_DEFAULT}: the registry's pinned hub id or "
                         f"local copy. profile reads <root>/models/base/<model> instead; it applies "
                         f"to the base arm only and is ignored when --base-model is given")
    ap.add_argument("--base-root", default="",
                    help="root for --base-layout profile; defaults to COB_PROFILE_ROOT, else this checkout")
    ap.add_argument("--model-name", default="",
                    help="registry name (qwen, granite, phi, ministral, gemma, smollm3) used to "
                         "resolve the base checkpoint and its pinned revision")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-seq-length", type=int, default=0, help="defaults to the mode's required v5 window")
    ap.add_argument("--smoke-rows", type=int, default=0, help="score only the first N rows per file; never selects")
    ap.add_argument("--dry-run", action="store_true", help="load rows and check the guard, no model")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from src.dataset import context as dataset_context
    if dataset_context.resolve().version != "v5":
        raise SystemExit(f"v5 evaluation needs {dataset_context.ENV_VAR}=v5")
    out_path, authorization = guard(args)
    rows = load_rows(args.mode, args.split, args.smoke_rows)
    identities = [identity(r) for r in rows]
    # both test arms verify the rows against the corpus-derived canonical digest before anything is
    # generated, so a wrong row set fails before the expensive work rather than after it
    canonical = verify_test_digest(args.mode, identities) if args.split == "test" else None
    print(f"{args.run} {args.mode} {args.split}: {len(rows)} rows{' (smoke)' if args.smoke_rows else ''} -> {out_path}")
    if args.dry_run:
        return 0

    window = args.max_seq_length or training_data.V5_WINDOWS[MODE_DATASET[args.mode]]
    training_data.check_window(MODE_DATASET[args.mode], window)
    checkpoint, revision = resolve_checkpoint(args)
    from unsloth import FastLanguageModel  # imported only when a model is really evaluated
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint, revision=revision,
                                                         max_seq_length=window,
                                                         load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    summary, scored, stop_info = generate_and_score(args, model, tokenizer, rows)

    result = {
        "dataset_version": "v5", "mode": args.mode, "split": args.split, "run": args.run, "epoch": args.epoch,
        "arm": "base" if is_base(args) else "tuned",
        "adapter": args.adapter, "tag": args.tag, "smoke": bool(args.smoke_rows), "rows": len(rows),
        "canonical_test": canonical,
        "row_ids_sha256": epoch_selection.row_ids_digest(identities), "window": window,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "authorization": authorization,
        "summary": summary, "selection_metrics": epoch_selection.selection_metrics(args.mode, summary),
        "generation_stop_rule": stop_info,
        "rows_detail": scored,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
