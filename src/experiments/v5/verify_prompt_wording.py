"""tier 1 verification of the v5 prompt wording harmonization, streamed row by row

  PYTHONPATH=. python -m src.experiments.v5.verify_prompt_wording --candidate data/finetune/.v5_wording_scratch
  PYTHONPATH=. python -m src.experiments.v5.verify_prompt_wording --candidate ... --examples-md reports/v5/wording/examples.md

the accepted corpus is data/finetune/v5, pinned by reports/v5/wording/pre_change/snapshot.json. the
candidate is a regenerated corpus in a scratch directory. nothing here loads a whole file

  mapping      the validated worded style is read from reports/v5/wording/validated_mapping.json,
               the literal operator-to-phrase mapping extracted from the reverse-auction prompt
               builder. the rewording rule below is written from that mapping, not imported from
               slm.py, so the check does not reuse the code it checks
  changed      sealed and edge files: the same row count and order; every field except user
               identical, value for value, including order_id, agent_id, stage, split,
               rule_decision, the completion label, category and role; and the new user prompt
               equal to the accepted prompt with only its TASK block reworded by the mapping.
               the new TASK block must hold no bare mapped operator and at least one worded phrase
  unchanged    reverse-auction, floor-boundary and supervisor files: sha256 equal to the accepted
               file and to the pre-change snapshot. reverse-auction and floor-boundary TASK blocks
               must also hold no bare mapped operator
  metadata     any metadata file that differs is listed; only the sealed and edge metadata may

only aggregate counts and at most a few failing identities are reported. writes
reports/v5/wording/verify_prompt_wording.json and exits non-zero on any failure
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from config import PROJECT_ROOT

ACCEPTED = PROJECT_ROOT / "data" / "finetune" / "v5"
WORDING_DIR = PROJECT_ROOT / "reports" / "v5" / "wording"
MAPPING_PATH = WORDING_DIR / "validated_mapping.json"
SNAPSHOT_PATH = WORDING_DIR / "pre_change" / "snapshot.json"
REPORT_PATH = WORDING_DIR / "verify_prompt_wording.json"
SPLITS = ("train", "validation", "test")
STAGES = ("material", "tailoring", "logistics")
TASK_START, TASK_END = "\nTASK\n", "\n\nReturn this JSON only:"
MAX_EXAMPLES = 3

CHANGED = [f"{stage}_{split}.jsonl" for stage in STAGES for split in SPLITS] + \
          [f"edge/{stage}_edge.jsonl" for stage in STAGES]
UNCHANGED_WORDED = [f"reverse_auction/{stage}_{split}.jsonl" for stage in STAGES for split in SPLITS] + \
                   [f"reverse_auction/floor_boundary/{stage}_{case}_floor.jsonl" for stage in STAGES
                    for case in ("below", "at", "above")]
UNCHANGED_OTHER = [f"supervisor/selection_{split}.jsonl" for split in SPLITS] + ["split/order_split.csv"]
METADATA_MAY_CHANGE = {"metadata.json", "edge/metadata.json"}


def load_mapping(path=MAPPING_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["mapping"]


class Wording:
    """the rewording rule, built from the literal mapping

    an operator ending in an opening bracket, max(, is reworded wherever it stands. every other
    operator is reworded only as a token with a space on each side, which is how the formulas are
    laid out and what keeps negative numbers, identifiers and assignments untouched"""

    def __init__(self, mapping):
        self.mapping = mapping
        bracketed = [re.escape(k) for k in mapping if k.endswith("(")]
        spaced = [re.escape(k) for k in sorted((k for k in mapping if not k.endswith("(")), key=len, reverse=True)]
        self.pattern = re.compile("|".join(bracketed + [f"(?<= )(?:{'|'.join(spaced)})(?= )"]))

    def reword(self, text):
        return self.pattern.sub(lambda m: self.mapping[m.group(0)], text)

    def bare_operators(self, text):
        """mapped operators still written as a bare symbol: a worded phrase carries its symbol in
        brackets, never between spaces, and a worded max( is preceded by its phrase"""
        bare = Counter()
        for m in self.pattern.finditer(text):
            token = m.group(0)
            if token.endswith("("):
                phrase = self.mapping[token]
                if text[max(0, m.end() - len(phrase)):m.end()] == phrase:
                    continue
            bare[token] += 1
        return bare

    def phrases(self, text):
        return sum(text.count(phrase) for phrase in self.mapping.values())


def task_bounds(user):
    start = user.index(TASK_START)
    return start, user.index(TASK_END, start)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_changed(relative, accepted_root, candidate_root, wording):
    stats = Counter()
    examples = []
    with open(accepted_root / relative, encoding="utf-8") as old_f, open(candidate_root / relative, encoding="utf-8") as new_f:
        for old_line, new_line in zip(old_f, new_f):
            stats["rows_compared"] += 1
            old, new = json.loads(old_line), json.loads(new_line)
            identity = f"{old.get('order_id')}/{old.get('agent_id')}"
            if list(old) != list(new):
                stats["field_set_or_order_mismatch"] += 1
                len(examples) < MAX_EXAMPLES and examples.append((identity, "fields"))
                continue
            for key in old:
                if key != "user" and old[key] != new[key]:
                    stats["non_prompt_field_mismatch"] += 1
                    stats[f"non_prompt_field_mismatch:{key}"] += 1
                    len(examples) < MAX_EXAMPLES and examples.append((identity, key))
            s, e = task_bounds(old["user"])
            expected = old["user"][:s] + wording.reword(old["user"][s:e]) + old["user"][e:]
            if new["user"] != expected:
                stats["prompt_not_the_mapped_rewording"] += 1
                len(examples) < MAX_EXAMPLES and examples.append((identity, "user"))
            else:
                stats["prompt_is_the_mapped_rewording"] += 1
            ns, ne = task_bounds(new["user"])
            task = new["user"][ns:ne]
            bare = wording.bare_operators(task)
            if bare:
                stats["rows_with_a_bare_mapped_operator"] += 1
            count = wording.phrases(task)
            stats["worded_phrases_total"] += count
            stats["rows_with_no_worded_phrase"] += count == 0
            stats["accepted_rows_that_were_symbolic"] += bool(wording.bare_operators(old["user"][s:e]))
        leftover_old, leftover_new = old_f.readline(), new_f.readline()
        if leftover_old or leftover_new:
            stats["row_count_mismatch"] += 1
    passed = (stats["rows_compared"] > 0 and not stats["row_count_mismatch"] and not stats["field_set_or_order_mismatch"]
              and not stats["non_prompt_field_mismatch"] and not stats["prompt_not_the_mapped_rewording"]
              and not stats["rows_with_a_bare_mapped_operator"] and not stats["rows_with_no_worded_phrase"])
    return {"kind": "changed", "passed": passed, "counts": dict(stats), "examples": examples}


def check_unchanged(relative, accepted_root, candidate_root, snapshot, wording, worded):
    accepted_hash, candidate_hash = sha256_file(accepted_root / relative), sha256_file(candidate_root / relative)
    result = {"kind": "unchanged", "accepted_sha256_equals_snapshot": accepted_hash == snapshot["files"][relative]["sha256"],
              "candidate_byte_identical": candidate_hash == accepted_hash}
    counts = Counter()
    if worded:
        with open(candidate_root / relative, encoding="utf-8") as f:
            for line in f:
                user = json.loads(line)["user"]
                s, e = task_bounds(user)
                counts["rows"] += 1
                counts["rows_with_a_bare_mapped_operator"] += bool(wording.bare_operators(user[s:e]))
                counts["worded_phrases_total"] += wording.phrases(user[s:e])
        result["counts"] = dict(counts)
    result["passed"] = (result["accepted_sha256_equals_snapshot"] and result["candidate_byte_identical"]
                        and not counts["rows_with_a_bare_mapped_operator"])
    return result


def check_metadata(accepted_root, candidate_root, snapshot):
    out = {}
    for relative in sorted(p for p in snapshot["files"] if p.endswith("metadata.json")):
        candidate = candidate_root / relative
        if not candidate.exists():
            out[relative] = {"candidate_present": False, "passed": False}
            continue
        same = sha256_file(candidate) == snapshot["files"][relative]["sha256"]
        out[relative] = {"identical": same, "may_change": relative in METADATA_MAY_CHANGE,
                         "passed": same or relative in METADATA_MAY_CHANGE}
        if not same:
            old = json.loads((accepted_root / relative).read_text(encoding="utf-8"))
            new = json.loads(candidate.read_text(encoding="utf-8"))
            out[relative]["keys_changed"] = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    return out


def examples_markdown(candidate_root, path):
    """complete example rows for the report's tier 2, written straight to a file"""
    picks = [("sealed", f"{stage}_train.jsonl", 1) for stage in STAGES] + \
            [("edge", "edge/material_edge.jsonl", 1), ("edge", "edge/tailoring_edge.jsonl", 1),
             ("edge", "edge/logistics_edge.jsonl", 1),
             ("reverse auction", "reverse_auction/logistics_train.jsonl", 1),
             ("reverse auction", "reverse_auction/material_train.jsonl", 1),
             ("reverse auction", "reverse_auction/tailoring_train.jsonl", 1),
             ("floor boundary", "reverse_auction/floor_boundary/material_below_floor.jsonl", 1),
             ("floor boundary", "reverse_auction/floor_boundary/tailoring_at_floor.jsonl", 1),
             ("floor boundary", "reverse_auction/floor_boundary/logistics_above_floor.jsonl", 1)]
    blocks = []
    for kind, relative, count in picks:
        with open(candidate_root / relative, encoding="utf-8") as f:
            for _ in range(count):
                row = json.loads(f.readline())
                shown = {k: v for k, v in row.items() if k not in ("system", "user", "completion")}
                blocks.append(f"#### {kind}: `{relative}`, {row['order_id']} / {row.get('agent_id')}\n\n"
                              f"fields: `{json.dumps(shown, ensure_ascii=False)}`\n\n"
                              f"```text\n[system]\n{row['system']}\n\n[user]\n{row['user']}\n\n[completion]\n{row['completion']}\n```\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(blocks), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="the regenerated corpus root to verify")
    parser.add_argument("--examples-md", default="")
    args = parser.parse_args()
    candidate_root = (PROJECT_ROOT / args.candidate) if not Path(args.candidate).is_absolute() else Path(args.candidate)
    wording = Wording(load_mapping())
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    files = {}
    for relative in CHANGED:
        files[relative] = check_changed(relative, ACCEPTED, candidate_root, wording)
        files[relative]["accepted_sha256_equals_snapshot"] = sha256_file(ACCEPTED / relative) == snapshot["files"][relative]["sha256"]
        files[relative]["passed"] &= files[relative]["accepted_sha256_equals_snapshot"]
    for relative in UNCHANGED_WORDED:
        files[relative] = check_unchanged(relative, ACCEPTED, candidate_root, snapshot, wording, worded=True)
    for relative in UNCHANGED_OTHER:
        if (candidate_root / relative).exists():
            files[relative] = check_unchanged(relative, ACCEPTED, candidate_root, snapshot, wording, worded=False)
        else:
            files[relative] = {"kind": "unchanged, not regenerated",
                               "accepted_sha256_equals_snapshot": sha256_file(ACCEPTED / relative) == snapshot["files"][relative]["sha256"]}
            files[relative]["passed"] = files[relative]["accepted_sha256_equals_snapshot"]
    metadata = check_metadata(ACCEPTED, candidate_root, snapshot)

    for relative, r in files.items():
        c = r.get("counts", {})
        print(f"{'PASS' if r['passed'] else 'FAIL'} {relative:62s} {r['kind']:24s} "
              f"rows {c.get('rows_compared', c.get('rows', '-'))} reworded {c.get('prompt_is_the_mapped_rewording', '-')} "
              f"non-prompt mismatches {c.get('non_prompt_field_mismatch', 0)} bare {c.get('rows_with_a_bare_mapped_operator', 0)} "
              f"phrases {c.get('worded_phrases_total', '-')}"
              + (f" byte-identical {r.get('candidate_byte_identical')}" if r["kind"] == "unchanged" else "")
              + (f" examples {r['examples']}" if r.get("examples") else ""))
    for relative, r in metadata.items():
        print(f"{'PASS' if r['passed'] else 'FAIL'} {relative:62s} metadata identical {r.get('identical')} "
              f"{'changed keys ' + str(r.get('keys_changed')) if r.get('keys_changed') else ''}")
    passed = all(r["passed"] for r in files.values()) and all(r["passed"] for r in metadata.values())
    result = {"candidate": str(candidate_root.relative_to(PROJECT_ROOT)).replace("\\", "/"),
              "mapping": wording.mapping, "files": files, "metadata": metadata, "passed": passed,
              "totals": {"changed_rows_compared": sum(r["counts"].get("rows_compared", 0) for r in files.values() if r["kind"] == "changed"),
                         "changed_rows_reworded_exactly": sum(r["counts"].get("prompt_is_the_mapped_rewording", 0)
                                                              for r in files.values() if r["kind"] == "changed"),
                         "non_prompt_field_mismatches": sum(r["counts"].get("non_prompt_field_mismatch", 0)
                                                            for r in files.values() if r["kind"] == "changed"),
                         "unchanged_files_byte_identical": sum(1 for r in files.values() if r.get("candidate_byte_identical")),
                         "unchanged_files_checked": sum(1 for r in files.values() if r["kind"].startswith("unchanged"))}}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.examples_md:
        examples_markdown(candidate_root, PROJECT_ROOT / args.examples_md)
    print(f"totals {result['totals']}\npassed {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
