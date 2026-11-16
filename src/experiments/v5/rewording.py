"""the v5 prompt wording harmonization: extract the mapping, regenerate in scratch, promote, clean up

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.rewording extract
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.rewording regenerate
  PYTHONPATH=. python -m src.experiments.v5.verify_prompt_wording --candidate data/finetune/.v5_wording_scratch
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.rewording promote
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.rewording cleanup

extract     writes reports/v5/wording/validated_mapping.json: the literal operator-to-phrase mapping
            and rule the reverse-auction prompts render with, read from slm.py as it stands
regenerate  runs every v5 generator unmodified, with its OUT_DIR pointed at
            data/finetune/.v5_wording_scratch, so the accepted corpus is never written. the edge
            and floor-boundary generators read the accepted sealed and reverse-auction files for
            their overlap and exclusion checks, which the wording change does not touch
promote     refuses unless the tier 1 verification report says passed and was made against this
            scratch. copies only the sealed and edge files that changed, plus their metadata, into
            data/finetune/v5, byte-checks each copy, rebuilds manifest.json and verifies it
cleanup     deletes the scratch corpus once promotion is recorded, so no second copy remains
"""

import io
import json
import shutil
import sys
from contextlib import redirect_stdout

from config import PROJECT_ROOT
from src.experiments.v5 import (
    generate_edge_examples, generate_finetune_data, generate_floor_boundary, generate_reverse_auction_data,
    generate_supervisor_data, manifest,
)
from src.experiments.v5.common import FINETUNE_V5, require_v5, sha256_file, write_json
from src.experiments.v5.verify_prompt_wording import CHANGED, MAPPING_PATH, METADATA_MAY_CHANGE, REPORT_PATH

SCRATCH = PROJECT_ROOT / "data" / "finetune" / ".v5_wording_scratch"
PROMOTION_PATH = PROJECT_ROOT / "reports" / "v5" / "wording" / "promotion.json"
GENERATORS = (("", generate_finetune_data), ("reverse_auction", generate_reverse_auction_data),
              ("supervisor", generate_supervisor_data), ("edge", generate_edge_examples),
              ("reverse_auction/floor_boundary", generate_floor_boundary))


def extract():
    from src.common import slm
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "src/common/slm.py _OPERATOR_WORDS and _OPERATOR_RE, the rewrite the reverse-auction prompts use",
        "mapping": dict(slm._OPERATOR_WORDS),
        "rule_as_written": slm._OPERATOR_RE.pattern,
        "rule_in_words": ("max( is reworded wherever it stands; every other operator only as a token with a space on "
                          "each side; only the TASK block is reworded, the profile, rfq and output schema are not"),
        "not_in_the_mapping": {"=": "an assignment in a formula line, rendered symbolically by the reverse-auction "
                                    "prompts too", "==": "the logistics route-days comparison, rendered symbolically "
                                                         "by the reverse-auction prompts too"},
    }
    MAPPING_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def regenerate():
    require_v5()
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    for relative, module in GENERATORS:
        out = SCRATCH / relative if relative else SCRATCH
        module.OUT_DIR = out
        with redirect_stdout(io.StringIO()):
            module.main()
        print(f"regenerated {relative or 'sealed'} into {out.relative_to(PROJECT_ROOT)}")
    shutil.copytree(FINETUNE_V5 / "split", SCRATCH / "split")


def promote():
    require_v5()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if not report["passed"] or report["candidate"] != str(SCRATCH.relative_to(PROJECT_ROOT)).replace("\\", "/"):
        raise SystemExit("the tier 1 verification has not passed against this scratch corpus, nothing promoted")
    copied = {}
    for relative in CHANGED + sorted(METADATA_MAY_CHANGE):
        source, target = SCRATCH / relative, FINETUNE_V5 / relative
        shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise SystemExit(f"{relative}: the promoted copy differs from the verified scratch file")
        copied[relative] = sha256_file(target)
    write_json(manifest.MANIFEST_PATH, manifest.build_manifest())
    problems = manifest.verify(json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8")))
    if problems:
        raise SystemExit(f"the rebuilt manifest does not verify: {problems[:5]}")
    PROMOTION_PATH.write_text(json.dumps({"promoted": copied, "manifest_sha256": sha256_file(manifest.MANIFEST_PATH),
                                          "manifest_verified": True,
                                          "verification_report": str(REPORT_PATH.relative_to(PROJECT_ROOT))},
                                         indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"promoted {len(copied)} files, manifest rebuilt and verified")


def cleanup():
    if not PROMOTION_PATH.exists():
        raise SystemExit("no promotion record, the scratch corpus is kept")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"deleted {SCRATCH.relative_to(PROJECT_ROOT)}")


def main():
    commands = {"extract": extract, "regenerate": regenerate, "promote": promote, "cleanup": cleanup}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(f"usage: rewording {{{','.join(commands)}}}")
    commands[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
