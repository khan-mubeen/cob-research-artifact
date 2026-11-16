"""run every test module in tests/, in one process

  PYTHONPATH=. python tests/run_all.py
  PYTHONPATH=. python tests/run_all.py --fast      skip the exhaustive v4 census

each module keeps its own main(), so any one of them still runs on its own and
the whole set still runs under pytest. this is the single command for the
regression check: the v4 golden harness plus the new foundation layers
"""

import argparse
import importlib
import os
import sys
import time
import traceback

MODULES = (
    "tests.test_auction_rounds",
    "tests.test_deadline_anchor",
    "tests.test_v4_golden",
    "tests.test_dataset_context",
    "tests.test_dataset_schemas",
    "tests.test_source_manifest",
    "tests.test_history_lookup",
    "tests.test_fx",
    "tests.test_logistics_sources",
    "tests.test_logistics_calibration",
    "tests.test_material_pricing",
    "tests.test_route_distance",
    "tests.test_v5_synthesis",
    "tests.test_freight_assignment",
    "tests.test_calibration_proposal_v3",
    "tests.test_l09_counterfactual",
    "tests.test_v5_wiring",
    "tests.test_v5_floor_boundary",
    "tests.test_v5_finetune_corpus",
    "tests.test_v5_supervisor_selection",
    "tests.test_v5_execution",
    "tests.test_v5_storage_hierarchy",
    "tests.test_v5_freeze_check",
    "tests.test_v5_evaluator_gaps",
    "tests.test_v5_base_test_arm",
    "tests.test_v5_layout_and_descriptive",
    "tests.test_v5_pinned_test_digest",
    "tests.test_v5_prompt_wording",
    "tests.test_v5_epoch_selection",
    "tests.test_v5_training_subsample",
)


def run(module_name):
    module = importlib.import_module(module_name)
    tests = [v for k, v in sorted(vars(module).items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            failed.append((test.__name__, traceback.format_exc()))
    return passed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip the exhaustive 50,000-order v4 census")
    args = parser.parse_args()
    if args.fast:
        os.environ["GOLDEN_CENSUS_ORDERS"] = "200"

    total_passed, all_failed = 0, []
    started = time.perf_counter()
    for name in MODULES:
        began = time.perf_counter()
        passed, failed = run(name)
        total_passed += passed
        all_failed.extend((name, test, trace) for test, trace in failed)
        mark = "ok  " if not failed else "FAIL"
        print(f"  {mark} {name:34s} {passed:3d} passed"
              f"{'' if not failed else f', {len(failed)} FAILED'}"
              f"   {time.perf_counter() - began:5.1f}s")

    print(f"\n{total_passed} passed, {len(all_failed)} failed "
          f"in {time.perf_counter() - started:.1f}s")
    for name, test, trace in all_failed:
        print(f"\n=== {name}.{test} ===\n{trace}")
    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
