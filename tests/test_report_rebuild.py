"""Reporting I/O regressions; no GPU, corpus or original profiling files needed.

Run with: python -m unittest tests.test_report_rebuild
"""
import contextlib
import csv
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.experiments.v5.latency_measurement import summarize as report


class PublishedReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = self.root / "records"
        self.records.mkdir()
        self.samples = {}
        for arm, n_out, total in (("base", 3, 50.0), ("selected", 5, 90.0)):
            rows = []
            for i in range(3):
                rows.append({
                    "row_index": i, "row_id": f"order-{i}", "input_tokens": 100,
                    "output_tokens": n_out, "total_ms": total * (3 if i == 0 else 1),
                    "ttft_ms": 10.0, "tpot_ms": (total - 10.0) / (n_out - 1),
                    "warmup": int(i == 0),
                })
            self.samples[arm] = rows
            with gzip.open(self.records / f"sealed_qwen_{arm}.csv.gz", "wt", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(report, "MODES", ("sealed",)).start()
        mock.patch.object(report, "MODELS", ("qwen",)).start()
        mock.patch.object(report, "meta", side_effect=lambda mode, model, arm: {
            "epoch": 2 if arm == "selected" else 0, "summary": {"rows": 3},
        }).start()
        # publish() changes these module-level output settings; restore them after each test.
        old_out, old_prefix = report.OUT, report.PREFIX
        self.addCleanup(setattr, report, "OUT", old_out)
        self.addCleanup(setattr, report, "PREFIX", old_prefix)

    def originals(self, mode, model, arm, wanted):
        return {
            r["row_index"]: (bool(r["warmup"]), r["total_ms"], r["output_tokens"],
                             r["row_id"], r["input_tokens"])
            for r in self.samples[arm] if r["row_index"] in wanted
        }

    def test_public_rebuild_matches_comparison_enabled_tables(self):
        full, public = self.root / "full", self.root / "public"
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(report, "official_rows", side_effect=self.originals):
                report.publish(self.records, full)
            with mock.patch.object(report, "official_rows", side_effect=AssertionError("original records read")):
                report.publish(self.records, public, compare_original=False)
        comparable = [p for p in full.iterdir() if "vs_original_profile" not in p.name]
        self.assertTrue(comparable)
        self.assertEqual({p.name for p in comparable}, {p.name for p in public.iterdir()})
        for path in comparable:
            self.assertEqual(path.read_bytes(), (public / path.name).read_bytes(), path.name)
        with (public / "latency_breakdown_cells.csv").open(newline="") as f:
            cell = next(csv.DictReader(f))
        self.assertEqual(float(cell["total_base"]), 50.0)
        self.assertEqual(float(cell["total_ft"]), 90.0)
        self.assertTrue((full / "latency_breakdown_vs_original_profile.csv").is_file())

    def test_default_does_not_silently_skip_missing_original_records(self):
        with mock.patch.object(report, "official_rows", side_effect=FileNotFoundError("original records")):
            with self.assertRaises(FileNotFoundError):
                report.publish(self.records, self.root / "default")

    def test_skip_refuses_a_stale_original_comparison(self):
        out = self.root / "existing"
        out.mkdir()
        old = out / "latency_breakdown_vs_original_profile.csv"
        old.write_text("previous comparison\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "separate output folder"):
            report.publish(self.records, out, compare_original=False)
        self.assertEqual(old.read_text(encoding="utf-8"), "previous comparison\n")

    def test_skip_flag_requires_published_records(self):
        with mock.patch("sys.argv", ["summarize.py", "--skip-original-profile"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    report.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
