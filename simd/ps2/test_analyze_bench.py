#!/usr/bin/env python3
"""Host-only regression for the full PS2 scalar/A/B verdict parser."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ps2_analyze", BASE / "analyze-bench.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def complete_log():
    lines = ["BENCH_START,R5900,PS2SDK_TIMER,warm24,rotating_order,median"]
    for category, variants in REPORT.VARIANTS.items():
        for index in range(REPORT.CASE_COUNT[category]):
            for vi, variant in enumerate(variants):
                # First SIMD candidate is consistently faster in both
                # median and worst case.  Others are slower or neutral.
                ticks = 150000 if vi == 0 else 100000 if vi == 1 else 154000
                lines.append(
                    f"CSV,{category},{variant},case{index},"
                    f"{64 + index},{index % 2},{ticks},100,"
                    f"{ticks // 100}"
                )
    lines.append("BENCH_END,failures=0")
    lines.append("LIBJPEG_PS2,DONE,PASS,tests=8,passed=8,bench_failures=0")
    return lines


class TestVerdicts(unittest.TestCase):
    def test_complete_case(self):
        rows, errors = REPORT.parse_lines(complete_log())
        self.assertFalse(errors)
        ratios, missing = REPORT.inspect_matrix(rows)
        self.assertFalse(missing)
        self.assertEqual(len(rows), 516)
        results = REPORT.verdicts(ratios)
        for cat, (_, candidate) in results.items():
            self.assertIsNotNone(candidate, cat)
            self.assertEqual(candidate[2], REPORT.VARIANTS[cat][1], cat)
            self.assertGreater(candidate[0], 1.05)

    def test_optional_fpu_accepted_only_when_complete(self):
        lines = complete_log()
        optional = [
            f"CSV,idct,fpu_approx,case{i},{64 + i},{i % 2},98000,100,980"
            for i in range(REPORT.CASE_COUNT["idct"])
        ]
        lines[-2:-2] = optional
        rows, errors = REPORT.parse_lines(lines)
        self.assertFalse(errors)
        ratios, missing = REPORT.inspect_matrix(rows)
        self.assertFalse(missing)
        self.assertEqual(len(rows), 532)
        self.assertEqual(len(ratios["idct", "fpu_approx"]), 16)
        self.assertNotEqual(REPORT.verdicts(ratios)["idct"][1][2],
                            "fpu_approx",
                            "approximate FPU must never win exact IDCT")
        partial = dict(rows)
        del partial[("idct", "fpu_approx", "case0", 64, 0)]
        _, missing = REPORT.inspect_matrix(partial)
        self.assertTrue(any("idct/fpu_approx" in x for x in missing))

    def test_optional_vu_and_dma_experiments(self):
        lines = complete_log()
        extras = [
            "CSV,vu_idct,scalar_matrix,dense,8,0,200000,100,2000",
            "CSV,vu_idct,vu0_macro,dense,8,0,240000,100,2400",
            "CSV,vif0_dma,cpu_store,upload256,256,0,100000,100,1000",
            "CSV,vif0_dma,vif0_dma,upload256,256,0,400000,100,4000",
        ]
        lines[-2:-2] = extras
        rows, errors = REPORT.parse_lines(lines)
        self.assertFalse(errors)
        ratios, missing = REPORT.inspect_matrix(rows)
        self.assertFalse(missing)
        self.assertEqual(len(rows), 520)
        self.assertAlmostEqual(ratios["vu_idct", "vu0_macro"][0], 5 / 6)
        self.assertAlmostEqual(ratios["vif0_dma", "vif0_dma"][0], 1 / 4)
        incomplete = [line for line in lines
                      if not line.startswith("CSV,vif0_dma,cpu_store,")]
        partial, _ = REPORT.parse_lines(incomplete)
        _, missing = REPORT.inspect_matrix(partial)
        self.assertTrue(any("vif0_dma/cpu_store" in error for error in missing))

    def test_missing_single_variant_invalid(self):
        lines = [line for line in complete_log()
                 if not line.startswith("CSV,merged,vector,case0,")]
        rows, _ = REPORT.parse_lines(lines)
        _, errors = REPORT.inspect_matrix(rows)
        self.assertTrue(any("merged/vector" in x for x in errors))

    def test_invalid_digest_prevents_decision(self):
        lines = complete_log()
        lines.insert(-2, "FAIL,ab_sample,color,RGBX,pmul8,sample=1")
        rows, errors = REPORT.parse_lines(lines)
        self.assertTrue(errors)
        self.assertIn("FAIL,ab_sample", " ".join(errors))

    def test_duplicate_and_wrong_timer(self):
        lines = complete_log()
        lines[0] = "BENCH_START,R5900,CP0_COUNT,wrong"
        lines.append(lines[1])
        _, errors = REPORT.parse_lines(lines)
        self.assertTrue(any("BENCH_START" in x for x in errors))
        self.assertTrue(any("repeated timing" in x for x in errors))

    def test_cli_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text("\n".join(complete_log()) + "\n", encoding="utf8")
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                code = REPORT.main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("PASS: complete", capture.getvalue())
            bad = complete_log()
            bad[-1] = "LIBJPEG_PS2,DONE,FAIL,tests=8,passed=7,bench_failures=1"
            path.write_text("\n".join(bad) + "\n", encoding="utf8")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(REPORT.main([str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
