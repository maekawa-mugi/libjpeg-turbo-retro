#!/usr/bin/env python3
"""Decide PS2 EE JPEG scalar/A/B winners from one complete console capture.

Timing method is adapted from openssl-retro/test/ps2/main.c: correctness
first, rotated execution orders, median per variant and output digest
after each timed batch.  This parser accepts ONLY complete 8/8 sessions,
with all variants on the same input matrix, before recommending a winner.

Example: python3 simd/ps2/analyze-bench.py ps2-console.log
"""
import argparse
import collections
import math
import re
import statistics
import sys

VARIANTS = {
    "merged": ("portable_c", "scalar", "pmul4", "pmul8",
               "addpack", "vector", "table"),
    "color": ("portable_c", "scalar", "pmul4", "pmul8", "regpack", "table"),
    "plain_up": ("portable_c", "mmi"),
    "fancy_up": ("portable_c", "mmi"),
    "downsample": ("portable_c", "mmi"),
    "idct": ("ijg_c", "evenoff", "evenon", "batch", "direct"),
    "quantize": ("ijg_c", "mmi", "regpipe"),
}
OPTIONAL_VARIANTS = {"idct": ("fpu_approx",)}

CASE_COUNT = {
    "merged": 32,
    "color": 16,
    "plain_up": 8,
    "fancy_up": 8,
    "downsample": 8,
    "idct": 16,
    "quantize": 12,
}


def parse_lines(lines):
    rows = {}
    failures = []
    started = ended = overall_ok = False
    for line_number, line in enumerate(lines, 1):
        if "BENCH_START,R5900,PS2SDK_TIMER" in line:
            if started:
                failures.append("Duplicate BENCH_START (multiple runs?)")
            started = True
        if "BENCH_END," in line:
            if ended:
                failures.append("Duplicate BENCH_END")
            ended = True
            match = re.search(r"BENCH_END,failures=(\d+)", line)
            if not match or int(match.group(1)):
                failures.append("BENCH_END reports errors")
        if ("TEST: OK! (9/9 groups passed)" in line or
                re.search(r"LIBJPEG_PS2,DONE,PASS,tests=8,passed=8,bench_failures=0", line)):
            overall_ok = True
        if "TEST: FAIL!" in line or re.search(r"\b(?:FAIL|SKIP),", line):
            failures.append(f"line {line_number}: {line.strip()[:160]}")
        match = re.search(r"\bCSV,([^\r\n]+)", line)
        if not match:
            continue
        fields = match.group(1).strip().split(",")
        if len(fields) != 8:
            failures.append(f"line {line_number}: expected 8 CSV fields")
            continue
        cat, variant, workload = fields[:3]
        if (cat not in VARIANTS or
                variant not in VARIANTS[cat] + OPTIONAL_VARIANTS.get(cat, ())):
            failures.append(f"line {line_number}: unexpected variant {cat}/{variant}")
            continue
        try:
            width, alignment, ticks, reps, per_call = map(int, fields[3:])
        except ValueError:
            failures.append(f"line {line_number}: invalid numeric CSV value")
            continue
        if width <= 0 or ticks <= 0 or reps <= 0 or per_call <= 0:
            failures.append(f"line {line_number}: zero or negative timing value")
            continue
        key = (cat, variant, workload, width, alignment)
        if key in rows:
            failures.append(f"line {line_number}: repeated timing {key}")
            continue
        rows[key] = ticks / reps

    if not started:
        failures.append("Missing BENCH_START")
    if not ended:
        failures.append("Missing BENCH_END")
    if not overall_ok:
        failures.append("Missing complete 8/8 PASS (or legacy 9/9 PASS)")
    return rows, failures


def inspect_matrix(rows):
    failures = []
    ratios = collections.defaultdict(list)
    for category, required in VARIANTS.items():
        optional = tuple(
            v for v in OPTIONAL_VARIANTS.get(category, ())
            if any(cat == category and variant == v
                   for cat, variant, _, _, _ in rows)
        )
        variants = required + optional
        data = {
            variant: {
                (workload, width, alignment): cost
                for (cat, v, workload, width, alignment), cost in rows.items()
                if cat == category and v == variant
            }
            for variant in variants
        }
        expected = CASE_COUNT[category]
        for variant in variants:
            if len(data[variant]) != expected:
                failures.append(
                    f"{category}/{variant}: got {len(data[variant])} "
                    f"of {expected} comparable workload rows"
                )
        baseline = data[variants[0]]
        for variant in variants[1:]:
            if data[variant].keys() != baseline.keys():
                failures.append(f"{category}/{variant}: incomplete/mismatched workloads")
            for case in baseline.keys() & data[variant].keys():
                ratios[(category, variant)].append(
                    baseline[case] / data[variant][case]
                )
    return ratios, failures


def verdicts(ratios, min_geomean=1.05, min_worst=0.95):
    summary = {}
    for category, required in VARIANTS.items():
        variants = required + tuple(
            v for v in OPTIONAL_VARIANTS.get(category, ())
            if len(ratios[(category, v)]) == CASE_COUNT[category]
        )
        contenders = []
        for variant in variants[1:]:
            samples = ratios[(category, variant)]
            if len(samples) != CASE_COUNT[category]:
                continue
            geomean = math.exp(statistics.mean(math.log(x) for x in samples))
            worst = min(samples)
            contenders.append((geomean, worst, variant, len(samples)))
        contenders.sort(reverse=True)
        finalists = [v for v in contenders
                     if v[0] >= min_geomean and v[1] >= min_worst]
        summary[category] = (contenders, finalists[0] if finalists else None)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Full single-ELF PS2 stdout transcript")
    parser.add_argument("--min-geomean", type=float, default=1.05,
                        help="Minimum geometric-mean speedup (default 1.05x)")
    parser.add_argument("--min-worst", type=float, default=0.95,
                        help="Minimum worst-case speedup (default 0.95x)")
    args = parser.parse_args(argv)

    with open(args.log, encoding="utf-8", errors="replace") as stream:
        rows, failures = parse_lines(stream)
    ratios, missing = inspect_matrix(rows)
    failures += missing
    summary = verdicts(ratios, args.min_geomean, args.min_worst)

    print("PS2 EE JPEG MMI A/B verdict report")
    print("Timer: PS2SDK GetTimerSystemTime(), balanced order, even median")
    print("Reference is a true portable C implementation for color/merged.")
    expected = sum(CASE_COUNT[k] * len(VARIANTS[k]) for k in VARIANTS)
    expected += sum(
        CASE_COUNT[k] for k, optional in OPTIONAL_VARIANTS.items()
        for v in optional
        if any(cat == k and variant == v for cat, variant, _, _, _ in rows)
    )
    print("Timing rows:", len(rows), "expected:", expected)
    print()
    for category, variants in VARIANTS.items():
        print(f"[{category}] baseline={variants[0]}")
        contenders, finalist = summary[category]
        for gm, worst, name, count in contenders:
            print(f"  {name:11s} geomean={gm:.3f}x "
                  f"worst={worst:.3f}x workloads={count}")
        if failures:
            print("  VERDICT: INVALID (incomplete/failed session)")
        elif finalist:
            print(f"  VERDICT: CANDIDATE {finalist[2]} "
                  f"(geomean={finalist[0]:.3f}x, worst={finalist[1]:.3f}x)")
        else:
            print(f"  VERDICT: RETAIN {variants[0]} (insufficient advantage)")
        print()

    if failures:
        print("INVALID SESSION: do not use timing to select a kernel",
              file=sys.stderr)
        for item in sorted(set(failures)):
            print("  " + item, file=sys.stderr)
        return 2
    print("PASS: complete 8/8 correctness and full A/B timing matrix.")
    print("Candidate status is provisional until repeated on real PS2 hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
