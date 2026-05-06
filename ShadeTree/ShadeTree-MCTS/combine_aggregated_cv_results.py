#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys


def combine_results(root_dir: str, output_path: str) -> int:
    pattern = os.path.join(root_dir, "*", "aggregated_cv_results.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files found at {pattern}", file=sys.stderr)
        return 1

    header_fields = None
    rows_written = 0

    with open(output_path, "w", newline="") as out_f:
        writer = None
        for path in paths:
            dataset = os.path.basename(os.path.dirname(path))
            with open(path, newline="") as in_f:
                reader = csv.DictReader(in_f)
                if not reader.fieldnames:
                    continue

                if header_fields is None:
                    header_fields = reader.fieldnames
                    writer = csv.DictWriter(
                        out_f, fieldnames=["dataset"] + header_fields
                    )
                    writer.writeheader()
                elif reader.fieldnames != header_fields:
                    raise ValueError(
                        f"Header mismatch in {path}: {reader.fieldnames} != {header_fields}"
                    )

                for row in reader:
                    writer.writerow({"dataset": dataset, **row})
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine aggregated_cv_results.csv across dataset folders."
    )
    parser.add_argument(
        "--root",
        default="/usr/xtmp/xz424/mcts/results_pw_ablation",
        help="Root results directory containing dataset subfolders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to <root>/combined_aggregated_cv_results.csv).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    output_path = args.output or os.path.join(
        root_dir, "combined_aggregated_cv_results.csv"
    )

    return combine_results(root_dir, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
