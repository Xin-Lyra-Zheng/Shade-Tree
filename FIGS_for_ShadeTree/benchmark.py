from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "log"


def _read_log(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and " " not in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _run(script: str, dataset: str, version: str, args) -> dict[str, str]:
    command = [
        sys.executable, str(HERE / script), "--dataset", dataset,
        "--version", version, "--max-rules", str(args.max_rules),
        "--max-trees", str(args.max_trees), "--max-depth", str(args.max_depth),
        "--min-samples-leaf", str(args.min_samples_leaf),
        "--n-internal-knots", str(args.n_internal_knots),
    ]
    if script == "figs_shadetree_v1.py":
        command += ["--n-rounds", str(args.n_rounds), "--val-metric", "logloss"]
    subprocess.run(command, cwd=HERE.parent, check=True)
    return _read_log(LOG_DIR / f"{dataset}_{version}.log")


def main():
    p = argparse.ArgumentParser(description="Benchmark FIGS, V1 and V2 on identical dataset settings")
    p.add_argument("--datasets", nargs="+", default=["netherlands", "bank_balanced"])
    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--max-trees", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--n-internal-knots", type=int, default=8)
    p.add_argument("--n-rounds", type=int, default=5)
    args = p.parse_args()

    rows = []
    for dataset in args.datasets:
        v1 = _run("figs_shadetree_v1.py", dataset, "benchmark_v1", args)
        v2 = _run("shape_figs_v2.py", dataset, "benchmark_v2", args)
        rows.extend([
            {"dataset": dataset, "model": "FIGS", "val_logloss": v1["figs_only_val_logloss"],
             "val_acc": v1["figs_only_val_acc"], "test_logloss": v1["figs_only_test_logloss"],
             "test_acc": v1["figs_only_test_acc"], "running_time_s": v1["figs_fit_seconds"]},
            {"dataset": dataset, "model": "V1", "val_logloss": v1["final_val_logloss"],
             "val_acc": v1["final_val_acc"], "test_logloss": v1["final_test_logloss"],
             "test_acc": v1["final_test_acc"], "running_time_s": v1["v1_total_fit_seconds"]},
            {"dataset": dataset, "model": "V2", "val_logloss": v2["final_val_logloss"],
             "val_acc": v2["final_val_acc"], "test_logloss": v2["final_test_logloss"],
             "test_acc": v2["final_test_acc"], "running_time_s": v2["v2_fit_seconds"]},
        ])

    fields = ["dataset", "model", "val_logloss", "val_acc", "test_logloss", "test_acc", "running_time_s"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    header = "| Dataset | Model | Val logloss | Val acc | Test logloss | Test acc | Running time (s) |"
    separator = "|---|---|---:|---:|---:|---:|---:|"
    table = [header, separator]
    for r in rows:
        table.append("| {dataset} | {model} | {val_logloss} | {val_acc} | {test_logloss} | {test_acc} | {running_time_s} |".format(**r))
    md_path = LOG_DIR / "benchmark_results.md"
    md_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table)); print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
