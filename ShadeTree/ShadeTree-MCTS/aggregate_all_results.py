import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def load_fold_artifacts(fold_dir: str) -> Tuple[Optional[pd.DataFrame], Dict, Dict]:
    """Load metrics, sparsity, and args (if present) for a fold directory."""
    metrics_path = os.path.join(fold_dir, "metrics.csv")
    sparsity_path = os.path.join(fold_dir, "sparsity.json")
    done_path = os.path.join(fold_dir, "RUN_COMPLETE.json")

    if not os.path.exists(metrics_path):
        return None, {}, {}

    try:
        metrics_df = pd.read_csv(metrics_path, index_col=0)
    except Exception:
        return None, {}, {}

    sparsity = {}
    if os.path.exists(sparsity_path):
        try:
            with open(sparsity_path, "r") as f:
                sparsity = json.load(f)
        except Exception:
            sparsity = {}

    args_dict: Dict = {}
    if os.path.exists(done_path):
        try:
            with open(done_path, "r") as f:
                payload = json.load(f)
            args_dict = payload.get("args", payload)
            if isinstance(payload, dict) and "param_str" in payload and "param_str" not in args_dict:
                args_dict["param_str"] = payload["param_str"]
        except Exception:
            args_dict = {}

    return metrics_df, sparsity, args_dict


def aggregate_metrics(metrics_dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(metrics_dfs)
    means = combined.groupby(by=combined.index).mean()
    stds = combined.groupby(by=combined.index).std(ddof=0)
    return means, stds


def extract_running_time_sec(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "running_time_sec" in df.columns:
        series = pd.to_numeric(df["running_time_sec"], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    if "running_time_sec" in df.index:
        try:
            row = df.loc["running_time_sec"]
            if isinstance(row, pd.Series):
                series = pd.to_numeric(row, errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[0])
            else:
                return float(row)
        except Exception:
            return None
    return None


def aggregate_sparsity(sparsity_dicts: List[Dict]) -> Tuple[pd.Series, pd.Series]:
    if not sparsity_dicts:
        empty = pd.Series(dtype=float)
        return empty, empty
    sparsity_df = pd.DataFrame(sparsity_dicts).apply(pd.to_numeric, errors="coerce")
    return (
        sparsity_df.mean(numeric_only=True).add_suffix("_mean"),
        sparsity_df.std(ddof=0).add_suffix("_std"),
    )


def extract_outer_seed(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"_outer_(\d+)", value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def resolve_outer_seed(fold_args: Dict, param_dir: str) -> Optional[int]:
    seed = extract_outer_seed(fold_args.get("param_str"))
    if seed is None:
        seed = extract_outer_seed(os.path.basename(param_dir))
    if seed is None:
        outer_split_seed = fold_args.get("outer_split_seed")
        if outer_split_seed is not None:
            try:
                seed = int(outer_split_seed)
            except ValueError:
                seed = None
    if seed is None:
        outer_split_seeds = fold_args.get("outer_split_seeds")
        if isinstance(outer_split_seeds, list) and len(outer_split_seeds) == 1:
            try:
                seed = int(outer_split_seeds[0])
            except ValueError:
                seed = None
    return seed


def aggregate_param_dir(
    param_dir: str,
    dataset: str,
    results_root: str,
    verbose: bool = False,
) -> Optional[List[Dict]]:
    fold_dirs = sorted(
        [
            entry.path
            for entry in os.scandir(param_dir)
            if entry.is_dir() and entry.name.startswith("fold_")
        ]
    )

    seed_groups: Dict[Optional[int], Dict[str, object]] = {}

    for fold_dir in fold_dirs:
        metrics_df, sparsity, fold_args = load_fold_artifacts(fold_dir)
        if metrics_df is None:
            if verbose:
                print(f"[skip] Missing metrics in {fold_dir}")
            continue
        outer_seed = resolve_outer_seed(fold_args, param_dir)
        group = seed_groups.setdefault(
            outer_seed,
            {"metrics_dfs": [], "sparsity_dicts": [], "args_dict": {}},
        )
        group["metrics_dfs"].append(metrics_df)
        group["sparsity_dicts"].append(sparsity)
        if not group["args_dict"] and fold_args:
            group["args_dict"] = fold_args

    if not seed_groups:
        return None

    rows: List[Dict] = []
    for outer_seed, payload in seed_groups.items():
        metrics_dfs = payload["metrics_dfs"]
        sparsity_dicts = payload["sparsity_dicts"]
        args_dict = payload["args_dict"]

        if not metrics_dfs:
            continue

        means, stds = aggregate_metrics(metrics_dfs)
        sparsity_means, sparsity_stds = aggregate_sparsity(sparsity_dicts)

        running_times = []
        for df in metrics_dfs:
            runtime = extract_running_time_sec(df)
            if runtime is not None:
                running_times.append(runtime)
        runtime_mean = float(np.mean(running_times)) if running_times else float("nan")
        runtime_std = float(np.std(running_times, ddof=0)) if running_times else float("nan")

        row: Dict = {"dataset": dataset}
        row.update(args_dict)
        row.pop("results_dir", None)
        row["param_dir"] = os.path.basename(param_dir)
        row["param_path"] = os.path.relpath(param_dir, results_root)
        if outer_seed is not None:
            row["outer_split_seed"] = outer_seed
            row["outer_split_seeds"] = [outer_seed]

        for split in means.columns:
            for metric in means.index:
                row[f"{split}_{metric}_mean"] = means.loc[metric, split]
                row[f"{split}_{metric}_std"] = stds.loc[metric, split]

        row.update(sparsity_means.to_dict())
        row.update(sparsity_stds.to_dict())
        row["running_time_sec_mean"] = runtime_mean
        row["running_time_sec_std"] = runtime_std
        row["num_folds_completed"] = len(metrics_dfs)
        rows.append(row)

    return rows if rows else None


def write_csv(rows: List[Dict], output_path: str, dedup_subset: Optional[List[str]] = None) -> None:
    if not rows:
        print(f"[warn] No rows to write for {output_path}")
        return

    all_cols = set().union(*[set(r.keys()) for r in rows])
    front_cols = [c for c in ["dataset", "param_dir", "param_path", "num_folds_completed"] if c in all_cols]
    other_cols = [c for c in all_cols if c not in front_cols]

    df = pd.DataFrame(rows)
    df = df.reindex(columns=front_cols + sorted(other_cols))

    if dedup_subset:
        existing_subset = [c for c in dedup_subset if c in df.columns]
        if existing_subset:
            df = df.drop_duplicates(subset=existing_subset, keep="last")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[done] Wrote {len(df)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MCTS results across datasets and hyperparameter runs."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="/usr/xtmp/xz424/mcts/results",
        help="Root directory that contains dataset subfolders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the combined CSV. Defaults to <results-root>/aggregated_all_results.csv.",
    )
    parser.add_argument(
        "--no-dataset-summaries",
        action="store_true",
        help="Skip writing per-dataset aggregated CSVs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging while scanning folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    output_path = args.output or os.path.join(results_root, "aggregated_all_results.csv")

    if not os.path.isdir(results_root):
        raise SystemExit(f"results_root does not exist or is not a directory: {results_root}")

    all_rows: List[Dict] = []
    dataset_entries = sorted(
        [d for d in os.scandir(results_root) if d.is_dir()],
        key=lambda d: d.name,
    )

    if not dataset_entries:
        print(f"[warn] No dataset directories found under {results_root}")

    for dataset_entry in dataset_entries:
        dataset = dataset_entry.name
        param_entries = sorted(
            [p for p in os.scandir(dataset_entry.path) if p.is_dir()],
            key=lambda p: p.name,
        )
        dataset_rows: List[Dict] = []

        for param_entry in param_entries:
            rows = aggregate_param_dir(param_entry.path, dataset, results_root, verbose=args.verbose)
            if rows is None:
                continue
            dataset_rows.extend(rows)
            all_rows.extend(rows)

        if not args.no_dataset_summaries and dataset_rows:
            dataset_out = os.path.join(dataset_entry.path, "aggregated_cv_results.csv")
            write_csv(dataset_rows, dataset_out, dedup_subset=["dataset", "param_path", "outer_split_seed"])

    write_csv(all_rows, output_path, dedup_subset=["dataset", "param_path", "outer_split_seed"])


if __name__ == "__main__":
    main()
