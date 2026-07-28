"""Paired inference and Pareto analysis for completed experiment results."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "n_samples", "signal_strength", "seed"]


def paired_bootstrap(values: np.ndarray, seed: int = 20260728, n_boot: int = 10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def add_excess_loss(results: pd.DataFrame) -> pd.DataFrame:
    oracle = (
        results.loc[results.model == "oracle", KEYS + ["log_loss"]]
        .rename(columns={"log_loss": "oracle_log_loss"})
        .drop_duplicates(KEYS)
    )
    output = results.merge(oracle, on=KEYS, how="left")
    output["excess_log_loss"] = output["log_loss"] - output["oracle_log_loss"]
    return output


def add_family_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = [results]
    additive = results[results.model.isin(["fsg", "ebm_main"])].copy()
    if not additive.empty:
        idx = additive.groupby(KEYS)["validation_log_loss"].idxmin()
        best = additive.loc[idx].copy()
        best["source_model"] = best["model"]
        best["model"] = "additive_gam_best"
        rows.append(best)
    competitors = results[
        ~results.model.isin(["oracle", "shadetree", "additive_gam_best"])
    ].copy()
    if not competitors.empty:
        idx = competitors.groupby(KEYS)["validation_log_loss"].idxmin()
        best = competitors.loc[idx].copy()
        best["source_model"] = best["model"]
        best["model"] = "best_non_shadetree"
        rows.append(best)
    return pd.concat(rows, ignore_index=True)


def pairwise_table(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    valid = results[(results.status == "ok") & (results.model != "oracle")]
    for condition, group in valid.groupby(["dataset", "n_samples", "signal_strength"]):
        models = sorted(group.model.unique())
        for model_a, model_b in combinations(models, 2):
            a = group[group.model == model_a][
                ["seed", "excess_log_loss", "rmse_logit"]
            ].rename(
                columns={
                    "excess_log_loss": "excess_a",
                    "rmse_logit": "rmse_a",
                }
            )
            b = group[group.model == model_b][
                ["seed", "excess_log_loss", "rmse_logit"]
            ].rename(
                columns={
                    "excess_log_loss": "excess_b",
                    "rmse_logit": "rmse_b",
                }
            )
            paired = a.merge(b, on="seed")
            if paired.empty:
                continue
            # Positive values favor A.
            delta = paired.excess_b - paired.excess_a
            mean_delta, ci_low, ci_high = paired_bootstrap(delta, n_boot=n_boot)
            denominator = max(float(paired.excess_b.mean()), 1e-12)
            relative_reduction = mean_delta / denominator
            diff_noninferiority = paired.excess_a - paired.excess_b
            _, _, ni_upper = paired_bootstrap(diff_noninferiority, n_boot=n_boot)
            ni_margin = max(0.005, 0.05 * max(float(paired.excess_b.mean()), 0.0))
            rmse_delta = paired.rmse_b - paired.rmse_a
            rmse_mean, rmse_low, rmse_high = paired_bootstrap(
                rmse_delta, n_boot=n_boot
            )
            rmse_relative = rmse_mean / max(float(paired.rmse_b.mean()), 1e-12)
            rows.append(
                {
                    "dataset": condition[0],
                    "n_samples": condition[1],
                    "signal_strength": condition[2],
                    "model_a": model_a,
                    "model_b": model_b,
                    "paired_seeds": len(paired),
                    "delta_excess_mean": mean_delta,
                    "delta_excess_ci_low": ci_low,
                    "delta_excess_ci_high": ci_high,
                    "relative_excess_reduction": relative_reduction,
                    "a_predictively_superior": bool(
                        ci_low > 0 and relative_reduction >= 0.10
                    ),
                    "noninferiority_margin": ni_margin,
                    "a_predictively_noninferior": bool(ni_upper < ni_margin),
                    "delta_rmse_mean": rmse_mean,
                    "delta_rmse_ci_low": rmse_low,
                    "delta_rmse_ci_high": rmse_high,
                    "relative_rmse_reduction": rmse_relative,
                    "a_better_function_recovery": bool(
                        rmse_low > 0 and rmse_relative >= 0.10
                    ),
                }
            )
    return pd.DataFrame(rows)


def pareto_table(tuning: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_test_log_loss",
        "candidate_total_active_components",
        "status",
    }
    if not required.issubset(tuning.columns):
        return pd.DataFrame()
    valid = tuning[
        (tuning.status == "ok")
        & np.isfinite(tuning.candidate_test_log_loss)
        & np.isfinite(tuning.candidate_total_active_components)
    ].copy()
    flags = []
    group_keys = ["dataset", "n_samples", "signal_strength", "seed"]
    for _, group in valid.groupby(group_keys):
        loss = group.candidate_test_log_loss.to_numpy()
        complexity = group.candidate_total_active_components.to_numpy()
        for i in range(len(group)):
            dominated = np.any(
                (loss <= loss[i])
                & (complexity <= complexity[i])
                & ((loss < loss[i]) | (complexity < complexity[i]))
            )
            flags.append((group.index[i], not dominated))
    valid["pareto"] = False
    for idx, flag in flags:
        valid.loc[idx, "pareto"] = flag
    return valid


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[(results.status == "ok") & (results.model != "oracle")]
    metrics = [
        "log_loss",
        "excess_log_loss",
        "brier",
        "auroc",
        "accuracy",
        "rmse_logit",
        "rmse_probability",
        "split_nodes",
        "mean_decisions",
        "active_shape_basis",
        "total_active_components",
        "fit_seconds",
    ]
    available = [m for m in metrics if m in valid]
    grouped = valid.groupby(["dataset", "n_samples", "signal_strength", "model"])
    mean = grouped[available].mean().add_suffix("_mean")
    std = grouped[available].std().add_suffix("_std")
    count = grouped.size().rename("completed_seeds")
    return pd.concat([count, mean, std], axis=1).reset_index()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("synthetic_experiment/results")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    results = pd.read_csv(args.results_dir / "results.csv")
    tuning_path = args.results_dir / "tuning.csv"
    results = add_excess_loss(results)
    results = add_family_rows(results)
    summary_table(results).to_csv(args.results_dir / "summary.csv", index=False)
    pairwise_table(results, args.bootstrap_replicates).to_csv(
        args.results_dir / "pairwise_tests.csv", index=False
    )
    if tuning_path.exists():
        tuning = pd.read_csv(tuning_path)
        pareto_table(tuning).to_csv(args.results_dir / "pareto_candidates.csv", index=False)
    print(f"Analysis written to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
