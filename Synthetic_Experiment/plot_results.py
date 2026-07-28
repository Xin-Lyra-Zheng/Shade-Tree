"""Generate the principal proposal figures from tidy result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .analyze_results import add_excess_loss


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sample_efficiency(results, output):
    valid = results[(results.status == "ok") & (results.model != "oracle")]
    for dataset, group in valid.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        subset = group[group.signal_strength == 1.0]
        for model, model_group in subset.groupby("model"):
            stats = model_group.groupby("n_samples").excess_log_loss.agg(["mean", "sem"])
            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=1.96 * stats["sem"],
                marker="o",
                label=model,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Excess test log loss")
        ax.set_title(dataset)
        ax.legend(fontsize=8)
        _save(fig, output / f"sample_efficiency_{dataset}.png")


def complexity_curves(tuning, output):
    required = {"candidate_total_active_components", "candidate_test_log_loss"}
    if not required.issubset(tuning):
        return
    valid = tuning[tuning.status == "ok"]
    for dataset, group in valid.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        subset = group[(group.signal_strength == 1.0)]
        for model, model_group in subset.groupby("model"):
            stats = (
                model_group.groupby("candidate_total_active_components")
                .candidate_test_log_loss.mean()
                .sort_index()
            )
            ax.plot(stats.index, stats.values, marker="o", label=model)
        ax.set_xlabel("Total active components")
        ax.set_ylabel("Test log loss")
        ax.set_title(f"Performance–complexity: {dataset}")
        ax.legend(fontsize=8)
        _save(fig, output / f"complexity_{dataset}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("synthetic_experiment/results")
    )
    args = parser.parse_args()
    output = args.results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    results = add_excess_loss(pd.read_csv(args.results_dir / "results.csv"))
    sample_efficiency(results, output)
    tuning_path = args.results_dir / "tuning.csv"
    if tuning_path.exists():
        complexity_curves(pd.read_csv(tuning_path), output)
    print(f"Figures written to {output.resolve()}")


if __name__ == "__main__":
    main()
