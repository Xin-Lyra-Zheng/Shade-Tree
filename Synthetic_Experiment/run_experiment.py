"""Run validation-selected synthetic experiments and write tidy CSV outputs."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from .config import MODELS, PROFILES, grid_for
from .datasets import generate_dataset, generate_evaluation_set
from .evaluation import Timer, evaluate_fitted_model, predictive_metrics
from .models import MissingDependencyError, build_model


def _split(data, seed: int):
    indices = np.arange(len(data.y))
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=seed,
        stratify=data.y,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.5,
        random_state=seed + 100_003,
        stratify=data.y[rest_idx],
    )
    return train_idx, val_idx, test_idx


def _select_model(
    name, candidates, X_train, y_train, X_val, y_val, X_test, y_test, seed
):
    best = None
    records = []
    for candidate_id, params in enumerate(candidates):
        try:
            model = build_model(name, params, random_state=seed)
            with Timer() as timer:
                model.fit(X_train, y_train, X_val, y_val)
            p_val = np.clip(model.predict_proba(X_val)[:, 1], 1e-12, 1 - 1e-12)
            val_loss = float(log_loss(y_val, p_val, labels=[0, 1]))
            p_test = np.clip(model.predict_proba(X_test)[:, 1], 1e-12, 1 - 1e-12)
            test_loss = float(log_loss(y_test, p_test, labels=[0, 1]))
            complexity = model.complexity(X_test)
            records.append(
                {
                    "candidate_id": candidate_id,
                    "params": json.dumps(params, sort_keys=True),
                    "validation_log_loss": val_loss,
                    "candidate_test_log_loss": test_loss,
                    **{f"candidate_{k}": v for k, v in complexity.items()},
                    "fit_seconds": timer.seconds,
                    "status": "ok",
                    "error": "",
                }
            )
            if best is None or val_loss < best[0]:
                best = (val_loss, model, params, timer.seconds)
        except Exception as exc:
            records.append(
                {
                    "candidate_id": candidate_id,
                    "params": json.dumps(params, sort_keys=True),
                    "validation_log_loss": np.nan,
                    "fit_seconds": np.nan,
                    "status": "unavailable" if isinstance(exc, MissingDependencyError) else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if isinstance(exc, MissingDependencyError):
                break
    return best, records


def _oracle_row(data, test_idx, mc):
    metrics = predictive_metrics(data.y[test_idx], data.true_probability[test_idx])
    row = {
        "model": "oracle",
        **metrics.__dict__,
        "rmse_logit": 0.0,
        "rmse_probability": 0.0,
        "split_nodes": 0,
        "max_depth": 0,
        "shape_functions": 0,
        "variables_used": 0,
        "active_shape_basis": 0,
        "mean_decisions": 0,
        "mean_active_shapes": 0,
        "total_active_components": 0,
        "fit_seconds": 0.0,
        "validation_log_loss": np.nan,
        "selected_params": "{}",
        "status": "ok",
        "error": "",
    }
    for region in np.unique(mc.region):
        row[f"rmse_logit_region_{int(region)}"] = 0.0
    return row


def run_condition(dataset, n_samples, signal_strength, seed, models, profile, mc_samples):
    data = generate_dataset(dataset, n_samples, seed, signal_strength)
    train_idx, val_idx, test_idx = _split(data, seed)
    mc = generate_evaluation_set(
        dataset,
        mc_samples,
        seed=seed + 1_000_003,
        signal_strength=signal_strength,
        intercept=data.intercept,
        n_features=data.X.shape[1],
    )
    common = {
        "dataset": dataset,
        "n_samples": n_samples,
        "signal_strength": signal_strength,
        "seed": seed,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "intercept": data.intercept,
    }
    results = [{**common, **_oracle_row(data, test_idx, mc)}]
    tuning = []
    for name in models:
        best, candidate_records = _select_model(
            name,
            grid_for(name, profile),
            data.X[train_idx],
            data.y[train_idx],
            data.X[val_idx],
            data.y[val_idx],
            data.X[test_idx],
            data.y[test_idx],
            seed,
        )
        for record in candidate_records:
            tuning.append({**common, "model": name, **record})
        if best is None:
            last = candidate_records[-1] if candidate_records else {}
            results.append(
                {
                    **common,
                    "model": name,
                    "status": last.get("status", "failed"),
                    "error": last.get("error", "No candidate completed."),
                }
            )
            continue
        val_loss, model, params, fit_seconds = best
        try:
            metrics = evaluate_fitted_model(
                model,
                data.X[test_idx],
                data.y[test_idx],
                mc.X,
                mc.true_logit,
                mc.true_probability,
                mc.region,
            )
            results.append(
                {
                    **common,
                    "model": name,
                    **metrics,
                    "fit_seconds": fit_seconds,
                    "validation_log_loss": val_loss,
                    "selected_params": json.dumps(params, sort_keys=True),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    **common,
                    "model": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results, tuning


def _append_csv(rows, path: Path):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_experiment/results"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--signal-strengths", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    datasets = args.datasets or profile["datasets"]
    models = args.models or list(MODELS)
    sample_sizes = args.sample_sizes or profile["sample_sizes"]
    strengths = args.signal_strengths or profile["signal_strengths"]
    seeds = args.seeds or profile["seeds"]
    mc_samples = args.mc_samples or profile["mc_samples"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.csv"
    tuning_path = args.output_dir / "tuning.csv"
    failure_path = args.output_dir / "runner_failures.log"

    for dataset in datasets:
        for n_samples in sample_sizes:
            for strength in strengths:
                for seed in seeds:
                    print(
                        f"[RUN] dataset={dataset} n={n_samples} "
                        f"signal={strength} seed={seed}"
                    )
                    try:
                        results, tuning = run_condition(
                            dataset,
                            n_samples,
                            strength,
                            seed,
                            models,
                            args.profile,
                            mc_samples,
                        )
                        _append_csv(results, result_path)
                        _append_csv(tuning, tuning_path)
                    except Exception:
                        message = (
                            f"dataset={dataset} n={n_samples} signal={strength} seed={seed}\n"
                            + traceback.format_exc()
                            + "\n"
                        )
                        with failure_path.open("a", encoding="utf-8") as handle:
                            handle.write(message)
                        if args.fail_fast:
                            raise
    print(f"Results: {result_path.resolve()}")
    print(f"Tuning:  {tuning_path.resolve()}")


if __name__ == "__main__":
    main()
