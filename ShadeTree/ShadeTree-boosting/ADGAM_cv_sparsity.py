"""
CV-only runner for ShadeTree that records sparsity per candidate and builds summaries
directly from CV candidates (no refit on full train).
"""
import os
import csv
import numpy as np
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from ShadeTree import (
    ShadeTree,
    prepare_data,
    set_global_seed,
    generate_unique_seeds,
    _clf_metrics_from_proba,
    _mean_std_str,
    _build_default_param_grid,
    _iter_grid,
    json_dumps,
    CV_FOLDS,
)


def _grid_tag(params):
    mg = str(params["min_gain_fraction"]).replace('.', 'p')
    return f"d{params['max_depth']}_mg{mg}_msl_{params['min_samples_leaf']}"


def _agg_metric_list(values):
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _sparsity_mean_std(s_list, key):
    vals = [s.get(key) for s in s_list if s is not None and key in s]
    if not vals:
        return np.nan, np.nan
    return _agg_metric_list(vals)


def append_cv_candidate_row_with_sparsity(dataset, method, run_idx, seed, params,
                                          tr_fold_metrics, vl_fold_metrics, te_fold_metrics,
                                          sparsity_list, mean_score, search_time_mean,
                                          structure_stats, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = {
        "dataset": dataset,
        "method": method,
        "run_idx": run_idx,
        "seed": seed,
        "grid_name": _grid_tag(params),
        "params_json": json_dumps(params),
        "cv_train_acc": _mean_std_str(tr_fold_metrics["acc"]),
        "cv_train_f1": _mean_std_str(tr_fold_metrics["f1"]),
        "cv_train_auroc": _mean_std_str(tr_fold_metrics["auroc"]),
        "cv_val_acc": _mean_std_str(vl_fold_metrics["acc"]),
        "cv_val_f1": _mean_std_str(vl_fold_metrics["f1"]),
        "cv_val_auroc": _mean_std_str(vl_fold_metrics["auroc"]),
        "selection_score": f"{mean_score:.6f}",
        "test_acc": _mean_std_str(te_fold_metrics["acc"]) if te_fold_metrics else "N/A",
        "test_f1": _mean_std_str(te_fold_metrics["f1"]) if te_fold_metrics else "N/A",
        "test_auroc": _mean_std_str(te_fold_metrics["auroc"]) if te_fold_metrics else "N/A",
        "sparsity_steps": _mean_std_str([s.get("step_sparsity") for s in sparsity_list]) if sparsity_list else "N/A",
        "sparsity_shapes": _mean_std_str([s.get("shape_func_sparsity") for s in sparsity_list]) if sparsity_list else "N/A",
        "sparsity_variables": _mean_std_str([s.get("variable_sparsity") for s in sparsity_list]) if sparsity_list else "N/A",
        "sparsity_decision": _mean_std_str([s.get("decision_sparsity") for s in sparsity_list]) if sparsity_list else "N/A",
        "search_time_mean": f"{search_time_mean:.6f}" if search_time_mean is not None else "N/A",
        "local_fraction": structure_stats.get("local_fraction", "N/A"),
        "iterations_run": structure_stats.get("iterations_run", "N/A"),
        "avg_leaf_depth": structure_stats.get("avg_leaf_depth", "N/A"),
        "max_leaf_depth": structure_stats.get("max_leaf_depth", "N/A"),
        "num_leaves": structure_stats.get("num_leaves", "N/A"),
        "avg_leaf_size": structure_stats.get("avg_leaf_size", "N/A"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    header = list(row.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def append_summary_row(dataset, method, runs_count, per_seed_metrics, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    row = {
        "dataset": dataset,
        "method": method,
        "runs": runs_count,
        "grid_name": "from_cv",
        "train_full_acc": _mean_std_str([m["train_acc"] for m in per_seed_metrics]),
        "train_full_f1": _mean_std_str([m["train_f1"] for m in per_seed_metrics]),
        "train_full_auroc": _mean_std_str([m["train_auroc"] for m in per_seed_metrics]),
        "test_acc": _mean_std_str([m["test_acc"] for m in per_seed_metrics]),
        "test_f1": _mean_std_str([m["test_f1"] for m in per_seed_metrics]),
        "test_auroc": _mean_std_str([m["test_auroc"] for m in per_seed_metrics]),
        "sparsity_steps": _mean_std_str([m["sparsity_steps"] for m in per_seed_metrics]),
        "sparsity_shapes": _mean_std_str([m["sparsity_shapes"] for m in per_seed_metrics]),
        "sparsity_variables": _mean_std_str([m["sparsity_variables"] for m in per_seed_metrics]),
        "sparsity_decision": _mean_std_str([m["sparsity_decision"] for m in per_seed_metrics]),
        "search_time_mean": _mean_std_str([m.get("search_time_mean") for m in per_seed_metrics]),
        "local_fraction": _mean_std_str([m.get("local_fraction") for m in per_seed_metrics]),
        "iterations_run": _mean_std_str([m.get("iterations_run") for m in per_seed_metrics]),
        "avg_leaf_depth": _mean_std_str([m.get("avg_leaf_depth") for m in per_seed_metrics]),
        "max_leaf_depth": _mean_std_str([m.get("max_leaf_depth") for m in per_seed_metrics]),
        "num_leaves": _mean_std_str([m.get("num_leaves") for m in per_seed_metrics]),
        "avg_leaf_size": _mean_std_str([m.get("avg_leaf_size") for m in per_seed_metrics]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    header = list(row.keys())
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def run_cv_only(args):
    base_results_dir = os.path.join(args.results_root, args.ablation_tag) if args.ablation_tag else args.results_root
    os.makedirs(base_results_dir, exist_ok=True)

    seeds = generate_unique_seeds(args.runs, args.random_state)
    per_seed_best = []

    param_grid = _build_default_param_grid(args)

    for run_idx, seed in enumerate(seeds, start=1):
        if args.verbose or args.runs > 1:
            print(f"\n=== Run {run_idx}/{args.runs} (seed={seed}) ===")
        set_global_seed(seed)
        args.random_state = seed

        X_train, X_val, X_test, y_train, y_val, y_test, data_info = prepare_data(args)
        if X_train is None:
            print("[WARN] prepare_data returned None; skip this run.")
            continue

        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        best_score = -np.inf
        best_payload = None  # store metrics for summary

        for params in _iter_grid(param_grid):
            fold_scores = []
            tr_fold_metrics = {"acc": [], "f1": [], "auroc": []}
            vl_fold_metrics = {"acc": [], "f1": [], "auroc": []}
            te_fold_metrics = {"acc": [], "f1": [], "auroc": []} if X_test is not None else None
            sparsity_list = []

            for fold_id, (tr_idx, vl_idx) in enumerate(cv.split(X_train_full, y_train_full), start=1):
                X_tr, X_vl = X_train_full[tr_idx], X_train_full[vl_idx]
                y_tr, y_vl = y_train_full[tr_idx], y_train_full[vl_idx]

                binner_to_use = None
                X_tr_bin = X_vl_bin = None
                feature_names_bin = None
                feat_names = getattr(data_info, "feature_names", None)
                if feat_names is None:
                    feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
                try:
                    import pandas as pd
                    if hasattr(data_info, "binner") and hasattr(data_info.binner, "config"):
                        from copy import deepcopy
                        cfg = deepcopy(data_info.binner.config)
                        binner_to_use = data_info.binner.__class__(cfg)
                        binner_to_use.fit(pd.DataFrame(X_tr, columns=feat_names))
                        X_tr_bin_np, feature_names_bin = binner_to_use.transform_numpy(X_tr, feature_names=feat_names)
                        X_vl_bin_np, _ = binner_to_use.transform_numpy(X_vl, feature_names=feat_names)
                        if X_tr_bin_np.shape[1] > 0:
                            X_tr_bin = X_tr_bin_np.astype(np.int8, copy=False)
                            X_vl_bin = X_vl_bin_np.astype(np.int8, copy=False)
                        else:
                            X_tr_bin = X_vl_bin = None
                            feature_names_bin = None
                except Exception as e:
                    if args.verbose:
                        print(f"[WARN] binner fit failed (fold {fold_id}): {e}")
                    X_tr_bin = X_vl_bin = None
                    binner_to_use = None
                    feature_names_bin = None

                model = ShadeTree(
                    M=params["M"],
                    eta=args.eta,
                    d1=params["d1"],
                    max_depth=params["max_depth"],
                    lambda_cost=params["min_gain_fraction"],
                    min_samples_leaf=params["min_samples_leaf"],
                    early_stopping_rounds=args.early_stopping_rounds,
                    tol=args.tol,
                    max_thresholds=params["max_thresholds_for_tree"],
                    k=params["k"],
                    loss=args.loss,
                    gam_mode=params["gam_mode"],
                    leaf_spline_alpha=params["leaf_spline_alpha"],
                    global_gam_max_support_size=args.global_gam_max_support_size,
                    leaf_fsg_max_support_size=params["leaf_fsg_max_support_size"],
                    leaf_fitter=args.leaf_fitter,
                    leaf_restrict=args.leaf_restrict_to_split_feature,
                    backward_fit=args.backward_fit,
                    backward_max_support=params["leaf_fsg_max_support_size"],
                    stochastic_coord=args.stochastic_coord,
                    stochastic_topk=args.stochastic_topk,
                    stochastic_seed=(args.stochastic_seed if args.stochastic_seed is not None else args.random_state),
                    structure_alpha=params["structure_alpha"],
                    step_mode=args.step_mode,
                    silence_parent=bool(getattr(args, "silence_parent", 1)),
                )

                model.fit(
                    X_tr, y_tr, X_vl, y_vl,
                    feature_names=feat_names,
                    X_train_bin=X_tr_bin, X_val_bin=X_vl_bin,
                    feature_names_bin=feature_names_bin,
                    binner=binner_to_use
                )

                proba_tr = model.predict_proba(X_tr)[:, 1]
                proba_vl = model.predict_proba(X_vl)[:, 1]
                m_tr = _clf_metrics_from_proba(y_tr, proba_tr)
                m_vl = _clf_metrics_from_proba(y_vl, proba_vl)

                m_te = None
                if X_test is not None and y_test is not None:
                    proba_te = model.predict_proba(X_test)[:, 1]
                    m_te = _clf_metrics_from_proba(y_test, proba_te)

                # choose reference split for decision sparsity
                if args.sparsity_split == "train":
                    X_ref = X_tr
                elif args.sparsity_split == "val":
                    X_ref = X_vl
                elif args.sparsity_split == "test":
                    X_ref = X_test if X_test is not None else X_tr
                else:  # trainval
                    X_ref = np.concatenate([X_tr, X_vl], axis=0)

                sparsity_metrics = model.compute_sparsity_metrics(X_reference=X_ref)

                fold_scores.append(m_vl["auroc"] if args.selection_score == "auroc" else m_vl["acc"])
                for k in ["acc", "f1", "auroc"]:
                    tr_fold_metrics[k].append(m_tr[k])
                    vl_fold_metrics[k].append(m_vl[k])
                    if te_fold_metrics is not None and m_te is not None:
                        te_fold_metrics[k].append(m_te[k])
                sparsity_list.append(sparsity_metrics)

            mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
            st_mean = getattr(model, "get_search_time_stats", None)
            st_mean = model.get_search_time_stats()[0] if st_mean else np.nan
            struct_metrics = {}
            if hasattr(model, "get_structure_metrics"):
                struct_metrics = model.get_structure_metrics()
            struct_metrics.setdefault("local_fraction", getattr(model, "get_local_fraction", lambda: np.nan)())
            struct_metrics.setdefault("iterations_run", getattr(model, "_iterations_run", np.nan))

            append_cv_candidate_row_with_sparsity(
                dataset=args.dataset,
                method=args.leaf_fitter,
                run_idx=run_idx,
                seed=seed,
                params=params,
                tr_fold_metrics=tr_fold_metrics,
                vl_fold_metrics=vl_fold_metrics,
                te_fold_metrics=te_fold_metrics,
                sparsity_list=sparsity_list,
                mean_score=mean_score,
                search_time_mean=st_mean,
                structure_stats=struct_metrics,
                csv_path=os.path.join(base_results_dir, f"cv_candidates_{args.loss}_{args.leaf_fitter}_{args.dataset}_{args.step_mode}_bf{args.backward_fit}_stoc{args.stochastic_coord}.csv")
            )

            better = mean_score > best_score
            if not better and np.isclose(mean_score, best_score, atol=1e-6) and best_payload is not None:
                if (params["M"], params["max_depth"], -params["min_gain_fraction"]) < \
                   (best_payload["params"]["M"], best_payload["params"]["max_depth"], -best_payload["params"]["min_gain_fraction"]):
                    better = True

            if better:
                best_score = mean_score
                sparsity_means = {k: _sparsity_mean_std(sparsity_list, k)[0] for k in ["step_sparsity","shape_func_sparsity","variable_sparsity","decision_sparsity"]}
                best_payload = {
                    "params": params,
                    "train_acc": float(np.mean(tr_fold_metrics["acc"])),
                    "train_f1": float(np.mean(tr_fold_metrics["f1"])),
                    "train_auroc": float(np.mean(tr_fold_metrics["auroc"])),
                    "test_acc": float(np.mean(te_fold_metrics["acc"])) if te_fold_metrics else np.nan,
                    "test_f1": float(np.mean(te_fold_metrics["f1"])) if te_fold_metrics else np.nan,
                    "test_auroc": float(np.mean(te_fold_metrics["auroc"])) if te_fold_metrics else np.nan,
                    "sparsity_steps": sparsity_means.get("step_sparsity", np.nan),
                    "sparsity_shapes": sparsity_means.get("shape_func_sparsity", np.nan),
                    "sparsity_variables": sparsity_means.get("variable_sparsity", np.nan),
                    "sparsity_decision": sparsity_means.get("decision_sparsity", np.nan),
                }

        if best_payload:
            per_seed_best.append(best_payload)

    if per_seed_best:
        append_summary_row(
            dataset=args.dataset,
            method=args.leaf_fitter,
            runs_count=len(per_seed_best),
            per_seed_metrics=per_seed_best,
            out_csv=os.path.join(base_results_dir, "summary_by_dataset_method_from_cv.csv"),
        )


def main():
    parser = argparse.ArgumentParser(description="CV-only ShadeTree runner with sparsity logging (no refit).")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--results_root', type=str, default='results')
    parser.add_argument('--ablation_tag', type=str, default='')

    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--eta', type=float, default=2.0)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--min_gain_fraction', type=float, default=0.01)
    parser.add_argument('--min_samples_leaf', type=int, default=200)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--d1', type=int, default=1)
    parser.add_argument('--max_thresholds_for_tree', type=int, default=15)
    parser.add_argument('--gam_mode', type=str, default='never', choices=['always','never','once'])
    parser.add_argument('--leaf_spline_alpha', type=float, default=0.1)
    parser.add_argument('--leaf_fitter', type=str, default='fsg', choices=['ebm','fsg'])
    parser.add_argument('--leaf_restrict_to_split_feature', action='store_true')
    parser.add_argument('--loss', type=str, default='logistic', choices=['logistic','exponential'])
    parser.add_argument('--early_stopping_rounds', type=int, default=5)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--backward_fit', action='store_true')
    parser.add_argument('--backward_max_support', type=int, default=8)
    parser.add_argument('--stochastic_coord', action='store_true')
    parser.add_argument('--stochastic_topk', type=int, default=3)
    parser.add_argument('--stochastic_seed', type=int, default=None)
    parser.add_argument('--structure_alpha', type=float, default=0.5)
    parser.add_argument('--step_mode', type=str, default='newton', choices=['grad','newton'])
    parser.add_argument('--silence_parent', type=int, default=1)
    parser.add_argument('--search_full_spline', action='store_true',
                        help='Use full-feature spline lookahead (much slower); default single-feature proxy.')
    parser.add_argument('--global_gam_max_support_size', type=int, default=100)
    parser.add_argument('--leaf_fsg_max_support_size', type=int, default=2)
    parser.add_argument('--binning_strategy', type=str, default='quantile')
    parser.add_argument('--num_bins', type=int, default=64)
    parser.add_argument('--max_thresholds_for_fsg', type=int, default=64)
    parser.add_argument('--sparsity_split', type=str, default='test',
                        choices=['train', 'val', 'trainval', 'test'],
                        help='Which split to use when computing decision sparsity.')

    # grid sweeps
    parser.add_argument('--grid_max_depths', type=str, default=None)
    parser.add_argument('--grid_min_gain', type=str, default=None)
    parser.add_argument('--grid_max_thresholds_for_tree', type=str, default=None)
    parser.add_argument('--grid_structure_alpha', type=str, default=None)
    parser.add_argument('--grid_leaf_fsg_max_support_size', type=str, default=None)

    parser.add_argument('--selection_score', type=str, default='auroc', choices=['auroc','acc'])

    args = parser.parse_args()
    run_cv_only(args)


if __name__ == "__main__":
    main()
