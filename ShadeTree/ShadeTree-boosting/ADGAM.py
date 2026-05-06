import cProfile
import io
import pstats
from math import sqrt
import cupy as cp
GPU_AVAILABLE = cp.is_available()
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score

from loss_functions import LogisticLoss
from _prepare_data import prepare_data
import csv
from datetime import datetime
from models import ShadeTree

import os
import random
import numpy as np
import argparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split, StratifiedKFold

CV_FOLDS = 3
SELECTION_SCORING = "auroc"
BASE_RESULTS_DIR = None

# Fix seeds & reproducibility
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def generate_unique_seeds(n, seed):
    if n == 1:
        return [int(seed)]
    random.seed(seed)
    population = range(0, 10000)
    unique_numbers = random.sample(population, n)
    return unique_numbers

# Printing metrics
def _safe_roc_auc(y_true, proba):
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return float('nan')

def _clf_metrics_from_proba(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    au  = _safe_roc_auc(y_true, proba)
    return {"acc": float(acc), "f1": float(f1), "auroc": float(au)}

def _clf_metrics(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    return _clf_metrics_from_proba(y, proba, threshold)

def _print_mean_std(block_name, rows):
    arr = np.array(rows, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        print(f"{block_name}: N/A")
        return
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr, ddof=0))
    print(f"{block_name}: {m:.4f} ± {s:.4f}")

def _mean_std_str(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return "N/A"
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr, ddof=0))
    return f"{m:.4f}±{s:.4f}"

def append_summary_row(dataset, method, runs_count,
                       train_full_metrics_list, test_metrics_list, sparsity_metrics_list,
                       args, csv_path="results/summary_by_dataset_method.csv"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def _grid_tag(args):
        mg = str(args.min_gain_fraction).replace('.', 'p')
        return f"d{args.max_depth}_mg{mg}_msl_{args.min_samples_leaf}"

    row = {
        "dataset": dataset,
        "method": method,
        "runs": runs_count,
        "grid_name": _grid_tag(args),
        "train_full_acc": _mean_std_str([m["acc"] for m in train_full_metrics_list]),
        "train_full_f1":  _mean_std_str([m["f1"]  for m in train_full_metrics_list]),
        "train_full_auroc": _mean_std_str([m["auroc"] for m in train_full_metrics_list]),
        "test_acc": _mean_std_str([m["acc"] for m in test_metrics_list]),
        "test_f1":  _mean_std_str([m["f1"]  for m in test_metrics_list]),
        "test_auroc": _mean_std_str([m["auroc"] for m in test_metrics_list]),
        "sparsity_steps": _mean_std_str([m.get("step_sparsity") for m in sparsity_metrics_list]) if sparsity_metrics_list else "N/A",
        "sparsity_shapes": _mean_std_str([m.get("shape_func_sparsity") for m in sparsity_metrics_list]) if sparsity_metrics_list else "N/A",
        "sparsity_variables": _mean_std_str([m.get("variable_sparsity") for m in sparsity_metrics_list]) if sparsity_metrics_list else "N/A",
        "sparsity_decision": _mean_std_str([m.get("decision_sparsity") for m in sparsity_metrics_list]) if sparsity_metrics_list else "N/A",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    header = list(row.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def append_detailed_row(dataset, method, run_idx, seed, args, best_params,
                        cv_train_metrics_lists, cv_val_metrics_lists,
                        test_metrics, sparsity_metrics=None,
                        csv_path="results/detailed_per_run.csv"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    def _grid_tag(args):
        mg = str(args.min_gain_fraction).replace('.', 'p')
        return f"d{args.max_depth}_mg{mg}"

    row = {
        "dataset": dataset,
        "method": method,
        "run_idx": run_idx,
        "seed": seed,
        "grid_name":_grid_tag(args),
        "best_params_json": json_dumps(best_params),
        "cv_train_acc":  _mean_std_str(cv_train_metrics_lists["acc"]),
        "cv_train_f1":   _mean_std_str(cv_train_metrics_lists["f1"]),
        "cv_train_auroc":_mean_std_str(cv_train_metrics_lists["auroc"]),
        "cv_val_acc":    _mean_std_str(cv_val_metrics_lists["acc"]),
        "cv_val_f1":     _mean_std_str(cv_val_metrics_lists["f1"]),
        "cv_val_auroc":  _mean_std_str(cv_val_metrics_lists["auroc"]),
        "test_acc":   f'{test_metrics["acc"]:.4f}',
        "test_f1":    f'{test_metrics["f1"]:.4f}',
        "test_auroc": f'{test_metrics["auroc"]:.4f}',
        "sparsity_steps":     (sparsity_metrics or {}).get("step_sparsity"),
        "sparsity_shapes":    (sparsity_metrics or {}).get("shape_func_sparsity"),
        "sparsity_variables": (sparsity_metrics or {}).get("variable_sparsity"),
        "sparsity_decision":  (sparsity_metrics or {}).get("decision_sparsity"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    header = list(row.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def append_cv_candidate_row(dataset, method, run_idx, seed, args, params,
                            tr_fold_metrics, te_fold_metrics, vl_fold_metrics, mean_score,
                            csv_path="results/cv_candidates.csv"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    def _grid_tag(args):
        mg = str(args.min_gain_fraction).replace('.', 'p')
        return f"d{args.max_depth}_mg{mg}_msl_{args.min_samples_leaf}"
    row = {
        "dataset": dataset,
        "method": method,
        "run_idx": run_idx,
        "seed": seed,
        "grid_name": _grid_tag(args),
        "params_json": json_dumps(params),
        "cv_train_acc": _mean_std_str(tr_fold_metrics["acc"]),
        "cv_train_f1": _mean_std_str(tr_fold_metrics["f1"]),
        "cv_train_auroc": _mean_std_str(tr_fold_metrics["auroc"]),
        "cv_val_acc": _mean_std_str(vl_fold_metrics["acc"]),
        "cv_val_f1": _mean_std_str(vl_fold_metrics["f1"]),
        "cv_val_auroc": _mean_std_str(vl_fold_metrics["auroc"]),
        "selection_score": f"{mean_score:.6f}",
        "test_acc":   (_mean_std_str(te_fold_metrics["acc"])   if te_fold_metrics else "N/A"),
        "test_f1":    (_mean_std_str(te_fold_metrics["f1"])    if te_fold_metrics else "N/A"),
        "test_auroc": (_mean_std_str(te_fold_metrics["auroc"]) if te_fold_metrics else "N/A"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    header = list(row.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def _plot_loss_curve(model, out_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skip loss curve plot.")
        return
    curve = getattr(model, "loss_curve_", None)
    if not curve or not curve.get("iter"):
        print(f"[WARN] No loss curve to plot for {out_path}")
        return
    iters = np.array(curve["iter"], dtype=int)
    tr = np.array(curve["train"], dtype=float)
    vl = np.array(curve["val"], dtype=float)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(iters, tr, label="train")
    if np.any(np.isfinite(vl)):
        plt.plot(iters, vl, label="val")
    if title:
        plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def json_dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)

def _parse_list_arg(raw, cast_fn):
    if raw is None:
        return None
    vals = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(cast_fn(p))
    return vals if vals else None

class ShadeTreeClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, M=100, eta=0.1, d1=1, lambda_cost=1e-5, min_samples_leaf=50, 
                 early_stopping_rounds=5, tol=1e-4, max_thresholds=20, max_depth=4, k=5,
                 loss='logistic', validation_size=0.15, random_state=42, 
                 gam_mode="always", leaf_spline_alpha=0.1):
        self.M = M
        self.eta = eta
        self.d1 = d1
        self.lambda_cost = lambda_cost
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.max_thresholds = max_thresholds
        self.max_depth = max_depth
        self.k = k
        self.loss = loss
        self.validation_size = validation_size
        self.random_state = random_state
        self.gam_mode = gam_mode
        self.leaf_spline_alpha = leaf_spline_alpha

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.model_ = ShadeTree(
            M=self.M, eta=self.eta, d1=self.d1, lambda_cost=self.lambda_cost,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping_rounds=self.early_stopping_rounds, tol=self.tol,
            max_thresholds=self.max_thresholds, max_depth=self.max_depth,
            k=self.k, loss=self.loss,
            gam_mode=self.gam_mode,
            leaf_spline_alpha=self.leaf_spline_alpha,
            verbose=False
        )
        feature_names = fit_params.get('feature_names')
        self.model_.fit(X, y, None, None, feature_names=feature_names)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict_proba(X)

def _build_default_param_grid(args):
    md_list = _parse_list_arg(getattr(args, "grid_max_depths", None), int) or [args.max_depth]
    mg_list = _parse_list_arg(getattr(args, "grid_min_gain", None), float) or [args.min_gain_fraction]
    silenced = bool(getattr(args, "silence_parent", 1))
    leaf_support_list = _parse_list_arg(getattr(args, "grid_leaf_fsg_max_support_size", None), int) or ([3] if silenced else [2])
    mt_list = _parse_list_arg(getattr(args, "grid_max_thresholds_for_tree", None), int) or [args.max_thresholds_for_tree]
    sa_list = _parse_list_arg(getattr(args, "grid_structure_alpha", None), float) or ([args.structure_alpha] if getattr(args, "structure_alpha", None) is not None else [0.5, 0.75])
    grid = {
        "M": [args.M],
        "max_depth": md_list,
        "min_gain_fraction": mg_list,
        "min_samples_leaf": [args.min_samples_leaf],
        "k": [args.k],
        "d1": [args.d1],
        "max_thresholds_for_tree": mt_list,
        "gam_mode": [args.gam_mode],
        "leaf_fsg_max_support_size": leaf_support_list,
        "leaf_spline_alpha" : [args.leaf_spline_alpha],
        "leaf_fitter": [args.leaf_fitter],
        "structure_alpha": sa_list,
    }
    return grid

def _iter_grid(grid_dict):
    from itertools import product
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    for combo in product(*vals):
        yield {k: v for k, v in zip(keys, combo)}

def _fit_one_fold_and_metrics(params, X_tr, y_tr, X_val, y_val,
                             data_info, args, run_idx=None, fold_id=None,
                             X_test=None, y_test=None):
    import pandas as pd
    X_tr_bin = X_val_bin = None
    feature_names_bin = None
    binner_to_use = None

    feat_names = getattr(data_info, "feature_names", None)
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
    X_tr_df = pd.DataFrame(X_tr,  columns=feat_names)
    X_val_df = pd.DataFrame(X_val, columns=feat_names)

    if hasattr(data_info, "binner") and hasattr(data_info.binner, "config"):
        try:
            from copy import deepcopy
            cfg = deepcopy(data_info.binner.config)
            binner_to_use = data_info.binner.__class__(cfg)
            binner_to_use.fit(X_tr_df)
            print("Binner fitted!")

            X_tr_bin_np, feature_names_bin = binner_to_use.transform_numpy(X_tr, feature_names=feat_names)
            X_val_bin_np, _ = binner_to_use.transform_numpy(X_val, feature_names=feat_names)

            if X_tr_bin_np.shape[1] == 0:
                X_tr_bin = X_val_bin = None
                feature_names_bin = None
            else:
                X_tr_bin = X_tr_bin_np.astype(np.int8, copy=False)
                X_val_bin = X_val_bin_np.astype(np.int8, copy=False)
        except Exception as e:
            print(f"[WARN] binner fitter failed: {e}")
            X_tr_bin = X_val_bin = None
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
        X_tr, y_tr, X_val, y_val,
        feature_names=feat_names,
        X_train_bin=X_tr_bin, X_val_bin=X_val_bin,
        feature_names_bin=feature_names_bin,
        binner=binner_to_use
    )

    if getattr(args, "plot_loss", False):
        tag = f"run_{run_idx:02d}_seed_{args.random_state}" if run_idx is not None else f"seed_{args.random_state}"
        root_dir = BASE_RESULTS_DIR or args.results_root
        out_dir = os.path.join(root_dir, args.loss_out_root, args.dataset, args.loss ,tag, "cv")
        out_path = os.path.join(out_dir, f"fold_{fold_id:02d}.png" if fold_id is not None else "cv_fold.png")
        _plot_loss_curve(model, out_path, title=f"CV Fold {fold_id} (run {run_idx})")

    proba_tr  = model.predict_proba(X_tr)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    m_tr = _clf_metrics_from_proba(y_tr,  proba_tr)
    m_vl = _clf_metrics_from_proba(y_val, proba_val)

    m_te = None
    if X_test is not None and y_test is not None:
        proba_te = model.predict_proba(X_test)[:, 1]
        m_te = _clf_metrics_from_proba(y_test, proba_te)

    score = m_vl["acc"]
    return float(score), m_tr, m_vl, m_te


def _refit_final_model(best_params, X_train_full, y_train_full, data_info, args, plot_ctx=None):
    import pandas as pd
    X_tr_bin = None
    feature_names_bin = None
    binner_to_use = None

    feat_names = getattr(data_info, "feature_names", None)
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(X_train_full.shape[1])]
    X_full_df = pd.DataFrame(X_train_full, columns=feat_names)

    if hasattr(data_info, "binner") and hasattr(data_info.binner, "config"):
        try:
            from copy import deepcopy
            cfg = deepcopy(data_info.binner.config)
            binner_to_use = data_info.binner.__class__(cfg)
            binner_to_use.fit(X_full_df)

            X_full_bin_np, feature_names_bin = binner_to_use.transform_numpy(
                X_train_full, feature_names=feat_names
            )
            if X_full_bin_np.shape[1] == 0:
                X_tr_bin = None
                feature_names_bin = None
            else:
                X_tr_bin = X_full_bin_np.astype(np.int8, copy=False)
        except Exception as e:
            print(f"[WARN] final binner fitter failed: {e}")
            X_tr_bin = None
            binner_to_use = None
            feature_names_bin = None

    model = ShadeTree(
        M=best_params["M"],
        eta=args.eta,
        d1=best_params["d1"],
        max_depth=best_params["max_depth"],
        lambda_cost=best_params["min_gain_fraction"],
        min_samples_leaf=best_params["min_samples_leaf"],
        early_stopping_rounds=args.early_stopping_rounds,
        tol=args.tol,
        max_thresholds=best_params["max_thresholds_for_tree"],
        k=best_params["k"],
        loss=args.loss,
        gam_mode=best_params["gam_mode"],
        leaf_spline_alpha=best_params["leaf_spline_alpha"],
        global_gam_max_support_size=args.global_gam_max_support_size,
        leaf_fsg_max_support_size=best_params["leaf_fsg_max_support_size"],
        verbose=args.verbose,
        leaf_fitter=args.leaf_fitter,
        leaf_restrict=args.leaf_restrict_to_split_feature,
        backward_fit=args.backward_fit,
        backward_max_support=best_params["leaf_fsg_max_support_size"],
        stochastic_coord=args.stochastic_coord,
        stochastic_topk=args.stochastic_topk,
        stochastic_seed=(args.stochastic_seed if args.stochastic_seed is not None else args.random_state),
        structure_alpha=best_params["structure_alpha"],
        step_mode=args.step_mode,
        silence_parent=bool(getattr(args, "silence_parent", 1)),
    )

    model.fit(
        X_train_full, y_train_full, None, None,
        feature_names=feat_names,
        X_train_bin=X_tr_bin, X_val_bin=None,
        feature_names_bin=feature_names_bin,
        binner=binner_to_use
    )

    if getattr(args, "plot_loss", False):
        tag = plot_ctx or f"seed_{args.random_state}"
        root_dir = BASE_RESULTS_DIR or args.results_root
        out_dir = os.path.join(root_dir, args.loss_out_root, args.dataset, tag)
        out_path = os.path.join(out_dir, "train_full.png")
        _plot_loss_curve(model, out_path, title=f"Train-Full ({tag})")

    return model

# Main
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ShadeTree with per-run 4-fold CV + hold-out test, and two CSV outputs.")
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--dataset', type=str, default='bank_balanced',
                        help="Dataset name (e.g., adult, bank, ...).")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random state for reproducibility.")
    parser.add_argument('--runs', type=int, default=5,
                        help="How many runs to execute with distinct randomly generated seeds.")

    parser.add_argument('--M', type=int, default=100, help="Max boosting iterations.")
    parser.add_argument('--eta', type=float, default=2, help="Learning rate (not swept in CV).")
    parser.add_argument('--max_depth', type=int, default=4, help="Max depth.")
    parser.add_argument('--min_gain_fraction', type=float, default=0.01, help="Min relative gain to split.")
    parser.add_argument('--min_samples_leaf', type=int, default=200, help="Min samples in a leaf.")
    parser.add_argument('--k', type=int, default=5, help="Top-K candidate nodes for local refine.")
    parser.add_argument('--d1', type=int, default=1, help="Depth for shallow search in global refine.")
    parser.add_argument('--max_thresholds_for_tree', type=int, default=15)
    parser.add_argument('--gam_mode', type=str, default='never', choices=['always','never','once'])
    parser.add_argument('--binning_strategy', type=str, default='quantile', choices=['quantile','uniform'])
    parser.add_argument('--num_bins', type=int, default=64)
    parser.add_argument('--max_thresholds_for_fsg', type=int, default=64)
    parser.add_argument('--global_gam_max_support_size', type=int, default=100)
    parser.add_argument('--leaf_fsg_max_support_size', type=int, default=2)
    parser.add_argument('--leaf_spline_alpha', type=float, default=0.1)
    parser.add_argument('--leaf_fitter', type=str, default='fsg', choices=['ebm','fsg'], help='Leaf model')
    parser.add_argument('--leaf_restrict_to_split_feature', action='store_true')
    parser.add_argument('--loss', type=str, default='exponential', choices=['logistic','exponential'], help='Loss function')

    parser.add_argument('--early_stopping_rounds', type=int, default=5, help="ES patience.")
    parser.add_argument('--tol', type=float, default=1e-4, help="ES tolerance.")
    parser.add_argument('--plot', action='store_true', help='Plot the final model')

    parser.add_argument('--plot_loss', action='store_true', help='Save per-iteration loss curves (if model exposes loss_curve_)')
    parser.add_argument('--loss_out_root', type=str, default='loss_plots', help='Directory root for loss curves')
    parser.add_argument('--emit_all_cv', action='store_true', help='Emit every CV candidate row to CSV')
    parser.add_argument('--cv_out_csv', type=str, default='results/cv_candidates.csv', help='CV candidates CSV path')
    parser.add_argument('--cv_only', action='store_true', help='Run CV only (no refit/test)')
    
    parser.add_argument('--backward_fit', action='store_true', help='Use backward fitting after training')
    parser.add_argument('--backward_max_support', type=int, default=8, help='Max support size fir fsg when using backward fitting')

    parser.add_argument('--stochastic_coord', action='store_true',
                        help='Enable stochastic coordinate selection (sample among top-k candidates).')
    parser.add_argument('--stochastic_topk', type=int, default=3,
                        help='Top-K candidates to consider when stochastic selection is enabled.')
    parser.add_argument('--stochastic_seed', type=int, default=None,
                        help='RNG seed for stochastic selection (None -> derive from global seed).')
    parser.add_argument('--structure_alpha', type=float, default=None,
                        help='alpha for controling density')
    
    parser.add_argument('--step_mode', type=str, default='newton', choices=['grad','newton'], help='Step mode for leaf fitting.')
    parser.add_argument('--silence_parent', type=int, default=1, help='Whether to silence parent leaf after split (1/0).')
    parser.add_argument('--grid_max_depths', type=str, default=None, help='Comma-separated max_depth values for grid search.')
    parser.add_argument('--grid_min_gain', type=str, default=None, help='Comma-separated min_gain_fraction values for grid search.')
    parser.add_argument('--grid_max_thresholds_for_tree', type=str, default=None, help='Comma-separated max_thresholds_for_tree values for grid search.')
    parser.add_argument('--grid_structure_alpha', type=str, default=None, help='Comma-separated structure_alpha values for grid search.')
    parser.add_argument('--grid_leaf_fsg_max_support_size', type=str, default=None, help='Comma-separated leaf_fsg_max_support_size values for grid search.')
    parser.add_argument('--results_root', type=str, default='results', help='Root folder to store outputs.')
    parser.add_argument('--ablation_tag', type=str, default='', help='Subfolder/tag under results_root for this ablation.')
    parser.add_argument('--sparsity_split', type=str, default='trainval',
                        choices=['train', 'val', 'trainval', 'test'],
                        help='Which split to use when computing decision sparsity.')
    
    args = parser.parse_args()
    if args.verbose:
        print(args)

    base_results_dir = os.path.join(args.results_root, args.ablation_tag) if args.ablation_tag else args.results_root
    global BASE_RESULTS_DIR
    BASE_RESULTS_DIR = base_results_dir

    seeds = generate_unique_seeds(args.runs, args.random_state)
    if args.verbose or args.runs > 1:
        print(f"[Info] Seeds for this session: {seeds}")

    train_full_metrics_across_runs = []
    test_metrics_across_runs = []
    sparsity_metrics_across_runs = []

    param_grid = _build_default_param_grid(args)

    for run_idx, seed in enumerate(seeds, start=1):
        if args.verbose or args.runs > 1:
            print(f"\n=== Run {run_idx}/{args.runs} (seed={seed}) ===")
        set_global_seed(seed)
        args.random_state = seed

        if args.verbose:
            print(f"--- Preparing dataset: {args.dataset} (seed={seed}) ---")
        X_train, X_val, X_test, y_train, y_val, y_test, data_info = prepare_data(args)
        if X_train is None:
            print("[WARN] prepare_data returned None; skip this run.")
            continue

        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        best_score = -np.inf
        best_params = None

        best_tr_fold_metrics = None
        best_vl_fold_metrics = None

        for params in _iter_grid(param_grid):
            fold_scores = []
            tr_fold_metrics = {"acc": [], "f1": [], "auroc": []}
            vl_fold_metrics = {"acc": [], "f1": [], "auroc": []}
            te_fold_metrics = {"acc": [], "f1": [], "auroc": []}

            for fold_id, (tr_idx, vl_idx) in enumerate(cv.split(X_train_full, y_train_full), start=1):
                X_tr, X_vl = X_train_full[tr_idx], X_train_full[vl_idx]
                y_tr, y_vl = y_train_full[tr_idx], y_train_full[vl_idx]

                score, m_tr, m_vl, m_te = _fit_one_fold_and_metrics(
                    params, X_tr, y_tr, X_vl, y_vl, data_info, args,
                    run_idx=run_idx, fold_id=fold_id, X_test=X_test, y_test=y_test
                )

                fold_scores.append(score)
                for k in ["acc","f1","auroc"]:
                    tr_fold_metrics[k].append(m_tr[k])
                    vl_fold_metrics[k].append(m_vl[k])
                if m_te is not None:
                    for k in ["acc","f1","auroc"]:
                        te_fold_metrics[k].append(m_te[k])

            mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")

            if getattr(args, "emit_all_cv", False):
                append_cv_candidate_row(
                    dataset=args.dataset,
                    method=args.leaf_fitter,
                    run_idx=run_idx,
                    seed=seed,
                    args=args,
                    params=params,
                    tr_fold_metrics=tr_fold_metrics,
                    vl_fold_metrics=vl_fold_metrics,
                    te_fold_metrics=te_fold_metrics,
                    mean_score=mean_score,
                    csv_path=os.path.join(base_results_dir, f"cv_candidates_{args.loss}_{args.leaf_fitter}_{args.dataset}_stoc.csv")
                )

            better = False
            if mean_score > best_score:
                better = True
            elif np.isclose(mean_score, best_score, atol=1e-6) and best_params is not None:
                if (params["M"], params["max_depth"], -params["min_gain_fraction"]) < \
                   (best_params["M"], best_params["max_depth"], -best_params["min_gain_fraction"]):
                    better = True

            if better:
                best_score = mean_score
                best_params = params
                best_tr_fold_metrics = tr_fold_metrics
                best_vl_fold_metrics = vl_fold_metrics

        if args.verbose:
            print(f"[CV] Best params (run {run_idx}): {best_params} | mean val AUROC {best_score:.4f}")

        if getattr(args, "cv_only", False):
            print(f"[CV-ONLY] Skipping refit/test for run {run_idx}.")
            continue

        final_model = _refit_final_model(
            best_params, X_train_full, y_train_full, data_info, args,
            plot_ctx=f"run_{run_idx:02d}_seed_{seed}"
        )

        # training performance
        train_full_metrics = _clf_metrics(final_model, X_train_full, y_train_full)
        train_full_metrics_across_runs.append(train_full_metrics)

        # hold-out test performance
        test_metrics = _clf_metrics(final_model, X_test, y_test)
        test_metrics_across_runs.append(test_metrics)

        # sparsity metrics (split controlled by --sparsity_split)
        if args.sparsity_split == 'train':
            X_ref = X_train
        elif args.sparsity_split == 'val':
            X_ref = X_val
        elif args.sparsity_split == 'test':
            X_ref = X_test
        else:  # trainval (default)
            X_ref = X_train_full

        sparsity_metrics = final_model.compute_sparsity_metrics(X_reference=X_ref)
        if sparsity_metrics.get("decision_sparsity") is None:
            sparsity_metrics["decision_sparsity"] = float("nan")
        sparsity_metrics_across_runs.append(sparsity_metrics)

        print(f"\n--- Final Evaluation (run {run_idx}, seed={seed}) ---")
        print(f"[Train+Val] ACC: {train_full_metrics['acc']:.4f} | F1: {train_full_metrics['f1']:.4f} | AUROC: {train_full_metrics['auroc']:.4f}")
        print(f"[Test     ] ACC: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUROC: {test_metrics['auroc']:.4f}")
        print(f"[Sparsity ] steps={sparsity_metrics['step_sparsity']} | shape_funcs={sparsity_metrics['shape_func_sparsity']} | vars={sparsity_metrics['variable_sparsity']} | decision_avg={sparsity_metrics['decision_sparsity']:.4f}")

        if args.plot:
          pass  # (final model plot can be added here)

        append_detailed_row(
            dataset=args.dataset,
            method=args.leaf_fitter,
            run_idx=run_idx,
            seed=seed,
            args=args,
            best_params=best_params,
            cv_train_metrics_lists=best_tr_fold_metrics,
            cv_val_metrics_lists=best_vl_fold_metrics,
            test_metrics=test_metrics,
            sparsity_metrics=sparsity_metrics,
            csv_path=os.environ.get("ShadeTree_DETAILED_CSV", os.path.join(base_results_dir, "detailed_per_run.csv"))
        )

    if len(train_full_metrics_across_runs) >= 1:
        print(f"\n=== Summary over {len(train_full_metrics_across_runs)} runs ===")
        print("[TRAIN+VAL]")
        _print_mean_std("ACC  ", [m["acc"] for m in train_full_metrics_across_runs])
        _print_mean_std("F1   ", [m["f1"]  for m in train_full_metrics_across_runs])
        _print_mean_std("AUROC", [m["auroc"] for m in train_full_metrics_across_runs])

        print("\n[TEST]")
        _print_mean_std("ACC  ", [m["acc"] for m in test_metrics_across_runs])
        _print_mean_std("F1   ", [m["f1"]  for m in test_metrics_across_runs])
        _print_mean_std("AUROC", [m["auroc"] for m in test_metrics_across_runs])

        print("\n[SPARSITY]")
        _print_mean_std("Steps", [float(m["step_sparsity"]) for m in sparsity_metrics_across_runs])
        _print_mean_std("ShapeFuncs", [float(m["shape_func_sparsity"]) for m in sparsity_metrics_across_runs])
        _print_mean_std("Variables", [float(m["variable_sparsity"]) for m in sparsity_metrics_across_runs])
        _print_mean_std("Decision", [float(m["decision_sparsity"]) for m in sparsity_metrics_across_runs])

        append_summary_row(
            dataset=args.dataset,
            method=args.leaf_fitter,
            runs_count=len(train_full_metrics_across_runs),
            args=args,
            train_full_metrics_list=train_full_metrics_across_runs,
            test_metrics_list=test_metrics_across_runs,
            sparsity_metrics_list=sparsity_metrics_across_runs,
            csv_path=os.environ.get("ShadeTree_SUMMARY_CSV", os.path.join(base_results_dir, "summary_by_dataset_method.csv"))
        )

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\n\n" + "="*30 + " CPROFILE ANALYSIS " + "="*30)
    print(s.getvalue())
