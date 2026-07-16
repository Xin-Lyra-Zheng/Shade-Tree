from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGS_FOR_SHADERTREE_DIR = ROOT / "FIGS_for_ShadeTree"
FIGS_DIR = FIGS_FOR_SHADERTREE_DIR
SHADERTREE_MCTS_DIR = FIGS_FOR_SHADERTREE_DIR
LOG_DIR = FIGS_FOR_SHADERTREE_DIR / "log"


@dataclass
class LeafSpec:
    leaf_id: str
    tree_idx: int
    path: list[tuple[int, float, bool]]
    constant_value: float
    model: object | None = None
    model_center: float = 0.0


def _ensure_dummy_interpret() -> None:
    if "interpret" in sys.modules and "interpret.glassbox" in sys.modules:
        return

    interpret_mod = types.ModuleType("interpret")
    glassbox_mod = types.ModuleType("interpret.glassbox")

    class ExplainableBoostingRegressor:  # pragma: no cover
        pass

    glassbox_mod.ExplainableBoostingRegressor = ExplainableBoostingRegressor
    interpret_mod.glassbox = glassbox_mod
    sys.modules["interpret"] = interpret_mod
    sys.modules["interpret.glassbox"] = glassbox_mod


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _binary_logloss(y: np.ndarray, raw_score: np.ndarray) -> float:
    p = np.clip(_sigmoid(raw_score), 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _binary_acc(y: np.ndarray, raw_score: np.ndarray) -> float:
    pred = (_sigmoid(raw_score) >= 0.5).astype(np.int32)
    return float(np.mean(pred == y.astype(np.int32)))


def _val_metric_value(metric_name: str, y: np.ndarray, raw_score: np.ndarray) -> float:
    if metric_name == "logloss":
        return _binary_logloss(y, raw_score)
    if metric_name == "acc":
        return _binary_acc(y, raw_score)
    raise ValueError(f"Unsupported val metric: {metric_name}")


def _is_improved(metric_name: str, old_value: float, new_value: float, min_delta: float) -> bool:
    if metric_name == "logloss":
        return new_value < (old_value - min_delta)
    if metric_name == "acc":
        return new_value > (old_value + min_delta)
    raise ValueError(f"Unsupported val metric: {metric_name}")


def _leaf_logit_contribution_from_value(value) -> float:
    """Return the binary raw-score contribution represented by a FIGS leaf.

    FIGSClassifier applies softmax to the two accumulated class scores.  For
    binary classification, P(class 1) = sigmoid(score_1 - score_0), so the
    difference below is an exact raw-score contribution, not an approximation.
    """
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    if arr.size >= 2:
        return float(arr[1] - arr[0])
    return float(arr[0])


def _collect_leaves(node, tree_idx: int, path: list[tuple[int, float, bool]], out: list[LeafSpec]) -> None:
    is_leaf = node.left is None and node.right is None
    if is_leaf:
        leaf_idx = len(out)
        out.append(
            LeafSpec(
                leaf_id=f"t{tree_idx}_l{leaf_idx}",
                tree_idx=tree_idx,
                path=list(path),
                constant_value=_leaf_logit_contribution_from_value(node.value),
            )
        )
        return

    feature = int(node.feature)
    threshold = float(node.threshold)

    if node.left is not None:
        _collect_leaves(node.left, tree_idx, path + [(feature, threshold, True)], out)
    if node.right is not None:
        _collect_leaves(node.right, tree_idx, path + [(feature, threshold, False)], out)


def _path_mask(X: np.ndarray, path: list[tuple[int, float, bool]]) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for feat, thresh, go_left in path:
        if go_left:
            mask &= X[:, feat] <= thresh
        else:
            mask &= X[:, feat] > thresh
    return mask


def _predict_leaf_model(model, X: np.ndarray, idx: np.ndarray, center: float) -> np.ndarray:
    contrib = np.zeros(X.shape[0], dtype=np.float64)
    if idx.size == 0:
        return contrib
    pred = np.asarray(model.predict(X[idx]), dtype=np.float64).reshape(-1)
    contrib[idx] = pred - center
    return contrib


def run(args) -> int:
    np.random.seed(args.random_state)
    rng = np.random.default_rng(args.random_state)

    sys.path.insert(0, str(FIGS_FOR_SHADERTREE_DIR))
    sys.path.insert(0, str(FIGS_DIR))
    sys.path.insert(0, str(SHADERTREE_MCTS_DIR))

    _ensure_dummy_interpret()

    prepare_data_mod = _load_module("prepare_data_v1", FIGS_FOR_SHADERTREE_DIR / "prepare_data.py")
    figs_mod = _load_module("figs_v1", FIGS_DIR / "FIGS.py")
    shape_fitter_mod = _load_module("shape_fitter_v1", SHADERTREE_MCTS_DIR / "shape_fitter.py")

    data_args = types.SimpleNamespace(
        dataset=args.dataset,
        random_state=args.random_state,
        binning_strategy=args.binning_strategy,
        num_bins=args.num_bins,
        max_thresholds_for_fsg=args.max_thresholds_for_fsg,
        verbose=args.verbose,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, data_info = prepare_data_mod.prepare_data(data_args)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    figs = figs_mod.FIGSClassifier(
        max_rules=args.max_rules,
        max_trees=args.max_trees,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    figs_start = time.perf_counter()
    figs.fit(X_train, y_train)
    figs_fit_seconds = time.perf_counter() - figs_start
    v1_start = time.perf_counter()

    leaves: list[LeafSpec] = []
    for t_idx, tree_root in enumerate(figs.trees_):
        _collect_leaves(tree_root, t_idx, [], leaves)

    train_masks = [
        _path_mask(X_train, lf.path) for lf in leaves
    ]
    val_masks = [
        _path_mask(X_val, lf.path) for lf in leaves
    ]
    test_masks = [
        _path_mask(X_test, lf.path) for lf in leaves
    ]

    leaf_train_contrib = []
    leaf_val_contrib = []
    leaf_test_contrib = []

    F_train = np.zeros(X_train.shape[0], dtype=np.float64)
    F_val = np.zeros(X_val.shape[0], dtype=np.float64)
    F_test = np.zeros(X_test.shape[0], dtype=np.float64)

    for i, lf in enumerate(leaves):
        c_train = np.zeros_like(F_train)
        c_val = np.zeros_like(F_val)
        c_test = np.zeros_like(F_test)
        c_train[train_masks[i]] = lf.constant_value
        c_val[val_masks[i]] = lf.constant_value
        c_test[test_masks[i]] = lf.constant_value

        leaf_train_contrib.append(c_train)
        leaf_val_contrib.append(c_val)
        leaf_test_contrib.append(c_test)

        F_train += c_train
        F_val += c_val
        F_test += c_test

    fitter = shape_fitter_mod.FastStepFitter(
        X_train,
        min_samples_leaf=args.min_samples_leaf,
        n_internal_knots=args.n_internal_knots,
    )

    # Guard the bridge between FIGS' two-score representation and this file's
    # scalar raw-score representation.  This must remain exact for binary FIGS.
    figs_val_proba = np.asarray(figs.predict_proba(X_val), dtype=np.float64)[:, 1]
    figs_test_proba = np.asarray(figs.predict_proba(X_test), dtype=np.float64)[:, 1]
    reconstructed_val_proba = _sigmoid(F_val)
    reconstructed_test_proba = _sigmoid(F_test)
    score_bridge_max_abs_error = max(
        float(np.max(np.abs(figs_val_proba - reconstructed_val_proba))),
        float(np.max(np.abs(figs_test_proba - reconstructed_test_proba))),
    )
    if score_bridge_max_abs_error > 1e-10:
        raise RuntimeError(
            "FIGS score reconstruction is inconsistent with predict_proba: "
            f"max_abs_error={score_bridge_max_abs_error:.3e}"
        )

    baseline_val_logloss = _binary_logloss(y_val, F_val)
    baseline_val_acc = _binary_acc(y_val, F_val)
    baseline_test_logloss = _binary_logloss(y_test, F_test)
    baseline_test_acc = _binary_acc(y_test, F_test)

    val_metric = _val_metric_value(args.val_metric, y_val, F_val)
    best_val_metric = val_metric

    lines = []
    lines.append(f"dataset={args.dataset}")
    lines.append(f"version={args.version}")
    lines.append(f"seed={args.random_state}")
    lines.append(f"n_train={X_train.shape[0]}")
    lines.append(f"n_val={X_val.shape[0]}")
    lines.append(f"n_test={X_test.shape[0]}")
    lines.append(f"n_features={X_train.shape[1]}")
    lines.append(f"figs_trees={len(figs.trees_)}")
    lines.append(f"figs_complexity={figs.complexity_}")
    lines.append(f"figs_fit_seconds={figs_fit_seconds:.6f}")
    lines.append(f"n_leaves={len(leaves)}")
    lines.append(f"val_metric={args.val_metric}")
    lines.append("shape_fitter=FastStepFitter")
    lines.append(f"shape_fitter_min_samples_leaf={args.min_samples_leaf}")
    lines.append(f"shape_fitter_n_internal_knots={args.n_internal_knots}")
    lines.append("shape_fitter_degree=0")
    lines.append("shape_fitter_feature_search=all_features_when_feature_idx_none")
    lines.append(f"shape_fitter_feature_count={X_train.shape[1]}")
    lines.append("optimization_objective=figs_binary_squared_error")
    lines.append("prediction_link=softmax_score_difference_equals_sigmoid")
    lines.append(f"figs_score_bridge_max_abs_error={score_bridge_max_abs_error:.3e}")
    lines.append(f"figs_only_val_logloss={baseline_val_logloss:.6f}")
    lines.append(f"figs_only_val_acc={baseline_val_acc:.6f}")
    lines.append(f"figs_only_test_logloss={baseline_test_logloss:.6f}")
    lines.append(f"figs_only_test_acc={baseline_test_acc:.6f}")

    for round_idx in range(args.n_rounds):
        order = rng.permutation(len(leaves))
        accepted = 0
        skipped = 0

        for leaf_idx in order:
            idx_train = np.flatnonzero(train_masks[leaf_idx])
            if idx_train.size < args.min_samples_leaf:
                skipped += 1
                continue

            old_train = leaf_train_contrib[leaf_idx]
            old_val = leaf_val_contrib[leaf_idx]
            old_test = leaf_test_contrib[leaf_idx]

            F_excl_train = F_train - old_train

            # FIGS grows regression stumps against one-hot residuals using
            # squared error.  In the binary score-difference space the matching
            # target is y1 - y0 = 2*y - 1.  Keeping this objective here avoids
            # switching to a logistic Newton objective only after FIGS is fit.
            binary_score_target = 2.0 * y_train - 1.0
            local_target = binary_score_target - F_excl_train
            weights = np.ones_like(local_target, dtype=np.float64)

            candidate_model = fitter.fit(
                idx_train.tolist(),
                local_target.astype(np.float64),
                weights,
                feature_idx=None,
            )

            # Do not center away the fitted intercept.  The fitted shape replaces
            # the complete old leaf contribution; subtracting its mean without
            # storing that mean separately would discard the leaf's constant
            # effect (especially when shrinkage == 1).
            center = 0.0

            idx_val = np.flatnonzero(val_masks[leaf_idx])
            idx_test = np.flatnonzero(test_masks[leaf_idx])

            cand_train = _predict_leaf_model(candidate_model, X_train, idx_train, center)
            cand_val = _predict_leaf_model(candidate_model, X_val, idx_val, center)
            cand_test = _predict_leaf_model(candidate_model, X_test, idx_test, center)

            new_train = (1.0 - args.shrinkage) * old_train + args.shrinkage * cand_train
            new_val = (1.0 - args.shrinkage) * old_val + args.shrinkage * cand_val
            new_test = (1.0 - args.shrinkage) * old_test + args.shrinkage * cand_test

            F_val_new = F_val - old_val + new_val
            new_metric = _val_metric_value(args.val_metric, y_val, F_val_new)

            if _is_improved(args.val_metric, val_metric, new_metric, args.min_delta):
                F_train = F_train - old_train + new_train
                F_val = F_val_new
                F_test = F_test - old_test + new_test

                leaf_train_contrib[leaf_idx] = new_train
                leaf_val_contrib[leaf_idx] = new_val
                leaf_test_contrib[leaf_idx] = new_test

                leaves[leaf_idx].model = candidate_model
                leaves[leaf_idx].model_center = center

                val_metric = new_metric
                best_val_metric = min(best_val_metric, val_metric)
                accepted += 1

        lines.append(
            f"round={round_idx + 1} accepted={accepted} skipped={skipped} "
            f"val_metric_value={val_metric:.6f} "
            f"val_logloss={_binary_logloss(y_val, F_val):.6f} val_acc={_binary_acc(y_val, F_val):.6f}"
        )

        if accepted == 0:
            lines.append(f"early_stop_round={round_idx + 1}")
            break

    final_val_logloss = _binary_logloss(y_val, F_val)
    final_val_acc = _binary_acc(y_val, F_val)
    final_test_logloss = _binary_logloss(y_test, F_test)
    final_test_acc = _binary_acc(y_test, F_test)
    v1_refit_seconds = time.perf_counter() - v1_start
    v1_total_fit_seconds = figs_fit_seconds + v1_refit_seconds

    lines.append(f"best_val_metric={best_val_metric:.6f}")
    lines.append(f"final_val_logloss={final_val_logloss:.6f}")
    lines.append(f"final_val_acc={final_val_acc:.6f}")
    lines.append(f"final_test_logloss={final_test_logloss:.6f}")
    lines.append(f"final_test_acc={final_test_acc:.6f}")
    lines.append(f"v1_refit_seconds={v1_refit_seconds:.6f}")
    lines.append(f"v1_total_fit_seconds={v1_total_fit_seconds:.6f}")

    if args.val_metric == "logloss":
        guarantee_ok = final_val_logloss <= baseline_val_logloss + 1e-12
    else:
        guarantee_ok = final_val_acc >= baseline_val_acc - 1e-12
    lines.append(f"val_not_worse_than_figs_only={guarantee_ok}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.dataset}_{args.version}.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("V1_OK")
    print(log_path)
    for line in lines:
        print(line)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FIGS-initialized ShadeTree-style leaf backfitting (V1).")
    p.add_argument("--dataset", type=str, default="blood_transfusion")
    p.add_argument("--version", type=str, default="v1")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument("--shrinkage", type=float, default=1.0)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--min-hessian", type=float, default=1e-4)
    p.add_argument("--val-metric", type=str, default="logloss", choices=["acc", "logloss"])

    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--max-trees", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=2)

    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--n-internal-knots", type=int, default=8)

    p.add_argument("--binning-strategy", type=str, default="quantile")
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--max-thresholds-for-fsg", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    raise SystemExit(run(cli_args))
