from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent.parent
V2_DIR = Path(__file__).resolve().parent
FIGS_DIR = V2_DIR
SHADERTREE_DIR = V2_DIR
LOG_DIR = Path(__file__).resolve().parent / "log"


@dataclass
class Leaf:
    leaf_id: int
    tree_idx: int
    depth: int
    path: list[tuple[int, float, bool]]
    train_contrib: np.ndarray
    val_contrib: np.ndarray
    test_contrib: np.ndarray
    model: object | None = None


@dataclass
class SplitCandidate:
    leaf: Leaf | None
    tree_idx: int
    depth: int
    feature: int
    threshold: float
    gain: float
    train_mask: np.ndarray


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_dummy_interpret() -> None:
    if "interpret.glassbox" in sys.modules:
        return
    interpret = types.ModuleType("interpret")
    glassbox = types.ModuleType("interpret.glassbox")
    glassbox.ExplainableBoostingRegressor = type("ExplainableBoostingRegressor", (), {})
    interpret.glassbox = glassbox
    sys.modules["interpret"] = interpret
    sys.modules["interpret.glassbox"] = glassbox


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _logloss(y, score):
    p = np.clip(_sigmoid(score), 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(y, score):
    return float(np.mean((_sigmoid(score) >= 0.5) == y.astype(bool)))


def _mask(X, path):
    out = np.ones(X.shape[0], dtype=bool)
    for feature, threshold, left in path:
        out &= X[:, feature] <= threshold if left else X[:, feature] > threshold
    return out


def _candidate(X, target, total, leaf, tree_idx, depth, mask, min_samples_leaf):
    idx = np.flatnonzero(mask)
    if idx.size < 2 * min_samples_leaf:
        return None
    old = np.zeros(idx.size) if leaf is None else leaf.train_contrib[idx]
    residual_without_leaf = target[idx] - (total[idx] - old)
    stump = DecisionTreeRegressor(
        max_depth=1, min_samples_leaf=min_samples_leaf, random_state=0
    ).fit(X[idx], residual_without_leaf)
    if stump.tree_.node_count < 3:
        return None
    feature = int(stump.tree_.feature[0])
    threshold = float(stump.tree_.threshold[0])
    pred = stump.predict(X[idx])
    old_sse = float(np.sum((residual_without_leaf - old) ** 2))
    new_sse = float(np.sum((residual_without_leaf - pred) ** 2))
    return SplitCandidate(leaf, tree_idx, depth, feature, threshold, old_sse - new_sse, mask)


def _fit_contribution(fitter, X, indices, local_target):
    out = np.zeros(X.shape[0], dtype=np.float64)
    if indices.size == 0:
        return None, out
    model = fitter.fit(indices.tolist(), local_target, np.ones_like(local_target), feature_idx=None)
    out[indices] = np.asarray(model.predict(X[indices]), dtype=np.float64).reshape(-1)
    return model, out


def run(args):
    np.random.seed(args.random_state)
    sys.path[:0] = [str(ROOT), str(V2_DIR), str(FIGS_DIR), str(SHADERTREE_DIR)]
    _ensure_dummy_interpret()
    prepare = _load("prepare_data_v2", V2_DIR / "prepare_data.py")
    figs_mod = _load("figs_baseline_v2", FIGS_DIR / "FIGS.py")
    shapes = _load("shape_fitter_v2", SHADERTREE_DIR / "shape_fitter.py")
    data_args = types.SimpleNamespace(
        dataset=args.dataset, random_state=args.random_state,
        binning_strategy=args.binning_strategy, num_bins=args.num_bins,
        max_thresholds_for_fsg=args.max_thresholds_for_fsg, verbose=args.verbose,
    )
    Xtr, Xv, Xte, ytr, yv, yte, _ = prepare.prepare_data(data_args)
    Xtr, Xv, Xte = (np.asarray(x, dtype=np.float64) for x in (Xtr, Xv, Xte))
    ytr, yv, yte = (np.asarray(y, dtype=np.float64) for y in (ytr, yv, yte))
    target = 2.0 * ytr - 1.0

    baseline = figs_mod.FIGSClassifier(
        max_rules=args.max_rules, max_trees=args.max_trees,
        max_depth=args.max_depth, random_state=args.random_state,
    )
    figs_start = time.perf_counter()
    baseline.fit(Xtr, ytr)
    figs_fit_seconds = time.perf_counter() - figs_start
    base_v = np.log(np.clip(baseline.predict_proba(Xv)[:, 1], 1e-12, 1) /
                    np.clip(baseline.predict_proba(Xv)[:, 0], 1e-12, 1))
    base_te = np.log(np.clip(baseline.predict_proba(Xte)[:, 1], 1e-12, 1) /
                     np.clip(baseline.predict_proba(Xte)[:, 0], 1e-12, 1))

    v2_start = time.perf_counter()
    fitter = shapes.FastStepFitter(
        Xtr, min_samples_leaf=args.min_samples_leaf,
        n_internal_knots=args.n_internal_knots,
    )
    Ftr = np.zeros(Xtr.shape[0]); Fv = np.zeros(Xv.shape[0]); Fte = np.zeros(Xte.shape[0])
    leaves: list[Leaf] = []
    next_leaf_id = 0
    tree_count = 0
    history = []
    best_checkpoint = (np.inf, Ftr.copy(), Fv.copy(), Fte.copy(), 0)

    for rule_idx in range(args.max_rules):
        candidates = []
        for leaf in leaves:
            if leaf.depth >= args.max_depth:
                continue
            c = _candidate(Xtr, target, Ftr, leaf, leaf.tree_idx, leaf.depth,
                           _mask(Xtr, leaf.path), args.min_samples_leaf)
            if c is not None:
                candidates.append(c)
        if tree_count < args.max_trees:
            all_mask = np.ones(Xtr.shape[0], dtype=bool)
            c = _candidate(Xtr, target, Ftr, None, tree_count, 0, all_mask,
                           args.min_samples_leaf)
            if c is not None:
                candidates.append(c)
        if not candidates:
            break
        chosen = max(candidates, key=lambda c: c.gain)
        if chosen.gain <= args.min_gain:
            break

        old_tr = np.zeros_like(Ftr) if chosen.leaf is None else chosen.leaf.train_contrib
        old_v = np.zeros_like(Fv) if chosen.leaf is None else chosen.leaf.val_contrib
        old_te = np.zeros_like(Fte) if chosen.leaf is None else chosen.leaf.test_contrib
        base_path = [] if chosen.leaf is None else chosen.leaf.path
        F_excl_tr, F_excl_v, F_excl_te = Ftr - old_tr, Fv - old_v, Fte - old_te
        if chosen.leaf is not None:
            leaves.remove(chosen.leaf)
        else:
            tree_count += 1

        new_leaves = []
        for go_left in (True, False):
            path = base_path + [(chosen.feature, chosen.threshold, go_left)]
            itr, iv, ite = (np.flatnonzero(_mask(X, path)) for X in (Xtr, Xv, Xte))
            local_target = target - F_excl_tr
            model, ctr = _fit_contribution(fitter, Xtr, itr, local_target)
            cv = np.zeros_like(Fv); cte = np.zeros_like(Fte)
            if iv.size:
                cv[iv] = np.asarray(model.predict(Xv[iv])).reshape(-1)
            if ite.size:
                cte[ite] = np.asarray(model.predict(Xte[ite])).reshape(-1)
            new_leaves.append(Leaf(next_leaf_id, chosen.tree_idx, chosen.depth + 1,
                                   path, ctr, cv, cte, model))
            next_leaf_id += 1
        leaves.extend(new_leaves)
        Ftr = F_excl_tr + sum((l.train_contrib for l in new_leaves), np.zeros_like(Ftr))
        Fv = F_excl_v + sum((l.val_contrib for l in new_leaves), np.zeros_like(Fv))
        Fte = F_excl_te + sum((l.test_contrib for l in new_leaves), np.zeros_like(Fte))
        val_loss = _logloss(yv, Fv)
        history.append((rule_idx + 1, chosen.gain, chosen.tree_idx, chosen.feature, val_loss))
        if val_loss < best_checkpoint[0] - args.min_delta:
            best_checkpoint = (val_loss, Ftr.copy(), Fv.copy(), Fte.copy(), rule_idx + 1)

    if best_checkpoint[4] > 0:
        _, Ftr, Fv, Fte, best_rules = best_checkpoint
    else:
        best_rules = len(history)
    v2_fit_seconds = time.perf_counter() - v2_start
    lines = [
        f"dataset={args.dataset}", "version=v2_in_loop", f"seed={args.random_state}",
        "optimization_objective=figs_binary_squared_error",
        "integration=fit_child_shapes_immediately_after_each_split",
        "candidate_ranking=constant_stump_train_sse_gain",
        f"figs_only_val_logloss={_logloss(yv, base_v):.6f}",
        f"figs_only_test_logloss={_logloss(yte, base_te):.6f}",
        f"figs_fit_seconds={figs_fit_seconds:.6f}",
    ]
    for rule, gain, tree_idx, feature, loss in history:
        lines.append(f"rule={rule} gain={gain:.6f} tree={tree_idx} feature={feature} val_logloss={loss:.6f}")
    lines += [
        f"best_rules={best_rules}", f"final_val_logloss={_logloss(yv, Fv):.6f}",
        f"final_val_acc={_accuracy(yv, Fv):.6f}",
        f"final_test_logloss={_logloss(yte, Fte):.6f}",
        f"final_test_acc={_accuracy(yte, Fte):.6f}",
        f"v2_fit_seconds={v2_fit_seconds:.6f}",
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{args.dataset}_{args.version}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("V2_OK"); print(path); print("\n".join(lines))
    return 0


def parser():
    p = argparse.ArgumentParser(description="In-loop Shape-FIGS V2")
    p.add_argument("--dataset", default="netherlands")
    p.add_argument("--version", default="v2")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--max-trees", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--n-internal-knots", type=int, default=8)
    p.add_argument("--min-gain", type=float, default=1e-8)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--binning-strategy", default="quantile")
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--max-thresholds-for-fsg", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
