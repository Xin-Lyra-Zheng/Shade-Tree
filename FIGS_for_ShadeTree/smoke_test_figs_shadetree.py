from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGS_FOR_SHADERTREE_DIR = ROOT / "FIGS_for_ShadeTree"
FIGS_DIR = FIGS_FOR_SHADERTREE_DIR
SHADERTREE_MCTS_DIR = FIGS_FOR_SHADERTREE_DIR
LOG_VERSION = "v1"


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


def main() -> int:
    lines = []
    dataset_name = "blood_transfusion"
    log_path = FIGS_FOR_SHADERTREE_DIR / f"{dataset_name}_{LOG_VERSION}.log"

    lines.append(f"ROOT={ROOT}")
    lines.append(f"FIGS_DIR_EXISTS={FIGS_DIR.exists()}")
    lines.append(f"SHADERTREE_MCTS_DIR_EXISTS={SHADERTREE_MCTS_DIR.exists()}")

    # Make local project modules importable without touching repo files.
    sys.path.insert(0, str(FIGS_DIR))
    sys.path.insert(0, str(SHADERTREE_MCTS_DIR))
    sys.path.insert(0, str(FIGS_FOR_SHADERTREE_DIR))

    _ensure_dummy_interpret()

    import cupy as cp

    lines.append(f"CUPY_AVAILABLE={cp.is_available()}")
    lines.append(f"GPU_COUNT={cp.cuda.runtime.getDeviceCount()}")

    # --- FIGS smoke test on a real dataset ---
    figs_mod = _load_module("figs_smoke_module", FIGS_DIR / "FIGS.py")
    from sklearn.metrics import accuracy_score
    from prepare_data import prepare_data

    dataset_args = SimpleNamespace(
        dataset=dataset_name,
        random_state=0,
        binning_strategy="quantile",
        num_bins=64,
        max_thresholds_for_fsg=64,
        verbose=True,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, data_info = prepare_data(dataset_args)
    X = np.vstack([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])

    figs = figs_mod.FIGSClassifier(max_rules=4, max_trees=3, max_depth=2, random_state=0)
    figs.fit(X, y)
    pred = figs.predict(X)
    acc = accuracy_score(y, pred)
    lines.append(f"FIGS_TREES={len(figs.trees_)}")
    lines.append(f"FIGS_COMPLEXITY={figs.complexity_}")
    lines.append(f"FIGS_ACC={acc:.4f}")
    lines.append(f"FIGS_DATASET={dataset_args.dataset}")
    lines.append(f"FIGS_FEATURES={len(data_info.feature_names)}")

    # --- ShapeFitter smoke test ---
    _ensure_dummy_interpret()
    shape_fitter = _load_module("shape_fitter_smoke_module", SHADERTREE_MCTS_DIR / "shape_fitter.py")

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(80, 4)).astype(np.float64)
    residuals = (X_train[:, 0] * 0.6 - X_train[:, 1] * 0.2 + rng.normal(scale=0.1, size=80)).astype(np.float64)
    weights = np.ones(80, dtype=np.float64)
    leaf_indices = list(range(0, 50))

    fitter = shape_fitter.FastStepFitter(X_train, min_samples_leaf=10, n_internal_knots=3)
    model = fitter.fit(leaf_indices, residuals, weights)
    preds = model.predict(X_train[:5])

    lines.append(f"SHAPEFITTER_MODEL_TYPE={type(model).__name__}")
    lines.append(f"SHAPEFITTER_PRED_SHAPE={tuple(preds.shape)}")
    lines.append(f"SHAPEFITTER_PRED_MEAN={float(np.mean(preds)):.6f}")

    # --- Direct CuPy GPU smoke test (must succeed) ---
    gpu_X_train = cp.asarray(X_train)
    gpu_residuals = cp.asarray(residuals)
    gpu_weights = cp.asarray(weights)
    gpu_leaf_indices = cp.asarray(leaf_indices, dtype=cp.int64)

    gpu_fitter = shape_fitter.GPUSplineLeafFitter(n_internal_knots=3, degree=3, min_samples_leaf=10)
    gpu_fitter.precompute_bases(gpu_X_train)
    gpu_leaf_residuals = gpu_residuals[gpu_leaf_indices]
    gpu_leaf_weights = gpu_weights[gpu_leaf_indices]
    gpu_best_ssr, gpu_best_j = gpu_fitter.evaluate_leaf_fast_batched(
        gpu_leaf_indices, gpu_leaf_residuals, gpu_leaf_weights
    )

    gpu_B_leaf = gpu_fitter.All_Bases_device[gpu_best_j, gpu_leaf_indices, :]
    gpu_sw = cp.sqrt(cp.maximum(gpu_leaf_weights, 1e-12)).reshape(-1, 1)
    gpu_B_t = gpu_B_leaf * gpu_sw
    gpu_y_t = gpu_leaf_residuals.reshape(-1, 1) * gpu_sw
    gpu_alphaI = gpu_fitter.alpha * cp.eye(gpu_B_leaf.shape[1], dtype=gpu_fitter.dtype)
    gpu_beta = cp.linalg.solve(gpu_B_t.T @ gpu_B_t + gpu_alphaI, gpu_B_t.T @ gpu_y_t).reshape(-1)

    gpu_model = shape_fitter.GPUUnivariateSplineModel(
        feature_index=int(gpu_best_j),
        degree=gpu_fitter.degree,
        n_internal_knots=gpu_fitter.n_internal_knots,
        xmin=float(cp.asnumpy(gpu_fitter._xmin[gpu_best_j])),
        xmax=float(cp.asnumpy(gpu_fitter._xmax[gpu_best_j])),
        coeffs=gpu_beta,
        constant=None,
    )
    gpu_preds = gpu_model.predict(gpu_X_train[:5])

    lines.append(f"GPU_SHAPEFITTER_BEST_J={int(gpu_best_j)}")
    lines.append(f"GPU_SHAPEFITTER_BEST_SSR={float(cp.asnumpy(gpu_best_ssr)):.6f}")
    lines.append(f"GPU_SHAPEFITTER_MODEL_TYPE={type(gpu_model).__name__}")
    lines.append(f"GPU_SHAPEFITTER_PRED_SHAPE={tuple(gpu_preds.shape)}")
    lines.append(f"GPU_SHAPEFITTER_PRED_MEAN={float(np.mean(gpu_preds)):.6f}")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("SMOKE_TEST_OK")
    print(log_path)
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
