from __future__ import annotations
import os
import shutil
import tempfile
import numpy as np
import logging
from contextlib import contextmanager
import importlib.util
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

try:
    import fastsparsegams as fsg
except Exception:  # ImportError or runtime issues
    fsg = None

# Optional: load binning logic from study_fastsparse/_binning.py
PercentileBinner = None
BinningConfig = None
_binning_path = os.path.join(os.path.dirname(__file__), "study_fastsparse", "_binning.py")
try:
    if os.path.exists(_binning_path):
        _spec = importlib.util.spec_from_file_location("_study_fastsparse_binning", _binning_path)
        if _spec and _spec.loader:
            _binning_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_binning_mod)
            PercentileBinner = getattr(_binning_mod, "PercentileBinner", None)
            BinningConfig = getattr(_binning_mod, "BinningConfig", None)
except Exception:
    PercentileBinner = None
    BinningConfig = None

try:
    from scipy import sparse as _scipy_sparse
except Exception:
    _scipy_sparse = None

@contextmanager
def sparse_triplet_ravel():
    """Ensure triplet inputs to scipy sparse constructors are 1D to avoid fsg/scipy shape issues."""
    if _scipy_sparse is None:
        yield
        return
    orig_csr, orig_csc = _scipy_sparse.csr_matrix, _scipy_sparse.csc_matrix

    def _wrap(maker):
        def _f(arg, *a, **k):
            if isinstance(arg, (tuple, list)) and len(arg) == 3:
                data, indices, indptr = arg
                arg = (np.asarray(data).ravel(),
                       np.asarray(indices).ravel(),
                       np.asarray(indptr).ravel())
            return maker(arg, *a, **k)
        return _f

    _scipy_sparse.csr_matrix = _wrap(orig_csr)
    _scipy_sparse.csc_matrix = _wrap(orig_csc)
    try:
        yield
    finally:
        _scipy_sparse.csr_matrix, _scipy_sparse.csc_matrix = orig_csr, orig_csc

# --- Work around CuPy TemporaryDirectory cleanup race (ignore errors) ---
if not getattr(tempfile, "_cupy_ignore_cleanup_patched", False):
    _orig_tmpdir = tempfile.TemporaryDirectory

    def _TemporaryDirectory_ignore_errors(*args, **kwargs):
        kwargs.setdefault("ignore_cleanup_errors", True)
        return _orig_tmpdir(*args, **kwargs)

    tempfile.TemporaryDirectory = _TemporaryDirectory_ignore_errors  # type: ignore
    tempfile._cupy_ignore_cleanup_patched = True  # type: ignore

# Part 1: GPU-based Univariate Spline Fitter 
def _as_cupy(x):
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    if isinstance(x, cp.ndarray): return x
    return cp.asarray(x)
def _as_numpy(x):
    if _CUPY_AVAILABLE and isinstance(x, cp.ndarray): return cp.asnumpy(x)
    return np.asarray(x)
def _scale_to_unit_interval(x: cp.ndarray, x_min: float, x_max: float) -> cp.ndarray:
    eps = cp.asarray(1e-12, dtype=x.dtype)
    return (x - x_min) / cp.maximum(x_max - x_min, eps)
def _make_clamped_uniform_knots(n_internal: int, degree: int) -> cp.ndarray:
    p = int(degree)
    K = int(n_internal)
    left  = cp.zeros(p + 1, dtype=cp.float64)
    right = cp.ones(p + 1, dtype=cp.float64)
    if K > 0:
        internal = cp.linspace(0.0, 1.0, K + 2, dtype=cp.float64)[1:-1]
        t = cp.concatenate([left, internal, right])
    else:
        t = cp.concatenate([left, right])
    return t
def _bspline_basis_1d(x: cp.ndarray, knots: cp.ndarray, degree: int) -> cp.ndarray:
    x = x.ravel()
    p = int(degree)
    m = int(knots.shape[0])
    n_bases = m - p - 1
    n = x.shape[0]
    B0 = cp.zeros((n_bases, n), dtype=cp.float64)
    for i in range(n_bases):
        t_i, t_ip1 = knots[i], knots[i + 1]
        left = x >= t_i
        right = x < t_ip1 if i < n_bases - 1 else x <= t_ip1
        B0[i, :] = (left & right)
    if p == 0: return B0.T
    B_prev = B0
    for k in range(1, p + 1):
        Bk = cp.zeros_like(B_prev)
        for i in range(n_bases):
            denom1 = knots[i + k] - knots[i]
            A = 0.0 if denom1 == 0 else ((x - knots[i]) / denom1) * B_prev[i, :]
            if i + 1 < n_bases:
                denom2 = knots[i + k + 1] - knots[i + 1]
                C = 0.0 if denom2 == 0 else ((knots[i + k + 1] - x) / denom2) * B_prev[i + 1, :]
            else: C = 0.0
            Bk[i, :] = A + C
        B_prev = Bk
    return B_prev.T

def _scale_to_unit_interval_np(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    eps = np.asarray(1e-12, dtype=x.dtype)
    return (x - x_min) / np.maximum(x_max - x_min, eps)
def _make_clamped_uniform_knots_np(n_internal: int, degree: int) -> np.ndarray:
    p = int(degree)
    K = int(n_internal)
    left = np.zeros(p + 1, dtype=np.float64)
    right = np.ones(p + 1, dtype=np.float64)
    if K > 0:
        internal = np.linspace(0.0, 1.0, K + 2, dtype=np.float64)[1:-1]
        t = np.concatenate([left, internal, right])
    else:
        t = np.concatenate([left, right])
    return t
def _bspline_basis_1d_np(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = x.ravel()
    p = int(degree)
    m = int(knots.shape[0])
    n_bases = m - p - 1
    n = x.shape[0]
    B0 = np.zeros((n_bases, n), dtype=np.float64)
    for i in range(n_bases):
        t_i, t_ip1 = knots[i], knots[i + 1]
        left = x >= t_i
        right = x < t_ip1 if i < n_bases - 1 else x <= t_ip1
        B0[i, :] = (left & right)
    if p == 0:
        return B0.T
    B_prev = B0
    for k in range(1, p + 1):
        Bk = np.zeros_like(B_prev)
        for i in range(n_bases):
            denom1 = knots[i + k] - knots[i]
            A = 0.0 if denom1 == 0 else ((x - knots[i]) / denom1) * B_prev[i, :]
            if i + 1 < n_bases:
                denom2 = knots[i + k + 1] - knots[i + 1]
                C = 0.0 if denom2 == 0 else ((knots[i + k + 1] - x) / denom2) * B_prev[i + 1, :]
            else:
                C = 0.0
            Bk[i, :] = A + C
        B_prev = Bk
    return B_prev.T

class GPUUnivariateSplineModel:
    def __init__(self, feature_index: int, degree: int, n_internal_knots: int,
                 xmin: float, xmax: float, coeffs: cp.ndarray | None,
                 constant: float | None = None):
        self.feature_index = int(feature_index)
        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.coeffs = coeffs
        self.constant = None if constant is None else float(constant)
        self._knots = _make_clamped_uniform_knots(self.n_internal_knots, self.degree)
    def predict(self, X) -> np.ndarray:
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=float)
        Xcp = _as_cupy(X)
        x = Xcp[:, self.feature_index].astype(cp.float64, copy=False)
        x01 = _scale_to_unit_interval(x, self.xmin, self.xmax)
        B = _bspline_basis_1d(x01, self._knots, self.degree)
        y = B @ self.coeffs
        return _as_numpy(y)

class CPUUnivariateSplineModel:
    def __init__(self, feature_index: int, degree: int, n_internal_knots: int,
                 xmin: float, xmax: float, coeffs: np.ndarray | None,
                 constant: float | None = None):
        self.feature_index = int(feature_index)
        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.coeffs = coeffs
        self.constant = None if constant is None else float(constant)
        self._knots = _make_clamped_uniform_knots_np(self.n_internal_knots, self.degree)
    def predict(self, X) -> np.ndarray:
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=float)
        Xnp = np.asarray(X)
        x = Xnp[:, self.feature_index].astype(np.float64, copy=False)
        x01 = _scale_to_unit_interval_np(x, self.xmin, self.xmax)
        B = _bspline_basis_1d_np(x01, self._knots, self.degree)
        y = B @ self.coeffs
        return np.asarray(y)

class GPUAdditiveSplineModel:
    """Small GAM made of multiple univariate spline terms (all on GPU)."""

    def __init__(
        self,
        feature_indices: list[int],
        degree: int,
        n_internal_knots: int,
        xmins: list[float],
        xmaxs: list[float],
        coeffs_list: list[cp.ndarray],
        constant: float | None = None,
    ):
        self.feature_indices = [int(fi) for fi in feature_indices]
        self.feature_index = self.feature_indices[0] if self.feature_indices else -1
        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.xmins = [float(v) for v in xmins]
        self.xmaxs = [float(v) for v in xmaxs]
        self.coeffs_list = [cp.asarray(c, dtype=cp.float64) for c in coeffs_list]
        self.constant = None if constant is None else float(constant)
        self._knots = _make_clamped_uniform_knots(self.n_internal_knots, self.degree)

    def predict(self, X) -> np.ndarray:
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=float)
        if not self.feature_indices:
            return np.zeros(X.shape[0], dtype=float)

        Xcp = _as_cupy(X)
        total = cp.zeros(Xcp.shape[0], dtype=cp.float64)
        for fi, beta, xmin, xmax in zip(self.feature_indices, self.coeffs_list, self.xmins, self.xmaxs):
            x = Xcp[:, fi].astype(cp.float64, copy=False)
            x01 = _scale_to_unit_interval(x, xmin, xmax)
            B = _bspline_basis_1d(x01, self._knots, self.degree)
            total = total + B @ beta
        return _as_numpy(total)

    def predict_component(self, X, component_idx: int) -> np.ndarray:
        """Predict contribution of a single component in the additive model."""
        if self.constant is not None or not self.feature_indices:
            return self.predict(X)
        if component_idx < 0 or component_idx >= len(self.feature_indices):
            return np.zeros(X.shape[0], dtype=float)
        fi = self.feature_indices[component_idx]
        beta = self.coeffs_list[component_idx]
        xmin = self.xmins[component_idx]
        xmax = self.xmaxs[component_idx]

        Xcp = _as_cupy(X)
        x = Xcp[:, fi].astype(cp.float64, copy=False)
        x01 = _scale_to_unit_interval(x, xmin, xmax)
        B = _bspline_basis_1d(x01, self._knots, self.degree)
        return _as_numpy(B @ beta)

class CPUAdditiveSplineModel:
    """Small GAM made of multiple univariate spline terms (CPU)."""

    def __init__(
        self,
        feature_indices: list[int],
        degree: int,
        n_internal_knots: int,
        xmins: list[float],
        xmaxs: list[float],
        coeffs_list: list[np.ndarray],
        constant: float | None = None,
    ):
        self.feature_indices = [int(fi) for fi in feature_indices]
        self.feature_index = self.feature_indices[0] if self.feature_indices else -1
        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.xmins = [float(v) for v in xmins]
        self.xmaxs = [float(v) for v in xmaxs]
        self.coeffs_list = [np.asarray(c, dtype=np.float64) for c in coeffs_list]
        self.constant = None if constant is None else float(constant)
        self._knots = _make_clamped_uniform_knots_np(self.n_internal_knots, self.degree)

    def predict(self, X) -> np.ndarray:
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=float)
        if not self.feature_indices:
            return np.zeros(X.shape[0], dtype=float)

        Xnp = np.asarray(X)
        total = np.zeros(Xnp.shape[0], dtype=np.float64)
        for fi, beta, xmin, xmax in zip(self.feature_indices, self.coeffs_list, self.xmins, self.xmaxs):
            x = Xnp[:, fi].astype(np.float64, copy=False)
            x01 = _scale_to_unit_interval_np(x, xmin, xmax)
            B = _bspline_basis_1d_np(x01, self._knots, self.degree)
            total = total + B @ beta
        return np.asarray(total)

    def predict_component(self, X, component_idx: int) -> np.ndarray:
        """Predict contribution of a single component in the additive model."""
        if self.constant is not None or not self.feature_indices:
            return self.predict(X)
        if component_idx < 0 or component_idx >= len(self.feature_indices):
            return np.zeros(X.shape[0], dtype=float)
        fi = self.feature_indices[component_idx]
        beta = self.coeffs_list[component_idx]
        xmin = self.xmins[component_idx]
        xmax = self.xmaxs[component_idx]

        Xnp = np.asarray(X)
        x = Xnp[:, fi].astype(np.float64, copy=False)
        x01 = _scale_to_unit_interval_np(x, xmin, xmax)
        B = _bspline_basis_1d_np(x01, self._knots, self.degree)
        return np.asarray(B @ beta)

class GPUSplineLeafFitter:
    def __init__(self, n_internal_knots: int = 6, degree: int = 3, alpha: float = 0.1,
                 min_samples_leaf: int = 50, dtype=cp.float64):
        self.n_internal_knots = int(n_internal_knots)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.min_samples_leaf = int(min_samples_leaf)
        self.dtype = dtype
        self._knots = _make_clamped_uniform_knots(self.n_internal_knots, self.degree)
        self.All_Bases_device = None
        self._xmin = None
        self._xmax = None
    def precompute_bases(self, X: np.ndarray | cp.ndarray) -> None:
        Xcp = _as_cupy(X).astype(self.dtype, copy=False)
        N, F = Xcp.shape
        self._xmin = cp.min(Xcp, axis=0).astype(self.dtype)
        self._xmax = cp.max(Xcp, axis=0).astype(self.dtype)
        S = self.n_internal_knots + self.degree + 1
        bases = []
        for j in range(F):
            xj = Xcp[:, j]
            x01 = _scale_to_unit_interval(xj, self._xmin[j], self._xmax[j])
            B = _bspline_basis_1d(x01, self._knots, self.degree).astype(self.dtype, copy=False)
            bases.append(B[:, None, :])
        B_all = cp.concatenate(bases, axis=1)
        self.All_Bases_device = B_all.transpose(1, 0, 2).copy()
    def evaluate_leaf_fast_batched(self, sample_idx, residuals, weights):
        if self.All_Bases_device is None:
            raise RuntimeError("Call precompute_bases(X) before evaluate_leaf_fast_batched.")
        idx = _as_cupy(sample_idx).astype(cp.int64, copy=False)
        r = _as_cupy(residuals).astype(self.dtype, copy=False).reshape(-1)
        w = _as_cupy(weights).astype(self.dtype, copy=False).reshape(-1)
        n_leaf = idx.shape[0]
        if n_leaf < self.min_samples_leaf:
            return cp.inf, -1
        B = self.All_Bases_device[:, idx, :]
        sw = cp.sqrt(cp.maximum(w, self.dtype(1e-12))).reshape(1, n_leaf, 1)
        B_t = B * sw
        y_t = r.reshape(1, n_leaf, 1) * sw
        F, _, S = B_t.shape
        alphaI = self.alpha * cp.eye(S, dtype=self.dtype)
        best_ssr, best_j = cp.inf, -1
        for j in range(F):
            Bj = B_t[j]
            A = Bj.T @ Bj + alphaI
            b = Bj.T @ y_t.reshape(n_leaf, 1)
            try:
                beta = cp.linalg.solve(A, b).reshape(-1)
                yhat = (B[j] @ beta).reshape(-1)
                ssr = cp.sum(w * (r - yhat) ** 2)
            except Exception:
                ssr = cp.inf
            if ssr < best_ssr:
                best_ssr, best_j = ssr, j
        return best_ssr, int(best_j)

class CPUSplineLeafFitter:
    def __init__(self, n_internal_knots: int = 6, degree: int = 3, alpha: float = 0.1,
                 min_samples_leaf: int = 50, dtype=np.float64):
        self.n_internal_knots = int(n_internal_knots)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.min_samples_leaf = int(min_samples_leaf)
        self.dtype = dtype
        self._knots = _make_clamped_uniform_knots_np(self.n_internal_knots, self.degree)
        self.All_Bases_device = None
        self._xmin = None
        self._xmax = None
    def precompute_bases(self, X: np.ndarray) -> None:
        Xnp = np.asarray(X, dtype=self.dtype)
        N, F = Xnp.shape
        self._xmin = np.min(Xnp, axis=0).astype(self.dtype)
        self._xmax = np.max(Xnp, axis=0).astype(self.dtype)
        S = self.n_internal_knots + self.degree + 1
        bases = []
        for j in range(F):
            xj = Xnp[:, j]
            x01 = _scale_to_unit_interval_np(xj, self._xmin[j], self._xmax[j])
            B = _bspline_basis_1d_np(x01, self._knots, self.degree).astype(self.dtype, copy=False)
            bases.append(B[:, None, :])
        B_all = np.concatenate(bases, axis=1)
        self.All_Bases_device = B_all.transpose(1, 0, 2).copy()
    def evaluate_leaf_fast_batched(self, sample_idx, residuals, weights):
        if self.All_Bases_device is None:
            raise RuntimeError("Call precompute_bases(X) before evaluate_leaf_fast_batched.")
        idx = np.asarray(sample_idx, dtype=np.int64)
        r = np.asarray(residuals, dtype=self.dtype).reshape(-1)
        w = np.asarray(weights, dtype=self.dtype).reshape(-1)
        n_leaf = idx.shape[0]
        if n_leaf < self.min_samples_leaf:
            return np.inf, -1
        B = self.All_Bases_device[:, idx, :]
        sw = np.sqrt(np.maximum(w, self.dtype(1e-12))).reshape(1, n_leaf, 1)
        B_t = B * sw
        y_t = r.reshape(1, n_leaf, 1) * sw
        F, _, S = B_t.shape
        alphaI = self.alpha * np.eye(S, dtype=self.dtype)
        best_ssr, best_j = np.inf, -1
        for j in range(F):
            Bj = B_t[j]
            A = Bj.T @ Bj + alphaI
            b = Bj.T @ y_t.reshape(n_leaf, 1)
            try:
                beta = np.linalg.solve(A, b).reshape(-1)
                yhat = (B[j] @ beta).reshape(-1)
                ssr = np.sum(w * (r - yhat) ** 2)
            except Exception:
                ssr = np.inf
            if ssr < best_ssr:
                best_ssr, best_j = ssr, j
        return best_ssr, int(best_j)

# Part 2: Abstraction Layer for MCTS Integration
logger = logging.getLogger('MCTS_ADTree')

class BaseShapeFitter:
    def __init__(self, X_train_full, **kwargs):
        self.max_features_per_shape = int(kwargs.pop("max_features_per_shape", 1))
        self.max_features_per_shape = max(1, self.max_features_per_shape)
        self.fitter = None
        self._use_gpu = False
        if _CUPY_AVAILABLE:
            try:
                self.fitter = GPUSplineLeafFitter(**kwargs)
                logger.info(f"Initializing {self.__class__.__name__} (degree={self.fitter.degree}) and pre-computing bases on GPU...")
                try:
                    self.fitter.precompute_bases(X_train_full)
                    self._use_gpu = True
                    logger.info("GPU bases pre-computation successful.")
                except Exception as e:
                    logger.warning(f"Pre-compute bases failed (first attempt). Error: {e}")
                    retry_tmp_root = os.environ.get("TMPDIR", None)
                    try:
                        retry_tmp_root = tempfile.mkdtemp(prefix="cupy_retry_", dir=retry_tmp_root)
                        os.environ["TMPDIR"] = retry_tmp_root
                        tempfile.tempdir = retry_tmp_root
                        logger.info(f"Retrying GPU base pre-computation with TMPDIR={retry_tmp_root}")
                        self.fitter.precompute_bases(X_train_full)
                        self._use_gpu = True
                        logger.info("GPU bases pre-computation successful on retry.")
                    except Exception as e2:
                        logger.error(f"GPU init failed; falling back to CPU. Error: {e2}")
                        self._init_cpu_fitter(X_train_full, **kwargs)
                    finally:
                        # Leave retry_tmp_root in place to avoid cleanup races that caused the failure.
                        pass
            except Exception as e:
                logger.error(f"GPU fitter init failed; falling back to CPU. Error: {e}")
                self._init_cpu_fitter(X_train_full, **kwargs)
        else:
            logger.warning("CuPy not available; falling back to CPU.")
            self._init_cpu_fitter(X_train_full, **kwargs)

    def _init_cpu_fitter(self, X_train_full, **kwargs):
        self.fitter = CPUSplineLeafFitter(**kwargs)
        self._use_gpu = False
        logger.info(f"Initializing {self.__class__.__name__} (degree={self.fitter.degree}) and pre-computing bases on CPU...")
        self.fitter.precompute_bases(X_train_full)
        logger.info("CPU bases pre-computation successful.")

    def _to_float(self, value):
        if self._use_gpu:
            return float(cp.asnumpy(value))
        return float(value)
    def _as_xp(self, x):
        if self._use_gpu:
            return _as_cupy(x)
        if _CUPY_AVAILABLE and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)
    def _constant_model(self, constant):
        if self._use_gpu:
            return GPUUnivariateSplineModel(-1, self.fitter.degree, self.fitter.n_internal_knots, 0.0, 1.0, coeffs=None, constant=constant)
        return CPUUnivariateSplineModel(-1, self.fitter.degree, self.fitter.n_internal_knots, 0.0, 1.0, coeffs=None, constant=constant)
    
    def _fit_constant(self, r_cupy, w_cupy):
        xp = cp if self._use_gpu else np
        c = self._to_float(xp.sum(w_cupy * r_cupy) / xp.maximum(xp.sum(w_cupy), 1e-12))
        return self._constant_model(c)

    def _fit_univariate(self, feature_idx, idx_cupy, r_cupy, w_cupy):
        if feature_idx == -1:
            return self._fit_constant(r_cupy, w_cupy)

        xp = cp if self._use_gpu else np
        B_leaf = self.fitter.All_Bases_device[feature_idx, idx_cupy, :]
        sw = xp.sqrt(xp.maximum(w_cupy, self.fitter.dtype(1e-12))).reshape(-1, 1)
        B_t = B_leaf * sw
        y_t = r_cupy * sw.reshape(-1)

        A = B_t.T @ B_t + self.fitter.alpha * xp.eye(B_leaf.shape[1], dtype=self.fitter.dtype)
        b = B_t.T @ y_t
        try:
            beta = xp.linalg.solve(A, b).reshape(-1)
            xmin = float(self.fitter._xmin[feature_idx])
            xmax = float(self.fitter._xmax[feature_idx])
            if self._use_gpu:
                return GPUUnivariateSplineModel(
                    feature_index=feature_idx, degree=self.fitter.degree,
                    n_internal_knots=self.fitter.n_internal_knots,
                    xmin=xmin, xmax=xmax, coeffs=beta, constant=None
                )
            return CPUUnivariateSplineModel(
                feature_index=feature_idx, degree=self.fitter.degree,
                n_internal_knots=self.fitter.n_internal_knots,
                xmin=xmin, xmax=xmax, coeffs=beta, constant=None
            )
        except Exception as e:
            logger.error(f"Linear solve failed during final fit for feature {feature_idx}. Reverting to constant. Error: {e}")
            return self._fit_constant(r_cupy, w_cupy)

    def _fit_additive(self, idx_cupy, r_cupy, w_cupy, k, preselected=None):
        n_leaf = idx_cupy.shape[0]
        if n_leaf < self.fitter.min_samples_leaf:
            return self._fit_constant(r_cupy, w_cupy)

        xp = cp if self._use_gpu else np
        preselected = [fi for fi in (preselected or []) if fi is not None and fi >= 0]
        B_all = self.fitter.All_Bases_device[:, idx_cupy, :]
        sw_vec = xp.sqrt(xp.maximum(w_cupy, self.fitter.dtype(1e-12))).reshape(-1)
        y_t = r_cupy * sw_vec
        alphaI = self.fitter.alpha * xp.eye(B_all.shape[2], dtype=self.fitter.dtype)

        ssr_list = []
        betas = []
        for j in range(B_all.shape[0]):
            Bj = B_all[j]
            Bj_t = Bj * sw_vec.reshape(-1, 1)
            try:
                A = Bj_t.T @ Bj_t + alphaI
                b = Bj_t.T @ y_t
                beta = xp.linalg.solve(A, b).reshape(-1)
                yhat = (Bj @ beta).reshape(-1)
                ssr = xp.sum(w_cupy * (r_cupy - yhat) ** 2)
            except Exception:
                beta = None
                ssr = xp.inf
            ssr_list.append(ssr)
            betas.append(beta)

        ssr_cpu = []
        for s in ssr_list:
            if xp.isfinite(s):
                ssr_cpu.append(self._to_float(s))
            else:
                ssr_cpu.append(float('inf'))
        selected = []
        for fi in preselected:
            if 0 <= fi < len(ssr_cpu):
                selected.append(int(fi))

        # Fill remaining slots with best-SSR features
        remaining = [j for j in range(len(ssr_cpu)) if j not in selected]
        remaining_sorted = sorted(remaining, key=lambda j: ssr_cpu[j])
        for j in remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(j)

        selected = selected[:k]
        selected = [fi for fi in selected if betas[fi] is not None and xp.isfinite(ssr_list[fi])]
        if not selected:
            return self._fit_constant(r_cupy, w_cupy)

        try:
            basis_list = [B_all[j] for j in selected]
            B_concat = xp.concatenate(basis_list, axis=1)  # (n_leaf, k*S)
            B_t = B_concat * sw_vec.reshape(-1, 1)
            A = B_t.T @ B_t + self.fitter.alpha * xp.eye(B_concat.shape[1], dtype=self.fitter.dtype)
            b = B_t.T @ y_t
            beta_all = xp.linalg.solve(A, b).reshape(-1)

            S = self.fitter.n_internal_knots + self.fitter.degree + 1
            coeffs_list = []
            offset = 0
            for _ in selected:
                coeffs_list.append(beta_all[offset:offset + S])
                offset += S

            xmins = [float(self.fitter._xmin[j]) for j in selected]
            xmaxs = [float(self.fitter._xmax[j]) for j in selected]
            if self._use_gpu:
                return GPUAdditiveSplineModel(
                    feature_indices=selected,
                    degree=self.fitter.degree,
                    n_internal_knots=self.fitter.n_internal_knots,
                    xmins=xmins,
                    xmaxs=xmaxs,
                    coeffs_list=coeffs_list
                )
            return CPUAdditiveSplineModel(
                feature_indices=selected,
                degree=self.fitter.degree,
                n_internal_knots=self.fitter.n_internal_knots,
                xmins=xmins,
                xmaxs=xmaxs,
                coeffs_list=coeffs_list
            )
        except Exception as e:
            logger.error(f"Additive spline solve failed; reverting to constant. Error: {e}")
            return self._fit_constant(r_cupy, w_cupy)

    def fit(self, leaf_data_indices, full_residuals, full_weights, feature_idx=None):
        if len(leaf_data_indices) == 0:
            return self._constant_model(0.0)

        idx_cupy = self._as_xp(leaf_data_indices)
        r_cupy = self._as_xp(full_residuals)[idx_cupy]
        w_cupy = self._as_xp(full_weights)[idx_cupy]

        # Univariate path (default / compatibility)
        if self.max_features_per_shape <= 1:
            if feature_idx is None:
                _, best_j = self.fitter.evaluate_leaf_fast_batched(idx_cupy, r_cupy, w_cupy)
                feature_idx = best_j
            return self._fit_univariate(feature_idx, idx_cupy, r_cupy, w_cupy)

        # Small-GAM path: allow multiple features per leaf
        preselected = [] if feature_idx is None else [feature_idx]
        return self._fit_additive(idx_cupy, r_cupy, w_cupy, k=self.max_features_per_shape, preselected=preselected)

class FastStepFitter(BaseShapeFitter):
    def __init__(self, X_train_full, **kwargs):
        kwargs['degree'] = 0
        kwargs['n_internal_knots'] = kwargs.get('n_internal_knots', 2)
        super().__init__(X_train_full, **kwargs)

# Part 3: Multi-Feature EBM Fitter for MCTS
class EBMModelWrapper:
    def __init__(self, ebm_model, feature_indices, feature_names, constant: float | None = None):
        self.ebm_model = ebm_model
        self.feature_indices = feature_indices
        self.feature_names = feature_names
        self.constant = constant
        self.feature_index = -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=float)
        if self.ebm_model is None or not self.feature_indices:
            return np.zeros(X.shape[0])
        X_subset = X[:, self.feature_indices]
        return self.ebm_model.predict(X_subset)

class EBMShapeFitter:
    # The constructor is parameterized to allow creating different configs (fast vs. high-quality).
    def __init__(self, X_train_full: np.ndarray, feature_names: list, min_samples_leaf: int = 50, 
                 interactions: int = 0, outer_bags: int = 1, inner_bags: int = 1, max_bins: int = 64, **kwargs):
        logger.info(f"Initializing Multi-Feature {self.__class__.__name__}...")
        if X_train_full is None:
            raise ValueError("EBMShapeFitter requires the full training dataset (X_train_full).")
        
        self.X_train_full = X_train_full
        self.feature_names = np.array(feature_names)
        self.min_samples_leaf = min_samples_leaf
        self.ebm_params = {
            'interactions': interactions,
            'inner_bags': inner_bags,
            'outer_bags': outer_bags,
            'max_bins': max_bins,
            'n_jobs': -1,
            'random_state': 42,
            **kwargs
        }
        logger.info(f"Multi-Feature EBM Fitter initialized with params: {self.ebm_params}")

    def fit(self, leaf_data_indices: list, full_residuals: np.ndarray, full_weights: np.ndarray, feature_idx: int | None = None):
        if len(leaf_data_indices) < self.min_samples_leaf:
            return None

        X_leaf = self.X_train_full[leaf_data_indices, :]
        r_leaf = full_residuals[leaf_data_indices]
        w_leaf = full_weights[leaf_data_indices]

        feature_stds = X_leaf.std(axis=0)
        variable_features_indices = np.where(feature_stds > 1e-6)[0].tolist()

        if not variable_features_indices:
            logger.debug("No variable features in this leaf. Fitting a constant model.")
            constant = np.average(r_leaf, weights=w_leaf if np.sum(w_leaf) > 0 else None)
            return EBMModelWrapper(ebm_model=None, feature_indices=[], feature_names=[], constant=constant)

        X_leaf_variable = X_leaf[:, variable_features_indices]
        variable_feature_names = self.feature_names[variable_features_indices].tolist()

        try:
            ebm = ExplainableBoostingRegressor(**self.ebm_params)
            ebm.feature_names = variable_feature_names
            ebm.fit(X_leaf_variable, r_leaf, sample_weight=w_leaf)
            
            return EBMModelWrapper(
                ebm_model=ebm,
                feature_indices=variable_features_indices,
                feature_names=variable_feature_names
            )
        except Exception as e:
            logger.error(f"EBM multi-feature fit failed. Reverting to constant. Error: {e}", exc_info=True)
            constant = np.average(r_leaf, weights=w_leaf if np.sum(w_leaf) > 0 else None)
            return EBMModelWrapper(ebm_model=None, feature_indices=[], feature_names=[], constant=constant)


# Part 4: Optional FastSparse-based CPU fitter for refinement
class FastSparseModel:
    def __init__(self, intercept: float, coeffs: np.ndarray,
                 binner=None, raw_feature_names=None, binner_feature_names=None, support_tol: float = 1e-6):
        self.intercept = float(intercept)
        self.coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        self.binner = binner
        self.raw_feature_names = list(raw_feature_names) if raw_feature_names is not None else None
        self.binner_feature_names = list(binner_feature_names) if binner_feature_names is not None else None
        self.support_tol = float(support_tol)

        support_idx = [i for i, c in enumerate(self.coeffs) if abs(c) > self.support_tol]
        self.support_indices = support_idx
        self.support_size = int(len(support_idx))
        # Treat intercept-only fits as a single "step" for sparsity accounting.
        self.step_count = max(1, self.support_size)
        self.feature_indices = self._map_support_to_raw_features(support_idx)
        self.feature_index = self.feature_indices[0] if len(self.feature_indices) == 1 else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xnp = np.asarray(X, dtype=np.float64)
        if self.binner is not None:
            Z, _ = self.binner.transform_numpy(Xnp, feature_names=self.raw_feature_names)
            Xnp = np.asarray(Z, dtype=np.float64)
        return (self.intercept + Xnp @ self.coeffs).reshape(-1)

    def _map_support_to_raw_features(self, support_idx):
        if not support_idx:
            return []
        if not self.raw_feature_names or not self.binner_feature_names:
            return [int(i) for i in support_idx]
        name_to_idx = {n: i for i, n in enumerate(self.raw_feature_names)}
        feats = []
        for col in support_idx:
            if col < 0 or col >= len(self.binner_feature_names):
                continue
            name = str(self.binner_feature_names[col])
            raw_name = name.split("<=")[0].strip()
            if raw_name.endswith("_isnan"):
                raw_name = raw_name[:-6]
            if raw_name in name_to_idx:
                feats.append(name_to_idx[raw_name])
                continue
            if "_" in raw_name:
                base = raw_name.split("_", 1)[0]
                if base in name_to_idx:
                    feats.append(name_to_idx[base])
                    continue
            import re
            m = re.search(r'(\d+)', raw_name)
            if m:
                feats.append(int(m.group(1)))
        return sorted(set(int(f) for f in feats))


class FastSparseShapeFitter:
    def __init__(self, X_train_full: np.ndarray, min_samples_leaf: int = 50, max_support_size: int = 3,
                 support_tol: float = 1e-6, algorithm: str = "CDPSI", loss: str = "SquaredError",
                 use_binner: bool = True, binning_config=None, **kwargs):
        self.available = fsg is not None
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_support_size = int(max_support_size)
        self.support_tol = float(support_tol)
        self.algorithm = str(algorithm)
        self.loss = str(loss)
        self.X_train_full = np.asarray(X_train_full, dtype=np.float64)
        self.use_binner = bool(use_binner)
        self.binner = None
        self.X_train_binned = None
        self.raw_feature_names = [f"f_{i}" for i in range(self.X_train_full.shape[1])]
        self.binner_feature_names = None
        if not self.available:
            logger.warning("fastsparsegams not available; FastSparseShapeFitter will fall back to None.")
        if self.use_binner and PercentileBinner is not None:
            try:
                cfg = binning_config if binning_config is not None else (BinningConfig() if BinningConfig else None)
                if cfg is not None:
                    self.binner = PercentileBinner(cfg)
                else:
                    self.binner = PercentileBinner()
                df = pd.DataFrame(self.X_train_full, columns=self.raw_feature_names)
                self.binner.fit(df)
                Z, names = self.binner.transform_numpy(self.X_train_full, feature_names=self.raw_feature_names)
                self.X_train_binned = np.asarray(Z, dtype=np.float64)
                self.binner_feature_names = list(names) if names is not None else None
            except Exception as e:
                logger.warning(f"FastSparse binner init failed; using raw X. Error: {e}")
                self.binner = None
                self.X_train_binned = None
                self.binner_feature_names = None

    def fit(self, leaf_data_indices: list, full_residuals: np.ndarray, full_weights: np.ndarray, feature_idx: int | None = None):
        if not self.available:
            return None
        if len(leaf_data_indices) < self.min_samples_leaf:
            return None

        idx = np.asarray(leaf_data_indices, dtype=int)
        if self.X_train_binned is not None:
            X_leaf = self.X_train_binned[idx]
        else:
            X_leaf = self.X_train_full[idx]
        r_leaf = np.asarray(full_residuals, dtype=np.float64)[idx]
        w_leaf = np.asarray(full_weights, dtype=np.float64)[idx]
        w_leaf = np.maximum(w_leaf, 1e-12)

        # Weighted design via sqrt weights
        sw = np.sqrt(w_leaf)
        Xw = X_leaf * sw[:, None]
        rw = r_leaf * sw

        try:
            import inspect
            args = inspect.getfullargspec(fsg.fit)
            fit_kwargs = {
                "X": Xw,
                "y": rw,
                "max_support_size": self.max_support_size,
            }
            if "penalty" in args.args:
                fit_kwargs["penalty"] = "L0"
            if "algorithm" in args.args:
                fit_kwargs["algorithm"] = self.algorithm
            if "loss" in args.args:
                fit_kwargs["loss"] = self.loss
            with sparse_triplet_ravel():
                path = fsg.fit(**fit_kwargs)
        except Exception as e:
            logger.error(f"fastsparsegams fit failed; skipping FastSparse refinement. Error: {e}")
            return None

        try:
            lambdas = list(path.lambda_0[0])
        except Exception:
            lambdas = []
        if not lambdas:
            return None

        best_intercept, best_coeffs, best_sse = None, None, np.inf
        for idx_lambda, lam in enumerate(lambdas):
            try:
                intercept = float(path.intercepts[0][idx_lambda])
                coeffs = path.coeffs[0][:, idx_lambda].toarray().reshape(-1)
                pred = (intercept + Xw @ coeffs).ravel()
                sse = float(np.sum((rw - pred) ** 2))
                if np.isfinite(sse) and sse < best_sse:
                    best_sse = sse
                    best_intercept = intercept
                    best_coeffs = coeffs
            except Exception:
                continue

        if best_coeffs is None:
            return None
        if np.sum(np.abs(best_coeffs) > self.support_tol) == 0:
            best_coeffs = np.zeros_like(best_coeffs)
        return FastSparseModel(
            best_intercept,
            best_coeffs,
            binner=self.binner,
            raw_feature_names=self.raw_feature_names,
            binner_feature_names=self.binner_feature_names,
            support_tol=self.support_tol,
        )
