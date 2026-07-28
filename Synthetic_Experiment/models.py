"""Uniform adapters for all models in the synthetic comparison."""

from __future__ import annotations

import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
MCTS_DIR = ROOT / "ShadeTree" / "ShadeTree-MCTS"


class MissingDependencyError(RuntimeError):
    pass


class LogisticAdapter:
    name = "logistic"

    def __init__(self, C: float = 1.0, random_state: int = 0, **_):
        self.model = LogisticRegression(
            C=C, max_iter=3000, solver="lbfgs", random_state=random_state
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def complexity(self, X=None):
        active = int(np.count_nonzero(np.abs(self.model.coef_) > 1e-10))
        return {
            "split_nodes": 0,
            "max_depth": 0,
            "shape_functions": active,
            "variables_used": active,
            "active_shape_basis": active,
            "mean_decisions": 0,
            "mean_active_shapes": active,
            "total_active_components": active,
        }


class EBMAdapter:
    """Classifier adapter for both interaction-free and pairwise EBM."""

    def __init__(
        self,
        interactions: int = 0,
        max_bins: int = 64,
        learning_rate: float = 0.05,
        max_rounds: int = 5000,
        min_samples_leaf: int = 2,
        random_state: int = 0,
        **_,
    ):
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except Exception as exc:
            raise MissingDependencyError(
                "EBM requires the 'interpret' package."
            ) from exc
        self.interactions = int(interactions)
        self.model = ExplainableBoostingClassifier(
            interactions=self.interactions,
            max_bins=int(max_bins),
            learning_rate=float(learning_rate),
            max_rounds=int(max_rounds),
            min_samples_leaf=int(min_samples_leaf),
            outer_bags=1,
            inner_bags=0,
            validation_size=0,
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decision_function(self, X):
        p = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    def complexity(self, X=None):
        term_features = list(getattr(self.model, "term_features_", []))
        term_scores = list(getattr(self.model, "term_scores_", []))
        active_flags = [
            bool(np.any(np.abs(np.asarray(scores)) > 1e-10)) for scores in term_scores
        ]
        main_terms = sum(
            len(term) == 1 and active
            for term, active in zip(term_features, active_flags)
        )
        interaction_terms = sum(
            len(term) > 1 and active
            for term, active in zip(term_features, active_flags)
        )
        active_cells = 0
        for scores in term_scores:
            arr = np.asarray(scores)
            active_cells += int(np.count_nonzero(np.abs(arr) > 1e-10))
        variables = len(
            {
                j
                for term, active in zip(term_features, active_flags)
                if active
                for j in term
            }
        )
        return {
            "split_nodes": 0,
            "max_depth": 0,
            "shape_functions": int(main_terms),
            "interaction_functions": int(interaction_terms),
            "variables_used": int(variables),
            "active_shape_basis": int(active_cells),
            "mean_decisions": 0,
            "mean_active_shapes": int(main_terms + interaction_terms),
            "total_active_components": int(active_cells),
        }


class FSGAdapter:
    """Fast Sparse GAM classifier using FSG's logistic solution path.

    Continuous features are represented by one-sided quantile step functions.
    The validation set selects one model from the returned L0 path.
    """

    def __init__(
        self,
        max_support_size: int = 20,
        n_bins: int = 16,
        random_state: int = 0,
        **_,
    ):
        try:
            import fastsparsegams as fsg
        except Exception as exc:
            raise MissingDependencyError(
                "FSG requires the 'fastsparsegams' package."
            ) from exc
        self.fsg = fsg
        self.max_support_size = int(max_support_size)
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)

    def _fit_binner(self, X):
        qs = np.linspace(0, 1, self.n_bins + 2)[1:-1]
        self.thresholds_ = []
        self.design_feature_ = []
        for j in range(X.shape[1]):
            thresholds = np.unique(np.quantile(X[:, j], qs))
            self.thresholds_.append(thresholds)
            self.design_feature_.extend([j] * len(thresholds))

    def _transform(self, X):
        columns = [
            (X[:, j, None] > thresholds[None, :]).astype(np.float64)
            for j, thresholds in enumerate(self.thresholds_)
            if len(thresholds)
        ]
        return np.concatenate(columns, axis=1) if columns else np.empty((len(X), 0))

    @staticmethod
    def _extract_path(path):
        lambdas = list(path.lambda_0[0])
        candidates = []
        for idx, _ in enumerate(lambdas):
            intercept = float(path.intercepts[0][idx])
            coef = np.asarray(path.coeffs[0][:, idx].toarray()).reshape(-1)
            candidates.append((intercept, coef))
        return candidates

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.metrics import log_loss

        if X_val is None or y_val is None:
            raise ValueError("FSGAdapter requires validation data for path selection.")
        self._fit_binner(np.asarray(X))
        Z = self._transform(np.asarray(X))
        Zv = self._transform(np.asarray(X_val))
        kwargs = dict(
            X=Z,
            y=np.asarray(y, dtype=float),
            penalty="L0",
            max_support_size=self.max_support_size,
            loss="Logistic",
        )
        signature = inspect.signature(self.fsg.fit)
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        try:
            path = self.fsg.fit(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                "FSG logistic fit failed. Verify that this fastsparsegams build "
                "supports loss='Logistic'."
            ) from exc
        candidates = self._extract_path(path)
        if not candidates:
            raise RuntimeError("FSG returned an empty regularization path.")
        losses = [
            log_loss(y_val, expit(intercept + Zv @ coef), labels=[0, 1])
            for intercept, coef in candidates
        ]
        self.intercept_, self.coef_ = candidates[int(np.argmin(losses))]
        return self

    def decision_function(self, X):
        return self.intercept_ + self._transform(np.asarray(X)) @ self.coef_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def complexity(self, X=None):
        active = np.flatnonzero(np.abs(self.coef_) > 1e-10)
        variables = len({self.design_feature_[i] for i in active})
        return {
            "split_nodes": 0,
            "max_depth": 0,
            "shape_functions": int(variables),
            "variables_used": int(variables),
            "active_shape_basis": int(len(active)),
            "mean_decisions": 0,
            "mean_active_shapes": int(variables),
            "total_active_components": int(len(active)),
        }


class _ConstantModel:
    feature_index = -1

    def __init__(self, constant: float):
        self.constant = float(constant)

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.constant, dtype=float)


class _ConstantFitter:
    """Drop-in replacement for MCTS.FastStepFitter."""

    def __init__(self, X_train_full, **kwargs):
        self.X_train_full = X_train_full

    def fit(self, leaf_data_indices, full_residuals, full_weights, feature_idx=None):
        idx = np.asarray(leaf_data_indices, dtype=int)
        if idx.size == 0:
            return _ConstantModel(0.0)
        residual = np.asarray(full_residuals)[idx]
        weight = np.maximum(np.asarray(full_weights)[idx], 1e-12)
        return _ConstantModel(np.sum(weight * residual) / np.sum(weight))


def _install_interpret_stub():
    """Allow the step-only MCTS modules to import when interpret is absent."""
    if "interpret.glassbox" in sys.modules:
        return
    try:
        import interpret.glassbox  # noqa: F401
        return
    except Exception:
        pass
    interpret = types.ModuleType("interpret")
    glassbox = types.ModuleType("interpret.glassbox")

    class UnavailableEBM:
        def __init__(self, *args, **kwargs):
            raise MissingDependencyError("EBM requires the 'interpret' package.")

    glassbox.ExplainableBoostingRegressor = UnavailableEBM
    interpret.glassbox = glassbox
    sys.modules.setdefault("interpret", interpret)
    sys.modules.setdefault("interpret.glassbox", glassbox)


def _load_mcts_modules():
    _install_interpret_stub()
    path = str(MCTS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import ADTree
    import MCTS

    return MCTS, ADTree


@contextmanager
def _temporary_fast_step(mcts_module, replacement):
    original = mcts_module.FastStepFitter
    mcts_module.FastStepFitter = replacement
    try:
        yield
    finally:
        mcts_module.FastStepFitter = original


@contextmanager
def _temporary_depth_limit(mcts_module, max_depth: int):
    """Enforce split depth because the upstream run_mcts argument is unused."""
    original = mcts_module.SearchState.available_actions

    def limited(self, action_cache, max_splits=float("inf")):
        actions = original(self, action_cache, max_splits=max_splits)
        # Shape/Split nodes alternate, so ShapeNode.depth // 2 is split depth.
        return [
            action
            for action in actions
            if self.node_map[action[1]].depth // 2 < int(max_depth)
        ]

    mcts_module.SearchState.available_actions = limited
    try:
        yield
    finally:
        mcts_module.SearchState.available_actions = original


class MCTSModelAdapter:
    """Adapter for paired constant-ADTree and ShadeTree MCTS experiments."""

    def __init__(
        self,
        constant_leaf: bool,
        max_split_nodes: int = 10,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        max_iters: int = 300,
        n_knots_step: int = 8,
        complexity_penalty: float = 0.0,
        num_thresholds: int = 20,
        random_state: int = 0,
        **_,
    ):
        self.constant_leaf = bool(constant_leaf)
        self.max_split_nodes = int(max_split_nodes)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_iters = int(max_iters)
        self.n_knots_step = int(n_knots_step)
        self.complexity_penalty = float(complexity_penalty)
        self.num_thresholds = int(num_thresholds)
        self.random_state = int(random_state)

    def _thresholds(self, X):
        percentiles = np.linspace(0, 100, self.num_thresholds + 2)[1:-1]
        result = {}
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            if len(values) > 1:
                result[j] = np.unique(np.percentile(values, percentiles)).tolist()
        return result

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("MCTS models require validation data.")
        import random

        np.random.seed(self.random_state)
        random.seed(self.random_state)
        MCTS, ADTree = _load_mcts_modules()
        self._ADTree = ADTree
        replacement = _ConstantFitter if self.constant_leaf else MCTS.FastStepFitter
        with _temporary_fast_step(MCTS, replacement), _temporary_depth_limit(
            MCTS, self.max_depth
        ):
            state, reward = MCTS.run_mcts(
                X_train_in=np.asarray(X, dtype=np.float64),
                y_train_in=np.asarray(y, dtype=np.int8),
                X_val_in=np.asarray(X_val, dtype=np.float64),
                y_val_in=np.asarray(y_val, dtype=np.int8),
                feature_names=[f"X{i + 1}" for i in range(X.shape[1])],
                feature_thresholds=self._thresholds(X),
                max_depth=self.max_depth,
                max_iters=self.max_iters,
                exploration_c=1.4,
                min_samples_leaf=self.min_samples_leaf,
                shape_fitter="step",
                n_knots_step=self.n_knots_step,
                complexity_penalty=self.complexity_penalty,
                patience=max(50, self.max_iters // 3),
                warmup_iters=min(50, self.max_iters // 4),
                log_frequency=max(50, self.max_iters),
                reward_metric="logloss",
                max_split_nodes=self.max_split_nodes,
                use_pw=1,
                c_pw=20,
                alpha_pw=0.4,
                eps_deepen=0.35,
                label_mode="binary",
                include_intermediate_shapes=False,
            )
        if state is None:
            raise RuntimeError("MCTS did not produce a valid model.")
        self.state_ = state
        self.tree_ = state.tree
        self.validation_reward_ = float(reward)
        return self

    def decision_function(self, X):
        return self.tree_.decision_function(np.asarray(X))

    def predict_proba(self, X):
        return self.tree_.predict_proba(np.asarray(X))

    def _counts(self):
        ShapeNode, SplitNode = self._ADTree.ShapeNode, self._ADTree.SplitNode
        splits = []
        shapes = []

        def walk(node):
            if node is None:
                return
            if isinstance(node, ShapeNode):
                shapes.append(node)
                for child in node.children:
                    walk(child)
            elif isinstance(node, SplitNode):
                splits.append(node)
                walk(node.true_child)
                walk(node.false_child)

        walk(self.tree_.root)
        return splits, shapes

    def _decisions_for_row(self, row):
        ShapeNode, SplitNode = self._ADTree.ShapeNode, self._ADTree.SplitNode

        def walk(node):
            if node is None:
                return 0
            if isinstance(node, ShapeNode):
                return sum(walk(child) for child in node.children)
            if isinstance(node, SplitNode):
                child = node.true_child if row[node.feature_idx] <= node.threshold else node.false_child
                return 1 + walk(child)
            return 0

        return walk(self.tree_.root)

    @staticmethod
    def _basis_count(model):
        def as_numpy(value):
            if hasattr(value, "get"):
                value = value.get()
            return np.asarray(value)

        if isinstance(model, _ConstantModel) or getattr(model, "feature_index", None) == -1:
            return 0
        if hasattr(model, "coeffs") and model.coeffs is not None:
            return int(np.count_nonzero(np.abs(as_numpy(model.coeffs)) > 1e-10))
        if hasattr(model, "coeffs_list"):
            return int(
                sum(np.count_nonzero(np.abs(as_numpy(c)) > 1e-10) for c in model.coeffs_list)
            )
        return 1

    def complexity(self, X=None):
        splits, shapes = self._counts()
        terminal = [node for node in shapes if node.is_terminal()]
        nonconstant = [
            node for node in terminal if getattr(node, "feature_idx", -1) != -1
        ]
        basis = sum(self._basis_count(node.shape_model) for node in nonconstant)
        variables = {
            node.feature_idx for node in nonconstant if getattr(node, "feature_idx", -1) >= 0
        }
        variables.update(split.feature_idx for split in splits)
        max_depth = max((node.depth for node in splits + shapes), default=0)
        if X is None or len(X) == 0:
            mean_decisions = float("nan")
        else:
            mean_decisions = float(
                np.mean([self._decisions_for_row(row) for row in np.asarray(X)])
            )
        return {
            "split_nodes": int(len(splits)),
            "max_depth": int(max_depth),
            "shape_functions": int(len(nonconstant)),
            "variables_used": int(len(variables)),
            "active_shape_basis": int(basis),
            "mean_decisions": mean_decisions,
            "mean_active_shapes": float(len(terminal)),
            "total_active_components": int(len(splits) + basis),
        }


def build_model(name: str, params: dict, random_state: int):
    params = dict(params)
    params["random_state"] = random_state
    if name == "logistic":
        return LogisticAdapter(**params)
    if name == "fsg":
        return FSGAdapter(**params)
    if name == "ebm_main":
        return EBMAdapter(interactions=0, **params)
    if name == "ebm_interact":
        return EBMAdapter(**params)
    if name == "adtree":
        return MCTSModelAdapter(constant_leaf=True, **params)
    if name == "shadetree":
        return MCTSModelAdapter(constant_leaf=False, **params)
    raise ValueError(f"Unknown model {name!r}.")
