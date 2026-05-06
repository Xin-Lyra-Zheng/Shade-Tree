# MCTS.py (classic MCTS: one expansion per iteration + immediate rollout)
import numpy as np
import math
from collections import defaultdict
from copy import deepcopy
import random
import logging
from tqdm import tqdm

from ADTree import ADTreeClassifier, SplitNode, ShapeNode
from util import evaluate_model, find_best_threshold
from shape_fitter import FastStepFitter, EBMShapeFitter, FastSparseShapeFitter

# Global switches
INCLUDE_INTERMEDIATE_SHAPES = False
REFINE_STEP_WITH_FASTSPARSE = False

# --- Helper functions ---
def _clip01(p):
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.clip(p, 1e-12, 1 - 1e-12)

def _calculate_xgboost_gains_fast(state, actions, lambda_reg=1.0):
    """
    Vectorized calculation of XGBoost split gains for a list of actions.
    Groups actions by parent_id to minimize data slicing overhead.
    """
    # Group actions by parent_id and feature_idx
    workload = defaultdict(lambda: defaultdict(list))
    action_map = {}

    for action in actions:
        _, parent_id, feat_idx, thresh = action
        workload[parent_id][feat_idx].append(thresh)
        action_map[(parent_id, feat_idx, thresh)] = action

    gains = {}

    for parent_id, feat_dict in workload.items():
        node_idxs = np.array(state.data_indices.get(parent_id, []), dtype=int)
        if len(node_idxs) == 0:
            continue

        # state.residuals = (y - p), state.weights = p(1-p)
        r_node = state.residuals[node_idxs]
        w_node = state.weights[node_idxs]

        G_total = np.sum(r_node)
        H_total = np.sum(w_node)

        # Parent score (no split)
        score_parent = (G_total**2) / (H_total + lambda_reg)

        for feat_idx, thresholds in feat_dict.items():
            # Get feature values from global X_train
            x_feat = X_train[node_idxs, feat_idx]

            # Sort for cumulative sum
            sort_order = np.argsort(x_feat)
            x_sorted = x_feat[sort_order]
            r_sorted = r_node[sort_order]
            w_sorted = w_node[sort_order]

            # Cumulative sums (scan)
            G_L_vec = np.cumsum(r_sorted)
            H_L_vec = np.cumsum(w_sorted)

            G_R_vec = G_total - G_L_vec
            H_R_vec = H_total - H_L_vec

            denom_L = H_L_vec + lambda_reg
            denom_R = H_R_vec + lambda_reg

            # Gain = 0.5 * [GL^2/HL + GR^2/HR - GT^2/HT] (omitting 0.5 constant)
            gain_vec = (G_L_vec**2 / denom_L) + (G_R_vec**2 / denom_R) - score_parent

            # Map specific thresholds to indices
            # np.searchsorted(side='right') gives index i where a[:i] <= thresh
            thresholds_arr = np.array(thresholds)
            split_indices = np.searchsorted(x_sorted, thresholds_arr, side='right') - 1
            split_indices = np.clip(split_indices, 0, len(x_sorted) - 1)

            calculated_gains = gain_vec[split_indices]

            for thr, g in zip(thresholds, calculated_gains):
                act = action_map[(parent_id, feat_idx, thr)]
                gains[act] = g

    return gains

# --- Global Config & Data (set by run_mcts) ---
X_train, y_train, X_val, y_val = [None] * 4
FEATURES, FEATURE_THRESHOLDS = [], {}
SEARCH_SHAPE_FITTER = None
COMPLEXITY_PENALTY = 0.0
logger = logging.getLogger('MCTS_ADTree')
MIN_SAMPLES_LEAF = None
LAST_BEST_STATE = None
# --- Diagnostics histories for visualization ---
REWARD_HISTORY = []  # per-iteration scalar reward
ROOT_STATS_HISTORY = []  # per-iteration snapshot of root action stats
# --- Globals for Pruning Logic ---
USE_XGBOOST_PRUNING = False
XGBOOST_IDEAL_RESIDUALS_SQ = None
PRUNING_MODE = 'proportion'
PRUNING_THRESHOLD = 1.2
# --- Globals for Ranking Logic ---
USE_XGB_RANKING = False
EPS_RANKING = 0.2
LAMBDA_RANKING = 1.0
SEARCH_SHAPE_FITTER = None

FITTER_COMPLEX = None
FITTER_SIMPLE = None
ADAPTIVE_THRESHOLD = 50

# --- Globals for Power-UCT ---
USE_POWER_UCT = False
POWER_P = 4.0
POWER_SUM = defaultdict(float)

# --- Label handling ---
TRAIN_LABEL_MODE = 'binary'
TRAIN_TARGET_VALUES = None
VAL_TARGET_VALUES = None
_WARNED_CONT_METRIC = False
CUSTOM_SAMPLE_WEIGHTS = None
# -------- Small utils for logging --------
def _count_split_nodes(node):
    if node is None: return 0
    if isinstance(node, SplitNode):
        return 1 + _count_split_nodes(node.true_child) + _count_split_nodes(node.false_child)
    if isinstance(node, ShapeNode):
        return sum(_count_split_nodes(c) for c in node.children)
    return 0

def _count_terminal_shape_nodes(node):
    if node is None: return 0
    if isinstance(node, ShapeNode):
        if not node.children: return 1
        return sum(_count_terminal_shape_nodes(c) for c in node.children)
    if isinstance(node, SplitNode):
        return _count_terminal_shape_nodes(node.true_child) + _count_terminal_shape_nodes(node.false_child)
    return 0

def _build_node_map_recursive(node, node_map):
    if node is None: return
    node_map[node.id] = node
    if isinstance(node, SplitNode):
        _build_node_map_recursive(node.true_child, node_map)
        _build_node_map_recursive(node.false_child, node_map)
    elif isinstance(node, ShapeNode):
        for child in node.children:
            _build_node_map_recursive(child, node_map)

def _convert_targets(y, mode):
    if y is None:
        return None
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if mode == 'binary':
        return (arr == 1).astype(np.float64)
    if mode == 'continuous':
        return arr
    raise ValueError(f"Unsupported label mode: {mode}")

def _sigmoid_scores(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return _clip01(1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20))))

def _compute_residuals_and_weights(scores, targets):
    if targets is None:
        raise RuntimeError("Training targets are not initialized.")
    if TRAIN_LABEL_MODE == 'binary':
        probas = _sigmoid_scores(scores)
        residuals = targets - probas
        weights = probas * (1.0 - probas)
        return residuals, weights
    residuals = targets - np.asarray(scores, dtype=np.float64).reshape(-1)
    global CUSTOM_SAMPLE_WEIGHTS
    if CUSTOM_SAMPLE_WEIGHTS is not None:
        weights = np.asarray(CUSTOM_SAMPLE_WEIGHTS, dtype=np.float64).reshape(-1)
        if weights.shape[0] != residuals.shape[0]:
            raise ValueError("CUSTOM_SAMPLE_WEIGHTS length mismatch with residuals.")
    else:
        weights = np.ones_like(residuals, dtype=np.float64)
    return residuals, weights

def _fmt_action(a, feature_names):
    atype, pid, fidx, thr = a
    fn = feature_names[fidx] if (fidx is not None and fidx >= 0) else "?"
    return f"{atype}@parent={pid} | {fn} <= {thr:.6g}"

class SearchState:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.tree = deepcopy(self.tree, memo)
        result.data_indices = deepcopy(self.data_indices, memo)
        result.node_map = {}
        _build_node_map_recursive(result.tree.root, result.node_map)
        result.scores = self.scores.copy()
        result.residuals = self.residuals.copy()
        result.weights = self.weights.copy()
        result._cached_hash = None
        result._cached_actions = None
        return result

    def __init__(self, tree, data_indices=None, scores=None):
        self.tree = tree
        self._cached_hash = None
        self._cached_actions = None
        self.node_map = {}
        _build_node_map_recursive(self.tree.root, self.node_map)

        if data_indices is None:
            self.data_indices = {self.tree.root.id: list(range(X_train.shape[0]))}
        else:
            self.data_indices = data_indices

        if scores is None:
            self.scores = np.ascontiguousarray(self.tree.decision_function(X_train), dtype=np.float64)
        else:
            self.scores = np.ascontiguousarray(scores, dtype=np.float64)

        train_targets = TRAIN_TARGET_VALUES
        residuals, weights = _compute_residuals_and_weights(self.scores, train_targets)
        self.residuals = np.ascontiguousarray(residuals, dtype=np.float64)
        self.weights = np.ascontiguousarray(weights, dtype=np.float64)

    # ---- Structural hash (semantic, independent of node.id) ----
    def hash(self):
        if self._cached_hash: return self._cached_hash
        def _serialize(node):
            if node is None: return "N"
            if isinstance(node, ShapeNode):
                res = f"(ShapeNode:f({node.feature_name})"
                sorted_children = sorted(node.children, key=lambda c: (c.feature_idx, c.threshold))
                for c in sorted_children:
                    res += _serialize(c)
                res += ")"
                return res
            elif isinstance(node, SplitNode):
                res = f"(SplitNode:{node.feature_idx}<={node.threshold:.10f}"
                res += _serialize(node.true_child)
                res += _serialize(node.false_child)
                res += ")"
                return res
            else:
                return "(N)"
        self._cached_hash = _serialize(self.tree.root)
        return self._cached_hash

    def get_all_shape_nodes(self):
        return [node for node in self.node_map.values() if isinstance(node, ShapeNode)]

    def available_actions(self, action_cache, max_splits=float('inf')):
        if self._cached_actions is not None: return self._cached_actions

        state_key = self.hash()
        if state_key in action_cache:
            self._cached_actions = action_cache[state_key]
            return self._cached_actions

        actions = []
        current_splits = _count_split_nodes(self.tree.root)
        if current_splits >= max_splits:
            self._cached_actions = []
            return []

        pruned_nodes = 0
        total_candidates = 0
        total_valid_thresholds = 0

        for node in self.get_all_shape_nodes():
            parent_id = node.id
            node_idxs = np.array(self.data_indices.get(parent_id, []), dtype=int)
            if len(node_idxs) < MIN_SAMPLES_LEAF:
                continue

            if USE_XGBOOST_PRUNING and XGBOOST_IDEAL_RESIDUALS_SQ is not None:
                ideal_ssr = np.sum(XGBOOST_IDEAL_RESIDUALS_SQ[node_idxs])
                true_ssr = np.sum(self.residuals[node_idxs] ** 2)

                prune_node = False
                if PRUNING_MODE == 'proportion' and ideal_ssr > 1e-9 and (true_ssr / ideal_ssr) < PRUNING_THRESHOLD:
                    prune_node = True
                elif PRUNING_MODE == 'difference' and (true_ssr - ideal_ssr) < PRUNING_THRESHOLD:
                    prune_node = True

                if prune_node:
                    pruned_nodes += 1
                    continue

            node_data = X_train[node_idxs, :]
            for feat_idx in FEATURES:
                thresholds = np.array(FEATURE_THRESHOLDS.get(feat_idx, []))
                if thresholds.size == 0:
                    continue
                total_candidates += thresholds.size

                feature_values = node_data[:, feat_idx]
                comparison_matrix = feature_values[:, np.newaxis] <= thresholds[np.newaxis, :]
                left_counts = np.sum(comparison_matrix, axis=0)
                right_counts = len(node_idxs) - left_counts

                valid_mask = (left_counts >= MIN_SAMPLES_LEAF) & (right_counts >= MIN_SAMPLES_LEAF)
                total_valid_thresholds += int(np.sum(valid_mask))
                for thresh in thresholds[valid_mask]:
                    actions.append(("add_split", parent_id, feat_idx, float(thresh)))

        logger.debug(f"[ACTIONS] splits_now={current_splits} | parents_pruned={pruned_nodes} | "
                     f"cand_thresholds={total_candidates} | valid_thresholds={total_valid_thresholds} | "
                     f"actions={len(actions)}")

        self._cached_actions = actions
        action_cache[state_key] = actions
        return actions

    def apply(self, action):
        """Apply an 'add_split' action with correct contextual residual logic."""
        action_type, parent_id, feat_idx, thresh = action

        new_state = deepcopy(self)
        parent_shape_node = new_state.node_map.get(parent_id)
        if not isinstance(parent_shape_node, ShapeNode):
            logger.error(f"Could not find parent ShapeNode with ID {parent_id}.")
            return None

        parent_idxs = np.array(new_state.data_indices[parent_id], dtype=int)
        parent_contribution = np.zeros_like(self.scores)
        if parent_idxs.size > 0 and (INCLUDE_INTERMEDIATE_SHAPES or parent_shape_node.is_terminal()):
            parent_contribution[parent_idxs] = parent_shape_node.predict_score(X_train[parent_idxs])

        if INCLUDE_INTERMEDIATE_SHAPES or parent_shape_node.is_terminal():
            scores_for_residual_calc = self.scores - parent_contribution
        else:
            scores_for_residual_calc = self.scores

        train_targets = TRAIN_TARGET_VALUES
        if TRAIN_LABEL_MODE == 'binary':
            probas_for_fitting = _sigmoid_scores(scores_for_residual_calc)
            residuals_for_fitting = np.ascontiguousarray(train_targets - probas_for_fitting, dtype=np.float64)
            weights_for_fitting = np.ascontiguousarray(probas_for_fitting * (1 - probas_for_fitting), dtype=np.float64)
        else:
            probas_for_fitting = None
            residuals_for_fitting = np.ascontiguousarray(train_targets - scores_for_residual_calc, dtype=np.float64)
            weights_for_fitting = np.ones_like(residuals_for_fitting, dtype=np.float64)

        feature_name = self.tree._feature_names[feat_idx]
        new_split_node = SplitNode(
            parent=parent_shape_node, depth=parent_shape_node.depth + 1,
            node_id=new_state.tree._get_next_node_id(),
            feature_idx=feat_idx, threshold=thresh, feature_name=feature_name
        )
        parent_shape_node.children.append(new_split_node)
        new_state.node_map[new_split_node.id] = new_split_node

        # Split parent indices
        mask = X_train[parent_idxs, feat_idx] <= thresh
        true_indices = parent_idxs[mask].tolist()
        false_indices = parent_idxs[~mask].tolist()

        if INCLUDE_INTERMEDIATE_SHAPES:
            new_total_scores = self.scores.copy()
        else:
            new_total_scores = scores_for_residual_calc.copy()

        for is_true_branch, indices in [(True, true_indices), (False, false_indices)]:
            n_samples = len(indices)

            if n_samples < ADAPTIVE_THRESHOLD:
                target_fitter = FITTER_SIMPLE
            else:
                target_fitter = FITTER_COMPLEX

            fitted_model = target_fitter.fit(indices, residuals_for_fitting, weights_for_fitting)
            if fitted_model is None:
                fitted_model = FITTER_SIMPLE.fit([], residuals_for_fitting, weights_for_fitting)

            best_feature_idx = getattr(fitted_model, 'feature_index', -1)
            f_name = "Constant" if best_feature_idx == -1 else self.tree._feature_names[best_feature_idx]

            new_shape_node = ShapeNode(
                parent=new_split_node, depth=new_split_node.depth + 1,
                node_id=new_state.tree._get_next_node_id(),
                feature_idx=best_feature_idx, feature_name=f_name, shape_model=fitted_model
            )
            if is_true_branch:
                new_split_node.true_child = new_shape_node
            else:
                new_split_node.false_child = new_shape_node

            new_state.node_map[new_shape_node.id] = new_shape_node
            new_state.data_indices[new_shape_node.id] = indices

            if len(indices) > 0:
                contrib = new_shape_node.predict_score(X_train[indices])
                new_total_scores[indices] += contrib
            else:
                logger.debug(f"[LEAF] branch={'T' if is_true_branch else 'F'} | n=0 (skipped fit)")

        # ---------- FAST LOCAL UPDATE: only update affected samples ----------
        # ---------- FAST LOCAL UPDATE ON AFFECTED SAMPLES ----------
        new_state.scores = new_total_scores

        # Only update residuals/weights for samples under parent_id
        affected = parent_idxs  # all samples that used parent's shape node

        scores_aff = new_total_scores[affected]
        if TRAIN_LABEL_MODE == 'binary':
            probas_aff = _sigmoid_scores(scores_aff)
            res_aff = train_targets[affected] - probas_aff
            w_aff = probas_aff * (1.0 - probas_aff)
        else:
            res_aff = train_targets[affected] - scores_aff
            w_aff = np.ones_like(res_aff, dtype=np.float64)

        # Copy old residuals/weights
        new_state.residuals = self.residuals.copy()
        new_state.weights = self.weights.copy()

        # Only rewrite affected portions
        new_state.residuals[affected] = res_aff
        new_state.weights[affected] = w_aff

        return new_state

def fast_accuracy(y_true_01, y_pred_01):
    return np.mean(y_true_01 == y_pred_01)
def fast_logloss(y_true_01, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return - np.mean(y_true_01 * np.log(p) + (1 - y_true_01) * np.log(1 - p))
def fast_auc(y_true_01, p):
    # Need both classes
    n1 = np.sum(y_true_01 == 1)
    n0 = len(y_true_01) - n1
    if n1 == 0 or n0 == 0:
        return 0.5

    # rank the scores
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(p)) + 1  # ranks start from 1

    # sum ranks of positive class
    sum_ranks_pos = np.sum(ranks[y_true_01 == 1])

    # AUC formula
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)

def evaluate_state_reward(state, X_eval, y_eval, reward_metric='auc', fast=False):
    """
    - fast=True: use hand-coded metrics (very fast), only compute reward metric
    - fast=False: use full slow metrics via evaluate_model (sklearn)
    """
    if X_eval is None or y_eval is None:
        return -np.inf, None

    num_splits = _count_split_nodes(state.tree.root)
    penalty = COMPLEXITY_PENALTY * num_splits

    if TRAIN_LABEL_MODE == 'binary':
        pred_proba = state.tree.predict_proba(X_eval)[:, 1]
        y_true_01 = (y_eval == 1).astype(int)

        if fast:
            if reward_metric == 'accuracy':
                y_pred = (pred_proba > 0.5).astype(int)
                perf = fast_accuracy(y_true_01, y_pred)
            elif reward_metric == 'auc':
                perf = fast_auc(y_true_01, pred_proba)
            elif reward_metric == 'logloss':
                perf = 1.0 - fast_logloss(y_true_01, pred_proba)
            elif reward_metric == 'accauc':
                y_pred = (pred_proba > 0.5).astype(int)
                acc = fast_accuracy(y_true_01, y_pred)
                auc = fast_auc(y_true_01, pred_proba)
                perf = 0.5 * (acc + auc)
            else:
                y_pred = (pred_proba > 0.5).astype(int)
                perf = fast_accuracy(y_true_01, y_pred)
            return perf - penalty, None

        y_pred_01 = (pred_proba > 0.5).astype(int)
        metrics = evaluate_model(y_eval, pred_proba, y_pred_01)

        if reward_metric == 'accauc':
            perf = 0.5 * (metrics['auc'] + metrics['accuracy'])
        elif reward_metric == 'auc':
            perf = metrics['auc']
        elif reward_metric == 'accuracy':
            perf = metrics['accuracy']
        else:
            perf = 1 - metrics['logloss']

        return perf - penalty, metrics

    # Continuous label mode -> use negative MSE as reward (with guardrails on requested metric)
    global _WARNED_CONT_METRIC
    if reward_metric not in {'mse', 'neg_mse'} and not _WARNED_CONT_METRIC:
        logger.warning(
            "TRAIN_LABEL_MODE='continuous' ignores reward_metric='%s'; falling back to negative MSE.",
            reward_metric
        )
        _WARNED_CONT_METRIC = True

    preds = state.tree.decision_function(X_eval)
    y_true = np.asarray(y_eval, dtype=np.float64).reshape(-1)
    errors = y_true - preds
    mse = float(np.mean(errors ** 2)) if len(errors) > 0 else np.inf
    perf = -mse
    if not fast:
        logger.info("[Regression] evaluate_state_reward -> reward_metric='%s', MSE=%.6f", reward_metric, mse)
    metrics = {'mse': mse}
    return perf - penalty, (None if fast else metrics)

def _rank_actions_with_xgb(state, actions):
    if not USE_XGB_RANKING or not actions:
        return None, None

    gains = _calculate_xgboost_gains_fast(state, actions, lambda_reg=LAMBDA_RANKING)
    ranked = sorted(actions, key=lambda a: gains.get(a, float('-inf')), reverse=True)
    return ranked, gains

def run_mcts(
    X_train_in, y_train_in, X_val_in, y_val_in,
    feature_names,
    feature_thresholds,
    max_depth=3, max_iters=500, exploration_c=1.4, min_samples_leaf=10,
    shape_fitter='step', n_knots_step=10,
    refine_step_with_fastsparse=False,
    complexity_penalty=0.0, patience=500, warmup_iters=100,
    log_frequency=50, reward_metric='auc', max_split_nodes=8,
    ebm_interactions=0, ebm_max_bins=32, ebm_outer_bags=4,
    use_xgboost_pruning=False,
    xgboost_ideal_residuals_sq=None,
    pruning_mode='proportion',
    pruning_threshold=1.2,
    use_pw=1,
    c_pw=100,
    alpha_pw=0.5,
    eps_deepen=0.35,
    X_test_in=None,
    y_test_in=None,
    use_xgb_ranking=False,
    eps_ranking=0.2,
    lambda_ranking=1.0,
    use_power_uct=False,
    power_p=4.0,
    label_mode='binary',
    include_intermediate_shapes=False,
    residual_sample_weights=None
):
    """
    Max-Backup MCTS (Best-Arm Identification / Simple Regret Optimization):
      * Q(s,a) stores the MAXIMUM reward observed, not the sum/mean.
      * Backprop updates Q = max(Q, reward).
      * Selection uses Q directly as the exploitation term.
      * Uses a proxy step fitter during search for sensitivity,
        then reverts to the requested fitter (e.g., Step) during refinement.

    """
    # ---------- Helpers ----------
    def _fmt_action_local(a):
        if a is None:
            return "<None>"
        a_type, parent_id, feat_idx, thresh = a
        fname = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"X{feat_idx}"
        return f"{a_type}@parent={parent_id} | {fname} <= {thresh}"

    def _count_terminals(node):
        if node is None: return 0
        if isinstance(node, ShapeNode):
            if not node.children:
                return 1
            return sum(_count_terminals(ch) for ch in node.children)
        elif isinstance(node, SplitNode):
            return _count_terminals(node.true_child) + _count_terminals(node.false_child)
        return 0

    # ---------- Globals setup ----------
    global X_train, y_train, X_val, y_val, FEATURES, FEATURE_THRESHOLDS, SEARCH_SHAPE_FITTER
    global COMPLEXITY_PENALTY, MIN_SAMPLES_LEAF, LAST_BEST_STATE
    global USE_XGBOOST_PRUNING, XGBOOST_IDEAL_RESIDUALS_SQ, PRUNING_MODE, PRUNING_THRESHOLD
    global REWARD_HISTORY, ROOT_STATS_HISTORY
    global FITTER_COMPLEX, FITTER_SIMPLE, ADAPTIVE_THRESHOLD
    global USE_XGB_RANKING, EPS_RANKING, LAMBDA_RANKING
    global USE_POWER_UCT, POWER_P, POWER_SUM
    global INCLUDE_INTERMEDIATE_SHAPES, REFINE_STEP_WITH_FASTSPARSE
    global TRAIN_LABEL_MODE, TRAIN_TARGET_VALUES, VAL_TARGET_VALUES, _WARNED_CONT_METRIC, CUSTOM_SAMPLE_WEIGHTS

    # Reset history buffers for visualization
    REWARD_HISTORY = []
    ROOT_STATS_HISTORY = []

    X_train, y_train, X_val, y_val = X_train_in, y_train_in, X_val_in, y_val_in
    FEATURES = list(range(X_train.shape[1]))
    FEATURE_THRESHOLDS = feature_thresholds
    COMPLEXITY_PENALTY = float(complexity_penalty)
    MIN_SAMPLES_LEAF = int(min_samples_leaf)
    TRAIN_LABEL_MODE = label_mode
    TRAIN_TARGET_VALUES = _convert_targets(y_train, TRAIN_LABEL_MODE)
    VAL_TARGET_VALUES = _convert_targets(y_val, TRAIN_LABEL_MODE)
    _WARNED_CONT_METRIC = False
    CUSTOM_SAMPLE_WEIGHTS = None
    if residual_sample_weights is not None:
        rsw = np.asarray(residual_sample_weights, dtype=np.float64).reshape(-1)
        if rsw.shape[0] != X_train.shape[0]:
            raise ValueError("residual_sample_weights must match X_train length.")
        CUSTOM_SAMPLE_WEIGHTS = rsw

    USE_XGBOOST_PRUNING = bool(use_xgboost_pruning)
    XGBOOST_IDEAL_RESIDUALS_SQ = xgboost_ideal_residuals_sq
    PRUNING_MODE = pruning_mode
    PRUNING_THRESHOLD = float(pruning_threshold)

    # ranking 全局开关
    USE_XGB_RANKING = bool(use_xgb_ranking)
    EPS_RANKING = float(eps_ranking)
    LAMBDA_RANKING = float(lambda_ranking)

    USE_POWER_UCT = bool(use_power_uct)
    POWER_P = float(power_p)
    POWER_SUM = defaultdict(float)
    INCLUDE_INTERMEDIATE_SHAPES = bool(include_intermediate_shapes)
    REFINE_STEP_WITH_FASTSPARSE = bool(refine_step_with_fastsparse)

    if USE_XGBOOST_PRUNING and XGBOOST_IDEAL_RESIDUALS_SQ is not None:
        logger.info(f"MCTS Search running with XGBoost pruning enabled (mode: {PRUNING_MODE}, threshold: {PRUNING_THRESHOLD}).")
    else:
        USE_XGBOOST_PRUNING = False

    # --- PROXY FITTER STRATEGY ---
    logger.info("Initializing proxy step fitter for MCTS search (knots=16).")
    FITTER_COMPLEX = FastStepFitter(
        X_train_in,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_internal_knots=3
    )

    FITTER_SIMPLE = FastStepFitter(
        X_train_in,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_internal_knots=3
    )

    ADAPTIVE_THRESHOLD = 50

    # ---------- PW & ε-deepening parameters ----------
    USE_PW = bool(use_pw)
    C_PW = float(c_pw)
    ALPHA_PW = float(alpha_pw)
    EPS_DEEPEN = float(eps_deepen)

    # ---------- Initialization ----------
    init_tree = ADTreeClassifier(include_intermediate_shapes=INCLUDE_INTERMEDIATE_SHAPES)
    init_tree.fit(X_train, y_train, feature_names)
    init_state = SearchState(init_tree)
    root_hash = init_state.hash()

    # Max-Backup specific initialization
    N = defaultdict(int)  # visit counts
    Q = {}                # max reward
    POWER_SUM = defaultdict(float)
    children = defaultdict(dict)      # node_hash -> {action -> child_hash or None}
    ACTION_CACHE = {}
    STATE_CACHE = {}                  # state_hash -> SearchState
    STATE_CACHE[init_state.hash()] = init_state

    # State visit counter for Progressive Widening
    STATE_VISITS = defaultdict(int)

    best_state, best_val_reward = None, -np.inf
    best_val_metrics = {}
    iters_without_improvement = 0

    debug_trace = logger.isEnabledFor(logging.DEBUG)
    trace_top_k = 1

    # ---------- Main loop ----------
    for it in tqdm(range(1, max_iters + 1), desc="MCTS Search", unit="iter"):
        path = []
        state = init_state
        expansion_steps = 0
        selection_steps = 0
        stop_reason = ""

        # ---- Selection Phase ----
        while True:
            node_key = state.hash()
            STATE_VISITS[node_key] += 1

            # Determine available actions (with caching)
            all_actions = state.available_actions(ACTION_CACHE, max_splits=max_split_nodes)

            # Terminal node check
            if not all_actions:
                stop_reason = "no_actions"
                break

            expanded_dict = children.get(node_key, {})  # { action -> child_hash or None }
            valid_expanded = [a for a, h in expanded_dict.items() if h is not None]
            num_expanded = len(valid_expanded)
            unexpanded = [a for a in all_actions if a not in expanded_dict]

            # Progressive Widening Logic
            if USE_PW:
                limit = max(1, int(math.floor(C_PW * (STATE_VISITS[node_key] ** ALPHA_PW))))
                can_expand_here = (len(unexpanded) > 0) and (num_expanded < limit)
            else:
                limit = None
                can_expand_here = len(unexpanded) > 0

            # epsilon-Deepening at Root Logic
            is_root = (node_key == root_hash)
            force_deepen_from_root = False
            if is_root and can_expand_here and len(valid_expanded) > 0:
                if random.random() < EPS_DEEPEN:
                    force_deepen_from_root = True
                    if debug_trace:
                        if USE_PW:
                            logger.debug(f"[ROOT-DEEPEN] Skip expanding at root this turn (limit={limit}, expanded={num_expanded}).")
                        else:
                            logger.debug(f"[ROOT-DEEPEN] Skip expanding at root this turn (expanded={num_expanded}).")

            # -------- EXPANSION --------
            if can_expand_here and not force_deepen_from_root:
                if USE_XGB_RANKING:
                    ranked_unexp, _ = _rank_actions_with_xgb(state, unexpanded)
                    if ranked_unexp is None or len(ranked_unexp) == 0:
                        action = random.choice(unexpanded)
                    else:
                        if random.random() < EPS_RANKING:
                            action = random.choice(unexpanded)
                        else:
                            action = ranked_unexp[0]
                else:
                    action = random.choice(unexpanded)

                if debug_trace:
                    if USE_PW:
                        logger.debug("[EXPAND-NEW] %s | PW(limit=%d, expanded=%d, visits=%d)",
                                     _fmt_action_local(action), limit, num_expanded, STATE_VISITS[node_key])
                    else:
                        logger.debug("[EXPAND-NEW] %s | PW(off, expanded=%d, visits=%d)",
                                     _fmt_action_local(action), num_expanded, STATE_VISITS[node_key])
                path.append((node_key, action))

                new_state = state.apply(action)
                expansion_steps += 1

                if new_state is not None:
                    child_hash = new_state.hash()
                    children.setdefault(node_key, {})[action] = child_hash
                    STATE_CACHE[child_hash] = new_state
                    state = new_state
                    stop_reason = "expanded_once"
                    break
                else:
                    # Mark invalid so it won't be chosen again
                    children.setdefault(node_key, {})[action] = None
                    stop_reason = "expand_failed"
                    break

            # -------- SELECTION (Max-UCB) --------
            if not valid_expanded:
                # Safety fallback: force expand if no valid children but unexpanded exist
                if len(unexpanded) > 0:
                    if USE_XGB_RANKING:
                        ranked_unexp, _ = _rank_actions_with_xgb(state, unexpanded)
                        if ranked_unexp is None or len(ranked_unexp) == 0 or random.random() < EPS_RANKING:
                            action = random.choice(unexpanded)
                        else:
                            action = ranked_unexp[0]
                    else:
                        action = random.choice(unexpanded)

                    if debug_trace:
                        logger.debug("[EXPAND-FALLBACK] %s", _fmt_action_local(action))
                    path.append((node_key, action))
                    new_state = state.apply(action)
                    expansion_steps += 1
                    if new_state is not None:
                        child_hash = new_state.hash()
                        children.setdefault(node_key, {})[action] = child_hash
                        STATE_CACHE[child_hash] = new_state
                        state = new_state
                        stop_reason = "expanded_once"
                        break
                    else:
                        children.setdefault(node_key, {})[action] = None
                        stop_reason = "expand_failed"
                        break
                stop_reason = "expanded_exhausted"
                break

            # UCB selection: Use Max Q instead of Mean Q
            total_visits = sum(N.get((node_key, a), 0) for a in valid_expanded)
            log_total = math.log(total_visits + 1)
            scored = []

            for a in valid_expanded:
                max_q = Q.get((node_key, a), -float('inf'))
                n = N.get((node_key, a), 0)
                # --- Power-UCT exploitation ---
                if USE_POWER_UCT and n > 0:
                    s_p = POWER_SUM.get((node_key, a), 0.0)
                    if s_p <= 0.0:
                        exploit = max_q
                    else:
                        mean_power = (s_p / n) ** (1.0 / POWER_P)
                        exploit = mean_power
                else:
                    exploit = max_q

                explore = exploration_c * math.sqrt(log_total / (n if n > 0 else 1e-9))

                ucb = exploit + explore
                scored.append((ucb, exploit, n, a))

            scored.sort(key=lambda x: x[0], reverse=True)

            # --- 这里改 tie-breaking：UCB 相同的 action 用 XGB rank + ε 随机 ---
            best_ucb = scored[0][0]
            tied = [(u, a) for (u, _, _, a) in scored if abs(u - best_ucb) < 1e-12]

            if USE_XGB_RANKING and len(tied) > 1:
                candidate_actions = [a for (_, a) in tied]
                ranked_tied, _ = _rank_actions_with_xgb(state, candidate_actions)
                if ranked_tied is None or len(ranked_tied) == 0:
                    action = candidate_actions[0]
                else:
                    if random.random() < EPS_RANKING:
                        action = random.choice(candidate_actions)
                    else:
                        action = ranked_tied[0]
            else:
                action = scored[0][3]

            if debug_trace:
                top_show = scored[:max(1, min(trace_top_k, len(scored)))]
                if USE_PW:
                    logger.debug(
                        "[SELECT] valid=%d | total_visits=%d | PW(limit=%d) | top_ucb: %s",
                        len(valid_expanded), total_visits, limit,
                        " | ".join([f"UCB={u:.4f} MAX={m:.4f} N={n} [{_fmt_action_local(a)}]" for (u, m, n, a) in top_show])
                    )
                else:
                    logger.debug(
                        "[SELECT] valid=%d | total_visits=%d | PW(off) | top_ucb: %s",
                        len(valid_expanded), total_visits,
                        " | ".join([f"UCB={u:.4f} MAX={m:.4f} N={n} [{_fmt_action_local(a)}]" for (u, m, n, a) in top_show])
                    )

            path.append((node_key, action))

            child_hash = expanded_dict[action]
            state = STATE_CACHE.get(child_hash)
            # Re-apply fallback if state somehow missing from cache
            if state is None:
                state = STATE_CACHE[path[-2][0]].apply(action)
                STATE_CACHE[child_hash] = state

            selection_steps += 1

        # ---- Evaluation (Rollout) ----
        if state is None:
            reward, metrics = -np.inf, {}
        else:
            reward, _ = evaluate_state_reward(state, X_val, y_val, reward_metric=reward_metric, fast=True)

        # ---- Backprop (Max-Backup Update) ----
        for node_key, action in reversed(path):
            old_max = Q.get((node_key, action), -float('inf'))
            Q[(node_key, action)] = max(old_max, reward)
            N[(node_key, action)] += 1

        # ---- Recording Diagnostics ----
        try:
            REWARD_HISTORY.append(float(reward))

            # Record snapshot of Root Action Statistics
            root_actions = children.get(root_hash, {})
            snapshot = {
                "iter": it,
                "actions": {}
            }
            for a in root_actions:
                n = N.get((root_hash, a), 0)
                curr_max = Q.get((root_hash, a), -float('inf'))
                if n == 0: curr_max = float('nan')

                snapshot["actions"][_fmt_action(a, feature_names)] = {
                    "N": int(n),
                    "Q": float(curr_max),
                    "mean_Q": float(curr_max),
                }
            ROOT_STATS_HISTORY.append(snapshot)
        except Exception:
            pass

        # ---- Track Best State ----
        if reward > best_val_reward:
            improved = True
            best_val_reward = reward
            best_state = state
            # best_val_metrics = metrics
            iters_without_improvement = 0
        else:
            improved = False
            iters_without_improvement += 1

        # ---- Logging ----
        if debug_trace:
            splits_now = _count_split_nodes(state.tree.root) if state else -1
            # auc = best_val_metrics.get('auc', float('nan'))
            logger.debug("[ITER %d] sel=%d exp=%d stop=%s | reward=%.4f %s | best=%.4f | splits=%d",
                         it, selection_steps, expansion_steps, stop_reason or "n/a",
                         reward, "(improved)" if improved else " ",
                         best_val_reward, splits_now)

        if it % log_frequency == 0 or it == max_iters:
            # best_auc = best_val_metrics.get('auc', 0.0)
            logger.info(f"Iter [{it}/{max_iters}] - Best Val Reward: {best_val_reward:.4f}")

        if it > warmup_iters and iters_without_improvement >= patience:
            logger.info(f"Early stopping triggered at iteration {it} after {patience} iterations without improvement.")
            break

    # ---------- Final Refinement Phase ----------
    logger.info("MCTS search complete. Starting final model refinement phase with safety checks...")

    if best_state is None:
        logger.error("Cannot perform refinement as no best state was found during search.")
        LAST_BEST_STATE = None
        CUSTOM_SAMPLE_WEIGHTS = None
        return None, -np.inf

    final_refined_state = deepcopy(best_state)
    baseline_reward, _ = evaluate_state_reward(final_refined_state, X_val, y_val, reward_metric)
    logger.info(f"Baseline validation reward before refinement: {baseline_reward:.4f}")

    # Determine Final Fitter based on Config (e.g., revert to Step if requested)
    if shape_fitter == 'step' and REFINE_STEP_WITH_FASTSPARSE:
        try:
            final_quality_fitter = FastSparseShapeFitter(
                X_train_in,
                min_samples_leaf=MIN_SAMPLES_LEAF
            )
        except Exception as e:
            logger.warning(f"FastSparse refinement initialization failed; fallback to FastStep. Error: {e}")
            final_quality_fitter = FastStepFitter(
                X_train_in,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                n_internal_knots=n_knots_step,
            )
    else:
        final_quality_fitter = FastStepFitter(
            X_train_in,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_internal_knots=n_knots_step,
        )

    terminal_nodes = [n for n in final_refined_state.get_all_shape_nodes() if n.is_terminal()]
    logger.info(f"Found {len(terminal_nodes)} terminal nodes to attempt to refine.")

    current_best_reward = baseline_reward

    for i, node in enumerate(terminal_nodes):
        original_simple_model = node.shape_model
        node_indices = final_refined_state.data_indices.get(node.id, [])
        if not node_indices:
            continue

        parent_contribution = node.predict_score(X_train)
        scores_of_other_leaves = final_refined_state.scores - parent_contribution
        train_targets = TRAIN_TARGET_VALUES
        if TRAIN_LABEL_MODE == 'binary':
            probas_from_other_leaves = _sigmoid_scores(scores_of_other_leaves)
            residuals_for_refinement = np.ascontiguousarray(train_targets - probas_from_other_leaves, dtype=np.float64)
            weights_for_refinement = np.ascontiguousarray(probas_from_other_leaves * (1 - probas_from_other_leaves), dtype=np.float64)
        else:
            probas_from_other_leaves = None
            residuals_for_refinement = np.ascontiguousarray(train_targets - scores_of_other_leaves, dtype=np.float64)
            weights_for_refinement = np.ones_like(residuals_for_refinement, dtype=np.float64)

        new_complex_model = final_quality_fitter.fit(node_indices, residuals_for_refinement, weights_for_refinement)

        if new_complex_model:
            node.shape_model = new_complex_model

            new_reward, _ = evaluate_state_reward(final_refined_state, X_val, y_val, reward_metric)

            if new_reward >= current_best_reward:
                current_best_reward = new_reward

                new_scores = scores_of_other_leaves.copy()
                if len(node_indices) > 0:
                    new_scores[node_indices] += new_complex_model.predict(X_train[node_indices])
                final_refined_state.scores = new_scores

                if hasattr(new_complex_model, "feature_indices"):
                    fi_list = list(getattr(new_complex_model, "feature_indices", []))
                    if len(fi_list) == 1:
                        node.feature_idx = int(fi_list[0])
                        node.feature_name = feature_names[node.feature_idx]
                    elif len(fi_list) == 0:
                        node.feature_idx = -1
                        node.feature_name = "Constant"
                    else:
                        node.feature_idx = -1
                        node.feature_name = "FastSparse"
                else:
                    best_feature_idx = getattr(new_complex_model, 'feature_index', -1)
                    node.feature_name = "Constant" if best_feature_idx == -1 else feature_names[best_feature_idx]
                    node.feature_idx = best_feature_idx
                if debug_trace:
                    logger.debug("[REFINE] node_id=%s ACCEPTED | reward: %.4f", node.id, new_reward)
            else:
                node.shape_model = original_simple_model
                if debug_trace:
                    logger.debug("[REFINE] node_id=%s REJECTED | reward would be: %.4f", node.id, new_reward)

    logger.info(f"Refinement complete. Final validation reward: {current_best_reward:.4f}")

    # ---------- EBM Post-Processing (Safeguarded Version) ----------
    if shape_fitter == 'ebm':
        logger.info("EBM Post-Processing: Replacing refined shape functions with EBMs (Safeguarded Greedy)...")
        final_ebm_fitter = EBMShapeFitter(
            X_train_in, feature_names=feature_names, min_samples_leaf=MIN_SAMPLES_LEAF,
            interactions=ebm_interactions, max_bins=ebm_max_bins, outer_bags=ebm_outer_bags
        )

        terminal_nodes_for_ebm = [n for n in final_refined_state.get_all_shape_nodes() if n.is_terminal()]

        for shape_node in terminal_nodes_for_ebm:
            leaf_indices = final_refined_state.data_indices.get(shape_node.id, [])
            if not leaf_indices:
                continue

            original_model = shape_node.shape_model
            original_name = shape_node.feature_name
            original_idx = shape_node.feature_idx

            ebm_model = final_ebm_fitter.fit(leaf_indices, final_refined_state.residuals, final_refined_state.weights)

            if ebm_model:
                shape_node.shape_model = ebm_model
                shape_node.feature_name = "EBM"
                shape_node.feature_idx = -1

                new_reward, _ = evaluate_state_reward(final_refined_state, X_val, y_val, reward_metric)

                if new_reward >= current_best_reward:
                    logger.info(f"[EBM-REFINE] node={shape_node.id} ACCEPTED | reward: {current_best_reward:.4f} -> {new_reward:.4f}")
                    current_best_reward = new_reward

                    X_leaf = X_train[leaf_indices]
                    pred_old = original_model.predict(X_leaf)
                    pred_new = ebm_model.predict(X_leaf)
                    diff = pred_new - pred_old

                    final_refined_state.scores[leaf_indices] += diff

                    train_targets = TRAIN_TARGET_VALUES
                    if TRAIN_LABEL_MODE == 'binary':
                        probas = _sigmoid_scores(final_refined_state.scores)
                        final_refined_state.residuals = train_targets - probas
                        final_refined_state.weights = probas * (1 - probas)
                    else:
                        final_refined_state.residuals = train_targets - final_refined_state.scores
                        final_refined_state.weights = np.ones_like(final_refined_state.residuals)
                else:
                    shape_node.shape_model = original_model
                    shape_node.feature_name = original_name
                    shape_node.feature_idx = original_idx

    logger.info("--- 📊 Final Refined Model Detailed Performance ---")
    if TRAIN_LABEL_MODE == 'binary':
        val_probs = final_refined_state.tree.predict_proba(X_val)[:, 1]
        best_thr = find_best_threshold(y_val, val_probs)
        logger.info(f"Optimal Threshold (based on Val): {best_thr:.4f}")

        def _log_dataset_metrics(name, X, y):
            if X is None or y is None:
                return
            probs = final_refined_state.tree.predict_proba(X)[:, 1]
            preds = (probs > best_thr).astype(int)
            m = evaluate_model(y, probs, preds)

            logger.info(
                f"[{name:<5}] "
                f"Acc: {m['accuracy']:.4f} | "
                f"AUC: {m['auc']:.4f} | "
                f"F1: {m['f1']:.4f} | "
                f"LogLoss: {m['logloss']:.4f}"
            )
    else:
        def _log_dataset_metrics(name, X, y):
            if X is None or y is None:
                return
            preds = final_refined_state.tree.decision_function(X)
            y_true = np.asarray(y, dtype=np.float64).reshape(-1)
            mse = float(np.mean((y_true - preds) ** 2)) if len(y_true) > 0 else float('inf')
            logger.info(f"[{name:<5}] MSE: {mse:.6f}")

    _log_dataset_metrics("Train", X_train, y_train)
    _log_dataset_metrics("Valid", X_val, y_val)
    if X_test_in is not None and y_test_in is not None:
        _log_dataset_metrics("Test", X_test_in, y_test_in)

    logger.info("-------------------------------------------------------")

    LAST_BEST_STATE = final_refined_state
    CUSTOM_SAMPLE_WEIGHTS = None
    return final_refined_state, current_best_reward
