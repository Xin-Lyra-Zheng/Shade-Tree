import json
import logging
import os
import random
import shutil
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from resilient_logger import ResilientFileHandler
import MCTS
import config
import iterative_residual_ensemble as residual_ensemble
import util

def setup_logger(log_dir):
    # Get the root logger for the application
    logger = logging.getLogger('MCTS_ADTree')

    # --- Critical: Prevent adding handlers multiple times ---
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

    # The formatter remains the same
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Ensure the directory for the log file exists
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "mcts_adtree_fold_run.log")

    # --- Use ONLY our ResilientFileHandler ---
    # This handler will write all logs (DEBUG and above) to the specified file.
    file_handler = ResilientFileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)  # The handler's level
    file_handler.setFormatter(formatter)

    # Add the single, resilient handler to the logger
    logger.addHandler(file_handler)

    # We no longer need a separate ConsoleHandler, as Slurm handles stdout.

    # Quieten the overly verbose ebm logger
    ebm_logger = logging.getLogger('interpret')
    ebm_logger.setLevel(logging.WARNING)

    return logger

# =========================
# Helpers (keep consistent with main_test)
# =========================
def _scores_to_proba(scores: np.ndarray, logger: logging.Logger) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if np.all(np.isfinite(s)) and s.min() >= 0.0 and s.max() <= 1.0 and (s.max() - s.min()) > 1e-8:
        logger.debug("[OOF] scores_to_proba: already in [0,1].")
        return s
    if np.any(s < 0) and np.any(s > 0):
        logger.debug("[OOF] scores_to_proba: logits detected -> sigmoid.")
        s_clip = np.clip(s, -50, 50)
        return 1.0 / (1.0 + np.exp(-s_clip))
    logger.debug("[OOF] scores_to_proba: rank-normalization fallback.")
    ranks = s.argsort().argsort().astype(float)
    return (ranks + 0.5) / len(s)

def state_predict_proba(state, X: np.ndarray, logger: logging.Logger) -> np.ndarray:
    tree = getattr(state, "tree", None)
    if tree is not None and hasattr(tree, "predict_proba"):
        try:
            proba = np.asarray(tree.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                logger.debug("[OOF] Using state.tree.predict_proba(X)[:,1].")
                return proba[:, 1].astype(float)
            elif proba.ndim == 1:
                logger.debug("[OOF] Using state.tree.predict_proba(X) -> 1D.")
                return proba.astype(float)
        except Exception as e:
            logger.debug(f"[OOF] state.tree.predict_proba failed: {e}")

    if hasattr(state, "predict_proba"):
        try:
            proba = np.asarray(state.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                logger.debug("[OOF] Using state.predict_proba(X)[:,1].")
                return proba[:, 1].astype(float)
            elif proba.ndim == 1:
                logger.debug("[OOF] Using state.predict_proba(X) -> 1D.")
                return proba.astype(float)
        except Exception as e:
            logger.debug(f"[OOF] state.predict_proba failed: {e}")

    if hasattr(util, "predict_proba_state"):
        try:
            logger.debug("[OOF] Using util.predict_proba_state(state, X).")
            return np.asarray(util.predict_proba_state(state, X), dtype=float).reshape(-1)
        except Exception as e:
            logger.debug(f"[OOF] util.predict_proba_state failed: {e}")

    if hasattr(util, "predict_proba"):
        try:
            p = np.asarray(util.predict_proba(state, X))
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1].astype(float)
            return p.reshape(-1).astype(float)
        except Exception as e:
            logger.debug(f"[OOF] util.predict_proba failed: {e}")

    if hasattr(state, "decision_function"):
        try:
            logger.debug("[OOF] Using decision_function -> proba.")
            scores = state.decision_function(X)
            return _scores_to_proba(scores, logger)
        except Exception as e:
            logger.debug(f"[OOF] decision_function failed: {e}")
    if hasattr(util, "decision_function_state"):
        try:
            logger.debug("[OOF] Using util.decision_function_state -> proba.")
            scores = util.decision_function_state(state, X)
            return _scores_to_proba(scores, logger)
        except Exception as e:
            logger.debug(f"[OOF] util.decision_function_state failed: {e}")

    for api_name in ["predict_score", "predict_scores", "predict_margin", "predict_raw"]:
        fn = getattr(state, api_name, None)
        if callable(fn):
            try:
                logger.debug(f"[OOF] Using state.{api_name}(X) -> proba.")
                scores = fn(X)
                return _scores_to_proba(scores, logger)
            except Exception:
                pass

    for api_name in ["predict_score_state", "predict_scores_state", "predict_margin_state", "predict_raw_state"]:
        fn = getattr(util, api_name, None)
        if callable(fn):
            try:
                logger.debug(f"[OOF] Using util.{api_name}(state, X) -> proba.")
                scores = fn(state, X)
                return _scores_to_proba(scores, logger)
            except Exception:
                pass

    raise RuntimeError("Cannot obtain probabilities for threshold scan.")

def select_global_thresholds_from_probs(y_true_pm1: np.ndarray, proba: np.ndarray,
                                        results_dir: str, logger: logging.Logger):
    os.makedirs(results_dir, exist_ok=True)
    y01 = (y_true_pm1 == 1).astype(int)
    p = np.asarray(proba, dtype=float).reshape(-1)

    unique_ps = np.unique(np.clip(p, 0.0, 1.0))
    grid = np.linspace(0.0, 0.999, 200)
    candidates = np.unique(np.concatenate([unique_ps, grid]))

    best_acc, thr_acc = -1.0, 0.5
    best_f1,  thr_f1  = -1.0, 0.5
    for t in candidates:
        pred = (p > t).astype(int)
        acc = accuracy_score(y01, pred)
        f1w = f1_score(y01, pred, average='weighted', zero_division=0)
        if acc > best_acc:
            best_acc, thr_acc = acc, float(t)
        if f1w > best_f1:
            best_f1, thr_f1 = f1w, float(t)

    logger.info(f("[OOF] Selected global threshold (accuracy-optimal): {thr_acc:.4f}"))
    logger.info(f("[OOF] Selected global threshold (F1-weighted-optimal): {thr_f1:.4f}"))

    out_csv = os.path.join(results_dir, "global_thresholds_train_scan.csv")
    try:
        pd.DataFrame({
            "metric": ["accuracy", "f1_weighted"],
            "threshold": [thr_acc, thr_f1],
            "score": [best_acc, best_f1],
        }).to_csv(out_csv, index=False)
    except Exception as e:
        logger.warning(f"[OOF] Failed to save thresholds CSV: {e}")
    return thr_acc, thr_f1

# =========================
# Single Fold Runner
# =========================
def run_single_fold(args, fold_data, param_str, fold_id, visualize_outer: bool):
    """
    Run one inner fold. `param_str` should already include the outer seed tag so results
    naturally go into results/<dataset>/<param_str_with_outer>/fold_<id>.

    visualize_outer: whether this outer seed is allowed to produce visualization files.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, data = fold_data

    results_root = getattr(args, "results_root", "/usr/xtmp/xz424/mcts/results")
    args.results_dir = os.path.join(results_root, args.dataset, param_str, f"fold_{fold_id}")

    # Cache files for skip/reuse
    metrics_cache_path = os.path.join(args.results_dir, "metrics.csv")
    sparsity_cache_path = os.path.join(args.results_dir, "sparsity.json")
    done_marker_path = os.path.join(args.results_dir, "RUN_COMPLETE.json")

    # If a previous run wrote metrics, reuse instead of recomputing.
    if os.path.exists(metrics_cache_path):
        try:
            cached_metrics = pd.read_csv(metrics_cache_path, index_col=0)
            cached_sparsity = {}
            if os.path.exists(sparsity_cache_path):
                with open(sparsity_cache_path, "r") as f:
                    cached_sparsity = json.load(f)
            if os.path.exists(done_marker_path):
                print(f"[SKIP] Fold {fold_id} for {param_str} already completed; reusing cached metrics from {args.results_dir}.")
            else:
                print(f"[SKIP] Fold {fold_id} for {param_str} has metrics.csv; skipping recompute from {args.results_dir}.")
            return cached_metrics, cached_sparsity
        except Exception as e:
            print(f"[SKIP] Cache load failed ({e}); rerunning fold {fold_id}.")

    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    logger = setup_logger(args.results_dir)
    logger.info(f"--- Starting Fold: {fold_id} for Exp: {param_str} ---")
    logger.info(f"Arguments: {vars(args)}")

    min_samples_leaf_count = max(1, int(args.min_samples_leaf_ratio * len(X_train)))

    feature_thresholds = {}
    if args.num_percentile_points > 0:
        percentiles = np.linspace(0, 100, args.num_percentile_points + 2)[1:-1]
        for i in range(X_train.shape[1]):
            unique_vals = np.unique(X_train[:, i])
            if len(unique_vals) > 1:
                thresholds = np.percentile(unique_vals, percentiles)
                feature_thresholds[i] = np.unique(thresholds).tolist()

    # --- XGBoost guidance for pruning / ensemble region scoring ---
    region_strategy = getattr(args, "ensemble_region_strategy", "mse")
    need_xgb_region_guidance = bool(
        getattr(args, "use_residual_ensemble", False) and region_strategy == "xgb_guided"
    )
    need_xgb_model = args.use_xgboost_pruning or need_xgb_region_guidance
    xgboost_ideal_residuals_sq = None
    xgb_region_residuals = None
    if need_xgb_model:
        logger.info("Training XGBoost guide model for %s.",
                    "pruning and ensemble regions" if args.use_xgboost_pruning and need_xgb_region_guidance
                    else ("pruning" if args.use_xgboost_pruning else "ensemble regions"))
        try:
            y_train_xgb = (y_train == 1).astype(np.float32)
            p0 = float(y_train_xgb.mean())
            p0 = max(1e-6, min(1 - 1e-6, p0))
            xgb_guide = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=args.random_state,
                n_jobs=-1,
                tree_method='hist',
                base_score=p0,
                use_label_encoder=False
            )
            xgb_guide.fit(X_train, y_train_xgb)
            xgb_probas = xgb_guide.predict_proba(X_train)[:, 1]
            xgb_residuals = y_train_xgb - xgb_probas
            if args.use_xgboost_pruning:
                xgboost_ideal_residuals_sq = np.ascontiguousarray(xgb_residuals ** 2, dtype=np.float64)
                logger.info("XGBoost guide residuals prepared for pruning.")
            if need_xgb_region_guidance:
                xgb_region_residuals = np.ascontiguousarray(xgb_residuals, dtype=np.float64)
                logger.info("XGBoost residuals cached for ensemble region scoring.")
        except Exception as e:
            logger.error(f"Failed to train XGBoost guide model. Disable dependent features. Error: {e}")
            args.use_xgboost_pruning = False
            xgboost_ideal_residuals_sq = None
            need_xgb_region_guidance = False
            xgb_region_residuals = None

    _t0 = time.time()

    def _log_split_metrics(df):
        for split in ['Train', 'Validation', 'Test']:
            if split not in df.columns:
                continue
            split_metrics = df[split]
            acc = float(split_metrics.get('accuracy', np.nan))
            auc = float(split_metrics.get('auc', np.nan))
            f1 = float(split_metrics.get('f1', np.nan))
            logger.info(
                f"[{split}] Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}"
            )

    if args.use_residual_ensemble:
        ensemble_model, ensemble_history = residual_ensemble.train_iterative_residual_ensemble(
            args=args,
            feature_names=data.feature_names,
            feature_thresholds=feature_thresholds,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            xgb_region_residuals=xgb_region_residuals,
        )
        running_time_sec = float(time.time() - _t0)
        metrics_df, sparsity = util.save_results(
            ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test, args,
            visualize=True, data_info=data
        )
        _log_split_metrics(metrics_df)
        try:
            metrics_df["running_time_sec"] = running_time_sec
        except Exception as e:
            logger.warning(f"Failed to attach running_time_sec: {e}")
        try:
            metrics_df.to_csv(metrics_cache_path)
        except Exception as e:
            logger.warning(f"Failed to update metrics.csv with running_time_sec: {e}")
        history_path = os.path.join(args.results_dir, "ensemble_round_history.json")
        try:
            with open(history_path, "w") as f:
                json.dump(ensemble_history, f, indent=2)
            logger.info(f"[Fold {fold_id}] Ensemble diagnostics saved to {history_path}")
        except Exception as e:
            logger.warning(f"Failed to save ensemble diagnostics: {e}")
        final_entry = ensemble_history[-1] if ensemble_history else {}
        final_val = (final_entry or {}).get("val_metrics") or {}
        logger.info(
            "[Fold %s] Ensemble size=%d | Final Val LogLoss=%s | Val AUC=%s",
            fold_id,
            getattr(ensemble_model, "size", 0),
            f"{final_val.get('logloss', float('nan')):.4f}" if final_val else "n/a",
            f"{final_val.get('auc', float('nan')):.4f}" if final_val else "n/a",
        )
        logger.info(f"[Fold {fold_id}] running_time_sec = {running_time_sec:.3f}s")
        logger.info(f"--- Finished Fold: {fold_id} ---")
        return metrics_df, sparsity

    MCTS.run_mcts(
        X_train_in=X_train, y_train_in=y_train, X_val_in=X_val, y_val_in=y_val,
        feature_names=data.feature_names,
        feature_thresholds=feature_thresholds,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        exploration_c=args.c_ucb,
        min_samples_leaf=min_samples_leaf_count,
        label_mode=args.label_mode,
        shape_fitter=args.shape_fitter,
        n_knots_step=args.n_knots_step,
        refine_step_with_fastsparse=args.refine_step_with_fastsparse,
        complexity_penalty=args.complexity_penalty,
        patience=args.patience,
        warmup_iters=args.warmup_iters,
        log_frequency=args.log_frequency,
        reward_metric=args.reward_metric,
        max_split_nodes=args.max_split_nodes,
        ebm_interactions=args.ebm_interactions,
        ebm_max_bins=args.ebm_max_bins,
        ebm_outer_bags=args.ebm_outer_bags,
        use_xgboost_pruning=args.use_xgboost_pruning,
        xgboost_ideal_residuals_sq=xgboost_ideal_residuals_sq,
        pruning_mode=args.pruning_mode,
        pruning_threshold=args.pruning_threshold,
        use_pw=args.use_pw,
        c_pw=args.c_pw,
        alpha_pw=args.alpha_pw,
        eps_deepen=args.eps_deepen,
        X_test_in=X_test, 
        y_test_in=y_test,
        use_power_uct=False,
        power_p=4.0,
        use_xgb_ranking=args.use_xgboost_ranking,
        eps_ranking=args.epsilon_ranking,
        lambda_ranking=args.lambda_ranking,
        include_intermediate_shapes=args.include_intermediate_shapes
    )

    best_state = MCTS.LAST_BEST_STATE
    if best_state is None:
        raise RuntimeError("MCTS run did not produce a valid final state.")

    # Always visualize for all outer seeds and all folds
    visualize = True

    metrics_df, sparsity = util.save_results(
        best_state, X_train, y_train, X_val, y_val, X_test, y_test, args,
        visualize=visualize, data_info=data
    )
    _log_split_metrics(metrics_df)
    try:
        metrics_df.to_csv(metrics_cache_path)
        with open(sparsity_cache_path, "w") as f:
            json.dump(sparsity, f, indent=2)
        with open(done_marker_path, "w") as f:
            json.dump({
                "param_str": param_str,
                "fold_id": fold_id,
                "timestamp": time.time(),
                "args": vars(args),
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write cache artifacts for skip logic: {e}")
    try:
        # We need to access the globals we populated in MCTS.py
        import MCTS as _MCTS_mod
        if visualize:
            # Plot Reward Curve
            if hasattr(_MCTS_mod, "REWARD_HISTORY") and _MCTS_mod.REWARD_HISTORY:
                util.plot_mcts_reward_curve(_MCTS_mod.REWARD_HISTORY, args.results_dir, filename_prefix="reward")

            # Plot Root Action Stats
            if hasattr(_MCTS_mod, "ROOT_STATS_HISTORY") and _MCTS_mod.ROOT_STATS_HISTORY:
                util.plot_root_action_stats(_MCTS_mod.ROOT_STATS_HISTORY, args.results_dir, filename_prefix="root_actions")

            logger.info(f"[Fold {fold_id}] MCTS diagnostics plots saved to {args.results_dir}")
        else:
            logger.info(f"[Fold {fold_id}] Visualization disabled for this outer seed or fold; skipping diagnostics plots.")
    except Exception as e:
        logger.warning(f"MCTS diagnostics plotting skipped due to: {e}")

    running_time_sec = float(time.time() - _t0)
    logger.info(f"[Fold {fold_id}] running_time_sec = {running_time_sec:.3f}s")
    try:
        metrics_df["running_time_sec"] = running_time_sec
    except Exception as e:
        logger.warning(f"Failed to attach running_time_sec: {e}")
    try:
        metrics_df.to_csv(metrics_cache_path)
    except Exception as e:
        logger.warning(f"Failed to update metrics.csv with running_time_sec: {e}")

    # --- train-scan threshold evaluation (same as main_test) ---
    try:
        prob_train = state_predict_proba(best_state, X_train, logger)
        thr_acc, thr_f1 = select_global_thresholds_from_probs(
            y_true_pm1=y_train,
            proba=prob_train,
            results_dir=args.results_dir,
            logger=logger
        )

        USE_TRAIN_SCAN_THRESHOLD_METRIC = "accuracy"  # or "f1_weighted"
        chosen_thr = thr_acc if USE_TRAIN_SCAN_THRESHOLD_METRIC == "accuracy" else thr_f1
        chosen_thr = float(np.clip(chosen_thr, 0.0, 0.999999))
        logger.info(f"[OOF/train-scan] Override eval threshold with {USE_TRAIN_SCAN_THRESHOLD_METRIC}: {chosen_thr:.4f}")

        prob_val  = state_predict_proba(best_state, X_val,  logger)
        prob_test = state_predict_proba(best_state, X_test, logger)

        def _eval_block(y_pm1, p):
            ypred01 = (p > chosen_thr).astype(int)
            return util.evaluate_model(y_pm1, p, ypred01)

        m_train = _eval_block(y_train, prob_train)
        m_val   = _eval_block(y_val,   prob_val)
        m_test  = _eval_block(y_test,  prob_test)

        metrics_oof = pd.DataFrame({"Train": m_train, "Validation": m_val, "Test": m_test})

        out_csv = os.path.join(
            args.results_dir,
            f"metrics_train_scan_threshold_{USE_TRAIN_SCAN_THRESHOLD_METRIC}.csv"
        )
        metrics_oof.to_csv(out_csv)
        logger.info(f"[OOF/train-scan] Saved metrics ({chosen_thr:.4f}) to: {out_csv}")
        logger.info(f"[OOF/train-scan] Test accuracy: {metrics_oof.loc['accuracy','Test']:.4f}")

        # Optional: append to per-fold summary if util provides this helper
        try:
            args_oof = deepcopy(args)
            setattr(args_oof, "threshold_source", f"train_scan_{USE_TRAIN_SCAN_THRESHOLD_METRIC}")
            if hasattr(util, "log_single_fold_results"):
                util.log_single_fold_results(args_oof, metrics_oof, sparsity, logger)
        except Exception as e:
            logger.warning(f"[OOF/train-scan] Failed to append to summary CSV: {e}")

    except Exception as e:
        logger.warning(f"[OOF/train-scan] Skipping due to: {e}")

    logger.info(f"--- Finished Fold: {fold_id} ---")
    return metrics_df, sparsity

# =========================
# Main (5 outer seeds * inner k-fold on 75%)
# =========================
def main():
    args = config.parse_args()

    # Build base param string
    param_str_base = f"fitter_{args.shape_fitter}_seed_{args.random_state}"
    param_str_base += f"_pen_{str(args.complexity_penalty).replace('.', 'p')}"
    param_str_base += f"_metric_{args.reward_metric}"
    param_str_base += f"_ratio_{str(args.min_samples_leaf_ratio).replace('.', 'p')}"
    param_str_base += f"_c_{str(args.c_ucb).replace('.', 'p')}"
    param_str_base += f"_pw_{args.use_pw}"
    param_str_base += f"_cpw_{str(args.c_pw).replace('.', 'p')}"
    if args.include_intermediate_shapes:
        param_str_base += "_inclinter"
    if args.refine_step_with_fastsparse:
        param_str_base += "_fsgrefine"
    if args.shape_fitter == 'step':
        param_str_base += f"_knots_{args.n_knots_step}"
    elif args.shape_fitter == 'ebm':
        param_str_base += f"_inter_{args.ebm_interactions}"
    if args.use_xgboost_pruning:
        param_str_base += f"_xgbprune_{args.pruning_mode}_{str(args.pruning_threshold).replace('.', 'p')}"
    if getattr(args, "use_residual_ensemble", False):
        strategy = getattr(args, "ensemble_region_strategy", "mse")
        param_str_base += f"_ensemble_{strategy}"

    print(f"Master runner starting for hyperparameter set: {param_str_base}")

    # Prepare data once
    X_full, y_full, data_info = util.prepare_data(args)

    # These seeds ONLY control the outer 75/25 split
    outer_seeds = list(args.outer_split_seeds)  # already a list[int]

    print(f"Using outer split seeds (75/25): {outer_seeds}")
    print(f"Inner CV k={args.k_folds} with shuffle=True, random_state={args.random_state}")

    all_metrics_dfs, all_sparsity_dicts = [], []

    # Loop over outer seeds
    for outer_idx, outer_seed in enumerate(outer_seeds, start=1):
        print("\n" + "="*80)
        print(f"🌰 OUTER SPLIT {outer_idx}/{len(outer_seeds)}  [seed={outer_seed}]")

        # Only the first outer seed is allowed to generate visualization files
        visualize_outer = (outer_idx == 1)

        # 75/25 stratified split controlled by outer_seed
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_full, y_full, test_size=0.25, stratify=y_full, random_state=outer_seed
        )

        # Inner K-fold or single validation
        skf_inner = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        if args.k_fold_validation:
            print(f"Beginning inner {args.k_folds}-Fold CV on the 75% pool...")
            split_sequence = [
                (fold_idx, split)
                for fold_idx, split in enumerate(skf_inner.split(X_train_val, y_train_val), start=1)
            ]
        else:
            print("k_fold_validation disabled: running a single train/val split on the 75% pool...")
            split_generator = skf_inner.split(X_train_val, y_train_val)
            try:
                first_split = next(split_generator)
            except StopIteration:
                raise RuntimeError("StratifiedKFold failed to produce a split.")
            split_sequence = [(1, first_split)]

        # Optionally restrict to a single inner fold (for per-job parallelization)
        fold_override = getattr(args, "inner_fold_override", None)
        if fold_override is not None:
            filtered_sequence = [item for item in split_sequence if item[0] == fold_override]
            if not filtered_sequence:
                raise ValueError(
                    f"--inner_fold_override={fold_override} is invalid. "
                    f"Valid folds: {[fid for fid, _ in split_sequence]}"
                )
            print(f"Running only inner fold {fold_override} due to --inner_fold_override.")
            split_sequence = filtered_sequence

        # For directory segregation: param_str add outer tag
        param_str_outer = f"{param_str_base}_outer_{outer_seed}"

        # Iterate inner folds (one or many)
        total_folds = args.k_folds if args.k_fold_validation else 1
        for fold_id, (tr_idx, va_idx) in split_sequence:
            print(f"\n--- 🚀 INNER FOLD {fold_id}/{total_folds} (outer_seed={outer_seed}) ---")
            X_train, X_val = X_train_val[tr_idx], X_train_val[va_idx]
            y_train, y_val = y_train_val[tr_idx], y_train_val[va_idx]

            np.random.seed(args.random_state)
            random.seed(args.random_state)

            fold_data = (X_train, y_train, X_val, y_val, X_test, y_test, data_info)
            try:
                metrics_df, sparsity = run_single_fold(
                    args, fold_data, param_str_outer, fold_id,
                    visualize_outer=visualize_outer
                )
                all_metrics_dfs.append(metrics_df)
                all_sparsity_dicts.append(sparsity)
            except Exception as e:
                print(f"!!!!!!!! FAILED inner fold {fold_id} for outer seed {outer_seed}. Error: {e} !!!!!!!!")
                import traceback
                traceback.print_exc()

    if not all_metrics_dfs:
        print("No folds completed successfully. Exiting.")
        return

    print("\n" + "="*80)
    print("All outer seeds & inner folds processed. Aggregating and logging results...")
    util.log_aggregated_cv_results(args, all_metrics_dfs, all_sparsity_dicts)

    print("*"*80)
    print(f"Master runner finished for {param_str_base}. Aggregated results have been saved.")
    print("*"*80)

if __name__ == "__main__":
    main() 
