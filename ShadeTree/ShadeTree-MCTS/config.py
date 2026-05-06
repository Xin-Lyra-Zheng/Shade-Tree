# config.py (supports outer 75/25 split seeds + all existing knobs)
import argparse

def parse_args():
    """
    Parse command-line arguments for MCTS ADTree experiments.
    - random_state: controls inner CV shuffling and model randomness.
    - outer_split_seeds: controls the OUTER 75/25 split (comma-separated).
    """
    parser = argparse.ArgumentParser(
        description="Configuration for MCTS-based ADTree with feature shape functions"
    )

    # --- Data & CV ---
    parser.add_argument('--dataset', type=str, default='bank',
                        help="Dataset to load.")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Inner CV shuffling & model randomness.")
    parser.add_argument('--label_mode', type=str, default='binary',
                        choices=['binary', 'continuous'],
                        help="How to interpret labels during MCTS search.")
    parser.add_argument('--k_folds', type=int, default=3,
                        help="Number of folds for inner cross-validation on the 75%% pool.")
    parser.add_argument('--k_fold_validation', action='store_true', default=False,
                        help="Enable inner k-fold validation (default False = single run).")
    parser.add_argument('--inner_fold_override', type=int, default=None,
                        help="If set, run only this 1-indexed inner fold (requires k-fold validation).")
    # NEW: outer 75/25 split seeds
    parser.add_argument('--outer_split_seeds', type=str,
                        default="1824,409,4506,4012,3657",
                        help="Comma-separated seeds for the OUTER 75/25 train_val/test split.")
    parser.add_argument('--results_root', type=str, default="/usr/xtmp/xz424/mcts/results",
                        help="Root directory where per-run results should be written.")

    # --- Fitter / Model structure ---
    parser.add_argument('--shape_fitter', type=str, default='step',
                        choices=['step', 'ebm'],
                        help="Shape function fitter to use.")
    parser.add_argument('--min_samples_leaf_ratio', type=float, default=0.01,
                        help="Minimum samples per leaf as a ratio of the dataset size.")
    parser.add_argument('--n_knots_step', type=int, default=8,
                        help="Number of internal knots for the step fitter.")
    parser.add_argument('--include_intermediate_shapes', action='store_true',
                        help="If set, include intermediate shape-node contributions in prediction (fit children on parent residuals).")
    parser.add_argument('--refine_step_with_fastsparse', action='store_true',
                        help="If set with shape_fitter=step, use fastsparsegams for final refinement instead of GPU step.")
    parser.add_argument('--ebm_interactions', type=int, default=0,
                        help="Number of interactions for EBM fitters.")
    parser.add_argument('--ebm_max_bins', type=int, default=16,
                        help="Max bins for EBM fitters.")
    parser.add_argument('--ebm_outer_bags', type=int, default=1,
                        help="Outer bags for EBM fitters.")
    parser.add_argument('--max_split_nodes', type=int, default=30,
                        help="Maximum number of split nodes allowed.")
    parser.add_argument('--use_residual_ensemble', action='store_true',
                        help="Enable iterative residual ensemble boosting instead of a single ShadeTree.")
    parser.add_argument('--ensemble_rounds', type=int, default=5,
                        help="Maximum number of boosting rounds for the residual ensemble.")
    parser.add_argument('--ensemble_eta', type=float, default=0.2,
                        help="Learning rate applied to each ensemble member.")
    parser.add_argument('--ensemble_min_region', type=int, default=200,
                        help="Minimum samples required for a candidate region to be considered.")
    parser.add_argument('--ensemble_region_strategy', type=str, default='mse',
                        choices=['mse', 'xgb_guided'],
                        help="Region scoring strategy for the ensemble (mse or xgb_guided).")
    parser.add_argument('--ensemble_patience', type=int, default=1,
                        help="Rounds of no validation improvement before early stopping the ensemble.")
    parser.add_argument('--ensemble_early_stop_metric', type=str, default='logloss',
                        choices=['logloss', 'auc', 'accuracy', 'f1'],
                        help="Metric to monitor for ensemble early stopping (uses validation split when available).")

    # --- MCTS search ---
    parser.add_argument('--max_iters', type=int, default=1000,
                        help="Maximum number of MCTS search iterations.")
    parser.add_argument('--max_depth', type=int, default=100,
                        help="Maximum depth of the generated tree.")
    parser.add_argument('--c_ucb', type=float, default=1.4,
                        help="Exploration-exploitation constant for UCB1.")
    parser.add_argument('--max_rollout_depth', type=int, default=5,
                        help="Maximum depth for rollout (if used).")
    parser.add_argument('--reward_metric', type=str, default='auc',
                        choices=['accauc', 'logloss', 'auc', 'accuracy'],
                        help="Reward metric for MCTS.")

    # --- Regularization & pruning ---
    parser.add_argument('--complexity_penalty', type=float, default=0.005,
                        help="Penalty coefficient for model complexity (per split node).")

    # XGBoost-guided pruning
    parser.add_argument('--use_xgboost_pruning', action='store_true',
                        help="Enable XGBoost-guided pruning heuristic.")
    parser.add_argument('--pruning_mode', type=str, default='proportion',
                        choices=['proportion', 'difference'],
                        help="Compare current SSR to XGB ideal SSR by proportion or difference.")
    parser.add_argument('--pruning_threshold', type=float, default=1.2,
                        help="Threshold for XGB pruning (mode-specific interpretation).")

    # --- Early stopping & logging ---
    parser.add_argument('--patience', type=int, default=200,
                        help="Patience for MCTS early stopping.")
    parser.add_argument('--warmup_iters', type=int, default=100,
                        help="Iterations before early stopping is active.")
    parser.add_argument('--log_frequency', type=int, default=100,
                        help="Log MCTS progress every N iterations.")
    parser.add_argument('--num_percentile_points', type=int, default=20,
                        help="#percentile points for split thresholds per feature.")

    parser.add_argument("--warmstart_exports", type=str, default="",
                    help="Comma-separated paths to ShadeTree export payloads (.json/.pkl).")
    parser.add_argument("--prior_lambda", type=float, default=8.0)
    parser.add_argument("--prior_gamma", type=float, default=0.9)
    parser.add_argument("--prior_cap", type=int, default=50)
    parser.add_argument("--prior_budget", type=int, default=40)
    parser.add_argument("--warmstart_metric_key", type=str, default="auc",
                        help="Metric key to read from export payloads / evaluator for warm-start reward.")
    parser.add_argument('--use_pw', type=int, default=1, choices=[0, 1],
                        help="Toggle Progressive Widening (1=on, 0=off).")
    parser.add_argument('--c_pw', type=float, default=100,
                        help="Progressive Widening scaling constant (C).")
    parser.add_argument('--alpha_pw', type=float, default=0.4,
                        help="Progressive Widening exponent (alpha).")
    parser.add_argument('--eps_deepen', type=float, default=0.35,
                        help="Probability of forcing deepening at the root node.")
    # --- XGBoost Stump Gain Ranking for Unexpanded Actions ---
    parser.add_argument('--use_xgboost_ranking', action='store_true',
                        help="Enable XGBoost stump gain ranking to prioritize splits during MCTS expansion.")
    parser.add_argument('--epsilon_ranking', type=float, default=0.2,
                        help="Probability of choosing a random action instead of the best ranked action (epsilon-greedy).")
    parser.add_argument('--lambda_ranking', type=float, default=1.0,
                        help="L2 regularization term (lambda) for XGBoost gain calculation.")
    args = parser.parse_args()

    # Parse outer seeds into list[int]
    try:
        seeds = [int(s.strip()) for s in args.outer_split_seeds.split(',') if s.strip() != ""]
        if not seeds:
            raise ValueError("No outer seeds parsed.")
        args.outer_split_seeds = seeds
    except Exception:
        # Fallback to the default 5-seed list
        args.outer_split_seeds = [1824, 409, 4506, 4012, 3657]

    return args
