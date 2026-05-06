import numpy as np
import pandas as pd
import time
from copy import deepcopy
from math import sqrt
import cupy as cp
GPU_AVAILABLE = cp.is_available()
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score

from loss_functions import SquaredErrorLoss, LogisticLoss, ExponentialLoss
import xgboost as xgb
import fastsparsegams as fsg
from scipy import sparse

from contextlib import contextmanager
import numpy as np

@contextmanager
def sparse_triplet_ravel():
    orig_csr, orig_csc = sparse.csr_matrix, sparse.csc_matrix
    def _wrap(maker):
        def _f(arg, *a, **k):
            if isinstance(arg, (tuple, list)) and len(arg) == 3:
                data, indices, indptr = arg
                arg = (np.asarray(data).ravel(),
                       np.asarray(indices).ravel(),
                       np.asarray(indptr).ravel())
            return maker(arg, *a, **k)
        return _f
    sparse.csr_matrix = _wrap(orig_csr)
    sparse.csc_matrix = _wrap(orig_csc)
    try:
        yield
    finally:
        sparse.csr_matrix, sparse.csc_matrix = orig_csr, orig_csc

class RuleSetNode:
    """
    Represents a node that has a baseline prediction model and a list of 
    additive sub-rules (RuleTrees). This is the core building block.
    """
    def __init__(self, depth, parent_rule=None, is_left_child=None, verbose=False, tree_id=None, tree=None):

        self.tree_id = tree_id
        self.depth = depth
        self.parent_rule = parent_rule
        self.is_left_child = is_left_child

        self.base_model = lambda x: np.zeros(x.shape[0])
        self.base_transformer = None
        self.base_feature_idx = -1
        self.base_used_features = []
        self.sub_rules = []
        
        self.num_samples = 0
        self.leaf_support_size = 0

        self._idx_cache = None
        self._idx_cache_gen = -1
        self._idx_cache_archive = []

    def get_all_nodes(self):
        """Traverse from this node (usually root) and return all RuleSetNodes in this tree."""
        nodes_to_visit = [self]
        all_nodes = []
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            all_nodes.append(current_node)
            for sub_rule in current_node.sub_rules:
                nodes_to_visit.append(sub_rule.left_child)
                nodes_to_visit.append(sub_rule.right_child)
        return all_nodes
    
    def predict(self, X_cpu):
        """Calculates total prediction by summing the baseline and all sub-rule predictions."""
        if X_cpu.shape[0] == 0:
            return np.array([])
            
        base_pred = np.zeros(X_cpu.shape[0])
        if self.base_model is not None:
            if isinstance(self.base_model, tuple):
                fsg_path, best_lam = self.base_model
                try:
                    candidate_lambdas = fsg_path.lambda_0[0]
                    best_idx = candidate_lambdas.index(best_lam)
                    intercept = fsg_path.intercepts[0][best_idx]
                    coeffs_vector = fsg_path.coeffs[0][:, best_idx].toarray()
                    base_pred = (intercept + X_cpu @ coeffs_vector).flatten()
                except Exception:
                    pass
            elif callable(self.base_model):
                base_pred = self.base_model(X_cpu)
            else:
                 feature_data = X_cpu[:, self.base_feature_idx].reshape(-1, 1)
                 spline_basis = self.base_transformer.transform(feature_data)
                 base_pred = self.base_model.predict(spline_basis)

        if self.sub_rules:
            sub_rule_preds = sum((rule.predict(X_cpu) for rule in self.sub_rules), np.zeros(X_cpu.shape[0]))
        else:
            sub_rule_preds = np.zeros(X_cpu.shape[0])
        
        return base_pred + sub_rule_preds

class RuleTree:
    """Represents a single binary split rule whose children are RuleSetNodes."""
    def __init__(self, split_rule, left_child_node, right_child_node):
        self.feature_idx, self.threshold = split_rule
        self.left_child = left_child_node
        self.right_child = right_child_node
        self.parent_node = None 

    def predict(self, X_cpu):
        if X_cpu.shape[0] == 0:
            return np.array([])
        predictions = np.zeros(X_cpu.shape[0])
        left_mask = X_cpu[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        if np.any(left_mask):
            predictions[left_mask] = self.left_child.predict(X_cpu[left_mask])
        if np.any(right_mask):
            predictions[right_mask] = self.right_child.predict(X_cpu[right_mask])
        return predictions

# class AdditiveTree:
#     """A container for a tree whose nodes are RuleSetNodes."""
#     def __init__(self, tree_id):
#         self.id = tree_id
#         self.root = RuleSetNode(tree=self, depth=0)

#     def predict(self, X_cpu):
#         return self.root.predict(X_cpu)

#     def get_all_ruleset_nodes(self):
#         """Traverses the tree to find all RuleSetNodes."""
#         nodes_to_visit = [self.root]
#         all_nodes = []
#         while nodes_to_visit:
#             current_node = nodes_to_visit.pop(0)
#             all_nodes.append(current_node)
#             for sub_rule in current_node.sub_rules:
#                 nodes_to_visit.append(sub_rule.left_child)
#                 nodes_to_visit.append(sub_rule.right_child)
#         return all_nodes

class ShadeTree:
    def _vprint(self, *a, **k):
        if getattr(self, 'verbose', False):
            print(*a, **k)

    def __init__(self, M=50, eta=0.05, d1=1, lambda_cost=0.01, min_samples_leaf=50, 
                 early_stopping_rounds=3, tol=1e-5, max_thresholds=20, max_depth=3, k=5, loss='logistic', 
                 step_mode="newton",
                 lb_static_enabled=True,
                 lb_alpha=1.0,
                 lb_abs_threshold=0.001,
                 local_exclude_root: bool = True,
                 local_exclude_root_threshold: float = 0.90,
                 use_upper_bound_for_local_topk: bool = True,
                 diag_enabled: bool = True, 
                 gam_mode: str = "never",
                 leaf_spline_alpha: float = 0.1,
                 global_gam_max_support_size: int = 5,
                 leaf_fsg_max_support_size: int = 5,
                 leaf_fitter: str = "fsg",
                 leaf_restrict: bool = False,
                 verbose=True,
                 backward_fit: bool = True,
                 backward_max_support: int = 8,
                 export_path=None,
                 stochastic_coord: bool = False,
                 stochastic_topk: int = 3,
                 searching_strategy: str = "lookahead_spline",
                 stochastic_seed: int | None = 42,
                 spline_refine_m: int = 3,
                 leaf_gain_threshold: float = 0.01,
                 structure_alpha: float = 0.5,
                 silence_parent: bool = True):

        self.M, self.eta, self.d1, self.min_gain_fraction = M, eta, d1, lambda_cost
        self.min_samples_leaf, self.early_stopping_rounds = min_samples_leaf, early_stopping_rounds
        self.tol, self.max_thresholds, self.max_depth = tol, max_thresholds, max_depth
        self.n_splines, self.spline_degree, self.k = 16, 0, k
        self.local_exclude_root = local_exclude_root
        self.local_exclude_root_threshold = float(local_exclude_root_threshold)
        self.use_upper_bound_for_local_topk = use_upper_bound_for_local_topk
        self.diag_enabled = bool(diag_enabled)
        self.verbose = bool(verbose)
        self._reset_diag_round()
        self.searching_strategy = str(searching_strategy)
        self.global_gam_max_support_size = global_gam_max_support_size
        self.leaf_fsg_max_support_size = leaf_fsg_max_support_size
        self.leaf_fitter = str(leaf_fitter)
        self.leaf_restrict = bool(leaf_restrict)
        self.ebm_params = {
            "interactions": 0,
            "max_bins": 64,
            "max_leaves": 2,
            "min_samples_leaf": 4, 
            "learning_rate": 0.01,
            "max_rounds": 1000,
            "outer_bags": 1,
            "random_state": 42,
        }

        self.gam_mode = str(gam_mode)
        self.leaf_spline_alpha = float(leaf_spline_alpha)

        self.backward_fit = backward_fit
        self.backward_max_support = backward_max_support

        self.stochastic_coord = stochastic_coord
        self.stochastic_topk = stochastic_topk
        self._rng = np.random.RandomState(int(stochastic_seed) if stochastic_seed is not None else 42)
        self.spline_refine_m = int(spline_refine_m)
        self.leaf_gain_threshold = float(leaf_gain_threshold)
        self.structure_alpha = float(structure_alpha)
        self.silence_parent = bool(silence_parent)

        if loss == 'squared_error':
            self.loss_function_ = SquaredErrorLoss()
        elif loss == 'logistic':
            self.loss_function_ = LogisticLoss()
        elif loss == 'exponential':
            self.loss_function_ = ExponentialLoss()
        else:
            raise ValueError("Loss function not supported.")
        self.step_mode = step_mode
        
        self.global_gam, self.global_gam_lambda = None, None
        self.initial_prediction_ = 0.0
        self.tree_ensemble = []
        self.feature_names = None
        self.next_tree_id = 0

        self.xp = cp if 'GPU_AVAILABLE' in globals() and GPU_AVAILABLE else np
        self.device = "GPU" if 'GPU_AVAILABLE' in globals() and GPU_AVAILABLE else "CPU"
        self._vprint(f"\n--- Initialized ShadeTree to run on {self.device} ---\n")
        self.All_Bases_device = None

        self.lb_static_enabled = lb_static_enabled
        self.lb_alpha = lb_alpha
        self.lb_abs_threshold = lb_abs_threshold
        self._static_ref = None
        self._train_size = None
        self._cache_epoch = 0

        self.loss_curve_ = {"iter": [], "train": [], "val": []}
    
    def _bump_cache_epoch(self):
        self._cache_epoch += 1


    def get_params(self, deep=True):
        loss_name = (
            'logistic' if isinstance(self.loss_function_, LogisticLoss)
            else 'exponential' if isinstance(self.loss_function_, ExponentialLoss)
            else 'squared_error'
        )
        return {
            "M": self.M, "eta": self.eta, "d1": self.d1,
            "lambda_cost": self.min_gain_fraction,
            "min_samples_leaf": self.min_samples_leaf,
            "early_stopping_rounds": self.early_stopping_rounds,
            "tol": self.tol, "max_thresholds": self.max_thresholds,
            "max_depth": self.max_depth, "k": self.k,
            "loss": loss_name,
            "structure_alpha": self.structure_alpha,
            "leaf_gain_threshold": self.leaf_gain_threshold,
            "step_mode": self.step_mode,
            "silence_parent": self.silence_parent,
        }

    def set_params(self, **params):
        for param, value in params.items():
            if param == "lambda_cost":
                setattr(self, "min_gain_fraction", value)
            else:
                setattr(self, param, value)
        if 'loss' in params:
            if params['loss'] == 'squared_error':
                self.loss_function_ = SquaredErrorLoss()
            elif params['loss'] == 'logistic':
                self.loss_function_ = LogisticLoss()
            elif params['loss'] == 'exponential':
                self.loss_function_ = ExponentialLoss()
            else:
                raise ValueError("Loss function not supported.")
        return self

    def _reset_diag_round(self):
        self._diag = {
            'thresholds_considered': 0, 'nodes_considered_local': 0,
            'nodes_after_root_filter': 0, 'nodes_after_upper_filter': 0,
            'local_topk_final': 0, 'actions_executed_global': 0,
            'actions_executed_local': 0, 'residual_sanity_mae': None,
        }

    def _log_diag_round_end(self, header: str = ""):
        if not self.diag_enabled or not getattr(self, 'verbose', False): return
        h = f"[DIAG] {header} " if header else "[DIAG] "
        d = self._diag
        print(h + f"local_nodes: considered={d['nodes_considered_local']}, after_root_filter={d['nodes_after_root_filter']}, topK={d['local_topk_final']}")

    def _gradient_response(self, y, raw):
        p = 1.0 / (1.0 + np.exp(-raw))
        r = y - p
        return r

    def _precompute_spline_bases(self, X_train_only):
        self._vprint(f"Pre-computing and batching spline bases for {self.device} (train-only).")
        try:
            cpu_bases_list = []
            spline_transformer = SplineTransformer(n_knots=self.n_splines, degree=self.spline_degree, include_bias=False)
            for j in range(X_train_only.shape[1]):
                cpu_bases_list.append(spline_transformer.fit_transform(X_train_only[:, j].reshape(-1, 1)))
            all_bases_cpu = np.stack(cpu_bases_list, axis=0)
            self.All_Bases_device = self.xp.asarray(all_bases_cpu)
        except Exception as e:
            self._vprint(f"Precompute failed: {e}")
        self._vprint(f"Batched pre-computation took {time.time() - start_time:.2f}s. Shape: {self.All_Bases_device.shape}")
    
    def fit(self, X_train, y_train, X_val, y_val,
            feature_names=None, X_train_bin=None, X_val_bin=None,
            feature_names_bin=None, binner=None,
            X_test=None, y_test=None, X_test_bin=None, export_path=None):

            # Cache bin / feature info
            self.fsg_X_train = X_train_bin
            self.fsg_X_val   = X_val_bin
            self.fsg_X_test  = X_test_bin if X_test_bin is not None else None
            self.fsg_feature_names = list(feature_names_bin) if feature_names_bin is not None else None
            self.fsg_binner  = binner
            self.feature_names = list(feature_names) if feature_names is not None else [f'f_{i}' for i in range(X_train.shape[1])]

            # Precompute spline bases for lookahead leaf eval
            try: self._precompute_spline_bases(X_train)
            except Exception: pass

            # references
            self._X_train_ref, self._X_val_ref = X_train, X_val
            self._y_train_ref, self._y_val_ref = y_train, y_val
            self.initial_prediction_ = self.loss_function_.initial_prediction(y_train)
            self.tree_ensemble = []
            self.next_tree_id = 0
            self._train_static_reference(X_train, y_train)

            best_val_loss = np.inf
            stagnant_rounds = 0
            best_iteration = 0
            best_model_state = None
            if not hasattr(self, "global_gam_transformer"): self.global_gam_transformer = None
            self._search_times = []
            self._current_raw_val_pred = None  # populated at end of each iteration; reused as next iteration's pre-split cache

            for m in range(self.M):
                self._reset_diag_round()
                print(f"\n{'='*15} Iteration {m+1}/{self.M} {'='*15}")

                current_raw_preds_train = self._get_raw_prediction(X_train, on_device=False)
                z_train, w_train = self.loss_function_.working_response(y_train, current_raw_preds_train, mode=self.step_mode)
                if w_train is None: w_train = np.ones_like(z_train)
                w_train = np.clip(w_train, 1e-12, 1e12)
                z_train = np.clip(z_train, -5.0, 5.0)

                # Val cache for _add_sub_rule_to_node lambda selection.
                # First iteration: compute fresh. Subsequent iterations: reuse the post-split
                # val pred stored at the end of the previous iteration (same model state).
                if X_val is not None and self._current_raw_val_pred is None:
                    self._current_raw_val_pred = np.asarray(
                        self._get_raw_prediction(X_val, on_device=False)).ravel()
                elif X_val is None:
                    self._current_raw_val_pred = None

                # Global GAM update (first iteration)
                if (self.gam_mode == "once" and m == 0):
                    X_for_fsg = self.fsg_X_train if self.fsg_X_train is not None else X_train
                    self._update_global_gam(X_for_fsg, z_train, w_train)
                    if self.fsg_X_train is not None: self.global_gam_transformer = self._make_binner_transformer()
                    print("\n>>> Global GAM only:")
                    self._print_metrics("GAM-Train", y_train, self._get_raw_prediction(X_train))
                    if X_val is not None: self._print_metrics("GAM-Val", y_val, self._get_raw_prediction(X_val))

                search_time = self._perform_best_action(X_train, z_train, w_train)
                # record search time (exclude warm-up iteration 0)
                if m > 0 and search_time is not None:
                    self._search_times.append(search_time)
                if search_time is not None:
                    self._vprint(f"Action search (candidate gen+scoring) took {search_time:.4f} seconds.")

                train_loss = self.loss_function_(y_train, self._get_raw_prediction(X_train))
                current_val_loss = np.inf
                if X_val is not None and y_val is not None:
                    _raw_val_post = np.asarray(self._get_raw_prediction(X_val, on_device=False)).ravel()
                    current_val_loss = self.loss_function_(y_val, _raw_val_post)
                    self._current_raw_val_pred = _raw_val_post  # reuse as next iteration's pre-split cache

                self.loss_curve_["iter"].append(m + 1)
                self.loss_curve_["train"].append(float(train_loss))
                self.loss_curve_["val"].append(float(current_val_loss))

                print(f"--- Train Loss: {train_loss:.6f}, Val Loss: {current_val_loss:.6f} (Best: {best_val_loss:.6f}) ---")

                if current_val_loss < best_val_loss - self.tol:
                    best_val_loss = current_val_loss
                    stagnant_rounds = 0
                    best_iteration = m
                    best_model_state = {
                        'tree_ensemble': deepcopy(self.tree_ensemble),
                        'global_gam': deepcopy(self.global_gam),
                        'global_gam_lambda': self.global_gam_lambda,
                        'initial_prediction': self.initial_prediction_,
                        'global_gam_transformer': self.global_gam_transformer,
                    }
                else:
                    stagnant_rounds += 1
                
                if self.early_stopping_rounds and stagnant_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {m+1}.")
                    break

            # Restore Best Model
            if best_model_state:
                print(f"Restoring best model from iteration {best_iteration + 1}...")
                self.tree_ensemble = best_model_state['tree_ensemble']
                self.global_gam = best_model_state['global_gam']
                self.global_gam_lambda = best_model_state['global_gam_lambda']
                self.initial_prediction_ = best_model_state['initial_prediction']
                self.global_gam_transformer = best_model_state.get('global_gam_transformer', None)

                if self.backward_fit:
                    self._vprint("[Backward fit] Optimizing leaves...")
                    self._backward_fit_leaves(X_train=X_train, y_train=y_train,
                                              X_val=X_val, y_val=y_val,
                                              max_support=self.backward_max_support, verbose=self.verbose)
                
                if export_path: export_best_model(self, export_path)

                print("\n>>> Best model metrics:")
                self._print_metrics("Best-Train", y_train, self._get_raw_prediction(X_train))
                if X_val is not None: self._print_metrics("Best-Val", y_val, self._get_raw_prediction(X_val))
                if X_test is not None: self._print_metrics("Best-Test", y_test, self._get_raw_prediction(X_test))

            return self

    def _print_metrics(self, name, y_true, raw_pred):
        try:
            if isinstance(self.loss_function_, (LogisticLoss, ExponentialLoss)):
                proba = self.loss_function_.transform_prediction(raw_pred)
                yhat = (proba >= 0.5).astype(int)
                acc = accuracy_score(y_true, yhat)
                try: auc = roc_auc_score(y_true, proba)
                except: auc = float("nan")
                print(f"[{name}] Acc={acc:.4f} | AUROC={auc:.4f}")
            else:
                rmse = float(np.sqrt(mean_squared_error(y_true, raw_pred)))
                print(f"[{name}] RMSE={rmse:.6f}")
        except: pass

    def predict_proba(self, X, on_device=False):
        raw_preds = self._get_raw_prediction(X, on_device=on_device)
        
        final_probs = self.loss_function_.transform_prediction(raw_preds)
        
        xp = self.xp if on_device else np
        
        prob_array = xp.vstack([1 - final_probs, final_probs]).T
        
        return prob_array
    
    def predict(self, X, on_device=False):
        proba = self.predict_proba(X, on_device=on_device)[:, 1]
        return np.round(proba).astype(int)
  
    def _get_raw_prediction(self, X, on_device=False):
        if X is None:
            return None 
        xp = self.xp
        is_gpu_array = isinstance(X, cp.ndarray)
        X_cpu = xp.asnumpy(X) if is_gpu_array else X
        
        initial_pred = np.full(X_cpu.shape[0], self.initial_prediction_)
        
        gam_pred_cpu = self._predict_gam_part(X_cpu)
        tree_preds_cpu = np.sum([tree.predict(X_cpu) for tree in self.tree_ensemble], axis=0)
        
        final_raw_preds = initial_pred + gam_pred_cpu + self.eta * tree_preds_cpu
        
        return xp.asarray(final_raw_preds) if on_device else final_raw_preds

    def _fit_leaf_with_fastsparse(self, X_leaf_cpu, y_residuals_leaf_cpu, sample_weights=None,
                                   restrict_to_feature=None,
                                   X_val_leaf=None, y_val_leaf=None, raw_val_leaf=None):
        import numpy as np
        X_leaf_cpu = np.asarray(X_leaf_cpu, dtype=np.float64)
        r = np.asarray(y_residuals_leaf_cpu, dtype=np.float64).reshape(-1) # Newton Targets
        n = r.shape[0]

        # Weights handling
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            w = np.clip(w, 1e-12, np.inf)
        else:
            w = np.ones(n, dtype=np.float64)

        # 1. Baseline: Weighted Mean (Constant Model)
        w_sum = np.sum(w)
        mean_val = np.sum(w * r) / (w_sum + 1e-12)
        sse_baseline = np.sum(w * (r - mean_val) ** 2)

        # Return mean if samples too small
        if (n < getattr(self, "min_samples_leaf", 1)) or (fsg is None):
            return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []

        # 2. Design Matrix Prep
        used_binner = False
        X_design = X_leaf_cpu
        if getattr(self, "fsg_binner", None) is not None:
            try:
                Z_leaf, _ = self.fsg_binner.transform_numpy(X_leaf_cpu, feature_names=getattr(self, "feature_names", None))
                X_design = np.asarray(Z_leaf, dtype=np.float64); used_binner = True
            except Exception: pass
        
        # Ignore restrict_to_feature for fitter; always allow full features
        design_cols = None
        X_fit = X_design

        if X_fit.shape[1] == 0:
             return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []

        # 3. WLS Transformation: multiply by sqrt(w)
        sw = np.sqrt(w)
        X_fit_w = X_fit * sw[:, None]
        r_w = r * sw

        # 3b. Filter out zero-variance (constant) columns — these cause FastSparse CDPSI to SIGSEGV
        # on small leaves where binary binner columns are all-0 or all-1 within the leaf.
        col_var = np.var(X_fit_w, axis=0)
        active_mask = col_var > 0
        if not np.any(active_mask):
            return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []
        if not np.all(active_mask):
            design_cols = list(np.where(active_mask)[0])
            X_fit_w = X_fit_w[:, active_mask]

        # 4. FSG Fit
        # max_support_size must be < n to avoid degenerate systems in CDPSI
        max_k = int(getattr(self, "leaf_fsg_max_support_size", 2))
        max_k = min(max_k, X_fit_w.shape[0] - 1, X_fit_w.shape[1])
        if max_k < 1:
            return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []
        try:
            with sparse_triplet_ravel():
                path = fsg.fit(X_fit_w, r_w, penalty="L0", max_support_size=max_k, loss="SquaredError")
        except:
            return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []

        if not getattr(path, "lambda_0", None):
             return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []

        # 5. Select Best Lambda
        # Prefer: val-based actual logistic loss (if val data provided)
        # Fallback: BIC on training data (penalises complexity, avoids always picking max-support)
        best_sse, best_lam, best_supp = np.inf, None, 0
        best_support_idx = None
        lambdas = list(path.lambda_0[0])

        use_val_selection = (X_val_leaf is not None and y_val_leaf is not None
                             and raw_val_leaf is not None
                             and len(y_val_leaf) > 0)

        if use_val_selection:
            # Prepare val design matrix (same binner as train if available)
            X_val_arr = np.asarray(X_val_leaf, dtype=np.float64)
            if used_binner and getattr(self, "fsg_binner", None) is not None:
                try:
                    Z_val, _ = self.fsg_binner.transform_numpy(X_val_arr,
                                    feature_names=getattr(self, "feature_names", None))
                    X_val_design = np.asarray(Z_val, dtype=np.float64)
                except Exception:
                    X_val_design = X_val_arr
            else:
                X_val_design = X_val_arr
            # Apply same column filter used for training (constant-column removal)
            if design_cols is not None:
                try:
                    X_val_design = X_val_design[:, design_cols]
                except Exception:
                    pass
            y_val_arr = np.asarray(y_val_leaf).reshape(-1)
            raw_val_arr = np.asarray(raw_val_leaf).reshape(-1)
            eta = float(getattr(self, "eta", 1.0))

            best_val_loss = np.inf
            for idx, lam in enumerate(lambdas):
                intercept = float(path.intercepts[0][idx])
                coeffs = path.coeffs[0][:, idx].toarray().reshape(-1)
                nz_idx = np.flatnonzero(coeffs)
                supp = int(np.count_nonzero(coeffs))

                # Predict on val in original (un-transformed) space
                try:
                    if X_val_design.shape[1] == coeffs.shape[0]:
                        pred_val = (intercept + X_val_design @ coeffs).ravel()
                    else:
                        pred_val = np.full(len(y_val_arr), mean_val)
                except Exception:
                    pred_val = np.full(len(y_val_arr), mean_val)

                raw_val_updated = raw_val_arr + eta * pred_val
                try:
                    val_loss = float(self.loss_function_(y_val_arr, raw_val_updated))
                except Exception:
                    val_loss = np.inf

                # Also track train SSE for gain_fraction computation below
                pred_tr = (intercept + X_fit_w @ coeffs).ravel()
                sse_tr = float(np.sum((r_w - pred_tr) ** 2))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_lam, best_supp = lam, supp
                    best_support_idx = nz_idx
                    best_sse = sse_tr  # keep train SSE of the chosen model for gain_fraction check
        else:
            # BIC-based selection on training data
            sigma2 = max(float(np.var(r_w)), 1e-12)
            best_bic = np.inf
            for idx, lam in enumerate(lambdas):
                intercept = float(path.intercepts[0][idx])
                coeffs = path.coeffs[0][:, idx].toarray().reshape(-1)
                nz_idx = np.flatnonzero(coeffs)
                supp = int(np.count_nonzero(coeffs))
                pred = (intercept + X_fit_w @ coeffs).ravel()
                sse = float(np.sum((r_w - pred) ** 2))
                bic = sse + np.log(max(n, 2)) * supp * sigma2
                if bic < best_bic:
                    best_bic = bic
                    best_sse, best_lam, best_supp = sse, lam, supp
                    best_support_idx = nz_idx

        # 6. Hard Fallback
        gain_fraction = (sse_baseline - best_sse) / (sse_baseline + 1e-12)
        threshold = getattr(self, "leaf_gain_threshold", 0.01)
        
        if (gain_fraction < threshold) or (best_supp == 0) or (best_lam is None):
            return (lambda x, _c=mean_val: np.full(np.asarray(x).shape[0], _c, dtype=float)), sse_baseline, 0, []

        # 7. Prediction Closure with Shrinkage
        def _leaf_predict_fn(X_input, _path=path, _lam=best_lam, _mean=mean_val, _self=self, 
                             _used=used_binner, _cols=design_cols):
            import numpy as _np
            cand = _path.lambda_0[0]
            try: bi = list(cand).index(_lam)
            except: bi = int(_np.argmin(_np.abs(_np.array(cand, dtype=float) - float(_lam))))
            intercept = float(_path.intercepts[0][bi])
            coeffs = _path.coeffs[0][:, bi].toarray().reshape(-1)

            Xin = _np.asarray(X_input, dtype=_np.float64)
            if _used and getattr(_self, "fsg_binner", None):
                Z, _ = _self.fsg_binner.transform_numpy(Xin, feature_names=getattr(_self, "feature_names", None))
                Xpred = _np.asarray(Z, dtype=_np.float64)
                if _cols: Xpred = Xpred[:, _cols]
            else:
                Xpred = Xin[:, _cols] if _cols else Xin
            
            if Xpred.shape[1] != coeffs.shape[0]: return _np.full(Xin.shape[0], _mean)

            raw = (intercept + Xpred @ coeffs).ravel()
            return raw

        used_features = self._map_leaf_support_to_raw_features(
            best_support_idx, design_cols, used_binner, X_leaf_cpu.shape[1], restrict_to_feature
        )
        return _leaf_predict_fn, best_sse, best_supp, used_features

    def _fit_leaf(self, X, y, sample_weights=None, restrict_to_feature=None,
                  X_val_leaf=None, y_val_leaf=None, raw_val_leaf=None):
        # Dispatcher
        if self.leaf_fitter == "fsg":
            return self._fit_leaf_with_fastsparse(X, y, sample_weights, restrict_to_feature,
                                                   X_val_leaf=X_val_leaf,
                                                   y_val_leaf=y_val_leaf,
                                                   raw_val_leaf=raw_val_leaf)
        else: # Fallback / EBM (not optimized for expert mode, omitted for brevity)
             w = sample_weights if sample_weights is not None else np.ones(len(y))
             c = float(np.average(y, weights=w))
             return (lambda x, _c=c: np.full(x.shape[0], _c)), 0.0, 0, []

    def _map_leaf_support_to_raw_features(self, support_idx, design_cols, used_binner, raw_dim, restrict_to_feature):
        """Map nonzero coefficient columns back to raw feature indices when possible."""
        if support_idx is None:
            return []
        feat_names = list(getattr(self, "feature_names", []) or [])
        if used_binner:
            # If caller forced a single raw feature, respect it.
            if restrict_to_feature is not None and int(restrict_to_feature) >= 0:
                return [int(restrict_to_feature)]
            names = getattr(self, "fsg_feature_names", None)
            feats = []
            if names:
                for col in support_idx:
                    ci = int(col)
                    if ci < 0 or ci >= len(names):
                        continue
                    name = str(names[ci])
                    raw_name = name.split("<=")[0].strip()
                    # map raw_name back to index if possible
                    try:
                        if raw_name in feat_names:
                            feats.append(feat_names.index(raw_name))
                            continue
                    except Exception:
                        pass
                    if raw_name.startswith("f_"):
                        try:
                            feats.append(int(raw_name.split("_", 1)[1]))
                            continue
                        except Exception:
                            pass
                    # last resort: digits in raw_name
                    import re
                    m = re.search(r'(\d+)', raw_name)
                    if m:
                        feats.append(int(m.group(1)))
            return feats
        # No binner: design_cols maps X_fit columns to raw columns.
        if design_cols:
            return [int(design_cols[int(i)]) for i in support_idx if 0 <= int(i) < len(design_cols)]
        return [int(i) for i in support_idx if 0 <= int(i) < raw_dim]

    def _columns_for_feature_in_binner(self, feature_idx):
        return None  # Placeholder; depends on specific binner implementation

    def _decision_feature_sets(self, X_cpu):
        """Return list of feature sets (one per sample) used along all tree paths."""
        import numpy as np
        X_np = np.asarray(X_cpu)
        n = X_np.shape[0]
        per_sample = [set() for _ in range(n)]
        for root in self.tree_ensemble:
            stack = [(root, np.arange(n), set())]
            while stack:
                node, idxs, path_feats = stack.pop()
                if idxs.size == 0:
                    continue
                if not node.sub_rules:
                    feats = set(path_feats)
                    base_feats = getattr(node, "base_used_features", None) or []
                    feats.update(int(f) for f in base_feats)
                    for i in idxs:
                        per_sample[int(i)].update(feats)
                    continue
                for sub_rule in node.sub_rules:
                    feat_idx = int(sub_rule.feature_idx)
                    new_path = set(path_feats); new_path.add(feat_idx)
                    mask = X_np[idxs, feat_idx] <= sub_rule.threshold
                    left_idx = idxs[mask]
                    right_idx = idxs[~mask]
                    if left_idx.size:
                        stack.append((sub_rule.left_child, left_idx, new_path))
                    if right_idx.size:
                        stack.append((sub_rule.right_child, right_idx, new_path))
        return per_sample


    def _per_sample_active_shape_counts(self, X_cpu):
        """Return the number of active leaf shape functions for each sample.
        Sums leaf shape function counts across trees (same feature in multiple
        trees counts multiple times — total shape function evaluations)."""
        X_np = np.asarray(X_cpu)
        n = X_np.shape[0]
        per_sample_counts = np.zeros(n, dtype=float)
        for root in self.tree_ensemble:
            stack = [(root, np.arange(n))]
            while stack:
                node, idxs = stack.pop()
                if idxs.size == 0:
                    continue
                if not node.sub_rules:
                    per_leaf_shape_count = len(set(getattr(node, "base_used_features", []) or []))
                    per_sample_counts[idxs] += per_leaf_shape_count
                    continue
                for sub_rule in node.sub_rules:
                    feat_idx = int(sub_rule.feature_idx)
                    mask = X_np[idxs, feat_idx] <= sub_rule.threshold
                    if mask.any():
                        stack.append((sub_rule.left_child, idxs[mask]))
                    if (~mask).any():
                        stack.append((sub_rule.right_child, idxs[~mask]))
        return per_sample_counts

    def compute_sparsity_metrics(self, X_reference=None):
        """Compute sparsity metrics. Optionally supply data to estimate decision sparsity."""
        leaves = [n for root in self.tree_ensemble for n in root.get_all_nodes() if not n.sub_rules]
        step_sparsity = sum(int(getattr(n, "leaf_support_size", 0)) for n in leaves)
        shape_func_sparsity = None
        if X_reference is not None:
            per_sample_shape_counts = self._per_sample_active_shape_counts(X_reference)
            shape_func_sparsity = float(np.mean(per_sample_shape_counts)) if per_sample_shape_counts.size else 0.0
        else:
            shape_func_sparsity = float(sum(len(set(getattr(n, "base_used_features", []) or [])) for n in leaves))

        used_features = set()
        for root in self.tree_ensemble:
            for node in root.get_all_nodes():
                for sub_rule in node.sub_rules:
                    used_features.add(int(sub_rule.feature_idx))
                if not node.sub_rules:
                    base_feats = getattr(node, "base_used_features", None) or []
                    used_features.update(int(f) for f in base_feats)
        variable_sparsity = len(used_features)

        decision_sparsity = None
        if X_reference is not None and step_sparsity > 0:
            per_sample_sets = self._decision_feature_sets(X_reference)
            decision_sparsity = float(np.mean([len(s) for s in per_sample_sets])) if per_sample_sets else 0.0

        return {
            "step_sparsity": step_sparsity,
            "shape_func_sparsity": shape_func_sparsity,
            "variable_sparsity": variable_sparsity,
            "decision_sparsity": decision_sparsity,
        }

    def _make_binner_transformer(self):
        binner = self.fsg_binner
        Xtr_ref = self._X_train_ref
        Xtr_bin = self.fsg_X_train
        def tf(X):
            if X is Xtr_ref and Xtr_bin is not None: return Xtr_bin
            Z, _ = binner.transform_numpy(X, feature_names=self.feature_names)
            return Z
        return tf

    def _perform_best_action(self, X_cpu, z_targets, weights_targets):
        xp = self.xp
        X_device = xp.asarray(X_cpu)
        z_device = xp.asarray(z_targets)
        w_device = xp.asarray(weights_targets) if weights_targets is not None else None

        import time
        t0 = time.time()

        global_candidates = self._evaluate_global_refinement(X_device, z_device, w_device)
        local_candidates  = self._evaluate_local_refinement(X_device, z_device, w_device)

        best_global = max(global_candidates, key=lambda t: float(t[0])) if global_candidates else None
        best_local = max(local_candidates, key=lambda t: float(t[0])) if local_candidates else None

        if not best_global and not best_local: return

        alpha = self.structure_alpha

        # Stochastic coordination: sample from union by gain density
        if getattr(self, "stochastic_coord", False):
            pool = []
            total_n = float(X_device.shape[0])
            for g, rule in (global_candidates or []):
                dens = float(g) / (total_n ** alpha) if total_n > 0 else 0.0
                if dens > 0: pool.append((dens, "global", rule))
            for g, action, _, size in (local_candidates or []):
                n_local = float(size) if size else 1.0
                dens = float(g) / (n_local ** alpha)
                if dens > 0: pool.append((dens, "local", action))

            if pool:
                weights = np.array([p[0] for p in pool], dtype=float)
                probs = weights / weights.sum()
                idx = int(self._rng.choice(len(pool), p=probs))
                kind, payload = pool[idx][1], pool[idx][2]
                search_time = time.time() - t0
                if kind == "global":
                    self._execute_global_refinement(payload, X_cpu, z_targets, weights_targets)
                else:
                    self._execute_local_refinement(payload, X_cpu, z_targets, weights_targets)
                return search_time

        chosen = None
        if best_global and best_local:
            gain_g, rule_g = float(best_global[0]), best_global[1]
            affected_global = float(X_device.shape[0])
            density_g = gain_g / (affected_global ** alpha)

            gain_l, action_l, _, size_l = best_local
            density_l = float(gain_l) / ((float(size_l) if size_l else 1.0) ** alpha)

            if density_g > density_l: chosen = ("global", rule_g)
            elif density_l > density_g: chosen = ("local", action_l)
            else: chosen = ("global", rule_g) if gain_g >= gain_l else ("local", action_l)
        elif best_global:
            chosen = ("global", best_global[1])
        else:
            chosen = ("local", best_local[1])

        kind, payload = chosen
        search_time = time.time() - t0
        if kind == "global":
            self._execute_global_refinement(payload, X_cpu, z_targets, weights_targets)
        else:
            self._execute_local_refinement(payload, X_cpu, z_targets, weights_targets)
        return search_time

    # Using reference model to get lb
    def _train_static_reference(self, X_train, y_train):
        try:
            y = np.asarray(y_train).reshape(-1)
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
            clf.fit(X_train, y)
            p = np.clip(clf.predict_proba(X_train)[:, 1], 1e-12, 1-1e-12)
            h = p*(1-p); z = (y-p)/h
            self._static_ref = {"bound_weighted": np.asarray(h*(z**2), dtype=float).reshape(-1),
                                "bound_unweighted": np.asarray((y-p)**2, dtype=float).reshape(-1)}
        except:
            self._static_ref = None

    def _lb_guess_indices(self, indices, use_weighted: bool):
        if self._static_ref is None or len(indices) == 0:
            return 0.0
        vec = self._static_ref.get("bound_weighted" if use_weighted else "bound_unweighted")
        if vec is None:
            return 0.0
        s = float(np.sum(vec[np.asarray(indices, dtype=int)]))
        return self.lb_alpha * s


    def _evaluate_global_refinement(self, X_device, z_device, w_device=None):
        """
        Global-level candidate enumeration.
        Returns list of (gain, rule) sorted by gain desc (length <= topk).
        """
        xp = self.xp
        n = X_device.shape[0]
        sample_indices = xp.arange(n)
        if w_device is None:
            cost_before = xp.sum(z_device ** 2)
        else:
            cost_before = xp.sum(w_device * (z_device ** 2))

        F = X_device.shape[1]
        candidates = []

        for feature_idx in range(F):
            feature_values_device = X_device[:, feature_idx]
            thresholds = self._get_candidate_thresholds(feature_values_device, z_device)
            if thresholds is None or len(thresholds) == 0:
                continue
            for t in thresholds:
                # cheap count filter
                left_mask = feature_values_device <= t
                if xp.sum(left_mask) < self.min_samples_leaf:
                    continue
                if xp.sum(~left_mask) < self.min_samples_leaf:
                    continue
                # precise eval
                cost_after = self._evaluate_split_cost(X_device, sample_indices, feature_idx, t, z_device, w_device)
                if cost_after is None or cost_after >= xp.inf:
                    continue
                gain = float(cost_before - cost_after)
                if gain <= 0.0:
                    continue
                rule = {'split_rule': (int(feature_idx), float(t))}
                candidates.append((gain, rule))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        topk = int(self.stochastic_topk) if (self.stochastic_topk is not None) else int(getattr(self, "k", 1))
        return candidates[:topk]

    def _evaluate_local_refinement(self, X_device, z_device, w_device=None):
        """
        Local-level candidate generation.
        For top nodes (coarse filter), compute the best split using depth=1 search and return per-node bests.
        Returns list of (gain, action, cost_before, size).
        """
        xp = self.xp
        best_list = []

        raw_pool = self._get_candidate_nodes_and_indices(X_device)
        scored = []
        for node, idx_dev in raw_pool:
            # Current node energy (matches split-eval cost_before)
            z_leaf = z_device[idx_dev]

            # Add parent contribution back only when we will silence parent at execution time.
            # If silence_parent=False, we keep parent.base_model and the new sub-rule should fit the
            # residual update (z_leaf) rather than (z_leaf + parent_contrib).
            try:
                X_leaf_cpu = xp.asnumpy(X_device[idx_dev]) if self.device == "GPU" else np.asarray(X_device[idx_dev])
            except Exception:
                X_leaf_cpu = None
            if getattr(self, "silence_parent", True) and X_leaf_cpu is not None and node.base_model is not None:
                try:
                    parent_contrib = node.base_model(X_leaf_cpu)
                    z_leaf = z_leaf + xp.asarray(parent_contrib, dtype=z_leaf.dtype)
                except Exception:
                    pass

            ssr_before = float(
                xp.sum(z_leaf ** 2) if (w_device is None)
                else xp.sum(w_device[idx_dev] * (z_leaf ** 2))
            )
            if ssr_before <= 0.0:
                continue

            # Teacher-bound energy on the same samples (cached on CPU)
            try:
                idx_cpu = xp.asnumpy(idx_dev) if self.device == "GPU" else idx_dev
            except Exception:
                idx_cpu = idx_dev

            # Use weighted lower bound only when using second-order weights
            use_w = (w_device is not None) and (getattr(self, "step_mode", "grad") != "grad")
            lb = self._lb_guess_indices(idx_cpu, use_weighted=use_w)
            potential = float(ssr_before - lb)

            # Keep only positive-potential nodes for coarse filtering
            if potential > float(getattr(self, "lb_abs_threshold", 0.0)):
                scored.append((potential, ssr_before, node, idx_dev))

        if not scored:
            return []

        # coarse top-k nodes
        scored.sort(key=lambda t: t[0], reverse=True)
        top_nodes = scored[: int(getattr(self, "k", len(scored)))]

        for potential, ssr_before, node, idx_dev in top_nodes:
            try:
                # use depth-1 search for best split for node
                w_leaf = None if (w_device is None) else w_device[idx_dev]
                # recompute z_leaf with parent contribution for this node
                z_leaf = z_device[idx_dev]
                try:
                    X_leaf_cpu = xp.asnumpy(X_device[idx_dev]) if self.device == "GPU" else np.asarray(X_device[idx_dev])
                except Exception:
                    X_leaf_cpu = None
                if getattr(self, "silence_parent", True) and X_leaf_cpu is not None and node.base_model is not None:
                    try:
                        parent_contrib = node.base_model(X_leaf_cpu)
                        z_leaf = z_leaf + xp.asarray(parent_contrib, dtype=z_leaf.dtype)
                    except Exception:
                        pass

                cost_after, best_structure = self._find_best_shallow_tree(
                    X_device, idx_dev, z_leaf, depth=1, upper_bound=ssr_before, weights_device=w_leaf
                )
            except Exception:
                continue
            if best_structure is None:
                continue
            gain = float(ssr_before - cost_after)
            if gain <= 0.0:
                continue
            action = {'node_to_split': node, 'rule': best_structure}
            size = int(idx_dev.shape[0]) if hasattr(idx_dev, 'shape') else len(idx_dev)
            best_list.append((gain, action, float(ssr_before), size))

        if not best_list:
            return []
        best_list.sort(key=lambda t: t[0], reverse=True)
        topk = int(self.stochastic_topk) if (self.stochastic_topk is not None) else int(getattr(self, "k", 1))
        return best_list[:topk]

    def _evaluate_split_cost(self, X_device, sample_indices, feature_idx, threshold, z_device, weights_device=None):
        """
        Evaluate cost_after for splitting `sample_indices` by (feature_idx <= threshold).
        Returns numeric cost_after (xp scalar or float). Uses existing _evaluate_leaf for leaf costs.
        """
        xp = self.xp
        # sample_indices may be xp array or numpy array
        try:
            feat_vals = X_device[sample_indices, feature_idx]
        except Exception:
            # fallback: build mask then index
            feat_vals = X_device[:, feature_idx][sample_indices]

        left_mask = feat_vals <= threshold
        right_mask = ~left_mask

        left_count = int(xp.sum(left_mask))
        right_count = int(xp.sum(right_mask))
        if left_count == 0 or right_count == 0:
            return xp.inf

        # produce left/right sample_indices in the same index-space as sample_indices expected by _evaluate_leaf
        left_idx = sample_indices[left_mask]
        right_idx = sample_indices[right_mask]

        # residuals for each side (note z_device is already aligned to sample_indices)
        z_left = z_device[left_mask]
        z_right = z_device[right_mask]

        w_left = None if (weights_device is None) else weights_device[left_mask]
        w_right = None if (weights_device is None) else weights_device[right_mask]

        cost_left, _ = self._evaluate_leaf(left_idx, z_left, w_left) if left_count >= self.min_samples_leaf else self._constant_leaf_cost(z_left, w_left)
        if cost_left == xp.inf:
            return xp.inf
        cost_right, _ = self._evaluate_leaf(right_idx, z_right, w_right) if right_count >= self.min_samples_leaf else self._constant_leaf_cost(z_right, w_right)
        return cost_left + cost_right

    def _find_best_shallow_tree(self, X_device, sample_indices, z_device, depth, upper_bound, weights_device):
        xp = self.xp
        if depth == 0:
            min_cost, _ = self._evaluate_leaf(sample_indices, z_device, weights_device)
            return min_cost, None

        min_total_cost = upper_bound
        best_structure = None
        F = X_device.shape[1]
        min_w_sum = float(getattr(self, "min_weighted_samples_leaf", 0.0)) if (weights_device is not None) else 0.0

        for feature_idx in range(F):
            feature_values_device = X_device[sample_indices, feature_idx]
            thresholds = self._get_candidate_thresholds(feature_values_device, z_device)
            for t in thresholds:
                if self.diag_enabled:
                    self._diag['thresholds_considered'] += 1

                left_mask  = feature_values_device <= t
                right_mask = ~left_mask
                if self.xp.sum(left_mask)  == 0: continue
                if self.xp.sum(right_mask) == 0: continue

                if min_w_sum > 0.0:
                    w_left_sum  = float(xp.sum(weights_device[left_mask]))
                    w_right_sum = float(xp.sum(weights_device[right_mask]))
                    if w_left_sum < min_w_sum or w_right_sum < min_w_sum:
                        continue

                left_indices  = sample_indices[left_mask]
                right_indices = sample_indices[right_mask]
                z_left  = z_device[left_mask]
                z_right = z_device[right_mask]
                w_left  = None if (weights_device is None) else weights_device[left_mask]
                w_right = None if (weights_device is None) else weights_device[right_mask]

                cost_left, _ = self._find_best_shallow_tree(
                    X_device, left_indices, z_left, depth-1, upper_bound=min_total_cost, weights_device=w_left
                ) if left_indices.shape[0] >= self.min_samples_leaf else (self._constant_leaf_cost(z_left, w_left), None)
                if cost_left >= min_total_cost:
                    continue

                cost_right, _ = self._find_best_shallow_tree(
                    X_device, right_indices, z_right, depth-1, upper_bound=(min_total_cost - cost_left), weights_device=w_right
                ) if right_indices.shape[0] >= self.min_samples_leaf else (self._constant_leaf_cost(z_right, w_right), None)
                total_cost = cost_left + cost_right
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_structure = {'split_rule': (feature_idx, float(t))}
        return min_total_cost, best_structure

    def _evaluate_leaf(self, sample_indices, z_device, weights_device, restrict_to_feature=None):
        xp = self.xp
        if len(sample_indices) < self.min_samples_leaf:
            return xp.inf, -1

        strategy = getattr(self, "searching_strategy", "lookahead_spline").lower()

        # Constant lookahead: best weighted mean
        if strategy == "lookahead_constant":
            if weights_device is None:
                mean = xp.mean(z_device)
                sse = xp.sum((z_device - mean) ** 2)
            else:
                wsum = xp.sum(weights_device)
                if wsum <= 0:
                    return xp.inf, -1
                mean = xp.sum(weights_device * z_device) / wsum
                sse = xp.sum(weights_device * (z_device - mean) ** 2)
            return float(sse), -1

        # Greedy: evaluate leaf by actually fitting FSG (same as final leaf fitter)
        if strategy == "greedy":
            try:
                idx_np = np.asarray(sample_indices, dtype=int)
                X_ref = getattr(self, "_X_train_ref", None)
                if X_ref is None:
                    return xp.inf, -1
                X_leaf = np.asarray(X_ref)[idx_np]
                z_np = np.asarray(z_device, dtype=float)
                w_np = None if weights_device is None else np.asarray(weights_device, dtype=float)
                _, best_sse, _, _ = self._fit_leaf_with_fastsparse(X_leaf, z_np, w_np, restrict_to_feature=None)
                return float(best_sse), -1
            except Exception:
                return xp.inf, -1

        # Default: spline lookahead (原有代理，单特征样条 argmin)
        # 注意：避免在 lookahead 中调用 FSG 真拟合，以免极慢。
        if self.All_Bases_device is None:
            return xp.sum(z_device ** 2), -1

        B = self.All_Bases_device[:, sample_indices, :]
        z_col = z_device.reshape((1, len(sample_indices), 1))

        if weights_device is None:
            Bt = xp.transpose(B, (0, 2, 1))
            A = Bt @ B 
            b = Bt @ z_col
        else:
            w = weights_device.reshape((1, len(sample_indices), 1))
            sqrt_w = xp.sqrt(w)
            B_t = B * sqrt_w
            y_t = z_col * sqrt_w
            Bt = xp.transpose(B_t, (0, 2, 1))
            A = Bt @ B_t
            b = Bt @ y_t

        K = B.shape[2]
        A += xp.eye(K, dtype=A.dtype) * 0.1
        try:
            coeffs = xp.linalg.solve(A, b)
        except Exception:
            return xp.inf, -1

        y_pred = (B @ coeffs)
        resid  = (z_col - y_pred)
        if weights_device is None:
            sse = xp.sum(resid ** 2, axis=1).flatten()
        else:
            w = weights_device.reshape((1, len(sample_indices), 1))
            sse = xp.sum(w * (resid ** 2), axis=1).flatten()

        j = int(xp.argmin(sse))
        return sse[j], j

    def _constant_leaf_cost(self, z_slice, w_slice):
        xp = self.xp
        if z_slice is None or len(z_slice) == 0:
            return xp.inf
        if w_slice is None:
            mean = xp.mean(z_slice)
            return float(xp.sum((z_slice - mean) ** 2))
        wsum = xp.sum(w_slice)
        if wsum <= 0:
            return xp.inf
        mean = xp.sum(w_slice * z_slice) / wsum
        return float(xp.sum(w_slice * (z_slice - mean) ** 2))


    def _execute_global_refinement(self, rule, X_cpu, z_train, w_train):
        if rule is None:
            return
        root = RuleSetNode(depth=0, tree_id=self.next_tree_id, verbose=getattr(self, "verbose", False))
        self.next_tree_id += 1

        self._add_sub_rule_to_node(root, rule, X_cpu, z_train, w_train)
        self.tree_ensemble.append(root)

    def _execute_local_refinement(self, action, X_cpu, z_train, w_train):
        if action is None or action['rule'] is None: 
            return
        self._add_sub_rule_to_node(action['node_to_split'], action['rule'], X_cpu, z_train, w_train)

    # def _add_sub_rule_to_node(self, parent_node, rule, X_cpu, y_cpu):
    #     feature_idx, threshold = rule['split_rule']
    #     idx = self.get_indices_for_node(parent_node, X_cpu)
    #     Xp = X_cpu[idx]
    #     yp = y_cpu[idx]

    #     raw_p = self._get_raw_prediction(Xp)
    #     r_p = self.loss_function_.negative_gradient(yp, raw_p)

    #     left_mask = Xp[:, feature_idx] <= threshold
    #     right_mask = ~left_mask

    #     X_left,  z_left  = Xp[left_mask],  r_p[left_mask]
    #     X_right, z_right = Xp[right_mask], r_p[right_mask]

    #     L = RuleSetNode(tree=parent_node.tree, depth=parent_node.depth+1, is_left_child=True)
    #     R = RuleSetNode(tree=parent_node.tree, depth=parent_node.depth+1, is_left_child=False)

    #     L.base_model, _, L.leaf_support_size = self._fit_leaf(X_left,  z_left,  sample_weights=None)
    #     R.base_model, _, R.leaf_support_size = self._fit_leaf(X_right, z_right, sample_weights=None)
    #     L.num_samples, R.num_samples = X_left.shape[0], X_right.shape[0]

    #     sub = RuleTree(rule['split_rule'], L, R)
    #     sub.parent_node = parent_node
    #     L.parent_rule = sub; R.parent_rule = sub
    #     parent_node.sub_rules.append(sub)
    #     self._bump_cache_epoch()
    #     if self.verbose:
    #         print(f"[LeafFit] feature={feature_idx}, thr={threshold:.6g} | left_support={L.leaf_support_size}, right_support={R.leaf_support_size}")

    def _add_sub_rule_to_node(self, parent_node, rule, X_cpu, z_global, w_global):
        feature_idx, threshold = rule['split_rule']
        idx = self.get_indices_for_node(parent_node, X_cpu)
        Xp, zp = X_cpu[idx], z_global[idx]
        wp = w_global[idx] if w_global is not None else np.ones_like(zp)
        parent_node.num_samples = Xp.shape[0]

        # Parent contribution
        parent_contribution = np.zeros_like(zp)
        if parent_node.base_model is not None:
            try:
                if callable(parent_node.base_model):
                     parent_contribution = parent_node.base_model(Xp)
            except: pass
        
        # If we silence parent, children must replace parent's contribution as well.
        # Otherwise (silence_parent=False), children are additive updates on top of the existing parent.
        if getattr(self, "silence_parent", True):
            targets_for_children = zp + parent_contribution
        else:
            targets_for_children = zp
        left_mask  = Xp[:, feature_idx] <= threshold
        
        L = RuleSetNode(depth=parent_node.depth+1, is_left_child=True,  tree_id=parent_node.tree_id,
                verbose=getattr(self, "verbose", False))
        R = RuleSetNode(depth=parent_node.depth+1, is_left_child=False, tree_id=parent_node.tree_id,
                verbose=getattr(self, "verbose", False))

        # Compute val data slices for val-based lambda selection in leaf fitter
        X_val_L = X_val_R = y_val_L = y_val_R = raw_val_L = raw_val_R = None
        _X_val = getattr(self, '_X_val_ref', None)
        _y_val = getattr(self, '_y_val_ref', None)
        if _X_val is not None and _y_val is not None:
            try:
                idx_val = self.get_indices_for_node(parent_node, _X_val)
                if len(idx_val) > 0:
                    Xv = np.asarray(_X_val)[idx_val]
                    yv = np.asarray(_y_val)[idx_val]
                    left_val_mask = Xv[:, feature_idx] <= threshold
                    X_val_L, y_val_L = Xv[left_val_mask], yv[left_val_mask]
                    X_val_R, y_val_R = Xv[~left_val_mask], yv[~left_val_mask]
                    # Use cached full-val predictions if available (set in fit() loop)
                    cached_raw_val = getattr(self, '_current_raw_val_pred', None)
                    if cached_raw_val is not None and len(cached_raw_val) == len(_X_val):
                        raw_val_all = np.asarray(cached_raw_val)[idx_val]
                    else:
                        raw_val_all = np.asarray(self._get_raw_prediction(Xv, on_device=False)).ravel()
                    raw_val_L = raw_val_all[left_val_mask]
                    raw_val_R = raw_val_all[~left_val_mask]
            except Exception:
                pass  # val slice computation is best-effort; fall back to BIC

        L.base_model, _, L.leaf_support_size, L.base_used_features = self._fit_leaf(
            Xp[left_mask], targets_for_children[left_mask], wp[left_mask],
            X_val_leaf=X_val_L, y_val_leaf=y_val_L, raw_val_leaf=raw_val_L)
        R.base_model, _, R.leaf_support_size, R.base_used_features = self._fit_leaf(
            Xp[~left_mask], targets_for_children[~left_mask], wp[~left_mask],
            X_val_leaf=X_val_R, y_val_leaf=y_val_R, raw_val_leaf=raw_val_R)
        L.num_samples, R.num_samples = int(np.sum(left_mask)), int(np.sum(~left_mask))

        sub = RuleTree(rule['split_rule'], L, R)
        sub.parent_node = parent_node
        L.parent_rule = sub; R.parent_rule = sub
        parent_node.sub_rules.append(sub)

        if getattr(self, "silence_parent", True):
            parent_node.base_model = lambda x: np.zeros(x.shape[0])
            parent_node.leaf_support_size = 0

    def get_search_time_stats(self):
        import numpy as np
        if not hasattr(self, "_search_times") or not self._search_times:
            return np.nan, np.nan
        arr = np.asarray(self._search_times, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    def get_indices_for_node(self, target_node, X_cpu):
        if target_node.parent_rule is None:
            return np.arange(X_cpu.shape[0])

        path, curr = [], target_node
        while curr.parent_rule is not None:
            path.append((curr.parent_rule, curr.is_left_child))
            curr = curr.parent_rule.parent_node
        path.reverse()

        current_indices = np.arange(X_cpu.shape[0])
        for rule, is_left in path:
            if len(current_indices) == 0:
                break
            X_subset = X_cpu[current_indices]
            mask = X_subset[:, rule.feature_idx] <= rule.threshold
            if not is_left:
                mask = ~mask
            current_indices = current_indices[mask]

        return current_indices

    def _get_candidate_nodes_and_indices(self, X_device):
      xp = self.xp
      X_cpu = xp.asnumpy(X_device) if self.device == 'GPU' else X_device

      all_nodes = [node for root in self.tree_ensemble for node in root.get_all_nodes()]

      potential = [
          n for n in all_nodes 
          if n.depth < self.max_depth and n.num_samples >= 2 * self.min_samples_leaf
      ] 
      if self.diag_enabled:
          self._diag['nodes_considered_local'] = len(potential)

      pool = []
      for node in potential:
          idx_cpu = self.get_indices_for_node(node, X_cpu)
          if len(idx_cpu) == 0:
              continue
          if self.diag_enabled:
              self._diag['nodes_after_root_filter'] += 1
          pool.append((node, xp.asarray(idx_cpu) if self.device == 'GPU' else idx_cpu))

      return pool

    def _update_global_gam(self, X_cpu, z_targets, sample_weights):
        # Weighted least squares style update for global GAM (once at iteration 0)
        if fsg is None: return
        if self.tree_ensemble or getattr(self, "_global_gam_fitted_once", False): return

        Ztr = np.asarray(X_cpu, dtype=np.float64)
        sw = np.sqrt(np.asarray(sample_weights, dtype=np.float64))
        Ztr_w = Ztr * sw[:, None]
        z_w = np.asarray(z_targets, dtype=np.float64) * sw
        
        try:
            path = fsg.fit(Ztr_w, z_w, penalty="L0", max_support_size=self.global_gam_max_support_size, loss="SquaredError")
        except: return

        lambdas = list(path.lambda_0[0])
        best_lam, best_score = None, np.inf
        for idx, lam in enumerate(lambdas):
            try:
                inter = float(path.intercepts[0][idx])
                coefs = path.coeffs[0][:, idx].toarray().ravel()
                pred = (inter + Ztr_w @ coefs).ravel()
                sse = float(np.sum((z_w - pred)**2))
                if sse < best_score: best_score, best_lam = sse, lam
            except: continue

        if best_lam is not None:
            self.global_gam = path
            self.global_gam_lambda = best_lam
            try: 
                bi = lambdas.index(best_lam)
                self.global_gam_coeffs_ = path.coeffs[0][:, bi].toarray().ravel()
            except: pass
        self._global_gam_fitted_once = True

    def _get_candidate_thresholds(self, feature_values_device, y_residuals_device):
        # GPU path: stay on-device to avoid costly transfers
        if self.device == 'GPU':
            fv = feature_values_device
            rz = y_residuals_device
            uniq = cp.unique(fv)
            if uniq.size <= 1:
                return cp.array([])
            if uniq.size <= self.max_thresholds:
                return (uniq[:-1] + uniq[1:]) / 2.0

            sort_idx = cp.argsort(fv)
            sf = fv[sort_idx]
            sr = rz[sort_idx]
            diff_feat = sf[:-1] != sf[1:]
            sign_change = cp.sign(sr[:-1]) != cp.sign(sr[1:])

            all_cand = (sf[:-1] + sf[1:]) / 2.0
            sign_cand = cp.unique(all_cand[diff_feat & sign_change])

            # Primary: use sign-change points only
            if sign_cand.size > 0:
                if sign_cand.size <= self.max_thresholds:
                    return sign_cand
                idx = cp.linspace(0, sign_cand.size - 1, self.max_thresholds, dtype=cp.int32)
                return sign_cand[idx]

            # Fallback: if no sign-change points exist, uniformly sample from all diff_feat points
            all_diff_cand = cp.unique(all_cand[diff_feat])
            if all_diff_cand.size > 0:
                if all_diff_cand.size <= self.max_thresholds:
                    return all_diff_cand
                idx = cp.linspace(0, all_diff_cand.size - 1, self.max_thresholds, dtype=cp.int32)
                return all_diff_cand[idx]
            return cp.array([])

        # CPU fallback (numpy)
        feature_values_cpu = feature_values_device
        y_residuals_cpu = y_residuals_device

        unique_features = np.unique(feature_values_cpu)
        if len(unique_features) <= 1: return np.array([])
        if len(unique_features) <= self.max_thresholds:
            return (unique_features[:-1] + unique_features[1:]) / 2.0

        sorted_indices = np.argsort(feature_values_cpu)
        sorted_features = feature_values_cpu[sorted_indices]
        sorted_residuals = y_residuals_cpu[sorted_indices]

        diff_feat = sorted_features[:-1] != sorted_features[1:]
        sign_change = np.sign(sorted_residuals[:-1]) != np.sign(sorted_residuals[1:])

        all_cand = (sorted_features[:-1] + sorted_features[1:]) / 2.0
        sign_cand = np.unique(all_cand[diff_feat & sign_change])

        # Primary: use sign-change points only
        if len(sign_cand) > 0:
            if len(sign_cand) <= self.max_thresholds:
                return sign_cand
            indices = np.linspace(0, len(sign_cand) - 1, self.max_thresholds, dtype=int)
            return sign_cand[indices]

        # Fallback: if no sign-change points exist, uniformly sample from all diff_feat points
        all_diff_cand = np.unique(all_cand[diff_feat])
        if len(all_diff_cand) > 0:
            if len(all_diff_cand) <= self.max_thresholds:
                return all_diff_cand
            indices = np.linspace(0, len(all_diff_cand) - 1, self.max_thresholds, dtype=int)
            return all_diff_cand[indices]
        return np.array([])
    
    def _predict_gam_part(self, X_cpu):
        X_view = self.global_gam_transformer(X_cpu) if self.global_gam_transformer else X_cpu
        if self.global_gam and self.global_gam_lambda and fsg:
            try:
                cands = self.global_gam.lambda_0[0]
                try: bi = list(cands).index(self.global_gam_lambda)
                except: bi = int(np.argmin(np.abs(np.array(cands, float)-float(self.global_gam_lambda))))
                inter = self.global_gam.intercepts[0][bi]
                coefs = self.global_gam.coeffs[0][:, bi].toarray()
                return (inter + X_view @ coefs).flatten()
            except: return np.zeros(X_cpu.shape[0])
        return np.zeros(X_cpu.shape[0])
    
    def _fit_leaf_fsg_auto(self, X_leaf_raw_or_design, y_target, max_support=8, restrict_to_feature=None):
        """
        Fit fastsparsegams on input that may be either:
          - raw features (n x p), in which case we will binarize via self.fsg_binner; OR
          - already design/binned matrix (n x D), in which case use as-is.

        Returns:
            leaf_predict_fn(X_rows_raw) -> preds (1d array)
            best_sse (float) on the training rows (w.r.t. y_target)
            support_size (int)
            path (object returned by fsg.fit) or None
            best_idx (int index in lambda path) or None
        Notes:
            - restrict_to_feature: None | int (orig feature index) | list of design cols.
              If int and we have a binner, we'll map to design cols via _columns_for_feature_in_binner.
        """
        X_in = np.asarray(X_leaf_raw_or_design, dtype=np.float64)
        y = np.asarray(y_target, dtype=np.float64).reshape(-1)
        n = y.shape[0]
        if n == 0:
            return (lambda X: np.zeros(np.asarray(X).shape[0], dtype=float)), np.inf, 0, None, None

        # determine if input is already design (binarized)
        is_design_input = False
        if getattr(self, "fsg_X_train", None) is not None:
            try:
                if X_in.shape[1] == np.asarray(self.fsg_X_train).shape[1]:
                    is_design_input = True
            except Exception:
                pass

        used_binner = False
        X_design = None

        # If input is not design but we have a binner, transform
        if (not is_design_input) and getattr(self, "fsg_binner", None) is not None:
            try:
                Z, _ = self.fsg_binner.transform_numpy(X_in, feature_names=getattr(self, "feature_names", None))
                X_design = np.asarray(Z, dtype=np.float64)
                used_binner = True
            except Exception:
                # fallback: treat input as raw and continue (no binner)
                X_design = None
                used_binner = False
        elif is_design_input:
            X_design = X_in.copy()
            used_binner = False  # already design; not using our binner for predictions
        else:
            X_design = None
            used_binner = False

        # handle restrict_to_feature: map original feature idx -> design columns if necessary
        design_cols_to_use = None
        if restrict_to_feature is not None:
            # if list-like, assume design columns list
            if isinstance(restrict_to_feature, (list, tuple, np.ndarray)):
                design_cols_to_use = list(restrict_to_feature)
            else:
                # single int: try map using binner if available
                try:
                    j = int(restrict_to_feature)
                    if getattr(self, "fsg_binner", None) is not None:
                        cols = self._columns_for_feature_in_binner(j)
                        if cols:
                            design_cols_to_use = list(cols)
                        else:
                            # fallback to using the single original column if input is raw
                            # but fsg.fit expects design columns; so if X_in was raw and no mapping, we will pick raw column
                            design_cols_to_use = [j] if (not is_design_input) else None
                    else:
                        # no binner -> assume input was raw and restrict by raw column index
                        design_cols_to_use = [j] if not is_design_input else None
                except Exception:
                    design_cols_to_use = None

        # choose matrix to fit on: prefer X_design if we have it, else use X_in (raw)
        if X_design is not None:
            X_fit = X_design
            fit_used_binner = used_binner
        else:
            X_fit = X_in
            fit_used_binner = False

        # If restricting to specific design columns, slice
        if design_cols_to_use:
            # sanity: remove out-of-bounds cols
            design_cols_to_use = [c for c in design_cols_to_use if 0 <= c < X_fit.shape[1]]
            if len(design_cols_to_use) == 0:
                # nothing to fit -> fallback to constant predictor
                c = float(np.mean(y))
                return (lambda X, _c=c: np.full(np.asarray(X).shape[0], _c, dtype=float)), float(np.sum((y - c) ** 2)), 0, None, None
            X_fit = X_fit[:, design_cols_to_use]

        # guard: if not enough samples or fsg missing, fallback to constant predictor
        if (X_fit.shape[1] == 0) or (not hasattr(fsg, "fit")) or (X_fit.shape[0] < max(1, getattr(self, "min_samples_leaf", 1))):
            c = float(np.mean(y))
            dummy = lambda X, _c=c: np.full(np.asarray(X).shape[0], _c, dtype=float)
            sse = float(np.sum((y - c) ** 2))
            return dummy, sse, 0, None, None

        max_k = int(max_support)
        leaf_algo = getattr(self, "leaf_fsg_algorithm", "CDPSI")
        leaf_loss = getattr(self, "leaf_fsg_loss", "SquaredError")

        # call fsg.fit on X_fit (which is design or selected cols)
        with sparse_triplet_ravel():
            try:
                path = fsg.fit(X_fit, y,
                               penalty="L0",
                               max_support_size=max_k,
                               algorithm=leaf_algo,
                               loss=leaf_loss)
            except Exception as e:
                # fit failed: fallback to constant predictor
                c = float(np.mean(y))
                dummy = lambda X, _c=c: np.full(np.asarray(X).shape[0], _c, dtype=float)
                sse = float(np.sum((y - c) ** 2))
                return dummy, sse, 0, None, None

        # choose best lambda index by SSE on training rows (consistent)
        best_idx = None
        best_sse = np.inf
        cand_lams = getattr(path, "lambda_0", None)
        if cand_lams is None or len(cand_lams) == 0:
            # no path -> fallback
            c = float(np.mean(y))
            dummy = lambda X, _c=c: np.full(np.asarray(X).shape[0], _c, dtype=float)
            sse = float(np.sum((y - c) ** 2))
            return dummy, sse, 0, path, None

        for idx in range(len(path.lambda_0[0])):
            try:
                intercept = float(path.intercepts[0][idx])
                coeffs = path.coeffs[0][:, idx].toarray().reshape(-1)
                pred = (intercept + X_fit @ coeffs).ravel()
                sse = float(np.sum((y - pred) ** 2))
                if np.isfinite(sse) and sse < best_sse:
                    best_sse = sse
                    best_idx = idx
            except Exception:
                continue

        if best_idx is None:
            c = float(np.mean(y))
            dummy = lambda X, _c=c: np.full(np.asarray(X).shape[0], _c, dtype=float)
            sse = float(np.sum((y - c) ** 2))
            return dummy, sse, 0, path, None

        # best coefficients on X_fit basis
        intercept = float(path.intercepts[0][best_idx])
        coeffs_vector = path.coeffs[0][:, best_idx].toarray().reshape(-1)
        tol = float(getattr(self, "leaf_support_tol", 1e-6))
        support_mask = np.abs(coeffs_vector) > tol
        support_size = int(np.sum(support_mask))
        
        fit_on_design = (X_design is not None)
        # Build prediction wrapper that accepts raw X rows
        def _leaf_predict_fn(X_rows_raw, _path=path, _best_idx=best_idx, _inter=intercept, _coefs=coeffs_vector,
                             _fit_used_binner=fit_used_binner, _design_cols=design_cols_to_use, _fit_on_design=fit_on_design):
            Xr = np.asarray(X_rows_raw, dtype=np.float64)

            # If the model was fitted on design (either because we transformed input or because user provided design),
            # we MUST transform incoming raw rows into the same design representation before applying coefficients.
            if _fit_on_design:
                # require binner available to transform raw -> design
                if getattr(self, "fsg_binner", None) is None:
                    raise RuntimeError("[LeafPredict] fitted on design but no fsg_binner available to transform raw inputs.")
                try:
                    Z, _ = self.fsg_binner.transform_numpy(Xr, feature_names=getattr(self, "feature_names", None))
                    Z = np.asarray(Z, dtype=np.float64)
                except Exception:
                    # if transform fails, raise or return zeros to be safe (choose raise for visibility)
                    raise
                Xpred = Z if _design_cols is None else Z[:, _design_cols]
            else:
                # fitted on raw matrix: use raw inputs (optionally subset columns)
                Xpred = Xr if _design_cols is None else Xr[:, _design_cols]

            # ensure shape match
            if Xpred.shape[1] != _coefs.shape[0]:
                raise ValueError(f"[LeafPredict] design cols mismatch: Xpred.shape[1]={Xpred.shape[1]} vs coef_len={_coefs.shape[0]}")
            return (float(_inter) + Xpred @ _coefs).ravel()

        return _leaf_predict_fn, float(best_sse), support_size, path, best_idx


    def _eval_node_base_on_design(self, node, X_design_rows):
        """Evaluate node.base_model on design-matrix rows (X_design may be binned or raw).
           Supports tuple(path, best_lambda), callable, and sklearn-like models.
        """
        m = getattr(node, "base_model", None)
        Xrows = np.asarray(X_design_rows)
        if m is None:
            return np.zeros(Xrows.shape[0], dtype=float)
        # tuple case => path object
        if isinstance(m, tuple) and len(m) == 2:
            path_obj, best_lam = m
            try:
                cand = path_obj.lambda_0[0]
                try:
                    bi = list(cand).index(best_lam)
                except Exception:
                    arr = np.array(cand, dtype=float)
                    bi = int(np.argmin(np.abs(arr - float(best_lam))))
                intercept = float(path_obj.intercepts[0][bi])
                coeffs = path_obj.coeffs[0][:, bi].toarray().reshape(-1)
                return (intercept + Xrows @ coeffs).ravel()
            except Exception:
                return np.zeros(Xrows.shape[0], dtype=float)
        # callable
        if callable(m):
            try:
                return np.asarray(m(Xrows)).ravel()
            except Exception:
                return np.zeros(Xrows.shape[0], dtype=float)
        # sklearn-like with predict
        if hasattr(m, "predict"):
            try:
                return np.asarray(m.predict(Xrows)).ravel()
            except Exception:
                return np.zeros(Xrows.shape[0], dtype=float)
        return np.zeros(Xrows.shape[0], dtype=float)

    def _backward_fit_leaves(self, X_train=None, y_train=None,
                               X_val=None, y_val=None,
                               max_support: int = 8,
                               require_improvement: bool = True,
                               verbose: bool = True,
                               max_rounds: int = 3):
        """
        Greedy backward refit: for each leaf (sorted by train size descending), fit FSG with
        max_support on that leaf's rows using the *same design matrix* used for training if
        available (self.fsg_X_train), otherwise raw X.
        Accept replacement only if full validation loss improves.
        Iterates for up to max_rounds rounds; stops early if no leaf is accepted in a round.
        Large leaves are processed first (Fix A) so their stable fits inform the residuals seen
        by small leaves. Multiple rounds (Fix B) give small leaves a second chance on the
        improved residuals.
        """
        Xtr = self._X_train_ref if X_train is None else X_train
        ytr = self._y_train_ref if y_train is None else y_train
        Xva = self._X_val_ref if X_val is None else X_val
        yva = self._y_val_ref if y_val is None else y_val

        if Xtr is None or ytr is None:
            if verbose: print("[BackwardRefit] missing training data; skipping.")
            return self
        if Xva is None or yva is None:
            if verbose: print("[BackwardRefit] missing validation data; skipping.")
            return self

        # choose design matrices (prefer binned)
        use_binned = getattr(self, "fsg_X_train", None) is not None
        if use_binned:
            Xtr_design = np.asarray(self.fsg_X_train)
            Xva_design = np.asarray(self.fsg_X_val) if getattr(self, "fsg_X_val", None) is not None else None
            # sanity check: rows must align with raw X
            try:
                assert Xtr_design.shape[0] == Xtr.shape[0], "fsg_X_train 行数必须等于 X_train 行数"
                if Xva_design is not None:
                    assert Xva_design.shape[0] == Xva.shape[0], "fsg_X_val 行数必须等于 X_val 行数"
            except AssertionError as e:
                if verbose: print(f"[BackwardRefit] 行对齐断言失败: {e}; 继续但可能出错.")
        else:
            Xtr_design = np.asarray(Xtr)
            Xva_design = np.asarray(Xva)

        # compute current raw preds & residuals
        raw_train = np.asarray(self._get_raw_prediction(Xtr)).ravel()
        r_train = self.loss_function_.negative_gradient(ytr, raw_train)
        raw_val_current = np.asarray(self._get_raw_prediction(Xva)).ravel()
        current_val_loss = float(self.loss_function_(yva, raw_val_current))
        if verbose:
            print(f"[BackwardRefit] start. current val loss = {current_val_loss:.6f} (use_binned={use_binned})")

        # --- Fix A: collect all eligible leaves and sort by train size (largest first) ---
        leaf_entries = []  # list of (n_train, t_idx, node)
        for t_idx, root in enumerate(self.tree_ensemble):
            for node in root.get_all_nodes():
                if getattr(node, "sub_rules", None) and len(node.sub_rules) != 0:
                    continue
                try:
                    idx_tr_pre = self.get_indices_for_node(node, Xtr)
                except Exception:
                    idx_tr_pre = np.array([], dtype=int)
                leaf_entries.append((len(idx_tr_pre), t_idx, node))
        leaf_entries.sort(key=lambda e: -e[0])  # descending by train size

        # --- Fix B: iterate for up to max_rounds; stop when no leaf is accepted ---
        leaf_count = 0
        for round_idx in range(max_rounds):
            n_accepted_this_round = 0
            if verbose:
                print(f"[BackwardRefit] === Round {round_idx + 1}/{max_rounds} ===")

            for n_tr_pre, t_idx, node in leaf_entries:
                # only leaves (no sub_rules)
                if getattr(node, "sub_rules", None) and len(node.sub_rules) != 0:
                    continue

                # get row indices (on original X); keep raw indices for passing into _fit function
                try:
                    idx_train = self.get_indices_for_node(node, Xtr)
                except Exception:
                    idx_train = np.array([], dtype=int)
                if len(idx_train) < self.min_samples_leaf:
                    if verbose:
                        print(f"[BackwardRefit] skip leaf (train size {len(idx_train)} < min_samples_leaf).")
                    continue

                try:
                    idx_val = self.get_indices_for_node(node, Xva)
                except Exception:
                    idx_val = np.array([], dtype=int)
                if len(idx_val) == 0:
                    if verbose:
                        print(f"[BackwardRefit] skip leaf (no val rows).")
                    continue

                # Prepare raw slices (pass raw into _fit_leaf_fsg_auto; it will auto-binarize if needed)
                X_leaf_tr_raw = np.asarray(Xtr)[idx_train]
                X_leaf_val_raw = np.asarray(Xva)[idx_val]

                # prepare design slices (for evaluating old models that expect design)
                X_leaf_tr_design = Xtr_design[idx_train] if Xtr_design is not None else None
                X_leaf_val_design = Xva_design[idx_val] if Xva_design is not None else None

                if verbose:
                    print(f"[BackwardRefit] trying leaf tree={t_idx} train={len(idx_train)} val={len(idx_val)} ...")

                # === determine restrict_to_feature to pass to fitter ===
                restrict_to = None
                # prefer node.base_feature_idx if present (original-feature index)
                if getattr(node, "base_feature_idx", None) is not None and int(node.base_feature_idx) >= 0:
                    try:
                        orig_j = int(node.base_feature_idx)
                        # if we have a binner and want design-col restrict, map to design cols
                        if use_binned and getattr(self, "_columns_for_feature_in_binner", None) is not None:
                            try:
                                cols = self._columns_for_feature_in_binner(orig_j)
                                if cols:
                                    restrict_to = list(cols)   # pass list of design cols
                                else:
                                    # cannot map -> pass original feature index (fitter will decide)
                                    restrict_to = orig_j
                            except Exception:
                                restrict_to = orig_j
                        else:
                            restrict_to = orig_j
                    except Exception:
                        restrict_to = None

                # Fit on raw (auto inside _fit_leaf_fsg_auto will transform if needed)
                try:
                    new_model_fn, new_sse, new_support, new_path, new_idx = \
                        self._fit_leaf_fsg_auto(X_leaf_tr_raw, r_train[idx_train], max_support=max_support, restrict_to_feature=restrict_to)
                except Exception as e:
                    if verbose: print(f"[BackwardRefit] fsg fit failed: {e}")
                    continue

                # evaluate on val: compute old contrib and new contrib for those val indices
                # old_contrib: try calling node.base_model on raw rows (most leaf predictors are wrappers)
                try:
                    old_contrib_val = None
                    if getattr(node, "base_model", None) is not None:
                        try:
                            # try raw interface first
                            old_contrib_val = np.asarray(node.base_model(np.asarray(Xva)[idx_val])).ravel()
                        except Exception:
                            # fallback: evaluate using design-based evaluator
                            if X_leaf_val_design is not None:
                                old_contrib_val = self._eval_node_base_on_design(node, X_leaf_val_design)
                            else:
                                # worst-case zeros
                                old_contrib_val = np.zeros(len(idx_val), dtype=float)
                    else:
                        old_contrib_val = np.zeros(len(idx_val), dtype=float)
                except Exception:
                    old_contrib_val = np.zeros(len(idx_val), dtype=float)

                # new_contrib: new_model_fn returned by _fit_leaf_fsg_auto accepts raw X (it wraps transform)
                try:
                    new_contrib_val = np.asarray(new_model_fn(X_leaf_val_raw)).ravel()
                except Exception:
                    if verbose: print("[BackwardRefit] new model predict failed; skipping leaf.")
                    continue

                # compute old/new train contributions before node.base_model is replaced
                old_contrib_tr = np.zeros(len(idx_train), dtype=float)
                try:
                    if getattr(node, "base_model", None) is not None:
                        old_contrib_tr = np.asarray(node.base_model(X_leaf_tr_raw)).ravel()
                except Exception:
                    pass
                try:
                    new_contrib_tr = np.asarray(new_model_fn(X_leaf_tr_raw)).ravel()
                except Exception:
                    new_contrib_tr = old_contrib_tr.copy()

                # update global raw for val indices (remember tree leaf contributions are scaled by eta)
                raw_val_candidate = raw_val_current.copy()
                raw_val_candidate[idx_val] = raw_val_candidate[idx_val] - (self.eta * old_contrib_val) + (self.eta * new_contrib_val)

                try:
                    val_loss_candidate = float(self.loss_function_(yva, raw_val_candidate))
                except Exception:
                    # fallback
                    val_loss_candidate = float(np.sqrt(mean_squared_error(yva, raw_val_candidate)))

                if verbose:
                    print(f"[BackwardRefit] val_loss_old={current_val_loss:.6f}, val_loss_new={val_loss_candidate:.6f}, new_support={new_support}")

                if (not require_improvement) or (val_loss_candidate < current_val_loss - 1e-12):
                    # accept: commit node.base_model to new model closure (and record path info)
                    # new_model_fn is already wrapper accepting raw X, safe to assign
                    node.base_model = new_model_fn
                    # also store path/idx for later introspection
                    try:
                        node.base_model_path = new_path
                        node.base_model_lambda_idx = new_idx
                    except Exception:
                        pass
                    node.leaf_support_size = int(new_support)
                    # incremental update of raw_train (avoids O(T*N) _get_raw_prediction)
                    raw_train[idx_train] += self.eta * (new_contrib_tr - old_contrib_tr)
                    r_train = self.loss_function_.negative_gradient(ytr, raw_train)
                    raw_val_current = raw_val_candidate
                    current_val_loss = val_loss_candidate
                    n_accepted_this_round += 1
                    if verbose:
                        print(f"[BackwardRefit] accepted replacement -> new val loss {current_val_loss:.6f}")
                else:
                    if verbose:
                        print(f"[BackwardRefit] rejected replacement.")

                leaf_count += 1

            if verbose:
                print(f"[BackwardRefit] Round {round_idx + 1} done: {n_accepted_this_round} leaf(ves) accepted.")
            if n_accepted_this_round == 0:
                if verbose:
                    print("[BackwardRefit] No improvement in this round; stopping early.")
                break

        if verbose:
            print(f"[BackwardRefit] finished. final val loss = {current_val_loss:.6f}")
        return self

    # Support counting 
    def _node_support_size(self, node):
        # Prefer stored leaf_support_size
        if hasattr(node, "leaf_support_size"):
            try:
                return int(node.leaf_support_size)
            except Exception:
                pass

        m = getattr(node, "base_model", None)

        # Tuple means (fsg_path, best_lambda)
        if isinstance(m, tuple) and len(m) == 2:
            path, best_lambda = m
            try:
                cand = path.lambda_0[0]
                try:
                    bi = list(cand).index(best_lambda)
                except Exception:
                    arr = np.array(cand, dtype=float)
                    bi = int(np.argmin(np.abs(arr - float(best_lambda))))
                coeffs = path.coeffs[0][:, bi].toarray()
                return int(np.count_nonzero(coeffs))
            except Exception:
                return 0

        # EBM proxy
        if hasattr(m, "feature_importances_"):
            try:
                fi = np.asarray(m.feature_importances_)
                return int(np.sum(fi > 0))
            except Exception:
                pass

        # Linear-like
        if hasattr(m, "coef_"):
            try:
                coef = np.asarray(m.coef_).ravel()
                return int(np.count_nonzero(coef))
            except Exception:
                pass
        return 0

    def support_summary(self, include_global: bool = True, return_breakdown: bool = True):
        total_support = 0
        global_support = 0
        leaves_support = 0
        per_tree = []
        num_leaf_nodes = 0

        # Global support
        if include_global:
            try:
                if hasattr(self, "global_gam_coeffs_"):
                    global_support = int(np.count_nonzero(np.asarray(self.global_gam_coeffs_)))
                elif (getattr(self, "global_gam", None) is not None and
                      getattr(self, "global_gam_lambda", None) is not None and fsg is not None):
                    cand = self.global_gam.lambda_0[0]
                    try:
                        bi = list(cand).index(self.global_gam_lambda)
                    except Exception:
                        arr = np.array(cand, dtype=float)
                        bi = int(np.argmin(np.abs(arr - float(self.global_gam_lambda))))
                    coeffs = self.global_gam.coeffs[0][:, bi].toarray().ravel()
                    global_support = int(np.count_nonzero(coeffs))
            except Exception:
                global_support = 0

        # Leaves
        for t_idx, root in enumerate(self.tree_ensemble):
            nodes_info = []
            tree_sum = 0
            for node in root.get_all_nodes():
                s = int(self._node_support_size(node))
                if s > 0 or getattr(node, "num_samples", 0) > 0:
                    num_leaf_nodes += 1
                tree_sum += s
                if return_breakdown:
                    nodes_info.append({
                        "depth": int(getattr(node, "depth", -1)),
                        "support": s,
                        "num_samples": int(getattr(node, "num_samples", 0)),
                    })
            per_tree.append({
                "tree_id": int(getattr(root, "tree_id", t_idx)),
                "sum_support": tree_sum,
                "nodes": nodes_info if return_breakdown else None
            })
            leaves_support += tree_sum

        total_support = global_support + leaves_support

        return {
            "total_support": int(total_support),
            "global_support": int(global_support),
            "leaves_support": int(leaves_support),
            "num_trees": int(len(self.tree_ensemble)),
            "num_leaf_nodes": int(num_leaf_nodes),
            "per_tree": per_tree if return_breakdown else None
        }
