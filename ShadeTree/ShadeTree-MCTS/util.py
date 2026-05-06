import os
import numpy as np
import pandas as pd
import logging
import argparse
import json
from threading import Lock
from types import SimpleNamespace
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss, f1_score, precision_score, recall_score)
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# Attempt to import from ADTree, but allow standalone execution
try:
    from ADTree import plot_ad_tree, ShapeNode, SplitNode
except ImportError:
    print("Warning: ADTree module not found. Plotting and sparsity functions will be disabled.")
    # Define placeholder class/function if ADTree is not available
    class ShapeNode: pass
    class SplitNode: pass
    def plot_ad_tree(*args, **kwargs):
        print("plot_ad_tree is not available.")

# --- Basic Setup ---
logger = logging.getLogger('MCTS_ADTree')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_lock = Lock()


# --- Result Logging and Plotting Functions ---

def log_aggregated_cv_results(args: argparse.Namespace, metrics_dfs: list, sparsity_dicts: list):
    """
    Aggregates results from all folds of a CV run and logs them to a master CSV file
    for the dataset, allowing for comparison across hyperparameter settings.
    """
    results_root = getattr(args, "results_root", "/usr/xtmp/xz424/mcts/results")
    summary_file_path = os.path.join(results_root, args.dataset, "aggregated_cv_results.csv")
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    
    # --- Aggregate Metrics ---
    combined_metrics_df = pd.concat(metrics_dfs)
    # The columns are ['Train', 'Validation', 'Test']. The index is the metric name.
    means = combined_metrics_df.groupby(by=combined_metrics_df.index).mean()
    # Use population std (ddof=0) so single-fold runs yield 0 instead of NaN.
    stds = combined_metrics_df.groupby(by=combined_metrics_df.index).std(ddof=0)

    def _extract_runtime_sec(df: pd.DataFrame) -> float | None:
        if df is None or df.empty:
            return None
        if "running_time_sec" in df.columns:
            series = pd.to_numeric(df["running_time_sec"], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])
        if "running_time_sec" in df.index:
            try:
                row = df.loc["running_time_sec"]
                if isinstance(row, pd.Series):
                    series = pd.to_numeric(row, errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[0])
                else:
                    return float(row)
            except Exception:
                return None
        return None

    runtime_values = []
    for df in metrics_dfs:
        runtime = _extract_runtime_sec(df)
        if runtime is not None:
            runtime_values.append(runtime)
    runtime_mean = float(np.mean(runtime_values)) if runtime_values else float("nan")
    runtime_std = float(np.std(runtime_values, ddof=0)) if runtime_values else float("nan")
    
    # --- Aggregate Sparsity ---
    sparsity_df = pd.DataFrame(sparsity_dicts)
    sparsity_df = sparsity_df.apply(
        pd.to_numeric,
        errors="coerce"     
    )
    sparsity_means = sparsity_df.mean(numeric_only=True).add_suffix("_mean")
    # Population std to avoid NaN when only one fold was logged.
    sparsity_stds = sparsity_df.std(ddof=0).add_suffix('_std')

    # --- Build Flat Dictionary for Logging ---
    # Start with all hyperparameters from the args namespace
    run_data = vars(deepcopy(args))
    # Make complex types hashable for deduplication.
    for key, value in list(run_data.items()):
        if isinstance(value, list):
            run_data[key] = tuple(value)
        elif isinstance(value, dict):
            run_data[key] = json.dumps(value, sort_keys=True)
    
    # Add aggregated metrics
    for split in ['Train', 'Validation', 'Test']:
        for metric in means.index:
            run_data[f'{split}_{metric}_mean'] = means.loc[metric, split]
            run_data[f'{split}_{metric}_std'] = stds.loc[metric, split]
            
    # Add aggregated sparsity
    run_data.update(sparsity_means.to_dict())
    run_data.update(sparsity_stds.to_dict())

    run_data["running_time_sec_mean"] = runtime_mean
    run_data["running_time_sec_std"] = runtime_std
        
    run_data['num_folds_completed'] = len(metrics_dfs)
    
    # Clean up non-essential args for a cleaner CSV
    run_data.pop('results_dir', None)
        
    # --- Save to CSV ---
    results_row_df = pd.DataFrame([run_data])
    with file_lock:
        try:
            if os.path.exists(summary_file_path):
                try:
                    existing_df = pd.read_csv(summary_file_path)
                except Exception as read_err:
                    logger.warning(f"Failed to read existing summary file; recreating. Error: {read_err}")
                    existing_df = None

                if existing_df is not None:
                    all_cols = list(existing_df.columns)
                    for col in results_row_df.columns:
                        if col not in all_cols:
                            all_cols.append(col)
                    combined = pd.concat([existing_df, results_row_df], ignore_index=True)
                    combined = combined.reindex(columns=all_cols)
                else:
                    combined = results_row_df
            else:
                combined = results_row_df

            # Deduplicate identical hyperparameter runs; keep the latest entry.
            dedup_keys = [
                'dataset', 'random_state', 'label_mode', 'k_folds', 'k_fold_validation',
                'inner_fold_override', 'outer_split_seeds', 'results_root', 'shape_fitter',
                'min_samples_leaf_ratio', 'n_knots_step',
                'include_intermediate_shapes', 'refine_step_with_fastsparse',
                'ebm_interactions', 'ebm_max_bins', 'ebm_outer_bags', 'max_split_nodes',
                'use_residual_ensemble', 'ensemble_rounds', 'ensemble_eta', 'ensemble_min_region',
                'ensemble_region_strategy', 'ensemble_patience', 'ensemble_early_stop_metric',
                'max_iters', 'max_depth', 'c_ucb', 'max_rollout_depth', 'reward_metric',
                'complexity_penalty', 'use_xgboost_pruning', 'pruning_mode', 'pruning_threshold',
                'patience', 'warmup_iters', 'log_frequency', 'num_percentile_points',
                'warmstart_exports', 'prior_lambda', 'prior_gamma', 'prior_cap', 'prior_budget',
                'warmstart_metric_key', 'c_pw', 'alpha_pw', 'eps_deepen',
                'use_xgboost_ranking', 'epsilon_ranking', 'lambda_ranking'
            ]
            missing_keys = [k for k in dedup_keys if k not in combined.columns]
            if missing_keys:
                logger.warning(f"Dedup keys missing in aggregated CSV: {missing_keys}")
            else:
                combined = combined.drop_duplicates(subset=dedup_keys, keep='last')

            combined.to_csv(summary_file_path, index=False)
            logger.info(f"Successfully logged aggregated CV results to {summary_file_path}")
        except Exception as e:
            logger.error(f"Failed to log aggregated CV results to {summary_file_path}. Error: {e}")


def find_best_threshold(y_true, y_pred_proba):
    """
    Finds the best classification threshold to maximize the F1 score.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    y_true_01 = (y_true == 1).astype(int)
    
    for threshold in thresholds:
        y_pred_01 = (y_pred_proba > threshold).astype(int)
        score = f1_score(y_true_01, y_pred_01, average='weighted', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
            
    return best_threshold


def plot_spline_model(spline_model, feature_name, ax=None, data_info=None, component_idx=None):
    """Plots a single spline or step function model (supports additive components)."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Resolve feature index/name for additive models
    if component_idx is not None and hasattr(spline_model, "feature_indices"):
        fi = spline_model.feature_indices[component_idx]
        feature_name = (getattr(data_info, "feature_names", None) or [])[fi] if (
            data_info is not None and hasattr(data_info, "feature_names") and fi < len(data_info.feature_names)
        ) else f"f{fi}"
        plot_model = spline_model
    else:
        plot_model = spline_model

    if plot_model.constant is not None:
        ax.axhline(y=spline_model.constant, color='r', linestyle='--', label=f'Constant Value: {spline_model.constant:.3f}')
        ax.set_title(f"Constant Shape Function")
    else:
        xmin, xmax = plot_model.xmin, plot_model.xmax
        plot_x = np.linspace(xmin, xmax, 200).reshape(-1, 1)
        
        max_feat = plot_model.feature_index
        if hasattr(spline_model, "feature_indices") and spline_model.feature_indices:
            try:
                max_feat = max(max_feat, max(spline_model.feature_indices))
            except Exception:
                pass
        num_features = max_feat + 1
        X_dummy = np.zeros((plot_x.shape[0], num_features))
        X_dummy[:, plot_model.feature_index] = plot_x[:, 0]
        if component_idx is not None and hasattr(spline_model, "predict_component"):
            plot_y = spline_model.predict_component(X_dummy, component_idx)
        else:
            plot_y = plot_model.predict(X_dummy)
        plot_x_show = plot_x 
        if (data_info is not None and 
            hasattr(data_info, 'scaler') and 
            data_info.scaler is not None):
            
            f_idx = plot_model.feature_index
            if f_idx < len(data_info.scaler.mean_):
                mean = data_info.scaler.mean_[f_idx]
                scale = data_info.scaler.scale_[f_idx]
                plot_x_show = plot_x * scale + mean

        if plot_model.degree == 0: # Step function
            plot_x_step = np.concatenate([plot_x_show, plot_x_show[-1:]])
            plot_y_step = np.concatenate([plot_y, plot_y[-1:]])
            ax.step(plot_x_step, plot_y_step, where='post', label='Step Fit (degree=0 spline)')
            ax.set_title(f"Step Shape Function for {feature_name}")
        else: # Spline function
            ax.plot(plot_x_show, plot_y, label=f'Spline Fit (degree={plot_model.degree})')
            ax.set_title(f"Spline Shape Function for {feature_name}")

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Shape Contribution (Log-Odds)")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_ebm_model(ebm_wrapper, filename_prefix):
    """
    [FIXED] Visualizes the global explanation of an EBM, saving each feature plot individually.
    This version correctly handles the figure object returned by the visualize() method
    in newer versions of the 'interpret' library.
    """
    if ebm_wrapper.ebm_model is None:
        logger.warning(f"Cannot plot EBM as it is a constant model. Skipping visualization for {filename_prefix}.")
        return
        
    try:
        ebm_global = ebm_wrapper.ebm_model.explain_global()
        
        if ebm_global is None or not hasattr(ebm_global, 'feature_names') or not ebm_global.feature_names:
            logger.warning(f"EBM explanation is empty or has no features to plot. Skipping visualization for {filename_prefix}.")
            return

        # In newer versions, feature names are part of the explanation object itself.
        # It's more reliable to use them directly.
        feature_names = ebm_global.feature_names
        
        if not feature_names: 
            logger.warning(f"EBM explanation has no features to plot. Skipping visualization for {filename_prefix}.")
            return

        # Iterate through all available plots in the explanation
        for i in range(len(feature_names)):
            try:
                # --- THIS IS THE CRITICAL FIX ---
                
                # The visualize() method now returns the plot/figure object directly.
                # We should capture it and use its savefig method.
                explanation_plot = ebm_global.visualize(key=i)
                
                # Check if the returned object has a 'figure' attribute (common pattern)
                if hasattr(explanation_plot, 'figure'):
                    figure_to_save = explanation_plot.figure
                else:
                    # If not, it might be the figure itself. We can get it via gcf().
                    # This provides backward compatibility.
                    figure_to_save = plt.gcf()

                # --- END OF THE FIX ---

                # Sanitize the feature name to create a valid filename
                feature_name_safe = "".join(c if c.isalnum() else "_" for c in feature_names[i])
                output_path = f"{filename_prefix}_feature_{feature_name_safe}.png"
                
                # Save the correct figure
                figure_to_save.savefig(output_path, bbox_inches="tight")
                
                # Close the figure to free up memory
                plt.close(figure_to_save)
                
                logger.info(f"Saved EBM feature plot to {output_path}")

            except Exception as e:
                feature_name_safe = feature_names[i] if i < len(feature_names) else "unknown"
                logger.error(f"Could not visualize EBM feature index {i} ('{feature_name_safe}'). Error: {e}")

    except Exception as e:
        logger.error(f"Failed to plot EBM model for {filename_prefix}. Error: {e}", exc_info=True)



# --- Data Loading Functions ---

# --- Data Loading Functions ---

def _load_single_csv(data_dir, config):
    if 'filename' not in config:
        raise ValueError("Config for _load_single_csv missing 'filename'.")
    data_path = os.path.join(data_dir, config['filename'])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find data file: {data_path}")
    
    df = pd.read_csv(
        data_path,
        sep=config.get('sep', ','),   # 默认逗号，某些数据集会传入 sep=';'
        engine='python'
    )

    # Titanic 特殊列处理
    if config.get('filename') == 'Titanic.csv' and 'name' in df.columns:
        df.drop(columns=['name'], inplace=True)

    target_col = config['target_col']
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {config.get('filename')}.")

    # 如果 target 是字符串，统一 strip 并去掉 '.'
    if df[target_col].dtype == object:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.replace('.', '', regex=False)
        )
    # 其他 object 列也 strip 一下
    for c in df.select_dtypes(include=['object']).columns:
        if c == target_col:
            continue
        df[c] = df[c].astype(str).str.strip()

    df.dropna(inplace=True)
    return df, None
def _load_data_file(data_dir, config):

    if 'filename' not in config:
        raise ValueError("Missing 'filename'.")
    if 'column_names' not in config:
        raise ValueError("Missing 'column_names' for .data loader.")
    if 'target_col' not in config:
        raise ValueError("Missing 'target_col'.")

    data_path = os.path.join(data_dir, config['filename'])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")

    df = pd.read_csv(
        data_path,
        header=None,
        names=config['column_names'],
        sep=config.get('sep', r',\s*'),
        engine='python',
        na_values=config.get('na_values', None)
    )

    tc = config['target_col']
    if tc not in df.columns:
        raise KeyError(
            f"Cannot find '{tc}' in {config['filename']}. "
            f"Actual columns: {list(df.columns)}"
        )

    if df[tc].dtype == object:
        df[tc] = (
            df[tc].astype(str)
                  .str.strip()
                  .str.replace('.', '', regex=False)
        )

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    df.dropna(inplace=True)

    return df, None

def _load_sklearn_dataset(data_dir, config):
    from sklearn.datasets import load_breast_cancer, load_iris
    name = str(config.get('sk_name', '')).lower()
    if name == 'breast_cancer':
        ds = load_breast_cancer(as_frame=True)
    elif name == 'iris':
        ds = load_iris(as_frame=True)
    else:
        raise ValueError(f"Unsupported sklearn dataset name: {name!r}")
    
    df = ds.frame.copy()
    target_col = config.get('target_col', 'target')
    if target_col not in df.columns:
        if 'target' in df.columns:
            df.rename(columns={'target': target_col}, inplace=True)
        else:
            # 兜底：如果 frame 里没有 target 列，就手动加
            df[target_col] = ds.target

    df.dropna(inplace=True)
    return df, None


def _load_adult_dataset(data_dir, config):
    """
    adult 原始 UCI：train/test 两个文件，统一读进来做一个大表，方便外层自己做 CV。
    """
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    train_path = os.path.join(data_dir, 'adult.data')
    df_train = pd.read_csv(
        train_path,
        header=None,
        names=column_names,
        sep=r',\s*',
        na_values='?',
        engine='python'
    )
    
    test_path = os.path.join(data_dir, 'adult.test')
    df_test = pd.read_csv(
        test_path,
        header=None,
        names=column_names,
        sep=r',\s*',
        na_values='?',
        skiprows=1,
        engine='python'
    )
    
    # 去掉 adult.test 里标签末尾的 '.'
    df_test[config['target_col']] = df_test[config['target_col']].astype(str).str.replace('.', '', regex=False)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    
    full_df = pd.concat([df_train, df_test], ignore_index=True)
    return full_df, None

def _load_heart_disease(data_dir, config):
    if 'filename' not in config:
        raise ValueError("Missing 'filename'.")
    if 'column_names' not in config:
        raise ValueError("Missing 'column_names' for heart_cleveland.")
    if 'target_col' not in config:
        raise ValueError("Missing 'target_col' for heart_cleveland (should be 'target').")

    data_path = os.path.join(data_dir, config['filename'])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")

    df = pd.read_csv(
        data_path,
        header=None,
        names=config['column_names'],
        sep=config.get('sep', r',\s*'),
        engine='python',
        na_values=config.get('na_values', '?')
    )

    df.dropna(inplace=True)

    if 'num' not in df.columns:
        raise KeyError("Expected column 'num' in heart disease data.")
    df['target'] = (df['num'].astype(float) > 0).astype(int)

    df.drop(columns=['num'], inplace=True)

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    tc = config['target_col']
    if tc not in df.columns:
        raise KeyError(f"Cannot find '{tc}' after binarization. Current cols: {list(df.columns)}")

    return df, None

def _load_generic_dataset(data_dir, config):
    if config.get('id'):
        dataset = fetch_ucirepo(id=config['id'])
        X_df = dataset.data.features
        y_df = dataset.data.targets
        df = pd.concat([X_df, y_df], axis=1)
        # UCI wine (id=186) 的 quality 二值化
        if config['id'] == 186:
            df['quality'] = (df['quality'] >= 6).astype(int)
    else:
        data_path = os.path.join(data_dir, config['filename'])
        df = pd.read_csv(
            data_path,
            sep=config.get('sep', ',')
        )
        # 本地 winequality-red / winequality-white，同样二值化 quality
        if 'winequality' in config.get('filename', '') and 'quality' in df.columns:
            df['quality'] = (df['quality'] >= 6).astype(int)
            
    df.dropna(inplace=True)
    return df, None
def _load_phishing_arff(data_dir, config):
    if 'filename' not in config:
        raise ValueError("Missing 'filename' for phishing arff loader.")
    if 'target_col' not in config:
        raise ValueError("Missing 'target_col' for phishing arff loader (should be 'target').")

    data_path = os.path.join(data_dir, config['filename'])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")

    attr_names = []
    data_rows = []
    in_data_section = False

    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()

            if line == "" or line.startswith('%'):
                continue

            if not in_data_section:
                if line.lower().startswith('@data'):
                    in_data_section = True
                    continue

                if line.lower().startswith('@attribute'):
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(f"Bad @attribute line: {line}")
                    colname = parts[1].strip().strip("'").strip('"')
                    attr_names.append(colname)
                continue

            if line != "":
                parts = [x.strip() for x in line.split(',')]
                if len(parts) != len(attr_names):
                    continue
                data_rows.append(parts)

    if not attr_names:
        raise ValueError("No @attribute lines found; cannot infer column names.")
    if not data_rows:
        raise ValueError("No data rows found after @data section.")

    df = pd.DataFrame(data_rows, columns=attr_names)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    label_candidates = [c for c in df.columns if c.lower() in ('result', 'class', 'target')]
    if len(label_candidates) == 0:
        raise KeyError(
            f"Cannot locate label column among {list(df.columns)} "
            f"(expected something like 'Result')."
        )
    raw_label_col = label_candidates[0]

    raw_vals = pd.to_numeric(df[raw_label_col], errors='coerce')
    df['target'] = np.where(raw_vals == 1, 1, 0).astype(int)

    if raw_label_col != 'target':
        df.drop(columns=[raw_label_col], inplace=True)

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    df.dropna(inplace=True)

    tc = config['target_col']
    if tc not in df.columns:
        raise KeyError(
            f"After relabeling phishing arff, cannot find '{tc}'. "
            f"Columns now: {list(df.columns)}"
        )

    return df, None

def prepare_data(args):
    german_credit_columns = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "class"
    ]
    heart_disease_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num"
    ]
    magic_telescope_columns = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class"
    ]
    DATA_CONFIG = {
        'adult': {
            'loader_func': _load_adult_dataset,
            'target_col': 'income',
            'pos_class': '>50K'
        },
        'bank': {
            'loader_func': _load_generic_dataset,
            'filename': 'bank.csv', 
            'target_col': 'y',
            'pos_class': 'yes', 'sep': ';'
        },
        'wine': {
            'loader_func': _load_generic_dataset,
            'id': 186, 
            'target_col': 'quality',
            'pos_class': 1
        },
        'winequality_red': {
            'loader_func': _load_generic_dataset,
            'filename': 'winequality-red.csv', 
            'target_col': 'quality',
            'pos_class': 1, 'sep': ';'
        },
        'winequality_white': {
            'loader_func': _load_generic_dataset,
            'filename': 'winequality-white.csv', 
            'target_col': 'quality',
            'pos_class': 1, 'sep': ';'
        },
        'adult_balanced': {
            'loader_func': _load_single_csv,
            'filename': 'adult_balanced.csv', 
            'target_col': 'target',
            'pos_class': '>50K'
        },
        'bank_balanced': {
            'loader_func': _load_single_csv,
            'filename': 'bank_balanced.csv', 
            'target_col': 'y',
            'pos_class': 'yes'
        },
        'bank_add': {
            'loader_func': _load_generic_dataset,
            'filename': 'bank-additional-full.csv', 
            'target_col': 'y',
            'pos_class': 'yes', 'sep': ';'
        },
        'bank_add_balanced': {
            'loader_func': _load_single_csv,
            'filename': 'bank_add_balanced.csv', 
            'target_col': 'target',
            'pos_class': 'yes', 'sep': ';'
        },
        'breast_cancer': {
            'loader_func': _load_sklearn_dataset,
            'sk_name': 'breast_cancer', 
            'target_col': 'target',
            'pos_class': 1
        },
        'iris': {
            'loader_func': _load_sklearn_dataset,
            'sk_name': 'iris', 
            'target_col': 'target',
            'pos_class': 1
        },
        'blood_transfusion': {
            'loader_func': _load_single_csv,
            'filename': 'blood_transfusion.csv', 
            'target_col': 'Class',
            'pos_class': 2,
        },
        'compas': {
            'loader_func': _load_single_csv,
            'filename': 'compas.csv', 
            'target_col': 'recid',
            'pos_class': 1,
        },
        'covertype': {
            'loader_func': _load_single_csv,
            'filename': 'covertype.csv', 
            'target_col': 'Cover_Type',
            'pos_class': 1,
        },
        'diabetes': {
            'loader_func': _load_single_csv,
            'filename': 'diabetes.csv', 
            'target_col': 'class',
            'pos_class': 'tested_positive',
        },
        'higgs': {
            'loader_func': _load_single_csv,
            'filename': 'higgs.csv', 
            'target_col': 'target',
            'pos_class': 'b\'0\'',
        },
        'netherlands': {
            'loader_func': _load_single_csv,
            'filename': 'netherlands.csv', 
            'target_col': 'recidivism_in_4y',
            'pos_class': 1,
        },
        'spambase': {
            'loader_func': _load_single_csv,
            'filename': 'spambase.csv', 
            'target_col': 'class',
            'pos_class': 1,
        },
        'Titanic': {
            'loader_func': _load_single_csv,
            'filename': 'Titanic.csv', 
            'target_col': 'survived',
            'pos_class': 1,
        },
        'news_headline1': {
            'loader_func': _load_single_csv,
            'filename': 'news_headline1.csv',
            'target_col': 'Y',
            'pos_class': 1,
        },
        'news_headline2': {
            'loader_func': _load_single_csv,
            'filename': 'news_headline2.csv',
            'target_col': 'Y',
            'pos_class': 1,
        },
        'news_headline3': {
            'loader_func': _load_single_csv,
            'filename': 'news_headline3.csv',
            'target_col': 'Y',
            'pos_class': 1,
        },
        'fico': {
            'loader_func': _load_single_csv,
            'filename': 'fico.csv',
            'target_col': 'PoorRiskPerformance',
            'pos_class': 1,
        },
        'higgs_20k': {
            'loader_func': _load_single_csv,
            'filename': 'higgs_20k.csv',
            'target_col': 'target',
            'pos_class': 1,
        },
        'german_credit':{
            "loader_func": _load_data_file,
            "filename": "german.data",
            "column_names": german_credit_columns,
            "target_col": "class",
            "pos_class": 1,
            "sep": r"\s+",
            "na_values": None,
        },
        'heart_disease_cleverland':{
            "loader_func": _load_heart_disease,
            "filename": "processed.cleveland.data",
            "column_names": heart_disease_columns,
            "target_col": "target",
            "pos_class": 1,
            "sep": r",\s*",
            "na_values": "?"
        },
        'telescope':{
            "loader_func": _load_data_file,
            "filename": "magic04.data",
            "column_names": magic_telescope_columns,
            "target_col": "class",
            "pos_class": "g",
            "sep": r",\s*",
            "na_values": None,
        },
        'TWCredit':{ 
            "loader_func": _load_single_csv,
            "filename": "TWCredit.csv",
            "target_col": "default.payment.next.month",
            "pos_class": 1, 
            "sep": ","
        },
        'phishing_website':{
            "loader_func": _load_phishing_arff,
            "filename": "Training Dataset.arff",
            "target_col": "target",
            "pos_class": 1
        }
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'dataset')
    
    if args.dataset not in DATA_CONFIG:
        raise ValueError(f"Configuration for dataset '{args.dataset}' not found.")

    config = DATA_CONFIG[args.dataset]
    df, _ = config['loader_func'](data_dir, config)

    y_all = df[config['target_col']]
    X_all = df.drop(columns=[config['target_col']])

    def _binarize(y: pd.Series, pos_class):
        """
        把标签二值化：
          - 正类：+1
          - 其他：-1
        兼容数值/字符串两种形式。
        """
        try:
            pos_numeric = float(pos_class)
            y_numeric = pd.to_numeric(y, errors='coerce')
            return np.where(y_numeric == pos_numeric, 1, -1).astype(int)
        except (ValueError, TypeError):
            y_str = y.astype(str).str.strip()
            pos_str = str(pos_class).strip()
            return np.where(y_str == pos_str, 1, -1).astype(int)

    y_all_binarized = _binarize(y_all, config['pos_class'])

    X_all = X_all.copy()
    cat_cols = X_all.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_all.select_dtypes(include=np.number).columns.tolist()

    ohe, scaler = None, None
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        ohe.fit(X_all[cat_cols])
    if num_cols:
        scaler = StandardScaler()
        scaler.fit(X_all[num_cols])

    df_parts = []
    if scaler is not None and num_cols:
        df_parts.append(
            pd.DataFrame(
                scaler.transform(X_all[num_cols]),
                index=X_all.index,
                columns=num_cols
            )
        )
    if ohe is not None and cat_cols:
        df_parts.append(
            pd.DataFrame(
                ohe.transform(X_all[cat_cols]),
                index=X_all.index,
                columns=ohe.get_feature_names_out(cat_cols)
            )
        )
    X_processed = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame(index=X_all.index)

    feature_names = X_processed.columns.tolist()
    # continuous_feature_indices 这里简单按原始 num_cols 名称匹配
    cont_indices = [i for i, name in enumerate(feature_names) if name in num_cols]
    data_info = SimpleNamespace(
        feature_names=feature_names,
        continuous_feature_indices=cont_indices,
        scaler=scaler
    )

    logger.info(f"Dataset '{args.dataset}': {len(X_processed)} total samples, {len(feature_names)} features.")
    
    return X_processed.to_numpy(dtype=np.float32), y_all_binarized, data_info


def evaluate_model(y_true, y_pred_proba, y_pred_01):
    y_true_01 = (y_true == 1).astype(int)
    metrics = {}
    try: metrics['auc'] = roc_auc_score(y_true_01, y_pred_proba)
    except ValueError: metrics['auc'] = 0.5
    try: metrics['accuracy'] = accuracy_score(y_true_01, y_pred_01)
    except ValueError: metrics['accuracy'] = 0.0
    try: metrics['brier'] = brier_score_loss(y_true_01, y_pred_proba)
    except ValueError: metrics['brier'] = 1.0
    try: metrics['logloss'] = log_loss(y_true_01, np.clip(y_pred_proba, 1e-15, 1 - 1e-15))
    except ValueError: metrics['logloss'] = 99.0
    
    try: metrics['f1'] = f1_score(y_true_01, y_pred_01, average='weighted', zero_division=0)
    except ValueError: metrics['f1'] = 0.0
    try: metrics['precision'] = precision_score(y_true_01, y_pred_01, average='weighted', zero_division=0)
    except ValueError: metrics['precision'] = 0.0
    try: metrics['recall'] = recall_score(y_true_01, y_pred_01, average='weighted', zero_division=0)
    except ValueError: metrics['recall'] = 0.0
    return metrics


# --- START OF FIX ---
def _count_nodes_recursive(node, counts):
    """[FIXED] A recursive helper to correctly count different node types based on the revised ADTree structure."""
    if node is None:
        return

    if isinstance(node, ShapeNode):
        counts['shape'] += 1
        # If the ShapeNode is an intermediate node, continue traversing its SplitNode children.
        for child_split_node in getattr(node, "children", []) or []:
            _count_nodes_recursive(child_split_node, counts)

    elif isinstance(node, SplitNode):
        counts['split'] += 1
        # A SplitNode's children are always ShapeNodes.
        _count_nodes_recursive(node.true_child, counts)
        _count_nodes_recursive(node.false_child, counts)


def _extract_feature_indices_from_node(node) -> set:
    """Collect feature indices referenced by a node or its shape model."""
    feats = set()
    f_idx = getattr(node, "feature_idx", None)
    if f_idx is not None and f_idx >= 0:
        feats.add(int(f_idx))

    shape_model = getattr(node, "shape_model", None)
    if shape_model is not None:
        sm_idx = getattr(shape_model, "feature_index", None)
        if sm_idx is not None and sm_idx >= 0:
            feats.add(int(sm_idx))
        sm_indices = getattr(shape_model, "feature_indices", None)
        if sm_indices:
            feats.update(int(fi) for fi in sm_indices if fi is not None and fi >= 0)
    return feats


def _collect_feature_indices(node) -> set:
    """Traverse the tree and gather all unique feature indices used in splits or shapes."""
    features = set()
    if node is None:
        return features

    if isinstance(node, SplitNode):
        features.update(_extract_feature_indices_from_node(node))
        features.update(_collect_feature_indices(node.true_child))
        features.update(_collect_feature_indices(node.false_child))
    elif isinstance(node, ShapeNode):
        features.update(_extract_feature_indices_from_node(node))
        for child_split in getattr(node, "children", []) or []:
            features.update(_collect_feature_indices(child_split))
    else:
        features.update(_extract_feature_indices_from_node(node))
        for child_attr in ["children", "true_child", "false_child"]:
            child = getattr(node, child_attr, None)
            if child is not None:
                features.update(_collect_feature_indices(child))
    return features


def _collect_terminal_shape_nodes(node):
    """Return all terminal ShapeNodes reachable from this node."""
    if node is None:
        return []
    if isinstance(node, ShapeNode):
        if node.is_terminal():
            return [node]
        term_nodes = []
        for child_split in getattr(node, "children", []) or []:
            term_nodes.extend(_collect_terminal_shape_nodes(child_split))
        return term_nodes
    if isinstance(node, SplitNode):
        return (
            _collect_terminal_shape_nodes(node.true_child)
            + _collect_terminal_shape_nodes(node.false_child)
        )
    # Fallback: attempt to follow common child attributes
    term_nodes = []
    for child_attr in ["children", "true_child", "false_child"]:
        child = getattr(node, child_attr, None)
        if child is not None:
            term_nodes.extend(_collect_terminal_shape_nodes(child))
    return term_nodes


def _features_on_path_to_root(node):
    """Collect unique feature indices encountered from the node up to the root."""
    feats = set()
    current = node
    while current is not None:
        if isinstance(current, SplitNode):
            f_idx = getattr(current, "feature_idx", None)
            if f_idx is not None and f_idx >= 0:
                feats.add(int(f_idx))
        else:
            feats.update(_extract_feature_indices_from_node(current))
        current = getattr(current, "parent", None)
    return feats


def _count_steps_in_step_shapes(shape_nodes, default_knots=None):
    """Count total steps across all step shape functions (degree == 0)."""
    total_steps = 0
    for node in shape_nodes:
        shape_model = getattr(node, "shape_model", None)
        if shape_model is None:
            continue
        try:
            step_count = getattr(shape_model, "step_count", None)
            if step_count is not None:
                total_steps += int(step_count)
                continue
        except Exception:
            pass

        degree = getattr(shape_model, "degree", None)
        if degree is None:
            continue
        try:
            if int(degree) != 0:
                continue
        except Exception:
            continue

        n_knots = getattr(shape_model, "n_internal_knots", None)
        if n_knots is None and default_knots is not None:
            n_knots = default_knots
        n_components = len(getattr(shape_model, "feature_indices", []) or [None])
        try:
            steps_per_comp = int(n_knots) + 1 if n_knots is not None else 1
        except Exception:
            steps_per_comp = 1
        total_steps += steps_per_comp * n_components
    return total_steps
# --- END OF FIX ---

def _count_steps_in_ebm_shapes(shape_nodes):
    """
    Estimate step counts for EBM leaf models using the model's own binning metadata
    (bin edges / labels / term scores) instead of relying on external heuristics.
    """
    def _count_from_term_scores(term_scores):
        total = 0
        seen = False
        for scores in term_scores:
            try:
                arr = np.asarray(scores)
            except Exception:
                continue
            if arr.size == 0:
                continue
            seen = True
            # Each term contributes one parameter per bin/cell.
            total += int(np.prod(arr.shape))
        return total if seen else 0

    def _count_from_edges(edge_list, subtract_one=True):
        total = 0
        seen = False
        for edges in edge_list:
            try:
                n = len(edges)
            except Exception:
                continue
            seen = True
            total += max(0, n - 1) if subtract_one else max(0, n)
        return total if seen else 0

    def _ebm_steps(shape_model):
        if shape_model is None:
            return 0
        ebm_model = getattr(shape_model, "ebm_model", None)
        if ebm_model is None:
            constant = getattr(shape_model, "constant", None)
            if constant is not None:
                return 1
            feats = getattr(shape_model, "feature_indices", []) or []
            return max(1, len(feats)) if feats is not None else 0

        # 1) Prefer term scores (covers both main effects and interactions).
        try:
            term_scores = getattr(ebm_model, "term_scores_", None)
            if term_scores is not None:
                steps = _count_from_term_scores(term_scores)
                if steps > 0:
                    return steps
        except Exception:
            pass

        # 2) Explicit bin edges/bounds on the model itself (main effects only).
        try:
            for attr, subtract_one in (("bin_edges_", True), ("feature_bounds_", True)):
                bin_edges = getattr(ebm_model, attr, None)
                if bin_edges is not None:
                    steps = _count_from_edges(bin_edges, subtract_one=subtract_one)
                    if steps > 0:
                        return steps
        except Exception:
            pass

        # 3) Inspect the preprocessor if present (newer interpret versions).
        try:
            preproc = getattr(ebm_model, "preprocessor_", None)
            if preproc is not None:
                for attr, subtract_one in (("col_bin_edges_", True), ("col_bin_bounds_", True)):
                    edges = getattr(preproc, attr, None)
                    if edges is not None:
                        steps = _count_from_edges(edges, subtract_one=subtract_one)
                        if steps > 0:
                            return steps
                bin_labels = getattr(preproc, "col_bin_labels_", None)
                if bin_labels is not None:
                    steps = _count_from_edges(bin_labels, subtract_one=False)
                    if steps > 0:
                        return steps
        except Exception:
            pass

        # 4) Last resort: at least one step per feature component.
        try:
            feats = getattr(shape_model, "feature_indices", []) or []
            return max(1, len(feats))
        except Exception:
            return 0

    total = 0
    for node in shape_nodes:
        shape_model = getattr(node, "shape_model", None)
        if hasattr(shape_model, "ebm_model"):
            total += _ebm_steps(shape_model)
        else:
            # Fallback for non-EBM shapes that remain in an EBM run.
            total += _count_steps_in_step_shapes([node])
    return total


def _count_leaf_shape_functions(shape_nodes):
    """Count shape functions represented by terminal shape nodes (EBM-aware)."""
    total = 0
    for node in shape_nodes:
        shape_model = getattr(node, "shape_model", None)
        if shape_model is None:
            continue
        ebm_model = getattr(shape_model, "ebm_model", None)
        if ebm_model is None:
            total += 1
            continue

        term_scores = getattr(ebm_model, "term_scores_", None)
        if term_scores is not None:
            try:
                total += len(term_scores)
                continue
            except Exception:
                pass
        feats = getattr(shape_model, "feature_indices", []) or []
        total += len(feats) if feats else 1
    return total


def calculate_sparsity_metrics(model, args):
    """[FIXED] Calculates sparsity metrics by correctly traversing the tree."""  
    counts = {'split': 0, 'shape': 0}
    _count_nodes_recursive(model.root, counts)
    
    num_split = counts['split']
    num_shape = counts['shape']
    all_shape_nodes = _find_all_shape_nodes(model.root)
    terminal_shape_nodes = _collect_terminal_shape_nodes(model.root)
    features_used = _collect_feature_indices(model.root)
    decision_feature_counts = [
        len(_features_on_path_to_root(node))
        for node in terminal_shape_nodes
    ]
    avg_features_per_decision = (
        float(np.mean(decision_feature_counts))
        if decision_feature_counts else np.nan
    )
    default_step_knots = args.n_knots_step if args.shape_fitter == 'step' else None
    total_step_count = _count_steps_in_step_shapes(all_shape_nodes, default_knots=default_step_knots)
    
    # Equivalent nodes: scale shape nodes by basis size and component count
    def _shape_equivalent_count(shape_model):
        if shape_model is None:
            return 0
        components = len(getattr(shape_model, "feature_indices", []) or [None])
        if args.shape_fitter == 'step':
            try:
                step_count = getattr(shape_model, "step_count", None)
                if step_count is not None:
                    return int(step_count)
            except Exception:
                pass
            n_knots = getattr(shape_model, "n_internal_knots", None)
            try:
                n_knots = int(n_knots)
            except Exception:
                n_knots = args.n_knots_step
            return components * (n_knots + 1)
        else:
            return components

    equivalent_nodes = num_split + sum(
        _shape_equivalent_count(getattr(node, "shape_model", None)) for node in all_shape_nodes
    )
    if args.shape_fitter == 'ebm':
        total_step_count = _count_steps_in_ebm_shapes(all_shape_nodes)
        equivalent_nodes = num_split + total_step_count
        leaf_shape_function_count = _count_leaf_shape_functions(terminal_shape_nodes)
        return {
            'num_split_nodes': num_split,
            'num_shape_nodes': num_shape,
            'equivalent_nodes': equivalent_nodes,
            'total_step_count': total_step_count,
            'num_leaf_shape_nodes': leaf_shape_function_count,
            'num_features_used': len(features_used),
            'avg_features_per_decision': avg_features_per_decision
        }
    
    return {
        'num_split_nodes': num_split,
        'num_shape_nodes': num_shape,
        'equivalent_nodes': equivalent_nodes,
        'total_step_count': total_step_count,
        'num_leaf_shape_nodes': len(terminal_shape_nodes),
        'num_features_used': len(features_used),
        'avg_features_per_decision': avg_features_per_decision
    }

def _find_all_shape_nodes(node):
    if node is None:
        return []
    out = []
    if isinstance(node, ShapeNode):
        out.append(node)
        for child_split in getattr(node, 'children', []) or []:
            out.extend(_find_all_shape_nodes(child_split))
        return out
    children = [getattr(node, 'true_child', None), getattr(node, 'false_child', None)]
    for ch in filter(None, children):
        out.extend(_find_all_shape_nodes(ch))
    return out


def save_results(model_or_state, X_train, y_train, X_val, y_val, X_test, y_test, args, visualize=True, data_info=None):
    """
    Evaluates the model, saves visualizations, and returns performance and sparsity metrics.
    """
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    predictor = model_or_state
    underlying_tree = getattr(model_or_state, "tree", None)
    if underlying_tree is not None and hasattr(underlying_tree, "predict_proba"):
        predictor = underlying_tree
    
    visualization_tree = None
    if hasattr(model_or_state, "build_visualization_tree"):
        try:
            visualization_tree = model_or_state.build_visualization_tree()
        except Exception as exc:
            logger.warning(f"Failed to build ensemble visualization tree: {exc}")
    if visualization_tree is None and underlying_tree is not None and hasattr(underlying_tree, "root"):
        visualization_tree = underlying_tree
    
    if visualize and visualization_tree is not None and hasattr(visualization_tree, "root"):
        plot_ad_tree(visualization_tree, filename=os.path.join(results_dir, "adtree_final_structure"))
        
        shape_nodes = _find_all_shape_nodes(visualization_tree.root)
        logger.info(f"Found {len(shape_nodes)} shape function nodes in the final model for visualization.")
        
        for node in shape_nodes:
            outname_prefix = os.path.join(results_dir, f"shape_func_node_{node.id}")
            
            if hasattr(node.shape_model, 'ebm_model'):
                plot_ebm_model(node.shape_model, outname_prefix)
            else:
                try:
                    sm = node.shape_model
                    comp_indices = getattr(sm, "feature_indices", None)
                    if comp_indices and len(comp_indices) > 1:
                        for ci, fi in enumerate(comp_indices):
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plot_spline_model(sm, f"f{fi}", ax=ax, data_info=data_info, component_idx=ci)
                            safe_name = f"{fi}_comp{ci}"
                            outname = f"{outname_prefix}_spline_{safe_name}.png"
                            fig.savefig(outname, bbox_inches="tight")
                            plt.close(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_spline_model(sm, node.feature_name, ax=ax, data_info=data_info)
                        safe_name = "".join(c if c.isalnum() else "_" for c in node.feature_name)
                        outname = f"{outname_prefix}_spline_{safe_name}.png"
                        fig.savefig(outname, bbox_inches="tight")
                        plt.close(fig)
                except Exception as e:
                    logger.error(f"Could not plot shape function for node {node.id}: {e}")
    elif visualize:
        logger.warning("Visualization requested, but model does not expose a tree structure. Skipping plots.")

    # --- Performance Evaluation ---
    y_train_pred_proba = predictor.predict_proba(X_train)[:, 1]
    y_val_pred_proba = predictor.predict_proba(X_val)[:, 1]
    
    best_threshold = find_best_threshold(y_val, y_val_pred_proba)
    if visualize:
        logger.info(f"Best threshold found on validation set: {best_threshold:.4f}")
    
    y_train_pred_01 = (y_train_pred_proba > best_threshold).astype(int)
    y_val_pred_01 = (y_val_pred_proba > best_threshold).astype(int)

    metrics_data = {
        'Train': evaluate_model(y_train, y_train_pred_proba, y_train_pred_01),
        'Validation': evaluate_model(y_val, y_val_pred_proba, y_val_pred_01),
    }
    
    if X_test is not None and y_test is not None:
        y_test_pred_proba = predictor.predict_proba(X_test)[:, 1]
        y_test_pred_01 = (y_test_pred_proba > best_threshold).astype(int)
        metrics_data['Test'] = evaluate_model(y_test, y_test_pred_proba, y_test_pred_01)
    
    df = pd.DataFrame(metrics_data)
    
    if visualization_tree is not None:
        sparsity = calculate_sparsity_metrics(visualization_tree, args)
    else:
        sparsity = {
            'num_split_nodes': np.nan,
            'num_shape_nodes': np.nan,
            'equivalent_nodes': np.nan,
        }
        if hasattr(model_or_state, "size"):
            sparsity['ensemble_size'] = model_or_state.size
    
    if visualize:
        print(f"\n--- Performance for Run: {os.path.basename(results_dir)} (threshold={best_threshold:.4f}) ---\n{df}\n")
        print(f"Sparsity Metrics: {sparsity}\n---------------------\n")
    
    return df, sparsity
# --- MCTS Diagnostics Plotting Functions (Ported from new version) ---

def plot_mcts_reward_curve(reward_history, results_dir, filename_prefix="reward"):
    """Plot MCTS convergence diagnostics.

    Plots:
    - reward per iteration (noisy exploration signal)
    - max reward per 10 iterations
    - best-so-far reward (running maximum).
    """
    if not reward_history:
        logger.warning("plot_mcts_reward_curve: empty reward_history, skip plotting.")
        return

    rewards = np.asarray(reward_history, dtype=float)
    iters = np.arange(1, len(rewards) + 1)

    # Running best-of-t curve (non-decreasing)
    best_so_far = np.maximum.accumulate(rewards)

    # Per-10-iteration maxima
    block_size = 10
    num_blocks = int(np.ceil(len(rewards) / block_size))
    block_max = []
    block_x = []
    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, len(rewards))
        block_max.append(rewards[start:end].max())
        block_x.append(end)

    plt.figure()
    plt.plot(iters, rewards, linewidth=0.6, alpha=0.25, label="reward per iteration")
    plt.plot(block_x, block_max, linewidth=1.6, label=f"max reward per {block_size} iterations")
    plt.plot(iters, best_so_far, linewidth=1.8, linestyle="--", label="best-so-far reward")

    plt.xlabel("Iteration")
    plt.ylabel("Validation reward")
    plt.title("MCTS reward and convergence diagnostics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{filename_prefix}_mcts_reward_curve.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved MCTS reward curve to {out_path}")


def plot_root_action_stats(root_stats_history, results_dir, filename_prefix="root_actions", top_k=8):
    """Plot root action visit counts and mean values over time.

    root_stats_history: list of dict snapshots. Each snapshot is a dict:
        {
          "iter": int,
          "actions": {
             action_repr: {"N": int, "Q": float, "mean_Q": float},
             ...
          }
        }
    """
    if not root_stats_history:
        logger.warning("plot_root_action_stats: empty history, skip plotting.")
        return

    # 1) Use the last snapshot to pick top_k actions by visit count
    last_actions = root_stats_history[-1]["actions"]
    sorted_actions = sorted(last_actions.items(), key=lambda kv: kv[1]["N"], reverse=True)
    top_actions = [a for a, stats in sorted_actions[:top_k]]

    iters = [snap["iter"] for snap in root_stats_history]

    # --- Visit counts ---
    plt.figure()
    for action in top_actions:
        visits = []
        for snap in root_stats_history:
            stats = snap["actions"].get(action)
            visits.append(stats["N"] if stats is not None else 0)
        plt.plot(iters, visits, label=action)
    plt.xlabel("Iteration")
    plt.ylabel("Visit count N(root,a)")
    plt.title("Root action visit counts over time (top actions)")
    plt.legend(fontsize=6)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    path_visits = os.path.join(results_dir, f"{filename_prefix}_root_action_visits.png")
    plt.savefig(path_visits)
    plt.close()
    logger.info(f"Saved root action visit plot to {path_visits}")

    # --- Mean Q values ---
    plt.figure()
    for action in top_actions:
        means = []
        for snap in root_stats_history:
            stats = snap["actions"].get(action)
            means.append(stats["mean_Q"] if stats is not None else np.nan)
        plt.plot(iters, means, label=action)
    plt.xlabel("Iteration")
    plt.ylabel("Mean value Q(root,a)/N(root,a)")
    plt.title("Root action mean value over time (top actions)")
    plt.legend(fontsize=6)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path_means = os.path.join(results_dir, f"{filename_prefix}_root_action_values.png")
    plt.savefig(path_means)
    plt.close()
    logger.info(f"Saved root action value plot to {path_means}")
