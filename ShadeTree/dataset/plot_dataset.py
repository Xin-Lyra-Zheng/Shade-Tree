import os
import pandas as pd
import numpy as np
import logging
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Required Machine Learning Libraries ---
try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    import pacmap
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please install the required libraries: pip install pacmap scikit-learn matplotlib")
    exit()

# --- Use the exact data loading logic from your project ---
# This ensures consistency with your experiments without changing util.py.
from util import (
    _load_single_csv,
    _load_sklearn_dataset,
    _load_adult_dataset,
    _load_generic_dataset
)

# Configure a basic logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_raw_data_using_util_loaders(dataset_name: str, config: dict):
    """
    Calls the appropriate internal loader function from util.py and correctly unpacks
    the returned tuple to get the raw dataframe.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'dataset')
    
    loader_func_name = config.get('loader_func_name')
    
    # Map the function name from the config to the actual imported function
    loader_map = {
        '_load_single_csv': _load_single_csv,
        '_load_sklearn_dataset': _load_sklearn_dataset,
        '_load_adult_dataset': _load_adult_dataset,
        '_load_generic_dataset': _load_generic_dataset
    }
    
    loader_func = loader_map.get(loader_func_name)
    if not loader_func:
        raise ValueError(f"Loader function '{loader_func_name}' is not recognized or found in util.py.")

    # The loaders in util.py return a tuple (e.g., (df, None)). We must capture it.
    returned_object = loader_func(data_dir, config)
    
    # We now correctly extract just the DataFrame, which is always the first element.
    raw_dataframe = returned_object[0]
    
    return raw_dataframe


def preprocess_features(X: pd.DataFrame):
    """
    Applies the same scaling and one-hot encoding logic from util.prepare_data
    to the feature set.
    """
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    X_processed_parts = []
    
    if num_cols:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X[num_cols])
        X_processed_parts.append(pd.DataFrame(X_num_scaled, columns=num_cols, index=X.index))
        
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        X_cat_ohe = ohe.fit_transform(X[cat_cols])
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
        X_processed_parts.append(pd.DataFrame(X_cat_ohe, columns=ohe_feature_names, index=X.index))
        
    if not X_processed_parts:
        return pd.DataFrame()

    return pd.concat(X_processed_parts, axis=1)


def main():
    """
    Main function to generate and save embedding plots for all datasets.
    """
    # NOTE: This DATA_CONFIG is replicated from util.py because it's defined locally
    # inside util.prepare_data and cannot be imported directly.
    DATA_CONFIG = {
        # 'adult': { 'loader_func_name': '_load_adult_dataset', 'target_col': 'income', 'pos_class': '>50K' },
        # 'bank': { 'loader_func_name': '_load_generic_dataset', 'filename': 'bank.csv', 'target_col': 'y', 'pos_class': 'yes' },
        # 'wine': { 'loader_func_name': '_load_generic_dataset', 'id': 186, 'target_col': 'quality', 'pos_class': 1 },
        # 'adult_balanced': { 'loader_func_name': '_load_single_csv', 'filename': 'adult_balanced.csv', 'target_col': 'target', 'pos_class': '>50K' },
        # 'bank_balanced': { 'loader_func_name': '_load_single_csv', 'filename': 'bank_balanced.csv', 'target_col': 'y', 'pos_class': 'yes' },
        # 'breast_cancer': { 'loader_func_name': '_load_sklearn_dataset', 'sk_name': 'breast_cancer', 'target_col': 'target', 'pos_class': 1 },
        # 'iris': { 'loader_func_name': '_load_sklearn_dataset', 'sk_name': 'iris', 'target_col': 'target', 'pos_class': 1 },
        # 'blood_transfusion': { 'loader_func_name': '_load_single_csv', 'filename': 'blood_transfusion.csv', 'target_col': 'Class', 'pos_class': 2 },
        'compas': { 'loader_func_name': '_load_single_csv', 'filename': 'compas.csv', 'target_col': 'recid', 'pos_class': 1 },
        # 'covertype': { 'loader_func_name': '_load_single_csv', 'filename': 'covertype.csv', 'target_col': 'Cover_Type', 'pos_class': 1 },
        'diabetes': { 'loader_func_name': '_load_single_csv', 'filename': 'diabetes.csv', 'target_col': 'class', 'pos_class': 'tested_positive' },
        # 'higgs': { 'loader_func_name': '_load_single_csv', 'filename': 'higgs.csv', 'target_col': 'target', 'pos_class': "b'1'"},
        'netherlands': { 'loader_func_name': '_load_single_csv', 'filename': 'netherlands.csv', 'target_col': 'recidivism_in_4y', 'pos_class': 1 },
        'spambase': { 'loader_func_name': '_load_single_csv', 'filename': 'spambase.csv', 'target_col': 'class', 'pos_class': 1 },
        # 'Titanic': { 'loader_func_name': '_load_single_csv', 'filename': 'Titanic.csv', 'target_col': 'survived', 'pos_class': 1 },
    }
    
    output_dir = "embedding_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting embedding generation. Plots will be saved to '{output_dir}/'")

    for dataset_name, config in DATA_CONFIG.items():
        logging.info(f"--- Processing dataset: {dataset_name} ---")
        
        try:
            # --- 1. Load Data ---
            df_raw = get_raw_data_using_util_loaders(dataset_name, config)
            
            if dataset_name == 'wine':
                from ucimlrepo import fetch_ucirepo
                logging.info("Re-loading 'wine' dataset to get original multi-class labels.")
                wine_ds = fetch_ucirepo(id=186)
                df_raw = pd.concat([wine_ds.data.features, wine_ds.data.targets], axis=1)

            df = df_raw.dropna().reset_index(drop=True)
            
            # --- 2. Dynamically set plotting parameters based on sample size ---
            num_samples = len(df)
            if num_samples > 1000:
                marker_alpha = 0.4
                marker_size = 2
            else:
                marker_alpha = 0.6
                marker_size = 5
            
            if num_samples < 10:
                logging.warning(f"Dataset '{dataset_name}' has fewer than 10 samples. Skipping.")
                continue

            # --- 3. Separate Features (X) and Original Target (y) ---
            target_col = config['target_col']
            y_original = df[target_col]
            X_original = df.drop(columns=[target_col])
            X_processed = preprocess_features(X_original)
            y_codes, class_names = pd.factorize(y_original)
            num_classes = len(class_names)
            logging.info(f"Found {num_classes} original classes.")

            # --- 4. [NEW] Define colormap based on number of classes ---
            if num_classes == 2:
                # Use a high-contrast Blue for Class 0 and Red for Class 1
                # Using specific hex codes for visually appealing shades
                custom_cmap = ListedColormap(['#1f77b4', '#d62728'])
                logging.info("Using custom [Blue, Red] colormap for binary classification.")
            else:
                # Fallback to a high-contrast multi-color map for other cases
                custom_cmap = plt.get_cmap('Set1', num_classes)
                logging.info(f"Using '{custom_cmap.name}' colormap for multi-class problem.")
            # --- End of new logic ---

            # --- 5. Compute Embeddings ---
            logging.info("Computing 3D PCA projection...")
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_processed)
            
            logging.info("Computing 2D PaCMAP embedding...")
            pacmap_transformer = pacmap.PaCMAP(n_components=2, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
            X_pacmap = pacmap_transformer.fit_transform(X_processed.to_numpy())

            # --- 6. Create and Save the Plot ---
            fig = plt.figure(figsize=(20, 8))
            dataset_title = dataset_name.replace("_", " ").title()
            fig.suptitle(f'Embedding for {dataset_title} Dataset', fontsize=28, y=1.02)

            ax1 = fig.add_subplot(1, 2, 1)
            # --- MODIFICATION: Use custom_cmap defined above ---
            scatter1 = ax1.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_codes, cmap=custom_cmap, alpha=marker_alpha, s=marker_size)
            ax1.set_title(f'{dataset_title} with PaCMAP (2D)')
            ax1.legend(handles=scatter1.legend_elements(num=num_classes)[0], labels=[f'Class {c}' for c in class_names])
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            # --- MODIFICATION: Use custom_cmap defined above ---
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_codes, cmap=custom_cmap, alpha=marker_alpha, s=marker_size)
            ax2.set_title('PCA Projection (3D)')
            ax2.set_xlabel('Principal Component 1')
            ax2.set_ylabel('Principal Component 2')
            ax2.set_zlabel('Principal Component 3')
            ax2.legend(handles=scatter2.legend_elements(num=num_classes)[0], labels=[f'Class {c}' for c in class_names], title="Classes")

            plt.tight_layout()
            output_path = os.path.join(output_dir, f'{dataset_name}_embedding.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            logging.info(f"Plot saved to '{output_path}'")

        except Exception as e:
            logging.error(f"Could not generate plot for '{dataset_name}'. Error: {e}", exc_info=True)
            
    logging.info("--- All datasets processed. ---")


if __name__ == "__main__":
    main()