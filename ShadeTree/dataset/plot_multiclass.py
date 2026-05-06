import os
import pandas as pd
import numpy as np
import logging
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Required Machine Learning Libraries ---
try:
    from sklearn.decomposition import PCA
    import pacmap
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please install the required libraries: pip install pacmap scikit-learn matplotlib")
    exit()

# --- Directly import the core data processing function from your project ---
# This ensures we are visualizing the exact same data the model trains on.
from util import prepare_data

# Configure a basic logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Main function to generate embedding plots for ONLY the datasets that are
    originally multi-class but are forced into a binary problem by util.py.
    """
    # This dictionary provides clear descriptions for the plot legends.
    DATASET_DESCRIPTIONS = {
        'wine': {'neg': 'Bad Quality (<6)', 'pos': 'Good Quality (>=6)'},
        'iris': {'neg': 'Not Class 1', 'pos': 'Class 1'}
    }
    
    # We explicitly define the list of datasets to process.
    datasets_to_plot = ['wine', 'iris', 'covertype']
    
    output_dir = "forced_binary_embedding_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting generation of forced-binary embeddings. Plots will be saved to '{output_dir}/'")

    for dataset_name in datasets_to_plot:
        logging.info(f"--- Processing dataset: {dataset_name} ---")
        
        try:
            # --- 1. Use util.prepare_data to get the final, binarized data ---
            # This is the most crucial step to ensure data consistency.
            dummy_args = SimpleNamespace(dataset=dataset_name)
            X_processed, y_binarized, data_info = prepare_data(dummy_args)

            # --- 2. Dynamically set plotting parameters ---
            num_samples = len(y_binarized)
            if num_samples > 1000:
                marker_alpha = 0.4
                marker_size = 2
            else:
                marker_alpha = 0.6
                marker_size = 5
            
            if num_samples < 10:
                logging.warning(f"Dataset '{dataset_name}' has too few samples (<10). Skipping plot.")
                continue

            # --- 3. Prepare colors and legends ---
            # y_binarized is -1 and 1. Convert it to 0 and 1 for plotting.
            y_codes = (y_binarized == 1).astype(int)
            
            # Define a high-contrast [Blue, Red] colormap for binary classification.
            custom_cmap = ListedColormap(['#1f77b4', '#d62728'])
            
            # Get legend labels from the description dictionary
            desc = DATASET_DESCRIPTIONS.get(dataset_name, {'neg': 'Class 0', 'pos': 'Class 1'})
            legend_labels = [f"{desc['neg']}", f"{desc['pos']}"]

            # --- 4. Compute dimensionality reduction ---
            logging.info(f"Dataset '{dataset_name}' has {num_samples} samples and {X_processed.shape[1]} features after processing.")
            logging.info("Computing 3D PCA projection...")
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_processed)
            
            logging.info("Computing 2D PaCMAP embedding...")
            pacmap_transformer = pacmap.PaCMAP(n_components=2, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
            X_pacmap = pacmap_transformer.fit_transform(X_processed)

            # --- 5. Create and save the plot ---
            fig = plt.figure(figsize=(20, 8))
            dataset_title = dataset_name.replace("_", " ").title()
            fig.suptitle(f'Forced Binary Embedding for {dataset_title} Dataset', fontsize=28, y=1.02)

            # Left plot: PaCMAP 2D
            ax1 = fig.add_subplot(1, 2, 1)
            scatter1 = ax1.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_codes, cmap=custom_cmap, alpha=marker_alpha, s=marker_size)
            ax1.set_title(f'{dataset_title} with PaCMAP (2D)')
            ax1.legend(handles=scatter1.legend_elements()[0], labels=legend_labels, title="Classes")
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Right plot: PCA 3D
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_codes, cmap=custom_cmap, alpha=marker_alpha, s=marker_size)
            ax2.set_title('PCA Projection (3D)')
            ax2.set_xlabel('Principal Component 1')
            ax2.set_ylabel('Principal Component 2')
            ax2.set_zlabel('Principal Component 3')
            ax2.legend(handles=scatter2.legend_elements()[0], labels=legend_labels, title="Classes")

            plt.tight_layout()
            output_path = os.path.join(output_dir, f'{dataset_name}_forced_binary_embedding.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            logging.info(f"Plot saved to '{output_path}'")

        except Exception as e:
            logging.error(f"Could not generate plot for '{dataset_name}'. Error: {e}", exc_info=True)
            
    logging.info("--- All specified datasets processed. ---")


if __name__ == "__main__":
    main()