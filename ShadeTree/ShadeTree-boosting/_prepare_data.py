import os
import pandas as pd
import numpy as np
from _binning import PercentileBinner, BinningConfig
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from types import SimpleNamespace

def _load_single_csv(data_dir, config):
    if 'filename' not in config:
        raise ValueError("Missing 'filename'.")
    data_path = os.path.join(data_dir, config['filename'])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")
    df = pd.read_csv(data_path, sep=config.get('sep', None), engine='python')
    # delete column "name" in Titanic dataset if exists
    if config['filename'] == 'Titanic.csv' and 'name' in df.columns:
        df.drop(columns=['name'], inplace=True)
    tc = config['target_col']
    if tc not in df.columns:
        raise KeyError(f"Cannot find '{tc}' in {config['filename']}. Actual column name: {list(df.columns)}")
    if df[tc].dtype == object:
        df[tc] = df[tc].astype(str).str.strip().str.replace('.', '', regex=False)
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    df.dropna(inplace=True)
    return df, None

def _load_sklearn_dataset(data_dir, config):
    from sklearn.datasets import load_breast_cancer, load_iris
    name = str(config.get('sk_name', '')).lower()
    if name == 'breast_cancer':
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.copy()
    elif name == 'iris':
        ds = load_iris(as_frame=True)
        df = ds.frame.copy()
    else:
        raise ValueError(f"Unsupported sklearn dataset name: {name!r}")

    target_col = config.get('target_col', 'target')
    if target_col not in df.columns:
        if 'target' in df.columns:
            df.rename(columns={'target': target_col}, inplace=True)
        else:
            raise KeyError(...)
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

def _load_adult_dataset(data_dir, config):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
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
    
    df_test[config['target_col']] = df_test[config['target_col']].str.replace('.', '', regex=False)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    return df_train, df_test

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
        # Check for a pre-downloaded local CSV first (avoids network calls on compute nodes)
        local_filename = config.get('filename')
        local_path = os.path.join(data_dir, local_filename) if local_filename else None
        if local_path and os.path.exists(local_path):
            sep = config.get('sep', ',')
            df = pd.read_csv(local_path, sep=sep)
        else:
            dataset = fetch_ucirepo(id=config['id'])
            X_df = dataset.data.features
            y_df = dataset.data.targets
            df = pd.concat([X_df, y_df], axis=1)
            df['quality'] = (df['quality'] >= 6).astype(int)
    else:
        data_path = os.path.join(data_dir, config['filename'])
        df = pd.read_csv(data_path, sep=config.get('sep', ','))

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

# def prepare_data(args):
#     german_credit_columns = [
#         "checking_status",
#         "duration",
#         "credit_history",
#         "purpose",
#         "credit_amount",
#         "savings_status",
#         "employment",
#         "installment_commitment",
#         "personal_status",
#         "other_parties",
#         "residence_since",
#         "property_magnitude",
#         "age",
#         "other_payment_plans",
#         "housing",
#         "existing_credits",
#         "job",
#         "num_dependents",
#         "own_telephone",
#         "foreign_worker",
#         "class"
#     ]
#     heart_disease_columns = [
#         "age",
#         "sex",
#         "cp",
#         "trestbps",
#         "chol",
#         "fbs",
#         "restecg",
#         "thalach",
#         "exang",
#         "oldpeak",
#         "slope",
#         "ca",
#         "thal",
#         "num"
#     ]
#     magic_telescope_columns = [
#         "fLength",
#         "fWidth",
#         "fSize",
#         "fConc",
#         "fConc1",
#         "fAsym",
#         "fM3Long",
#         "fM3Trans",
#         "fAlpha",
#         "fDist",
#         "class"
#     ]
#     DATA_CONFIG = {
#         'adult': {
#             'loader_func': _load_adult_dataset,
#             'target_col': 'income',
#             'pos_class': '>50K'
#         },
#         'bank': {
#             'loader_func': _load_generic_dataset,
#             'filename': 'bank.csv', 
#             'target_col': 'y',
#             'pos_class': 'yes', 'sep': ';'
#         },
#         'wine': {
#             'loader_func': _load_generic_dataset,
#             'id': 186, 
#             'target_col': 'quality',
#             'pos_class': 1
#         },
#         'winequality_red': {
#             'loader_func': _load_generic_dataset,
#             'filename': 'winequality-red.csv', 
#             'target_col': 'quality',
#             'pos_class': 1, 'sep': ';'
#         },
#         'winequality_white': {
#             'loader_func': _load_generic_dataset,
#             'filename': 'winequality-white.csv', 
#             'target_col': 'quality',
#             'pos_class': 1, 'sep': ';'
#         },
#         'adult_balanced': {
#             'loader_func': _load_single_csv,
#             'filename': 'adult_balanced.csv', 
#             'target_col': 'target',
#             'pos_class': '>50K'
#         },
#         'bank_balanced': {
#             'loader_func': _load_single_csv,
#             'filename': 'bank_balanced.csv', 
#             'target_col': 'y',
#             'pos_class': 'yes'
#         },
#         'bank_add': {
#             'loader_func': _load_generic_dataset,
#             'filename': 'bank-additional-full.csv', 
#             'target_col': 'y',
#             'pos_class': 'yes', 'sep': ';'
#         },
#         'bank_add_balanced': {
#             'loader_func': _load_single_csv,
#             'filename': 'bank_add_balanced.csv', 
#             'target_col': 'target',
#             'pos_class': 'yes', 'sep': ';'
#         },
#         'breast_cancer': {
#             'loader_func': _load_sklearn_dataset,
#             'sk_name': 'breast_cancer', 
#             'target_col': 'target',
#             'pos_class': 1
#         },
#         'iris': {
#             'loader_func': _load_sklearn_dataset,
#             'sk_name': 'iris', 
#             'target_col': 'target',
#             'pos_class': 1
#         },
#         'blood_transfusion': {
#             'loader_func': _load_single_csv,
#             'filename': 'blood_transfusion.csv', 
#             'target_col': 'Class',
#             'pos_class': 2,
#         },
#         'compas': {
#             'loader_func': _load_single_csv,
#             'filename': 'compas.csv', 
#             'target_col': 'recid',
#             'pos_class': 1,
#         },
#         'covertype': {
#             'loader_func': _load_single_csv,
#             'filename': 'covertype.csv', 
#             'target_col': 'Cover_Type',
#             'pos_class': 1,
#         },
#         'diabetes': {
#             'loader_func': _load_single_csv,
#             'filename': 'diabetes.csv', 
#             'target_col': 'class',
#             'pos_class': 'tested_positive',
#         },
#         'higgs': {
#             'loader_func': _load_single_csv,
#             'filename': 'higgs.csv', 
#             'target_col': 'target',
#             'pos_class': 'b\'0\'',
#         },
#         'netherlands': {
#             'loader_func': _load_single_csv,
#             'filename': 'netherlands.csv', 
#             'target_col': 'recidivism_in_4y',
#             'pos_class': 1,
#         },
#         'spambase': {
#             'loader_func': _load_single_csv,
#             'filename': 'spambase.csv', 
#             'target_col': 'class',
#             'pos_class': 1,
#         },
#         'Titanic': {
#             'loader_func': _load_single_csv,
#             'filename': 'Titanic.csv', 
#             'target_col': 'survived',
#             'pos_class': 1,
#         },
#         'news_headline1': {
#             'loader_func': _load_single_csv,
#             'filename': 'news_headline1.csv',
#             'target_col': 'Y',
#             'pos_class': 1,
#         },
#         'news_headline2': {
#             'loader_func': _load_single_csv,
#             'filename': 'news_headline2.csv',
#             'target_col': 'Y',
#             'pos_class': 1,
#         },
#         'news_headline3': {
#             'loader_func': _load_single_csv,
#             'filename': 'news_headline3.csv',
#             'target_col': 'Y',
#             'pos_class': 1,
#         },
#         'fico': {
#             'loader_func': _load_single_csv,
#             'filename': 'fico.csv',
#             'target_col': 'PoorRiskPerformance',
#             'pos_class': 1,
#         },
#         'higgs_20k': {
#             'loader_func': _load_single_csv,
#             'filename': 'higgs_20k.csv',
#             'target_col': 'target',
#             'pos_class': 1,
#         },
#         'german_credit':{
#             "loader_func": _load_data_file,
#             "filename": "german.data",
#             "column_names": german_credit_columns,
#             "target_col": "class",
#             "pos_class": 1,
#             "sep": r"\s+",
#             "na_values": None,
#         },
#         'heart_disease_cleverland':{
#             "loader_func": _load_heart_disease,
#             "filename": "processed.cleveland.data",
#             "column_names": heart_disease_columns,
#             "target_col": "target",
#             "pos_class": 1,
#             "sep": r",\s*",
#             "na_values": "?"
#         },
#         'telescope':{
#             "loader_func": _load_data_file,
#             "filename": "magic04.data",
#             "column_names": magic_telescope_columns,
#             "target_col": "class",
#             "pos_class": "g",
#             "sep": r",\s*",
#             "na_values": None,
#         },
#         'TWCredit':{ 
#             "loader_func": _load_single_csv,
#             "filename": "TWCredit.csv",
#             "target_col": "default.payment.next.month",
#             "pos_class": 1, 
#             "sep": ","
#         },
#         'phishing_website':{
#             "loader_func": _load_phishing_arff,
#             "filename": "Training Dataset.arff",
#             "target_col": "target",
#             "pos_class": 1
#         }
#     }

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(script_dir, 'dataset')
#     if args.dataset not in DATA_CONFIG:
#         print(f"DATA_CONFIG:{DATA_CONFIG}") if getattr(args, 'verbose', False) else None
#         raise ValueError(f"Cannot find configuration for '{args.dataset}'.")

#     config = DATA_CONFIG[args.dataset]
#     df_train, df_test = config['loader_func'](data_dir, config)

#     if args.dataset == 'adult' and df_test is not None:
#         y_train_full = df_train[config['target_col']]
#         X_train_full = df_train.drop(columns=config['target_col'])

#         y_test = df_test[config['target_col']]
#         X_test = df_test.drop(columns=config['target_col'])

#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train_full, y_train_full,
#             test_size=0.25,
#             random_state=args.random_state,
#             stratify=y_train_full
#         )
#     else:
#         print(f"Splitting '{args.dataset}' dataset (50/25/25)...")
#         y_all = df_train[config['target_col']]
#         X_all = df_train.drop(columns=config['target_col'])

#         X_train_val, X_test, y_train_val, y_test = train_test_split(
#             X_all, y_all, test_size=0.25, random_state=args.random_state, stratify=y_all
#         )
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train_val, y_train_val, test_size=1/3, random_state=args.random_state, stratify=y_train_val
#         )

#     def _binarize(y: pd.Series, pos_class):
#         """
#         Robustly binarizes the target column to 1 and 0, handling both numeric and string labels.
#         """
#         try:
#             # Attempt to treat pos_class as a number
#             pos_numeric = float(pos_class)
#             # If successful, treat the entire 'y' series as numeric for comparison
#             y_numeric = pd.to_numeric(y, errors='coerce')
#             result = np.where(y_numeric == pos_numeric, 1, 0)
#         except (ValueError, TypeError):
#             # If pos_class is a string (e.g., 'yes'), treat both as strings
#             y_str = y.astype(str).str.strip()
#             pos_str = str(pos_class).strip()
#             result = np.where(y_str == pos_str, 1, 0)
            
#         return result.astype(int)

#     pos_class = config['pos_class']
#     y_train = _binarize(y_train, pos_class)
#     y_val   = _binarize(y_val,   pos_class)
#     y_test  = _binarize(y_test,  pos_class)

#     X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()
#     cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
#     num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

#     ohe = ModuleNotFoundError
#     ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
#     ohe.fit(X_train[cat_cols])

#     feature_names = X_train.columns.tolist()
#     data_info = SimpleNamespace(feature_names=feature_names)

#     # Binning for FSG
#     cfg = BinningConfig(strategy=str(getattr(args, "binning_strategy", "quantile")),
#                         num_bins=int(getattr(args, "num_bins", 64)),
#                         max_thresholds_per_feature=int(getattr(args, "max_thresholds_for_fsg", 64)),
#                         dtype="int8")
#     binner = PercentileBinner(cfg).fit(X_train)
#     Xtr_bin, feat_names_bin = binner.transform_numpy(X_train, feature_names=feature_names)
#     Xva_bin, _ = binner.transform_numpy(X_val, feature_names=feature_names)
#     Xte_bin, _ = binner.transform_numpy(X_test, feature_names=feature_names)
#     data_info.X_train_bin = Xtr_bin.astype(np.int8, copy=False)
#     data_info.X_val_bin   = Xva_bin.astype(np.int8, copy=False)
#     data_info.X_test_bin  = Xte_bin.astype(np.int8, copy=False)
#     data_info.feature_names_bin = feat_names_bin
#     print(f"feature_name_bin:{feat_names_bin}")
#     data_info.binner = binner

#     X_all_bin = np.vstack([Xtr_bin, Xva_bin, Xte_bin])

#     y_all = np.concatenate([y_train, y_val, y_test])

#     split_col = (
#         ['train'] * len(y_train) +
#         ['val']   * len(y_val)   +
#         ['test']  * len(y_test)
#     )

#     df_all = pd.DataFrame(X_all_bin, columns=feat_names_bin)
#     df_all[config['target_col']] = y_all.astype(int)
#     df_all['split'] = split_col

#     out_path_all = os.path.join(data_dir, f"{args.dataset}_all_bin_{args.random_state}.csv")
#     df_all.to_csv(out_path_all, index=False)
#     print(f"[saved combined binned dataset] {out_path_all} with shape {df_all.shape}")

#     print(f"Dataset '{args.dataset}': {len(X_train)+len(X_val)+len(X_test)} samples, {len(feature_names)} features.")
#     print(f"Training set({len(X_train)}), Validation Set({len(X_val)}), Test Set({len(X_test)})")

#     return (X_train.to_numpy(dtype=np.float32),
#             X_val.to_numpy(dtype=np.float32),
#             X_test.to_numpy(dtype=np.float32),
#             y_train, y_val, y_test, data_info)

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
            'filename': 'wine.csv',   # local cache: red+white merged, quality>=6->1
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
        print(f"DATA_CONFIG:{DATA_CONFIG}") if getattr(args, 'verbose', False) else None
        raise ValueError(f"Cannot find configuration for '{args.dataset}'.")

    config = DATA_CONFIG[args.dataset]
    df_train, df_test = config['loader_func'](data_dir, config)

    if args.dataset == 'adult' and df_test is not None:
        y_train_full = df_train[config['target_col']]
        X_train_full = df_train.drop(columns=config['target_col'])

        y_test = df_test[config['target_col']]
        X_test = df_test.drop(columns=config['target_col'])

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.25,
            random_state=args.random_state,
            stratify=y_train_full
        )
    else:
        print(f"Splitting '{args.dataset}' dataset (50/25/25)...")
        y_all = df_train[config['target_col']]
        X_all = df_train.drop(columns=config['target_col'])

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=args.random_state, stratify=y_all
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=1/3, random_state=args.random_state, stratify=y_train_val
        )

    def _binarize(y: pd.Series, pos_class):
        """
        Robustly binarizes the target column to 1 and 0, handling both numeric and string labels.
        """
        try:
            pos_numeric = float(pos_class)
            y_numeric = pd.to_numeric(y, errors='coerce')
            result = np.where(y_numeric == pos_numeric, 1, 0)
        except (ValueError, TypeError):
            y_str = y.astype(str).str.strip()
            pos_str = str(pos_class).strip()
            result = np.where(y_str == pos_str, 1, 0)

        return result.astype(int)

    pos_class = config['pos_class']
    y_train = _binarize(y_train, pos_class)
    y_val   = _binarize(y_val,   pos_class)
    y_test  = _binarize(y_test,  pos_class)

    X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    if len(cat_cols) > 0:
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
        ohe.fit(X_train[cat_cols])
    else:
        ohe = None

    feature_names = X_train.columns.tolist()
    data_info = SimpleNamespace(feature_names=feature_names)

    cfg = BinningConfig(strategy=str(getattr(args, "binning_strategy", "quantile")),
                        num_bins=int(getattr(args, "num_bins", 64)),
                        max_thresholds_per_feature=int(getattr(args, "max_thresholds_for_fsg", 64)),
                        dtype="int8")
    binner = PercentileBinner(cfg).fit(X_train)
    Xtr_bin, feat_names_bin = binner.transform_numpy(X_train, feature_names=feature_names)
    Xva_bin, _ = binner.transform_numpy(X_val, feature_names=feature_names)
    Xte_bin, _ = binner.transform_numpy(X_test, feature_names=feature_names)
    data_info.X_train_bin = Xtr_bin.astype(np.int8, copy=False)
    data_info.X_val_bin   = Xva_bin.astype(np.int8, copy=False)
    data_info.X_test_bin  = Xte_bin.astype(np.int8, copy=False)
    data_info.feature_names_bin = feat_names_bin
    # Map each binned column back to its raw feature index for sparsity grouping
    f2g = []
    for name in feat_names_bin:
        raw_name = str(name).split("<=")[0].strip()
        try:
            idx = feature_names.index(raw_name)
        except ValueError:
            try:
                idx = int(raw_name.split("_", 1)[1]) if raw_name.startswith("f_") else int(raw_name.lstrip("f"))
            except Exception:
                idx = 0
        f2g.append(idx)
    data_info.fsg_feature_to_group = np.asarray(f2g, dtype=int)
    print(f"feature_name_bin:{feat_names_bin}")
    data_info.binner = binner

    X_all_bin = np.vstack([Xtr_bin, Xva_bin, Xte_bin])
    y_all = np.concatenate([y_train, y_val, y_test])

    split_col = (
        ['train'] * len(y_train) +
        ['val']   * len(y_val)   +
        ['test']  * len(y_test)
    )

    df_all = pd.DataFrame(X_all_bin, columns=feat_names_bin)
    df_all[config['target_col']] = y_all.astype(int)
    df_all['split'] = split_col

    print(f"Dataset '{args.dataset}': {len(X_train)+len(X_val)+len(X_test)} samples, {len(feature_names)} features.")
    print(f"Training set({len(X_train)}), Validation Set({len(X_val)}), Test Set({len(X_test)})")

    if len(num_cols) > 0:
        X_train_num = X_train[num_cols].astype(np.float32)
        X_val_num   = X_val[num_cols].astype(np.float32)
        X_test_num  = X_test[num_cols].astype(np.float32)
    else:
        X_train_num = pd.DataFrame(index=X_train.index)
        X_val_num   = pd.DataFrame(index=X_val.index)
        X_test_num  = pd.DataFrame(index=X_test.index)

    if len(cat_cols) > 0:
        if hasattr(ohe, "get_feature_names_out"):
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
        else:
            cat_feature_names = ohe.get_feature_names(cat_cols)

        X_train_cat = pd.DataFrame(
            ohe.transform(X_train[cat_cols]),
            index=X_train.index,
            columns=cat_feature_names
        ).astype(np.float32)

        X_val_cat = pd.DataFrame(
            ohe.transform(X_val[cat_cols]),
            index=X_val.index,
            columns=cat_feature_names
        ).astype(np.float32)

        X_test_cat = pd.DataFrame(
            ohe.transform(X_test[cat_cols]),
            index=X_test.index,
            columns=cat_feature_names
        ).astype(np.float32)
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_val_cat   = pd.DataFrame(index=X_val.index)
        X_test_cat  = pd.DataFrame(index=X_test.index)

    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_val   = pd.concat([X_val_num,   X_val_cat],   axis=1)
    X_test  = pd.concat([X_test_num,  X_test_cat],  axis=1)

    data_info.feature_names = X_train.columns.tolist()

    return (X_train.to_numpy(dtype=np.float32),
            X_val.to_numpy(dtype=np.float32),
            X_test.to_numpy(dtype=np.float32),
            y_train, y_val, y_test, data_info)
