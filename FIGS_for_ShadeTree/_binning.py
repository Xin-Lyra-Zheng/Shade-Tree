"""Local percentile-binning subset required by prepare_data.py."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BinningConfig:
    strategy: str = "quantile"
    num_bins: int = 100
    min_unique: int = 5
    min_bin_frac: float = 0.0
    cap_quantiles: tuple[float, float] = (0.0, 1.0)
    generate: str = "binary"
    drop_original: bool = True
    one_hot_categories: bool = True
    max_thresholds_per_feature: Optional[int] = None
    dtype: str = "int8"
    na_policy: str = "separate"


class PercentileBinner:
    def __init__(self, config=None):
        self.config = config or BinningConfig()
        self.thresholds_ = {}
        self.categorical_levels_ = {}
        self.feature_order_ = []

    def fit(self, X):
        df = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)
        self.feature_order_ = list(df.columns)
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) >= self.config.min_unique:
                values = s.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if self.config.strategy == "quantile":
                    q = np.linspace(0, 1, self.config.num_bins + 1)[1:-1]
                    thresholds = np.unique(np.quantile(values, q, method="nearest")) if values.size else []
                else:
                    thresholds = np.linspace(values.min(), values.max(), self.config.num_bins + 1)[1:-1] if values.size else []
                thresholds = np.asarray(thresholds, dtype=float)
                limit = self.config.max_thresholds_per_feature
                if limit and thresholds.size > limit:
                    take = np.linspace(0, thresholds.size - 1, limit).round().astype(int)
                    thresholds = thresholds[take]
                self.thresholds_[col] = thresholds.tolist()
            else:
                self.categorical_levels_[col] = s.astype("string").dropna().astype(str).unique().tolist()
        return self

    def transform_numpy(self, X, feature_names=None, return_sparse=False):
        values = X.values if hasattr(X, "values") else np.asarray(X)
        feature_names = list(X.columns) if hasattr(X, "columns") else (feature_names or self.feature_order_)
        positions = {name: i for i, name in enumerate(feature_names)}
        blocks, names = [], []
        for col, levels in self.categorical_levels_.items():
            series = pd.Series(values[:, positions[col]]).astype("string").to_numpy()
            for level in levels:
                blocks.append((series == level).astype(self.config.dtype)[:, None])
                names.append(f"{col}_{level}")
            if self.config.na_policy == "separate":
                blocks.append(pd.isna(series).astype(self.config.dtype)[:, None])
                names.append(f"{col}_isnan")
        for col, threshold_list in self.thresholds_.items():
            series = values[:, positions[col]].astype(float)
            thresholds = np.asarray(threshold_list, dtype=float)
            if thresholds.size:
                blocks.append((series[:, None] <= thresholds[None, :]).astype(self.config.dtype))
                names.extend([f"{col}<={t:g}" for t in threshold_list])
            if self.config.na_policy == "separate":
                blocks.append(np.isnan(series).astype(self.config.dtype)[:, None])
                names.append(f"{col}_isnan")
        result = np.concatenate(blocks, axis=1) if blocks else np.empty((values.shape[0], 0), dtype=self.config.dtype)
        if return_sparse:
            from scipy import sparse
            result = sparse.csr_matrix(result)
        return result, names
