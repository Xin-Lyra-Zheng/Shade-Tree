from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

@dataclass
class BinningConfig:
    strategy: str = "quantile"  # "quantile" or "uniform"
    num_bins: int = 100
    min_unique: int = 5
    min_bin_frac: float = 0.0
    cap_quantiles: Tuple[float, float] = (0.0, 1.0)
    generate: str = "binary"    # "binary", "onehot", or "both"
    drop_original: bool = True
    one_hot_categories: bool = True
    max_thresholds_per_feature: Optional[int] = None
    dtype: str = "int8"
    na_policy: str = "separate"
    keep: Optional[List[str]] = None
    continuous_override: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

class PercentileBinner:
    def __init__(self, config: Optional[BinningConfig] = None):
        self.config = config or BinningConfig()
        self.thresholds_: Dict[str, List[float]] = {}
        self.categorical_levels_: Dict[str, List[str]] = {}
        self.continuous_features_: List[str] = []
        self.feature_order_: List[str] = []  # order seen in fit()
        self.fitted_: bool = False

    def _is_continuous(self, s: pd.Series) -> bool:
        if self.config.continuous_override and s.name in self.config.continuous_override:
            return True
        if self.config.exclude and s.name in self.config.exclude:
            return False
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            nunique = s.nunique(dropna=True)
            return nunique >= self.config.min_unique
        return False

    def fit(self, df: pd.DataFrame) -> "PercentileBinner":
        cfg = self.config
        X = df.copy()
        self.feature_order_ = list(X.columns)

        for col in X.columns:
            if cfg.exclude and col in cfg.exclude:
                continue
            s = X[col]
            if self._is_continuous(s):
                self.continuous_features_.append(col)
                x = s.to_numpy()
                x = x[np.isfinite(x)]
                if x.size == 0:
                    self.thresholds_[col] = []
                    continue

                lo_q, hi_q = cfg.cap_quantiles
                if lo_q > 0 or hi_q < 1:
                    lo = np.nanquantile(x, lo_q)
                    hi = np.nanquantile(x, hi_q)
                    x = np.clip(x, lo, hi)

                if cfg.strategy == "quantile":
                    qs = np.linspace(0, 1, cfg.num_bins + 1)[1:-1]
                    thr = np.unique(np.quantile(x, qs, method="nearest"))
                elif cfg.strategy == "uniform":
                    lo = np.nanmin(x); hi = np.nanmax(x)
                    if lo == hi:
                        thr = np.array([])
                    else:
                        thr = np.linspace(lo, hi, cfg.num_bins + 1)[1:-1]
                else:
                    raise ValueError("strategy must be 'quantile' or 'uniform'")

                if cfg.max_thresholds_per_feature and thr.size > cfg.max_thresholds_per_feature:
                    idx = np.linspace(0, thr.size - 1, cfg.max_thresholds_per_feature).round().astype(int)
                    thr = thr[idx]

                self.thresholds_[col] = [float(t) for t in np.unique(thr)]
            else:
                s_cat = pd.Series(df[col], dtype="object").astype("string")
                levels = s_cat.dropna().astype(str).unique().tolist()
                self.categorical_levels_[col] = levels

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame):
        assert self.fitted_, "Call fit() before transform()."
        cfg = self.config
        X = df.copy()
        out = []

        if cfg.keep:
            keep_cols = [c for c in cfg.keep if c in X.columns]
            if keep_cols:
                out.append(X[keep_cols])

        # categorical one-hot
        if cfg.one_hot_categories and self.categorical_levels_:
            for col, levels in self.categorical_levels_.items():
                if col not in X.columns:
                    continue
                s = X[col].astype("object").astype("string")
                dummies = pd.get_dummies(s, prefix=col, sparse=False, dtype=cfg.dtype)
                # ensure all fitted levels exist
                for lv in levels:
                    cname = f"{col}_{lv}"
                    if cname not in dummies.columns:
                        dummies[cname] = 0
                if cfg.na_policy == "separate" and X[col].isna().any():
                    dummies[f"{col}_isnan"] = X[col].isna().astype(cfg.dtype)
                out.append(dummies[[f"{col}_{lv}" for lv in levels]])

        # continuous -> binary thresholds
        for col, thr in self.thresholds_.items():
            if col not in X.columns:
                continue
            s = X[col].to_numpy()
            lo_q, hi_q = cfg.cap_quantiles
            if lo_q > 0 or hi_q < 1:
                finite = np.isfinite(s)
                if finite.any():
                    lo = np.nanquantile(s[finite], lo_q)
                    hi = np.nanquantile(s[finite], hi_q)
                    s = np.clip(s, lo, hi, out=np.array(s, copy=True))

            cols = {}
            for t in thr:
                cname = f"{col}<={t:g}"
                cols[cname] = (s <= t).astype(cfg.dtype)
            block = pd.DataFrame(cols, index=X.index)
            if cfg.na_policy == "separate" and np.isnan(s).any():
                block[f"{col}_isnan"] = np.isnan(s).astype(cfg.dtype)
            out.append(block)

        if not out:
            return pd.DataFrame(index=df.index), []

        X_new = pd.concat(out, axis=1)
        feature_names = X_new.columns.tolist()
        return X_new, feature_names

    def transform_numpy(self, X, feature_names: Optional[List[str]] = None, return_sparse: bool = False):
        from scipy import sparse

        if hasattr(X, "values"):
            feature_vals = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            feature_vals = np.asarray(X)
            if feature_names is None:
                feature_names = list(self.feature_order_)

        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        cfg = self.config
        blocks = []
        names: List[str] = []

        # keep original columns
        if cfg.keep:
            keep_cols = [c for c in cfg.keep if c in name_to_idx]
            if keep_cols:
                K = np.column_stack([feature_vals[:, name_to_idx[c]] for c in keep_cols])
                K = K.astype(cfg.dtype, copy=False)
                blocks.append(K)
                names.extend(keep_cols)

        # one hot cat
        if cfg.one_hot_categories and self.categorical_levels_:
            for col, levels in self.categorical_levels_.items():
                if col not in name_to_idx:
                    continue
                j = name_to_idx[col]
                sv = pd.Series(feature_vals[:, j]).astype("string").to_numpy()  # 统一成 string
                for lv in levels:
                    blocks.append((sv == lv).astype(cfg.dtype)[:, None])
                    names.append(f"{col}_{lv}")
                if cfg.na_policy == "separate":
                    isn = pd.isna(sv)
                    blocks.append(isn.astype(cfg.dtype)[:, None])
                    names.append(f"{col}_isnan")

        # continous values
        for col, thr_list in self.thresholds_.items():
            if col not in name_to_idx or len(thr_list) == 0:
                continue
            j = name_to_idx[col]
            s = feature_vals[:, j]

            lo_q, hi_q = cfg.cap_quantiles
            if lo_q > 0 or hi_q < 1:
                finite = np.isfinite(s) if np.issubdtype(s.dtype, np.floating) else np.ones_like(s, dtype=bool)
                if finite.any():
                    lo = np.nanquantile(s[finite], lo_q)
                    hi = np.nanquantile(s[finite], hi_q)
                    s = np.clip(s, lo, hi, out=np.array(s, copy=True))

            thr = np.asarray(thr_list, dtype=float)[None, :]
            x = s[:, None].astype(float, copy=False)
            block = (x <= thr).astype(cfg.dtype, copy=False)
            blocks.append(block)
            names.extend([f"{col}<={t:g}" for t in thr_list])

            if cfg.na_policy == "separate":
                isn = np.isnan(s) if np.issubdtype(s.dtype, np.floating) else pd.isna(s)
                blocks.append(isn.astype(cfg.dtype)[:, None])
                names.append(f"{col}_isnan")

        Z = np.concatenate(blocks, axis=1) if blocks else np.empty((feature_vals.shape[0], 0), dtype=cfg.dtype)
        if return_sparse:
            Z = sparse.csr_matrix(Z)
        return Z, names
    
    def get_feature_names_out(self, original_feature_names: Optional[List[str]] = None) -> List[str]:
        if original_feature_names is None:
            original_feature_names = list(self.feature_order_)
        names: List[str] = []

        if self.config.keep:
            keep_cols = [c for c in self.config.keep if c in original_feature_names]
            names.extend(keep_cols)

        if self.config.one_hot_categories and self.categorical_levels_:
            for col in original_feature_names:
                levels = self.categorical_levels_.get(col)
                if levels:
                    names.extend([f"{col}_{lv}" for lv in levels])
                    if self.config.na_policy == "separate":
                        names.append(f"{col}_isnan")

        for col in original_feature_names:
            thr_list = self.thresholds_.get(col, [])
            if thr_list:
                names.extend([f"{col}<={t:g}" for t in thr_list])
                if self.config.na_policy == "separate":
                    names.append(f"{col}_isnan")

        return names
    
    def get_feature_columns_for(self, feature: str, original_feature_names: Optional[List[str]] = None) -> List[int]:
        """
        Given a feature name, return its column index in the output of transform_numpy
        """
        names = self.get_feature_names_out(original_feature_names)

        # For "keep"
        if feature in names:
            return [names.index(feature)]

        # For one-hot features
        prefix = f"{feature}_"
        cat_idxs = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if cat_idxs:
            # Finding positive classes
            pos = {"1", "true", "True", "yes", "Yes", "success", "t", "y", "positive", "pos"}
            neg = {"0", "false", "False", "no", "No", "failure", "f", "n", "negative", "neg"}

            for i in cat_idxs:
                suf = names[i][len(prefix):].lower()
                if suf in pos:
                    return [i]

            if len(cat_idxs) == 2:
                s0 = names[cat_idxs[0]][len(prefix):].lower()
                s1 = names[cat_idxs[1]][len(prefix):].lower()
                if s0 in neg and s1 not in neg:
                    return [cat_idxs[1]]
                if s1 in neg and s0 not in neg:
                    return [cat_idxs[0]]

            # If failed, roll back to the first column
            levels = self.categorical_levels_.get(feature, [])
            if levels:
                cand = f"{feature}_{levels[0]}"
                if cand in names:
                    return [names.index(cand)]

            return [cat_idxs[0]]

        # For continuous features
        thr_prefix = f"{feature}<="
        thr_idxs = [i for i, n in enumerate(names) if n.startswith(thr_prefix)]
        isn = f"{feature}_isnan"
        if isn in names:
            thr_idxs.append(names.index(isn))
        return thr_idxs
    
    def to_json(self, path: str):
        payload = {
            "config": self.config.__dict__,
            "thresholds_": self.thresholds_,
            "categorical_levels_": self.categorical_levels_,
            "continuous_features_": self.continuous_features_,
            "feature_order_": self.feature_order_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = BinningConfig(**payload["config"])
        obj = cls(cfg)
        obj.thresholds_ = {k: list(v) for k, v in payload["thresholds_"].items()}
        obj.categorical_levels_ = {k: list(v) for k, v in payload["categorical_levels_"].items()}
        obj.continuous_features_ = list(payload["continuous_features_"])
        obj.feature_order_ = list(payload.get("feature_order_", []))
        obj.fitted_ = True
        return obj
