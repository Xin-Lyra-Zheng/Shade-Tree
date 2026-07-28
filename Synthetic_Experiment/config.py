"""Prespecified experiment grids."""

from __future__ import annotations

from itertools import product


MODELS = (
    "logistic",
    "fsg",
    "ebm_main",
    "ebm_interact",
    "adtree",
    "shadetree",
)


PARAM_GRIDS = {
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "fsg": {
        "max_support_size": [5, 10, 20, 40],
        "n_bins": [8, 16],
    },
    "ebm_main": {
        "max_bins": [32, 64],
        "learning_rate": [0.03, 0.05],
        "max_rounds": [2000],
    },
    "ebm_interact": {
        "interactions": [5, 10, 20],
        "max_bins": [32, 64],
        "learning_rate": [0.03, 0.05],
        "max_rounds": [2000],
    },
    "adtree": {
        "max_split_nodes": [3, 5, 10, 20, 40],
        "max_depth": [6, 10],
        "min_samples_leaf": [20],
        "max_iters": [500],
        "num_thresholds": [20],
        "complexity_penalty": [0.0, 0.002],
    },
    "shadetree": {
        "max_split_nodes": [3, 5, 10, 20, 40],
        "max_depth": [6, 10],
        "min_samples_leaf": [20],
        "max_iters": [500],
        "num_thresholds": [20],
        "n_knots_step": [4, 8],
        "complexity_penalty": [0.0, 0.002],
    },
}


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def grid_for(model: str, profile: str) -> list[dict]:
    candidates = expand_grid(PARAM_GRIDS[model])
    if profile == "smoke":
        item = dict(candidates[0])
        if model in {"adtree", "shadetree"}:
            item.update(max_iters=15, max_split_nodes=3, num_thresholds=5)
        if model == "ebm_interact":
            item["interactions"] = 5
        return [item]
    return candidates


PROFILES = {
    "smoke": {
        "datasets": ["additive", "xor", "gated"],
        "sample_sizes": [500],
        "signal_strengths": [1.0],
        "seeds": [0],
        "mc_samples": 2000,
    },
    "full": {
        "datasets": ["additive", "xor", "logic", "gated", "region"],
        "sample_sizes": [500, 1000, 2000, 5000, 10000],
        "signal_strengths": [0.5, 1.0, 2.0],
        "seeds": list(range(10)),
        "mc_samples": 50000,
    },
}
