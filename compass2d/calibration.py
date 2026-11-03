"""Conformal quantiles with equal weight per group (CDF pooling, Dunn et al., 2022)."""

import lightgbm
import numpy as np


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Smallest score whose normalized cumulative weight reaches level (upstream's helper)."""
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, level * cumulative[-1] - 1e-12)
    return float(scores[order][min(index, len(scores) - 1)])


def group_weights(groups: np.ndarray) -> np.ndarray:
    """1 / (K n_k): the weighted ECDF of the scores becomes the mean of the group ECDFs."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse])


def group_quantile(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    density_ratio: np.ndarray | None = None,
) -> float:
    """Quantile (1 - alpha) of the pooled group CDFs, optionally reweighted toward the test region.

    Dunn et al. define CDF pooling without a finite-sample correction and report undercoverage at
    k <= 35 groups. Combining it with a density ratio has no established guarantee.
    """
    weights = group_weights(groups)
    if density_ratio is not None:
        weights = weights * density_ratio
    return weighted_quantile(scores, weights, 1 - alpha)


def leave_one_group_out_quantiles(
    scores: np.ndarray, groups: np.ndarray, alpha: float
) -> np.ndarray:
    """Per-sample quantile calibrated on every group except the sample's own."""
    quantiles = np.empty(len(scores))
    for group in np.unique(groups):
        held_out = groups == group
        quantiles[held_out] = group_quantile(scores[~held_out], groups[~held_out], alpha)
    return quantiles


def density_ratio(
    calibration_features: np.ndarray, test_features: np.ndarray, seed: int, n_jobs: int
) -> np.ndarray:
    """p_test / p_calibration from a LightGBM domain classifier, predicted in-sample as upstream."""
    features = np.vstack([calibration_features, test_features])
    labels = np.concatenate([np.zeros(len(calibration_features)), np.ones(len(test_features))])
    classifier = lightgbm.LGBMClassifier(
        n_estimators=100, n_jobs=n_jobs, verbosity=-1, random_state=seed
    ).fit(features, labels)
    probability = classifier.predict_proba(calibration_features)[:, 1]
    return probability / (1 - probability + 1e-9)
