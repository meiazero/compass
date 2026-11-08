"""Conformal quantiles over groups: CDF pooling and repeated subsampling (Dunn et al., 2022)."""

from collections.abc import Callable

import lightgbm
import numpy as np

# p-values are means of rationals, so an exact alpha must not fall to rounding.
TOLERANCE = 1e-12


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


def subsampled_quantile(
    scores: np.ndarray, groups: np.ndarray, alpha: float, n_subsamples: int, seed: int
) -> float:
    """Largest score whose p-value, averaged over subsamples of one score per group, is >= alpha.

    Repeated subsampling, Dunn et al. §4.3: coverage of a new group is at least 1 - 2 alpha for any
    number of groups (Theorem 10), and near 1 - alpha in their simulations.
    """
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    k = len(counts)
    # Above every drawn score the p-value is 1 / (k + 1), so with too few groups nothing bounds it.
    if 1 / (k + 1) >= alpha - TOLERANCE:
        raise ValueError(f"{k} groups bound no interval at alpha {alpha}; need k > 1/alpha - 1")
    by_group = scores[np.argsort(inverse, kind="stable")]
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    drawn = np.sort(by_group[starts + (rng.random((n_subsamples, k)) * counts).astype(int)], axis=1)
    candidates = np.unique(drawn)
    at_least = k - np.stack([np.searchsorted(row, candidates, side="left") for row in drawn])
    p_values = ((1 + at_least) / (k + 1)).mean(axis=0)
    return float(candidates[p_values >= alpha - TOLERANCE].max())


def leave_one_group_out_quantiles(
    scores: np.ndarray, groups: np.ndarray, rule: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Per-sample quantile, by rule(scores, groups), calibrated on every group but the sample's."""
    quantiles = np.empty(len(scores))
    for group in np.unique(groups):
        held_out = groups == group
        quantiles[held_out] = rule(scores[~held_out], groups[~held_out])
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
