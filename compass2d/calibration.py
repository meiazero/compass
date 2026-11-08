"""Conformal quantiles over groups (Dunn et al., 2022), optionally under covariate shift (Tibshirani
et al., 2019)."""

from collections.abc import Callable

import lightgbm
import numpy as np
import sklearn.model_selection

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


def group_quantile(scores: np.ndarray, groups: np.ndarray, alpha: float) -> float:
    """Quantile (1 - alpha) of the pooled group CDFs.

    Dunn et al. define CDF pooling without a finite-sample correction and report undercoverage at
    k <= 35 groups.
    """
    return weighted_quantile(scores, group_weights(groups), 1 - alpha)


def weighted_quantiles(
    scores: np.ndarray, weights: np.ndarray, test_weights: np.ndarray, alpha: float
) -> np.ndarray:
    """Per test point, the (1 - alpha) quantile of Tibshirani et al. (2019), Corollary 1.

    Calibration scores carry w(X_i) and the test point carries w(x) at +inf, so the quantile is
    inf wherever the calibration has too little mass near x. Rescaling every weight changes
    nothing, so an odds ratio without the class prior is enough.
    """
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] + test_weights
    index = np.searchsorted(cumulative, (1 - alpha) * total - TOLERANCE * total)
    inside = index < len(scores)
    return np.where(inside, scores[order][np.where(inside, index, 0)], np.inf)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's (sum w)^2 / sum w^2: how many equally weighted scores the weights are worth."""
    return float(weights.sum() ** 2 / (weights**2).sum())


def group_weighted_quantiles(
    scores: np.ndarray,
    groups: np.ndarray,
    calibration_ratio: np.ndarray,
    test_ratio: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """weighted_quantiles with groups as the units: a calibration group weighs the mean density
    ratio of its cells, spread evenly over them as in CDF pooling, and the test point weighs its
    own ratio as a new group of one. The combination has no established guarantee."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return weighted_quantiles(scores, calibration_ratio / counts[inverse], test_ratio, alpha)


def group_masses(groups: np.ndarray, calibration_ratio: np.ndarray) -> np.ndarray:
    """Each group's weight in group_weighted_quantiles: the mean density ratio of its cells."""
    _, inverse = np.unique(groups, return_inverse=True)
    return np.bincount(inverse, calibration_ratio) / np.bincount(inverse)


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
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    calibration_groups: np.ndarray,
    test_groups: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Odds p_test / p_calibration at every calibration and test point, from a LightGBM domain
    classifier cross-fitted by group, so no point is scored by a classifier that saw its group."""
    features = np.vstack([calibration_features, test_features])
    labels = np.concatenate([np.zeros(len(calibration_features)), np.ones(len(test_features))])
    groups = np.concatenate(
        [[f"calibration {g}" for g in calibration_groups], [f"test {g}" for g in test_groups]]
    )
    probability = np.empty(len(labels))
    folds = sklearn.model_selection.GroupKFold(n_splits=n_folds)
    for train, held_out in folds.split(features, labels, groups):
        classifier = lightgbm.LGBMClassifier(
            n_estimators=100, n_jobs=n_jobs, verbosity=-1, random_state=seed
        ).fit(features[train], labels[train])
        probability[held_out] = classifier.predict_proba(features[held_out])[:, 1]
    odds = probability / (1 - probability + 1e-9)
    return odds[: len(calibration_features)], odds[len(calibration_features) :]
