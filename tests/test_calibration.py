"""Weighted conformal quantiles under a known covariate shift."""

import numpy as np

import compass2d.calibration

ALPHA = 0.1


def shifted_problem(
    rng: np.random.Generator, n_calibration: int, n_test: int
) -> tuple[np.ndarray, ...]:
    """x ~ N(0, 1) in calibration and N(1.5, 1) in test; noise grows with |x|, so the shift
    moves test points to where calibration scores are rare and small."""
    x_calibration = rng.normal(0.0, 1.0, n_calibration)
    x_test = rng.normal(1.5, 1.0, n_test)
    scores = np.abs(rng.normal(0, 0.2 + np.abs(x_calibration)))
    test_scores = np.abs(rng.normal(0, 0.2 + np.abs(x_test)))
    # N(1.5, 1) / N(0, 1) density ratio, known exactly here.
    ratio = np.exp(1.5 * x_calibration - 1.125)
    test_ratio = np.exp(1.5 * x_test - 1.125)
    return scores, ratio, test_scores, test_ratio


def test_weighted_quantiles_cover_under_covariate_shift() -> None:
    """Tibshirani et al. (2019), Corollary 1, averaged over calibration draws as it is stated."""
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(300):
        scores, ratio, test_scores, test_ratio = shifted_problem(rng, 100, 50)
        q = compass2d.calibration.weighted_quantiles(scores, ratio, test_ratio, ALPHA)
        covered.append(test_scores <= q)
    # 15,000 test points: about three binomial SDs below the target.
    assert np.mean(covered) >= 1 - ALPHA - 0.008


def test_groups_of_one_reduce_to_cell_weights() -> None:
    scores, ratio, _, test_ratio = shifted_problem(np.random.default_rng(1), 200, 20)
    by_group = compass2d.calibration.group_weighted_quantiles(
        scores, np.arange(len(scores)), ratio, test_ratio, ALPHA
    )
    by_cell = compass2d.calibration.weighted_quantiles(scores, ratio, test_ratio, ALPHA)
    assert np.array_equal(by_group, by_cell)
