"""Weighted conformal quantiles under a known covariate shift."""

import numpy as np
import pytest

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


def split_conformal(scores: np.ndarray, _groups: np.ndarray) -> float:
    rank = int(np.ceil((1 - ALPHA) * (len(scores) + 1)))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def test_mondrian_covers_each_stratum_where_a_pooled_quantile_does_not() -> None:
    """Stratum 1 is noisier and rarer, so one pooled quantile undercovers it."""
    rng = np.random.default_rng(3)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        strata = (rng.random(n) < 0.2).astype(int)
        return np.abs(rng.normal(0, np.where(strata == 1, 4.0, 1.0))), strata

    scores, strata = draw(4000)
    test_scores, test_strata = draw(20000)
    groups = np.arange(len(scores))
    q = compass2d.calibration.mondrian_quantiles(
        scores, groups, strata, test_strata, split_conformal
    )
    pooled = split_conformal(scores, groups)
    for stratum in (0, 1):
        inside = test_strata == stratum
        assert np.mean(test_scores[inside] <= q[inside]) >= 1 - ALPHA - 0.012
    assert np.mean(test_scores[test_strata == 1] <= pooled) < 1 - ALPHA - 0.1
    absent = compass2d.calibration.mondrian_quantiles(
        scores, groups, strata, np.array([2]), split_conformal
    )
    assert np.isinf(absent).all()


def test_density_ratio_is_one_constant_when_nothing_shifts() -> None:
    """Same features everywhere, groups of unequal size: every fold must agree on the ratio."""
    calibration_groups = np.repeat(np.arange(10), 40)
    test_groups = np.repeat(np.arange(5), [300, 100, 100, 100, 100])
    calibration_ratio, test_ratio = compass2d.calibration.density_ratio(
        np.zeros((len(calibration_groups), 3)),
        np.zeros((len(test_groups), 3)),
        calibration_groups,
        test_groups,
        n_folds=5,
        seed=0,
        n_jobs=1,
    )
    ratio = np.concatenate([calibration_ratio, test_ratio])
    assert np.allclose(ratio, ratio[0])


def test_density_ratio_refuses_a_fold_without_one_population() -> None:
    with pytest.raises(ValueError, match="lacks one population"):
        compass2d.calibration.density_ratio(
            np.zeros((40, 3)),
            np.zeros((10, 3)),
            np.repeat(np.arange(4), 10),
            np.zeros(10, dtype=int),
            n_folds=5,
            seed=0,
            n_jobs=1,
        )
