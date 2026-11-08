"""Asymmetric COMPASS on synthetic logits whose errors are skewed."""

import math

import torch

import compass2d.compass
import compass2d.metrics

ALPHA = 0.1
BETA_RANGE = 10.0
THRESHOLD = 1e-5


def identity(z: torch.Tensor) -> torch.Tensor:
    return z


def area(logits: torch.Tensor) -> torch.Tensor:
    return compass2d.metrics.hard_area(logits, pixel_area=1.0)


def skewed_cells(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy logits and true areas; a per-cell bias overestimates slightly and often, and
    underestimates by a lot but rarely, so both one-sided scores carry information."""
    truth = torch.randn(n, 1, 16, 16, generator=generator) - 0.3
    bias = 0.15 - 0.3 * torch.empty(n, 1, 1, 1).exponential_(generator=generator)
    logits = truth + 0.8 * torch.randn(n, 1, 16, 16, generator=generator) + bias
    return logits, area(truth)


def test_asymmetric_interval_reaches_target_coverage() -> None:
    generator = torch.Generator().manual_seed(0)
    z, target = skewed_cells(2000, generator)
    ones = torch.ones(len(z), 1, 1, 1)
    (lower, lower_saturated), (upper, upper_saturated) = (
        compass2d.compass.asymmetric_binary_search_scores(
            identity, area, z, ones, target, BETA_RANGE, THRESHOLD
        )
    )
    assert not (lower_saturated | upper_saturated).any()
    rank = math.ceil((1 - ALPHA / 2) * (len(z) + 1)) - 1
    beta_lower, beta_upper = lower.sort().values[rank], upper.sort().values[rank]

    z_test, target_test = skewed_cells(2000, generator)
    shift = torch.ones(len(z_test))
    low = compass2d.compass.shifted(identity, area, z_test, ones, -beta_lower * shift)
    high = compass2d.compass.shifted(identity, area, z_test, ones, beta_upper * shift)
    coverage = ((low <= target_test) & (target_test <= high)).double().mean().item()
    # Theorem 3 bounds coverage below by 1 - alpha; 0.02 is about three binomial SDs at n = 2000.
    assert coverage >= 1 - ALPHA - 0.02


def test_direction_toward_less_area_saturates_the_upper_side() -> None:
    z, target = skewed_cells(200, torch.Generator().manual_seed(1))
    (_, _), (_, upper_saturated) = compass2d.compass.asymmetric_binary_search_scores(
        identity, area, z, -torch.ones(len(z), 1, 1, 1), target, BETA_RANGE, THRESHOLD
    )
    assert upper_saturated[target > area(z)].all()
