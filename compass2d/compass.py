"""Symmetric and asymmetric endpoint COMPASS, adapted from Cheung et al. (2026), Appendix C.

COMPASS-L and COMPASS-J share this code: L takes z = logits, decode = identity, delta = ones. A
channel-sized delta of shape (B, C, 1, 1) broadcasts over space, so every pixel of a channel moves
by the same amount, as upstream does with `delta.view(-1, 1, 1)`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

Decode = Callable[[torch.Tensor], torch.Tensor]
Metric = Callable[[torch.Tensor], torch.Tensor]


def channel_delta(delta: torch.Tensor) -> torch.Tensor:
    """(B, C) direction to the broadcastable (B, C, 1, 1)."""
    return delta.view(delta.shape[0], -1, 1, 1)


def shifted(
    decode: Decode, metric: Metric, z: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Metric at z + beta * delta, with one signed beta per sample."""
    return metric(decode(z + beta.to(z).view(-1, 1, 1, 1) * delta))


def endpoints(
    decode: Decode, metric: Metric, z: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval [min, max] of the metric at z -+ beta * delta, with one beta per sample."""
    down, up = shifted(decode, metric, z, delta, -beta), shifted(decode, metric, z, delta, beta)
    return torch.minimum(down, up), torch.maximum(down, up)


def _bisect(
    covers: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    beta_range: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal beta for which covers(beta) holds, per sample, and whether it hit beta_range.

    Like upstream, a sample never covered keeps beta_range as its score. The flag is new: a
    saturated score understates how far the target really is, which invalidates width comparisons.
    """
    low = torch.zeros_like(target, dtype=torch.float64)
    high = torch.full_like(target, beta_range, dtype=torch.float64)
    for _ in range(math.ceil(math.log2(beta_range / threshold))):
        middle = (low + high) / 2
        covered = covers(middle)
        high = torch.where(covered, middle, high)
        low = torch.where(covered, low, middle)
    return high, high >= beta_range


@torch.no_grad()
def binary_search_scores(
    decode: Decode,
    metric: Metric,
    z: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    beta_range: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal beta whose interval covers the target, and whether the search hit beta_range."""

    def covers(beta: torch.Tensor) -> torch.Tensor:
        lower, upper = endpoints(decode, metric, z, delta, beta)
        return (lower <= target) & (target <= upper)

    return _bisect(covers, target, beta_range, threshold)


@torch.no_grad()
def asymmetric_binary_search_scores(
    decode: Decode,
    metric: Metric,
    z: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    beta_range: float,
    threshold: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Algorithm 6: (score, saturated) for the lower bound, then for the upper bound.

    Calibrate each side at 1 - alpha / 2, or both at 1 - alpha through joint_score. The endpoints
    stand in for Theorem 3's one-sided envelopes only when the metric ascends along delta, which
    verify_monotonicity checks.
    """
    lower = _bisect(
        lambda beta: shifted(decode, metric, z, delta, -beta) <= target,
        target,
        beta_range,
        threshold,
    )
    upper = _bisect(
        lambda beta: shifted(decode, metric, z, delta, beta) >= target,
        target,
        beta_range,
        threshold,
    )
    return lower, upper


def joint_score(
    lower: tuple[torch.Tensor, torch.Tensor],
    upper: tuple[torch.Tensor, torch.Tensor],
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One score for both sides, max(beta_lo, beta_hi / ratio), and whether either side saturated.

    The score is <= q exactly when [shifted(-q), shifted(+q * ratio)] covers the target, so one
    quantile at 1 - alpha replaces Theorem 3's two at 1 - alpha / 2, which repeated subsampling
    refuses below 2 / alpha groups. The ratio must come from cells outside the calibration set.
    The upper side is censored at beta_range / ratio, so a quantile reaching
    min(beta_range, beta_range / ratio) supports no finite interval.
    """
    if not ratio > 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    (beta_lower, lower_saturated), (beta_upper, upper_saturated) = lower, upper
    return torch.maximum(beta_lower, beta_upper / ratio), lower_saturated | upper_saturated


@dataclass
class Monotonicity:
    ascending: torch.Tensor
    any_direction: torch.Tensor
    flat: torch.Tensor
    gain: torch.Tensor


@torch.no_grad()
def verify_monotonicity(
    decode: Decode,
    metric: Metric,
    z: torch.Tensor,
    delta: torch.Tensor,
    beta_max: float,
    n_steps: int,
) -> Monotonicity:
    """Algorithm 2, per sample. Absent from upstream's code, which only plots a sweep.

    `flat` is kept apart because a constant response passes the ascending test trivially, which is
    common on cells with no mining and no prediction.
    """
    curve = torch.stack(
        [metric(decode(z + beta * delta)) for beta in torch.linspace(-beta_max, beta_max, n_steps)]
    )
    steps = curve[1:] - curve[:-1]
    ascending = (steps >= 0).all(0)
    return Monotonicity(
        ascending=ascending,
        any_direction=ascending | (steps <= 0).all(0),
        flat=(steps == 0).all(0),
        gain=curve[-1] - curve[0],
    )
