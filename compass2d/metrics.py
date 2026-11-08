"""Area of a sigmoid binary segmentation, hard and differentiable, in the unit of `pixel_area`."""

import torch

# Upstream's temp_scale value; the paper only says "a sigmoid function".
UPSTREAM_TEMPERATURE = 10_000.0


def hard_area(logits: torch.Tensor, pixel_area: float) -> torch.Tensor:
    """Sigmoid at 0.5 is logit at 0; float64 so the search compares exact pixel multiples."""
    return (logits >= 0).flatten(1).sum(1).to(torch.float64) * pixel_area


def soft_area(logits: torch.Tensor, pixel_area: float, temperature: float) -> torch.Tensor:
    """Sum in float64: float32 rounding would break the monotonicity test on a smooth response."""
    probabilities = torch.sigmoid(temperature * logits).to(torch.float64)
    return probabilities.flatten(1).sum(1) * pixel_area
