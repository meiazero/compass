"""COMPASS-J perturbation directions: Jacobian PCA (Cheung et al., 2026) and a linear SVM normal."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.svm
import torch


@dataclass
class Directions:
    pca_components: np.ndarray  # (L, C)
    pca_mean: np.ndarray  # (C,)
    pca_explained_variance: np.ndarray  # (L,)
    svm_w: np.ndarray  # (C,), oriented toward more area

    def save(self, path: Path) -> None:
        with path.open("wb") as file:
            np.savez(file, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Directions":
        with np.load(path) as stored:
            return cls(**{name: stored[name] for name in stored.files})


def _scale(delta: np.ndarray) -> np.ndarray:
    """Divide by the channel std, so one beta_range is meaningful for both directions."""
    std = delta.std(axis=-1, ddof=1, keepdims=True)
    if (std == 0).any():
        raise ValueError("direction is constant across channels; its scale is undefined")
    return delta / std


def jacobian_sum(
    decode: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Per-sample gradient of the metric summed over space, (B, C).

    In eval mode each area depends only on its own sample, so one backward pass over the sum
    yields every per-sample gradient. Requires batch norm in eval mode.
    """
    z = z.detach().requires_grad_(True)
    gradient = torch.autograd.grad(metric(decode(z)).sum(), z)[0]
    return gradient.flatten(2).sum(2).detach()


def sample_pixels(
    z: torch.Tensor, mask: torch.Tensor, n_pixels: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature vectors at uniformly sampled pixels and their labels, from a batch.

    The mask becomes the majority label of each feature pixel, by area downsampling.
    """
    labels = torch.nn.functional.interpolate(mask, size=z.shape[-2:], mode="area") >= 0.5
    features, targets = [], []
    n_total = z.shape[-2] * z.shape[-1]
    for sample_z, sample_labels in zip(z, labels, strict=True):
        chosen = torch.randperm(n_total, generator=generator)[:n_pixels].to(z.device)
        features.append(sample_z.flatten(1).T[chosen])
        targets.append(sample_labels.flatten()[chosen])
    return torch.cat(features), torch.cat(targets)


def fit_pca(jacobians: np.ndarray, n_components: int) -> sklearn.decomposition.PCA:
    return sklearn.decomposition.PCA(n_components=n_components).fit(jacobians)


def fit_svm(
    features: np.ndarray,
    labels: np.ndarray,
    mean_jacobian: np.ndarray,
    c: float,
    class_weight: str | None,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """Return a linear SVM normal in raw feature coordinates, oriented by the mean Jacobian.

    Channels differ in scale by orders of magnitude, so the fit is standardized and the normal
    mapped back: w_raw = w_standardized / sigma describes the same boundary.
    """
    scaler = sklearn.preprocessing.StandardScaler()
    svm = sklearn.svm.LinearSVC(
        C=c, class_weight=class_weight, max_iter=max_iter, random_state=seed, dual="auto"
    )
    svm.fit(scaler.fit_transform(features), labels)
    w = svm.coef_.ravel() / scaler.scale_
    projection = w @ mean_jacobian
    if projection == 0:
        raise ValueError("mean Jacobian is orthogonal to w; cannot orient toward more area")
    return _scale(w if projection > 0 else -w)


def cell_deltas(direction: str, jacobians: np.ndarray, fitted: Directions) -> np.ndarray:
    """Per-cell channel-sized direction, (N, C).

    Upstream's delta_pca adds the PCA mean back through inverse_transform on uncentered Jacobians,
    so PCA stays defined on empty cells whose Jacobian is zero. The paper's V V^T J / ||.||_2 has
    no mean term and would be NaN there.
    """
    if direction == "pca":
        coefficients = jacobians @ fitted.pca_components.T
        return _scale(coefficients @ fitted.pca_components + fitted.pca_mean)
    if direction == "svm":
        return np.broadcast_to(fitted.svm_w, jacobians.shape)
    raise ValueError(f"unknown direction: {direction!r}")


def abs_cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(abs(first @ second) / (np.linalg.norm(first) * np.linalg.norm(second)))
