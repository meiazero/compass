"""Splits a segmentation model into encoder f and decoder g at a named module."""

import torch

LOGITS = "logits"


@torch.no_grad()
def encode(model: torch.nn.Module, layer: str, x: torch.Tensor) -> torch.Tensor:
    """Feature map z = f(x) at the named module, or the logits for COMPASS-L."""
    if layer == LOGITS:
        return model(x)
    captured = []
    handle = model.get_submodule(layer).register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        model(x)
    finally:
        handle.remove()
    return captured[0]


def decode(model: torch.nn.Module, layer: str, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Logits g(z); x only supplies the U-Net skip connections below the replaced layer.

    ponytail: reruns the encoder on every call, which is cheap on an A100; caching its output would
    need the library's decoder internals, whose signature changes between releases.
    """
    if layer == LOGITS:
        return z
    handle = model.get_submodule(layer).register_forward_hook(lambda _module, _inputs, _output: z)
    try:
        return model(x)
    finally:
        handle.remove()
