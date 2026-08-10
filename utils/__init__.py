"""Inference, evaluation, training, stitching, and visualization utilities."""

from .inference import (
    build_model_from_checkpoint,
    load_checkpoint,
    load_pretrained_model,
    reconstruct_patches,
    select_device,
)
from .metrics import (
    average_patch_metrics,
    format_average_metrics,
    format_metrics,
    reconstruction_metrics,
)
from .stitching import AdaptiveImageStitcher, ExperimentalImageStitcher
from .training import train_model
from .visualization import plot_global_stitching, plot_patch_comparison

__all__ = [
    "load_checkpoint",
    "build_model_from_checkpoint",
    "load_pretrained_model",
    "reconstruct_patches",
    "select_device",
    "average_patch_metrics",
    "format_average_metrics",
    "format_metrics",
    "reconstruction_metrics",
    "AdaptiveImageStitcher",
    "ExperimentalImageStitcher",
    "train_model",
    "plot_global_stitching",
    "plot_patch_comparison",
]
