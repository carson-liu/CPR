"""Reconstruction metrics shared by patch and global evaluation."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def reconstruction_metrics(target, prediction):
    """Return MSE, MAE, SSIM, and PSNR for two images."""

    if target.shape != prediction.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {target.shape} != {prediction.shape}"
        )
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0
    return {
        "MSE": float(np.mean((prediction - target) ** 2)),
        "MAE": float(np.mean(np.abs(prediction - target))),
        "SSIM": float(
            structural_similarity(target, prediction, data_range=data_range)
        ),
        "PSNR": float(
            peak_signal_noise_ratio(target, prediction, data_range=data_range)
        ),
    }


def format_metrics(metrics, multiline=False):
    """Format a metric dictionary for terminal or figure output."""

    separator = "\n" if multiline else " | "
    return separator.join(
        [
            f"MSE {metrics['MSE']:.4f}   MAE {metrics['MAE']:.4f}",
            f"SSIM {metrics['SSIM']:.3f}   PSNR {metrics['PSNR']:.2f} dB",
        ]
    )


def average_patch_metrics(targets, predictions):
    """Return mean and standard deviation of metrics across all patches."""

    if targets.shape != predictions.shape or targets.ndim != 3:
        raise ValueError("Expected matching patch stacks with shape (N, H, W)")
    per_patch = [
        reconstruction_metrics(target, prediction)
        for target, prediction in zip(targets, predictions)
    ]
    return {
        key: {
            "mean": float(np.mean([metrics[key] for metrics in per_patch])),
            "std": float(np.std([metrics[key] for metrics in per_patch])),
        }
        for key in per_patch[0]
    }


def format_average_metrics(summary):
    """Format mean ± standard deviation patch metrics."""

    return " | ".join(
        [
            f"MSE {summary['MSE']['mean']:.4f} ± {summary['MSE']['std']:.4f}",
            f"MAE {summary['MAE']['mean']:.4f} ± {summary['MAE']['std']:.4f}",
            f"SSIM {summary['SSIM']['mean']:.3f} ± {summary['SSIM']['std']:.3f}",
            f"PSNR {summary['PSNR']['mean']:.2f} ± {summary['PSNR']['std']:.2f} dB",
        ]
    )
