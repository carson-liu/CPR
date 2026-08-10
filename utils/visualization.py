"""Paper-style patch and stitched-result visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import format_metrics, reconstruction_metrics


def _save_figure(figure, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")


def plot_patch_comparison(
    diffraction,
    amp_true,
    amp_pred,
    phase_true,
    phase_pred,
    indices=(0, 33, 65, 110, 143),
    grid_size=12,
    save_path=None,
):
    """Plot patch comparisons and return their individual metrics."""

    column_titles = [
        "Diffraction (log10)",
        "Amplitude GT",
        "Amplitude CPR",
        "Phase GT",
        "Phase CPR",
    ]
    figure, axes = plt.subplots(
        len(indices), 5, figsize=(13.5, 3 * len(indices)), constrained_layout=True
    )
    patch_metrics = []
    for row, index in enumerate(indices):
        amp_metrics = reconstruction_metrics(amp_true[index], amp_pred[index])
        phase_metrics = reconstruction_metrics(
            phase_true[index], phase_pred[index]
        )
        scan_row, scan_column = divmod(index, grid_size)
        patch_metrics.append(
            {
                "index": index,
                "scan_position": (scan_row, scan_column),
                "amplitude": amp_metrics,
                "phase": phase_metrics,
            }
        )

        images = [
            np.log10(diffraction[index] + 1e-8),
            amp_true[index],
            amp_pred[index],
            phase_true[index],
            phase_pred[index],
        ]
        for column, image in enumerate(images):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(column_titles[column], fontsize=11)

        axes[row, 0].text(
            -0.08,
            0.5,
            f"Patch {index}\n({scan_row}, {scan_column})",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        axes[row, 2].text(
            0.5,
            -0.08,
            format_metrics(amp_metrics, multiline=True),
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        axes[row, 4].text(
            0.5,
            -0.08,
            format_metrics(phase_metrics, multiline=True),
            transform=axes[row, 4].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    figure.suptitle("CPR patch reconstruction", fontsize=15)
    _save_figure(figure, save_path)
    return figure, patch_metrics


def plot_global_stitching(
    amp_true,
    amp_pred,
    phase_true,
    phase_pred,
    amp_metrics,
    phase_metrics,
    save_path=None,
):
    """Plot stitched ground truth and CPR results with global metrics."""

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)

    axes[0, 0].imshow(amp_true)
    axes[0, 0].set_title("Amplitude GT")
    axes[0, 1].imshow(amp_pred)
    axes[0, 1].set_title(
        f"Amplitude CPR\n{format_metrics(amp_metrics, multiline=True)}",
        fontsize=10,
    )
    axes[1, 0].imshow(phase_true)
    axes[1, 0].set_title("Phase GT")
    axes[1, 1].imshow(phase_pred)
    axes[1, 1].set_title(
        f"Phase CPR\n{format_metrics(phase_metrics, multiline=True)}",
        fontsize=10,
    )

    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle("CPR global stitched reconstruction", fontsize=15)
    _save_figure(figure, save_path)
    return figure
