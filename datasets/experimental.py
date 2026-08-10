"""Experimental X-ray dataset using the original measurement values."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from .common import PreparedData, ReconstructionTestSet


EXPERIMENTAL_DIFFRACTION_FILE = "20191008_39_diff_reduced.npz"
EXPERIMENTAL_OBJECT_FILE = "20191008_39_amp_pha_10nm_full.npy"


def _check_paths(root):
    root = Path(root)
    diffraction_path = root / EXPERIMENTAL_DIFFRACTION_FILE
    object_path = root / EXPERIMENTAL_OBJECT_FILE
    missing = [
        path.name for path in (diffraction_path, object_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing experimental data files: {missing}")
    return diffraction_path, object_path


def _extract_experimental_test_set(diffraction, complex_object, grid_size):
    test_start = complex_object.shape[0] - grid_size
    patch_shape = diffraction.shape[-2:]
    test_diffraction = diffraction[test_start:, :grid_size].reshape(
        -1, *patch_shape
    )
    test_object = complex_object[test_start:, :grid_size]
    test_amplitude = np.abs(test_object).reshape(-1, *patch_shape)
    test_phase = np.angle(test_object).reshape(-1, *patch_shape)
    return ReconstructionTestSet(
        diffraction=test_diffraction.astype(np.float32, copy=False),
        amplitude=test_amplitude.astype(np.float32, copy=False),
        phase=test_phase.astype(np.float32, copy=False),
        grid_size=grid_size,
    )


def prepare_experimental_data(root=".", grid_size=60, validation_size=805):
    """Prepare the experimental training, validation, and test splits."""

    diffraction_path, object_path = _check_paths(root)
    with np.load(diffraction_path) as archive:
        diffraction = archive["arr_0"]
    complex_object = np.load(object_path)
    amplitude = np.abs(complex_object)
    phase = np.angle(complex_object)

    scan_shape = amplitude.shape[:2]
    patch_shape = amplitude.shape[-2:]
    test_start = scan_shape[0] - grid_size
    mask = np.ones(scan_shape, dtype=bool)
    mask[test_start:, :grid_size] = False

    x_train = diffraction[mask].reshape(-1, 1, *patch_shape)
    amp_train = amplitude[mask].reshape(-1, 1, *patch_shape)
    phase_train = phase[mask].reshape(-1, 1, *patch_shape)
    permutation = np.random.RandomState(0).permutation(len(x_train))
    x_train = x_train[permutation]
    amp_train = amp_train[permutation]
    phase_train = phase_train[permutation]

    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(amp_train, dtype=torch.float32),
        torch.as_tensor(np.cos(phase_train), dtype=torch.float32),
        torch.as_tensor(np.sin(phase_train), dtype=torch.float32),
    )
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError(
            f"Invalid experimental validation size: {validation_size}"
        )
    train_dataset, validation_dataset = random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )

    return PreparedData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_set=_extract_experimental_test_set(
            diffraction, complex_object, grid_size
        ),
        max_signal=float(np.percentile(diffraction, 99.5)),
    )


def load_experimental_test_set(root=".", grid_size=60):
    """Load the experimental test region without data normalization."""

    diffraction_path, object_path = _check_paths(root)
    with np.load(diffraction_path) as archive:
        diffraction = archive["arr_0"]
    complex_object = np.load(object_path, mmap_mode="r")
    return _extract_experimental_test_set(
        diffraction, complex_object, grid_size
    )
