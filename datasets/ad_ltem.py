"""AD_LTEM-derived simulated dataset used by the CPR experiments."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from .common import PreparedData, ReconstructionTestSet


AD_LTEM_FILES = {
    "diffraction": "75_diff_n.npy",
    "amplitude": "75_amp_n.npy",
    "phase": "75_ph_n.npy",
}


def _load_arrays(root):
    root = Path(root)
    paths = {name: root / filename for name, filename in AD_LTEM_FILES.items()}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AD_LTEM data files: {missing}")
    arrays = {name: np.load(path) for name, path in paths.items()}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("AD_LTEM diffraction, amplitude, and phase shapes differ")
    return arrays


def _extract_test_set(arrays, grid_size):
    patch_shape = arrays["diffraction"].shape[-2:]

    def extract(array):
        return (
            array[-grid_size:, -grid_size:]
            .reshape(-1, *patch_shape)
            .astype(np.float32, copy=False)
        )

    return ReconstructionTestSet(
        diffraction=extract(arrays["diffraction"]),
        amplitude=extract(arrays["amplitude"]),
        phase=extract(arrays["phase"]),
        grid_size=grid_size,
    )


def load_ad_ltem_test_set(root=".", grid_size=12):
    """Load the lower-right AD_LTEM test region used in the paper."""

    return _extract_test_set(_load_arrays(root), grid_size)


def prepare_ad_ltem_data(root=".", grid_size=12, validation_size=100):
    """Prepare the AD_LTEM training, validation, and test splits."""

    arrays = _load_arrays(root)
    scan_shape = arrays["amplitude"].shape[:2]
    patch_shape = arrays["amplitude"].shape[-2:]
    test_start = scan_shape[0] - grid_size
    mask = np.ones(scan_shape, dtype=bool)
    mask[test_start:, test_start:] = False

    diffraction = arrays["diffraction"][mask].reshape(-1, 1, *patch_shape)
    amplitude = arrays["amplitude"][mask].reshape(-1, 1, *patch_shape)
    phase = arrays["phase"][mask].reshape(-1, 1, *patch_shape)

    permutation = np.random.RandomState(0).permutation(len(diffraction))
    diffraction = diffraction[permutation]
    amplitude = amplitude[permutation]
    phase = phase[permutation]

    dataset = TensorDataset(
        torch.as_tensor(diffraction, dtype=torch.float32),
        torch.as_tensor(amplitude, dtype=torch.float32),
        torch.as_tensor(np.cos(phase), dtype=torch.float32),
        torch.as_tensor(np.sin(phase), dtype=torch.float32),
    )
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError(f"Invalid AD_LTEM validation size: {validation_size}")
    train_dataset, validation_dataset = random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )

    return PreparedData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_set=_extract_test_set(arrays, grid_size),
        max_signal=float(np.percentile(arrays["diffraction"], 99.5)),
    )
