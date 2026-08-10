"""Shared containers for CPR data pipelines."""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ReconstructionTestSet:
    """Diffraction inputs and reconstruction targets for one test region."""

    diffraction: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    grid_size: int


@dataclass(frozen=True)
class PreparedData:
    """Training/validation datasets plus dataset-specific metadata."""

    train_dataset: Dataset
    validation_dataset: Dataset
    test_set: ReconstructionTestSet
    max_signal: float
