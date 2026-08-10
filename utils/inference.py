"""Checkpoint loading and batched CPR inference."""

from pathlib import Path

import numpy as np
import torch

from models import CPRNet


def select_device():
    """Select CUDA, Apple MPS, or CPU in that order."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path):
    """Safely load a CPR state-dict checkpoint on CPU."""

    return torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )


def build_model_from_checkpoint(checkpoint, device=None):
    """Build CPRNet from an already loaded checkpoint dictionary."""

    device = torch.device(device) if device is not None else select_device()
    model = CPRNet(max_signal=checkpoint["max_signal"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_pretrained_model(checkpoint_path, device=None):
    """Load the public state-dict checkpoint into CPRNet."""

    checkpoint = load_checkpoint(checkpoint_path)
    return build_model_from_checkpoint(checkpoint, device=device)


def reconstruct_patches(model, diffraction, batch_size=64, device=None):
    """Reconstruct amplitude and phase for a stack of diffraction patches."""

    if diffraction.ndim != 3:
        raise ValueError("Expected diffraction with shape (N, height, width)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    amplitude_chunks = []
    phase_chunks = []

    with torch.inference_mode():
        for start in range(0, len(diffraction), batch_size):
            batch = torch.from_numpy(
                diffraction[start : start + batch_size]
            ).unsqueeze(1)
            batch = batch.to(device)
            pred_amp, pred_cos, pred_sin = model(batch)
            pred_phase = torch.atan2(pred_sin, pred_cos)
            amplitude_chunks.append(pred_amp.squeeze(1).cpu().numpy())
            phase_chunks.append(pred_phase.squeeze(1).cpu().numpy())

    return np.concatenate(amplitude_chunks), np.concatenate(phase_chunks)
