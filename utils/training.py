"""Shared CPR training loop for AD_LTEM and experimental data."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CPRLoss, CPRNet

from .inference import select_device


def _run_epoch(
    model,
    data_loader,
    loss_function,
    device,
    optimizer=None,
    scheduler=None,
):
    training = optimizer is not None
    model.train(training)
    accumulated = {}
    description = "Train" if training else "Validation"

    for diffraction, amplitude, cos_phase, sin_phase in tqdm(
        data_loader, desc=description, leave=False
    ):
        diffraction = diffraction.to(device)
        amplitude = amplitude.to(device)
        cos_phase = cos_phase.to(device)
        sin_phase = sin_phase.to(device)

        with torch.set_grad_enabled(training):
            pred_amp, pred_cos, pred_sin = model(diffraction)
            loss, loss_details = loss_function(
                pred_amp,
                pred_cos,
                pred_sin,
                amplitude,
                cos_phase,
                sin_phase,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        for name, value in loss_details.items():
            accumulated[name] = accumulated.get(name, 0.0) + value

    return {
        name: value / len(data_loader) for name, value in accumulated.items()
    }


def train_model(
    prepared_data,
    dataset_name,
    checkpoint_path,
    epochs=25,
    batch_size=64,
    learning_rate=1e-3,
    num_workers=4,
    device=None,
):
    """Train CPR with the optimizer, scheduler, and loss from the notebooks."""

    device = torch.device(device) if device is not None else select_device()
    train_loader = DataLoader(
        prepared_data.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        prepared_data.validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = CPRNet(max_signal=prepared_data.max_signal).to(device)
    loss_function = CPRLoss(
        amp_weight=1.0,
        phase_weight=1.3,
        consistency_weight=0.1,
        grad_weight=0.12,
        circular_weight=0.6,
        ssim_weight=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterations_per_epoch = math.floor(
        len(prepared_data.train_dataset) / batch_size
    ) + 1
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=learning_rate / 10,
        max_lr=learning_rate,
        step_size_up=6 * iterations_per_epoch,
        cycle_momentum=False,
        mode="triangular2",
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    print(
        f"Dataset={dataset_name} | device={device} | epochs={epochs} | "
        f"batch_size={batch_size} | learning_rate={learning_rate}"
    )
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_losses = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        validation_losses = _run_epoch(
            model, validation_loader, loss_function, device
        )

        if validation_losses["total"] < best_validation_loss:
            best_validation_loss = validation_losses["total"]
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "max_signal": prepared_data.max_signal,
                    "input_size": prepared_data.test_set.diffraction.shape[-1],
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )

        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{epochs} | {elapsed:.1f}s | lr={current_lr:.6f} | "
            f"train={train_losses['total']:.5f} | "
            f"validation={validation_losses['total']:.5f} | "
            f"best={best_validation_loss:.5f}"
        )

    print(f"Best checkpoint: {checkpoint_path}")
    return model
