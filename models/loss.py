"""Circular phase loss used in the paper experiments."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPRLoss(nn.Module):
    def __init__(
        self,
        amp_weight=1.5,
        phase_weight=1.2,
        consistency_weight=0.1,
        grad_weight=0.1,
        circular_weight=0.5,
        ssim_weight=0.3,
        use_mae=False,
    ):
        super().__init__()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        self.consistency_weight = consistency_weight
        self.grad_weight = grad_weight
        self.circular_weight = circular_weight
        self.ssim_weight = ssim_weight
        self.use_mae = use_mae

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.base_loss = self.l1_loss if use_mae else self.mse_loss

    def ssim_loss(self, pred, target, window_size=11):
        c1 = 0.01**2
        c2 = 0.03**2
        sigma = 1.5
        gauss = torch.Tensor(
            [
                np.exp(
                    -(x - window_size // 2) ** 2 / (2 * sigma**2)
                )
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(pred.device)
        channels = pred.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1)

        mu_pred = F.conv2d(
            pred, kernel, padding=window_size // 2, groups=channels
        )
        mu_target = F.conv2d(
            target, kernel, padding=window_size // 2, groups=channels
        )
        mu_pred_sq = mu_pred**2
        mu_target_sq = mu_target**2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = (
            F.conv2d(
                pred * pred,
                kernel,
                padding=window_size // 2,
                groups=channels,
            )
            - mu_pred_sq
        )
        sigma_target_sq = (
            F.conv2d(
                target * target,
                kernel,
                padding=window_size // 2,
                groups=channels,
            )
            - mu_target_sq
        )
        sigma_pred_target = (
            F.conv2d(
                pred * target,
                kernel,
                padding=window_size // 2,
                groups=channels,
            )
            - mu_pred_target
        )

        numerator = (2 * mu_pred_target + c1) * (
            2 * sigma_pred_target + c2
        )
        denominator = (mu_pred_sq + mu_target_sq + c1) * (
            sigma_pred_sq + sigma_target_sq + c2
        )
        return 1.0 - torch.mean(numerator / denominator)

    def phase_consistency_loss(self, cos_pred, sin_pred):
        magnitude = cos_pred**2 + sin_pred**2
        return torch.mean((magnitude - 1.0) ** 2)

    def circular_phase_loss(
        self, cos_pred, sin_pred, cos_true, sin_true
    ):
        cos_diff = cos_pred * cos_true + sin_pred * sin_true
        return torch.mean(1.0 - cos_diff)

    def gradient_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        loss_dx = self.l1_loss(pred_dx, target_dx)
        loss_dy = self.l1_loss(pred_dy, target_dy)
        return (loss_dx + loss_dy) / 2.0

    def forward(
        self,
        pred_amp,
        pred_cos,
        pred_sin,
        true_amp,
        true_cos,
        true_sin,
    ):
        amp_l1 = self.base_loss(pred_amp, true_amp)
        amp_grad = self.gradient_loss(pred_amp, true_amp)
        if self.ssim_weight > 0:
            amp_ssim = self.ssim_loss(pred_amp, true_amp)
            loss_amplitude = (
                amp_l1
                + self.grad_weight * amp_grad
                + self.ssim_weight * amp_ssim
            )
        else:
            amp_ssim = torch.tensor(0.0).to(pred_amp.device)
            loss_amplitude = amp_l1 + self.grad_weight * amp_grad

        cos_l1 = self.base_loss(pred_cos, true_cos)
        sin_l1 = self.base_loss(pred_sin, true_sin)
        circular_loss = self.circular_phase_loss(
            pred_cos, pred_sin, true_cos, true_sin
        )
        cos_grad = self.gradient_loss(pred_cos, true_cos)
        sin_grad = self.gradient_loss(pred_sin, true_sin)

        if self.ssim_weight > 0:
            cos_ssim = self.ssim_loss(pred_cos, true_cos)
            sin_ssim = self.ssim_loss(pred_sin, true_sin)
            phase_ssim = (cos_ssim + sin_ssim) / 2.0
        else:
            phase_ssim = torch.tensor(0.0).to(pred_cos.device)

        loss_phase = (
            cos_l1
            + sin_l1
            + self.circular_weight * circular_loss
            + self.grad_weight * (cos_grad + sin_grad)
        )
        if self.ssim_weight > 0:
            loss_phase = loss_phase + self.ssim_weight * phase_ssim

        loss_consistency = self.phase_consistency_loss(pred_cos, pred_sin)
        total_loss = (
            self.amp_weight * loss_amplitude
            + self.phase_weight * loss_phase
            + self.consistency_weight * loss_consistency
        )
        loss_dict = {
            "total": total_loss.item(),
            "amplitude": loss_amplitude.item(),
            "amp_l1": amp_l1.item(),
            "amp_grad": amp_grad.item(),
            "amp_ssim": amp_ssim.item() if self.ssim_weight > 0 else 0.0,
            "phase": loss_phase.item(),
            "cos_l1": cos_l1.item(),
            "sin_l1": sin_l1.item(),
            "circular": circular_loss.item(),
            "cos_grad": cos_grad.item(),
            "sin_grad": sin_grad.item(),
            "phase_ssim": (
                phase_ssim.item() if self.ssim_weight > 0 else 0.0
            ),
            "consistency": loss_consistency.item(),
        }
        return total_loss, loss_dict


# Compatibility name used in the original experiment notebooks.
ImprovedPhaseLoss = CPRLoss
