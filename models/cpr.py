"""CPR network used in the paper experiments."""

import numpy as np
import torch
import torch.nn as nn


class SaturationAwareDualGainScaling(nn.Module):
    """Saturation-Aware Dual-Gain Scaling (SADGS)."""

    def __init__(
        self,
        max_signal,
        gain_levels=np.array([0.001, 4]),
        saturation_threshold=4095,
        headroom_factor=0.85,
    ):
        super().__init__()
        optimal_scale = (
            saturation_threshold
            * headroom_factor
            / (2 ** gain_levels[0] * max_signal)
        )
        self.gain_factors = [
            optimal_scale * (2**level) for level in gain_levels
        ]
        self.gain_1, self.gain_2 = self.gain_factors
        self.saturation_threshold = saturation_threshold

    def forward(self, x):
        branch_1 = torch.clamp(
            self.gain_1 * x, 0, self.saturation_threshold
        )
        branch_2 = torch.clamp(
            self.gain_2 * x, 0, self.saturation_threshold
        )
        return branch_1, branch_2


class CircularCoordinateDecoder(nn.Module):
    """Decoder for amplitude, cosine, or sine with one skip connection."""

    def __init__(self, in_channels, out_channels=1, activation="sigmoid"):
        super().__init__()
        half_channels = in_channels // 2

        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.decoder2 = nn.Sequential(
            nn.Conv2d(
                in_channels + half_channels,
                half_channels,
                3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                half_channels, half_channels, 3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.decoder3 = nn.Sequential(
            nn.Conv2d(
                half_channels, half_channels, 3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                half_channels, half_channels, 3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.final = nn.Conv2d(
            half_channels, out_channels, 3, stride=1, padding=1
        )

        activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        self.activation = activations.get(activation, nn.Identity)()

    def forward(self, x, skip):
        x = self.up1(self.decoder1(x))
        x = torch.cat([x, skip], dim=1)
        x = self.up2(self.decoder2(x))
        x = self.up3(self.decoder3(x))
        return self.activation(self.final(x))


class CPRNet(nn.Module):
    """Circular Phase Representation network."""

    def __init__(self, max_signal, nconv=32):
        super().__init__()
        self.range_expander = SaturationAwareDualGainScaling(max_signal)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, nconv, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 4, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, nconv, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 4, 5, stride=2, padding=2),
            nn.ReLU(),
        )

        self.projection = nn.Conv2d(
            nconv * 4 * 2, nconv * 4, 1, bias=False
        )
        self.decoder1 = CircularCoordinateDecoder(
            nconv * 4, 1, "sigmoid"
        )
        self.decoder2 = CircularCoordinateDecoder(nconv * 4, 1, "tanh")
        self.decoder3 = CircularCoordinateDecoder(nconv * 4, 1, "tanh")
        self.skip_conv = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv * 2, 3, stride=2, padding=1),
        )

    def forward(self, x):
        x1, x2 = self.range_expander(x)
        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        skip = self.skip_conv(x)

        x = self.projection(torch.cat([enc1, enc2], dim=1))
        amp = self.decoder1(x, skip)
        cos_ph = self.decoder2(x, skip)
        sin_ph = self.decoder3(x, skip)

        magnitude = torch.sqrt(cos_ph**2 + sin_ph**2 + 1e-8)
        cos_ph = cos_ph / magnitude
        sin_ph = sin_ph / magnitude
        return amp, cos_ph, sin_ph


# Compatibility names used by the original experiment notebooks/checkpoints.
SaturationAwareScaling = SaturationAwareDualGainScaling
recon_model = CPRNet
