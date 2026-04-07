from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with optional force-regression head.

    If `force_dim` is None, forward returns only stress map tensor.
    If `force_dim` is an int, forward returns (stress_map, force_pred).
    """

    def __init__(
        self,
        in_channels: int = 1,
        stress_out_channels: int = 1,
        encoder_channels: Sequence[int] = (64, 128, 256, 512),
        force_dim: Optional[int] = None,
        force_hidden_dim: int = 256,
        force_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must contain at least 2 stages")

        channels = list(encoder_channels)

        self.in_conv = DoubleConv(in_channels, channels[0])
        self.down_blocks = nn.ModuleList([Down(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        decoder_spec: List[Tuple[int, int, int]] = []
        in_c = channels[-1] * 2
        for skip_c in reversed(channels):
            out_c = skip_c
            decoder_spec.append((in_c, skip_c, out_c))
            in_c = out_c

        self.up_blocks = nn.ModuleList([Up(in_c, skip_c, out_c) for in_c, skip_c, out_c in decoder_spec])
        self.stress_head = nn.Conv2d(channels[0], stress_out_channels, kernel_size=1)

        self.force_dim = force_dim
        if force_dim is not None:
            self.force_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.force_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels[-1] * 2, force_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(force_dropout),
                nn.Linear(force_hidden_dim, force_dim),
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        skips: List[torch.Tensor] = []

        x = self.in_conv(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        bottleneck = self.bottleneck(x)

        x = bottleneck
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        stress_pred = self.stress_head(x)

        if self.force_dim is None:
            return stress_pred

        force_pred = self.force_head(self.force_pool(bottleneck))
        return stress_pred, force_pred


def build_unet(**kwargs) -> UNet:
    return UNet(**kwargs)
