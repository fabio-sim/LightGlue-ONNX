import torch
import torch.nn.functional as F
from torch import nn


class TrivialUpsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class TrivialDownsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm: nn.Module
        nonl: nn.Module

        if skip_norm_and_gate:
            norm = nn.Sequential()
            nonl = nn.Sequential()
        else:
            norm = nn.InstanceNorm2d(in_)
            nonl = nn.PReLU(in_)

        dropout = nn.Sequential()
        conv = nn.Conv2d(in_, out_, size, padding="same", bias=True)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    in_: int
    out_: int

    def __init__(self, in_: int, out_: int, size: int = 5, is_first: bool = False) -> None:
        self.in_ = in_  # type: ignore[unresolved-attribute]
        self.out_ = out_  # type: ignore[unresolved-attribute]

        downsample: nn.Module
        if is_first:
            downsample = nn.Sequential()
            conv = Conv(in_, out_, size, skip_norm_and_gate=True)
        else:
            downsample = TrivialDownsample()
            conv = Conv(in_, out_, size)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    bottom_: int
    horizontal_: int
    cat_: int
    out_: int

    def __init__(self, bottom_: int, horizontal_: int, out_: int, size: int = 5) -> None:
        super().__init__()

        self.bottom_ = bottom_  # type: ignore[unresolved-attribute]
        self.horizontal_ = horizontal_  # type: ignore[unresolved-attribute]
        self.cat_ = bottom_ + horizontal_  # type: ignore[unresolved-attribute]
        self.out_ = out_  # type: ignore[unresolved-attribute]

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: torch.Tensor, hor: torch.Tensor) -> torch.Tensor:
        bot_big = self.upsample(bot)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
