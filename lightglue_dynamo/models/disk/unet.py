import itertools

import torch
from torch import nn

from .blocks import ThinUnetDownBlock, ThinUnetUpBlock


class Unet(nn.Module):
    up: list[int]
    down: list[int]
    in_features: int

    def __init__(self, in_features: int = 1, up: list[int] = [], down: list[int] = [], size: int = 5) -> None:  # noqa: B006
        super().__init__()
        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up  # type: ignore[unresolved-attribute]
        self.down = down  # type: ignore[unresolved-attribute]
        self.in_features = in_features  # type: ignore[unresolved-attribute]

        down_dims = [in_features, *down]
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(itertools.pairwise(down_dims)):
            down_block = ThinUnetDownBlock(d_in, d_out, size=size, is_first=i == 0)
            self.path_down.append(down_block)

        bot_dims = [down[-1], *up]
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for _, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up, strict=False)):
            up_block = ThinUnetUpBlock(d_bot, d_hor, d_out, size=size)
            self.path_up.append(up_block)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        features = [inp]
        for layer in self.path_down:
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal, strict=False):
            f_bot = layer(f_bot, f_hor)

        return f_bot
