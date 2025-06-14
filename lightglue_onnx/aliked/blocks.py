from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair
import torchvision
# from custom_ops import get_patches


def get_patches(
        input_map: torch.Tensor, locations: torch.Tensor, patch_size: int = 3, device: str = ""
):
    # TODO: check shapes of input params.

    # Convert patch center locations to integer to support indexing.
    locations_long = locations.long()

    # Create grid for patch, for indexing.
    window_low = 0
    window_high = patch_size
    y_grid, x_grid = torch.meshgrid(
        torch.arange(window_low, window_high),
        torch.arange(window_low, window_high),
    )

    x_coords = locations_long[:, 0] - 1
    y_coords = locations_long[:, 1] - 1
    x_indices = x_coords.unsqueeze(-1).unsqueeze(-1) + x_grid.to(device)
    y_indices = y_coords.unsqueeze(-1).unsqueeze(-1) + y_grid.to(device)

    input_shape = input_map.shape
    x_indices = torch.clip(x_indices, 0, input_shape[2] - 1)
    y_indices = torch.clip(y_indices, 0, input_shape[1] - 1)
    patches = input_map[:, y_indices, x_indices]
    patches = patches.permute(1, 0, 2, 3)
    return patches


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        mask=False,
        device="cpu"
    ):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.mask = mask
        self.device = device
        self.channel_num = (
            3 * kernel_size * kernel_size
            if mask
            else 2 * kernel_size * kernel_size
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.channel_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0
        max_offset = torch.tensor(max_offset, device=self.device)

        out = self.offset_conv(x)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
        else:
            offset = out
            mask = None
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=mask,
        )
        return x


def get_conv(
    inplanes,
    planes,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    conv_type="conv",
    mask=False,
    device="",
):
    if conv_type == "conv":
        conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "dcn":
        conv = DeformableConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=_pair(padding),
            bias=bias,
            mask=mask,
            device=device,
        )
    else:
        raise TypeError
    return conv


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = get_conv(
            in_channels,
            out_channels,
            kernel_size=3,
            conv_type=conv_type,
            mask=mask,
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(
            out_channels,
            out_channels,
            kernel_size=3,
            conv_type=conv_type,
            mask=mask,
        )
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# modified based on torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
        device: str = "cpu",
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "ResBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_conv(
            inplanes, planes, kernel_size=3, conv_type=conv_type, mask=mask, device=device
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = get_conv(
            planes, planes, kernel_size=3, conv_type=conv_type, mask=mask, device=device
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class SDDH(nn.Module):
    def __init__(
        self,
        dims: int,
        kernel_size: int = 3,
        n_pos: int = 8,
        gate=nn.ReLU(),
        conv2D=False,
        mask=False,
        device="cpu",
    ):
        super(SDDH, self).__init__()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        self.conv2D = conv2D
        self.mask = mask
        self.device = device
        # self.get_patches_func = get_patches.apply

        # estimate offsets
        self.channel_num = 3 * n_pos if mask else 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                dims,
                self.channel_num,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            gate,
            nn.Conv2d(
                self.channel_num,
                self.channel_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # sampled feature conv
        self.sf_conv = nn.Conv2d(
            dims, dims, kernel_size=1, stride=1, padding=0, bias=False
        )

        # convM
        if not conv2D:
            # deformable desc weights
            agg_weights = torch.nn.Parameter(torch.rand(n_pos, dims, dims))
            self.register_parameter("agg_weights", agg_weights)
        else:
            self.convM = nn.Conv2d(
                dims * n_pos,
                dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, x, keypoints):
        # x: [B,C,H,W]
        # keypoints: list, [[N_kpts,2], ...] (w,h)
        b, c, h, w = x.shape
        keypoints = list(torch.unbind(keypoints, dim=0))
        wh = torch.tensor([[w - 1, h - 1]], device=x.device)
        max_offset = max(h, w) / 4.0
        max_offset = torch.tensor(max_offset, device=self.device)

        offsets = []
        descriptors = []
        # get offsets for each keypoint
        for ib in range(b):
            xi, kptsi = x[ib], keypoints[ib]
            kptsi_wh = (kptsi / 2 + 0.5) * wh
            N_kpts = len(kptsi)

            # xi.shape: (128, 376, 1241) - (D, H, W)
            # kpts_wh.shape: (1441, 2) - (N, 2), x,y -> W, H
            # kernel_size: 3 - int
            # patch.shape: (N, D, kernel_size, kernel_size)

            if self.kernel_size > 1:
                # patch = self.get_patches_func(
                #     xi, kptsi_wh.long(), self.kernel_size
                # )  # [N_kpts, C, K, K]
                # sad treba usporediti nakom permute!!!
                patch = get_patches(
                    xi, kptsi_wh.long(), self.kernel_size, device=self.device
                )  # [N_kpts, C, K, K]
            else:
                kptsi_wh_long = kptsi_wh.long()
                patch = (
                    xi[:, kptsi_wh_long[:, 1], kptsi_wh_long[:, 0]]
                    .permute(1, 0)
                    .reshape(N_kpts, c, 1, 1)
                )

            offset = self.offset_conv(patch).clamp(
                -max_offset, max_offset
            )  # [N_kpts, 2*n_pos, 1, 1]
            if self.mask:
                offset = (
                    offset[:, :, 0, 0]
                    .view(N_kpts, 3, self.n_pos)
                    .permute(0, 2, 1)
                )  # [N_kpts, n_pos, 3]
                offset = offset[:, :, :-1]  # [N_kpts, n_pos, 2]
                mask_weight = torch.sigmoid(
                    offset[:, :, -1]
                )  # [N_kpts, n_pos]
            else:
                offset = (
                    offset[:, :, 0, 0]
                    .view(N_kpts, 2, self.n_pos)
                    .permute(0, 2, 1)
                )  # [N_kpts, n_pos, 2]
            offsets.append(offset)  # for visualization

            # get sample positions
            pos = kptsi_wh.unsqueeze(1) + offset  # [N_kpts, n_pos, 2]
            pos = 2.0 * pos / wh[None] - 1
            pos = pos.reshape(1, N_kpts * self.n_pos, 1, 2)

            # sample features
            features = F.grid_sample(
                xi.unsqueeze(0), pos, mode="bilinear", align_corners=True
            )  # [1,C,(N_kpts*n_pos),1]
            features = features.reshape(c, N_kpts, self.n_pos, 1).permute(
                1, 0, 2, 3
            )  # [N_kpts, C, n_pos, 1]
            if self.mask:
                features = torch.einsum("ncpo,np->ncpo", features, mask_weight)

            features = torch.selu_(self.sf_conv(features)).squeeze(
                -1
            )  # [N_kpts, C, n_pos]
            # convM
            if not self.conv2D:
                descs = torch.einsum(
                    "ncp,pcd->nd", features, self.agg_weights
                )  # [N_kpts, C]
            else:
                features = features.reshape(N_kpts, -1)[
                    :, :, None, None
                ]  # [N_kpts, C*n_pos, 1, 1]
                descs = self.convM(features).squeeze()  # [N_kpts, C]

            # normalize
            descs = F.normalize(descs, p=2.0, dim=1)
            descriptors.append(descs)

        descriptors = torch.stack(descriptors, dim=0)
        offsets = torch.stack(offsets, dim=0)

        return descriptors, offsets
