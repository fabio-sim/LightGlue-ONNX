# Descriptor implementation adapted from ALIKED (BSD-3-Clause).
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from ..ops.shape_utils import shape_as_tensor


def _get_patches(tensor: torch.Tensor, keypoints: torch.Tensor, size: int) -> torch.Tensor:
    batch, channels, height, width = tensor.shape
    corners = (keypoints - size / 2 + 1).long()
    x_corner = corners[..., 0].clamp(0, width - 1 - size)
    y_corner = corners[..., 1].clamp(0, height - 1 - size)
    offset = torch.arange(size, device=tensor.device)
    y_offset, x_offset = torch.meshgrid(offset, offset, indexing="ij")
    x = x_corner[..., None, None] + x_offset
    y = y_corner[..., None, None] + y_offset
    linear_indices = (y * width + x).flatten(1)
    linear_indices = linear_indices[:, None].expand(-1, channels, -1)
    sampled = torch.gather(tensor.flatten(2), 2, linear_indices)
    return sampled.reshape(batch, channels, keypoints.shape[1], size, size).permute(0, 2, 1, 3, 4)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, portable: bool = False) -> None:
        super().__init__()
        self.portable = portable
        self.offset_conv = nn.Conv2d(in_channels, 18, 3, padding=1)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        maximum = shape_as_tensor(tensor)[-2:].to(tensor).max() / 4.0
        offsets = self.offset_conv(tensor).clamp(-maximum, maximum)
        if self.portable:
            return self._portable_deform_conv2d(tensor, offsets)
        return torchvision.ops.deform_conv2d(
            tensor, offsets, self.regular_conv.weight, self.regular_conv.bias, padding=(1, 1)
        )

    def _portable_deform_conv2d(self, tensor: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Decompose the 3x3 deformable convolution into standard GridSample operations.

        TensorRT requires a separately distributed plugin for the standard ONNX
        DeformConv node. This path is slower in eager PyTorch, but exports without
        custom operators and keeps the normal torchvision implementation available.
        """
        _batch, _channels, height, width = tensor.shape
        dtype = tensor.dtype
        y_base, x_base = torch.meshgrid(
            torch.arange(height, device=tensor.device, dtype=dtype),
            torch.arange(width, device=tensor.device, dtype=dtype),
            indexing="ij",
        )
        spatial_shape = shape_as_tensor(tensor)[-2:].to(tensor)
        samples: list[torch.Tensor] = []
        for kernel_index in range(9):
            kernel_y, kernel_x = divmod(kernel_index, 3)
            y = y_base + kernel_y - 1 + offsets[:, 2 * kernel_index]
            x = x_base + kernel_x - 1 + offsets[:, 2 * kernel_index + 1]
            grid_x = 2 * x / (spatial_shape[1] - 1).clamp(min=1) - 1
            grid_y = 2 * y / (spatial_shape[0] - 1).clamp(min=1) - 1
            grid = torch.stack((grid_x, grid_y), dim=-1)
            samples.append(F.grid_sample(tensor, grid, mode="bilinear", align_corners=True))
        sampled = torch.stack(samples, dim=1)
        weight = self.regular_conv.weight.flatten(2)
        output = torch.einsum("nkchw,ock->nohw", sampled, weight)
        if self.regular_conv.bias is not None:
            output = output + self.regular_conv.bias.reshape(1, -1, 1, 1)
        return output


def _fuse_conv_bn(
    convolution: nn.Conv2d | DeformableConv2d, batch_norm: nn.BatchNorm2d
) -> nn.Conv2d | DeformableConv2d:
    if isinstance(convolution, DeformableConv2d):
        # BatchNorm is applied to the output channels after deformable sampling,
        # so its affine transform can be folded into the ordinary projection.
        convolution.regular_conv = fuse_conv_bn_eval(convolution.regular_conv, batch_norm)
        return convolution
    return fuse_conv_bn_eval(convolution, batch_norm)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, deformable: bool = False, portable: bool = False
    ) -> None:
        super().__init__()
        conv = lambda source, target: (
            DeformableConv2d(source, target, portable=portable)
            if deformable
            else nn.Conv2d(source, target, 3, padding=1, bias=False)
        )
        self.gate = nn.SELU()
        self.conv1 = conv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.gate(self.bn1(self.conv1(tensor)))
        return self.gate(self.bn2(self.conv2(tensor)))

    def fuse_batch_norm(self) -> None:
        if isinstance(self.bn1, nn.BatchNorm2d):
            self.conv1 = _fuse_conv_bn(self.conv1, self.bn1)
            self.bn1 = nn.Identity()
        if isinstance(self.bn2, nn.BatchNorm2d):
            self.conv2 = _fuse_conv_bn(self.conv2, self.bn2)
            self.bn2 = nn.Identity()


class ResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, deformable: bool = False, portable: bool = False
    ) -> None:
        super().__init__()
        conv = lambda source, target: (
            DeformableConv2d(source, target, portable=portable)
            if deformable
            else nn.Conv2d(source, target, 3, padding=1, bias=False)
        )
        self.gate = nn.SELU()
        self.conv1 = conv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(tensor)
        tensor = self.gate(self.bn1(self.conv1(tensor)))
        tensor = self.bn2(self.conv2(tensor))
        return self.gate(tensor + identity)

    def fuse_batch_norm(self) -> None:
        if isinstance(self.bn1, nn.BatchNorm2d):
            self.conv1 = _fuse_conv_bn(self.conv1, self.bn1)
            self.bn1 = nn.Identity()
        if isinstance(self.bn2, nn.BatchNorm2d):
            self.conv2 = _fuse_conv_bn(self.conv2, self.bn2)
            self.bn2 = nn.Identity()


class SparseDescriptorHead(nn.Module):
    def __init__(self, dimensions: int = 128, kernel_size: int = 3, positions: int = 16) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.positions = positions
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dimensions, 2 * positions, kernel_size),
            nn.SELU(),
            nn.Conv2d(2 * positions, 2 * positions, 1),
        )
        self.sf_conv = nn.Conv2d(dimensions, dimensions, 1, bias=False)
        self.agg_weights = nn.Parameter(torch.rand(positions, dimensions, dimensions))

    def forward(self, features: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        batch, channels = features.shape[:2]
        shape = shape_as_tensor(features)
        scale = torch.stack((shape[-1], shape[-2])).to(features) - 1
        point_count = keypoints.shape[1]
        pixel_points = (keypoints / 2 + 0.5) * scale
        # ALIKED intentionally quantizes patch centers before estimating offsets.
        patches = _get_patches(features, pixel_points.long(), self.kernel_size)
        patches = patches.flatten(0, 1)
        maximum = shape[-2:].to(features).max() / 4.0
        offsets = self.offset_conv(patches).clamp(-maximum, maximum)
        offsets = offsets[:, :, 0, 0].reshape(batch, point_count, 2, self.positions).transpose(2, 3)
        positions = pixel_points[:, :, None] + offsets
        positions = (2 * positions / scale - 1).reshape(batch, point_count * self.positions, 1, 2)
        sampled = F.grid_sample(features, positions, mode="bilinear", align_corners=True)
        sampled = F.selu(self.sf_conv(sampled)).reshape(batch, channels, point_count, self.positions)
        sampled = sampled.permute(0, 2, 1, 3)
        descriptor = torch.einsum("bncp,pcd->bnd", sampled, self.agg_weights)
        return F.normalize(descriptor, p=2, dim=2)


class ALIKEDDescriptor(nn.Module):
    weights_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"

    def __init__(self, weights: str | Path | None = weights_url, *, portable_deform_conv: bool = False) -> None:
        super().__init__()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.gate = nn.SELU()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ResBlock(16, 32)
        self.block3 = ResBlock(32, 64, deformable=True, portable=portable_deform_conv)
        self.block4 = ResBlock(64, 128, deformable=True, portable=portable_deform_conv)
        self.conv1 = nn.Conv2d(16, 32, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 32, 1, bias=False)
        self.desc_head = SparseDescriptorHead()
        if weights is not None:
            self._load_weights(weights)

    def _load_weights(self, weights: str | Path) -> None:
        if isinstance(weights, str) and weights.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(weights, map_location="cpu", weights_only=True)
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        filtered = {key: value for key, value in state.items() if not key.startswith("score_head.")}
        self.load_state_dict(filtered, strict=True)

    def fuse_batch_norm(self) -> None:
        """Fold inference BatchNorm parameters into regular and deformable convolutions."""
        if self.training:
            raise RuntimeError("BatchNorm folding requires ALIKEDDescriptor.eval()")
        for module in self.modules():
            if isinstance(module, (ConvBlock, ResBlock)):
                module.fuse_batch_norm()

    def _dense_features(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(image)
        x2 = self.block2(self.pool2(x1))
        x3 = self.block3(self.pool4(x2))
        x4 = self.block4(self.pool4(x3))
        features = torch.cat(
            [
                self.gate(self.conv1(x1)),
                F.interpolate(self.gate(self.conv2(x2)), scale_factor=2, mode="bilinear", align_corners=True),
                F.interpolate(self.gate(self.conv3(x3)), scale_factor=8, mode="bilinear", align_corners=True),
                F.interpolate(self.gate(self.conv4(x4)), scale_factor=32, mode="bilinear", align_corners=True),
            ],
            dim=1,
        )
        return F.normalize(features, p=2, dim=1)

    def forward(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        shape = shape_as_tensor(image)
        scale = torch.stack((shape[-1], shape[-2])).to(image) - 1
        normalized_keypoints = 2 * keypoints / scale - 1
        return self.desc_head(self._dense_features(image), normalized_keypoints)


class RaCoALIKED(nn.Module):
    normalize_by_long_edge = True

    def __init__(
        self,
        num_keypoints: int = 2048,
        candidate_multiplier: int = 2,
        max_num_candidates: int | None = 3840,
        raco_weights: str | Path | None = None,
        aliked_weights: str | Path | None = None,
        nms_radius: int = 3,
        subpixel_sampling: bool = True,
        subpixel_temperature: float = 0.5,
        sort_by_ranker: bool = True,
        topk_chunk_size: int | None = 65536,
        portable_deform_conv: bool = False,
    ) -> None:
        super().__init__()
        from .raco import RaCo

        self.raco = RaCo(
            num_keypoints=num_keypoints,
            candidate_multiplier=candidate_multiplier,
            max_num_candidates=max_num_candidates,
            nms_radius=nms_radius,
            subpixel_sampling=subpixel_sampling,
            subpixel_temperature=subpixel_temperature,
            sort_by_ranker=sort_by_ranker,
            topk_chunk_size=topk_chunk_size,
            weights=raco_weights or RaCo.weights_url,
        )
        self.aliked = ALIKEDDescriptor(
            weights=aliked_weights or ALIKEDDescriptor.weights_url, portable_deform_conv=portable_deform_conv
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keypoints, detection_scores, ranker_scores = self.raco(image)
        descriptors = self.aliked(image, keypoints)
        return keypoints, detection_scores, descriptors, ranker_scores

    def fuse_batch_norm(self) -> None:
        if self.training:
            raise RuntimeError("BatchNorm folding requires RaCoALIKED.eval()")
        self.raco.fuse_batch_norm()
        self.aliked.fuse_batch_norm()
