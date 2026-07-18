from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from ..ops.shape_utils import shape_as_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.gate(self.bn1(self.conv1(tensor)))
        return self.gate(self.bn2(self.conv2(tensor)))

    def fuse_batch_norm(self) -> None:
        if isinstance(self.bn1, nn.BatchNorm2d):
            self.conv1 = fuse_conv_bn_eval(self.conv1, self.bn1)
            self.bn1 = nn.Identity()
        if isinstance(self.bn2, nn.BatchNorm2d):
            self.conv2 = fuse_conv_bn_eval(self.conv2, self.bn2)
            self.bn2 = nn.Identity()


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate = nn.SELU(inplace=True)
        self.match_dims = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = self.match_dims(tensor)
        tensor = self.gate(self.bn1(self.conv1(tensor)))
        tensor = self.bn2(self.conv2(tensor))
        return self.gate(tensor + identity)

    def fuse_batch_norm(self) -> None:
        if isinstance(self.bn1, nn.BatchNorm2d):
            self.conv1 = fuse_conv_bn_eval(self.conv1, self.bn1)
            self.bn1 = nn.Identity()
        if isinstance(self.bn2, nn.BatchNorm2d):
            self.conv2 = fuse_conv_bn_eval(self.conv2, self.bn2)
            self.bn2 = nn.Identity()


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, bias=False)


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)


def _subpixel_offsets(logits: torch.Tensor, indices: torch.Tensor, nms_radius: int, temperature: float) -> torch.Tensor:
    batch, _, _, _ = logits.shape
    patches = F.unfold(logits, kernel_size=nms_radius, padding=nms_radius // 2)
    patches = patches.gather(2, indices[:, None].expand(batch, nms_radius**2, -1))
    probabilities = F.softmax(patches / temperature, dim=1)
    coordinates = torch.linspace(-(nms_radius - 1) / 2, (nms_radius - 1) / 2, nms_radius, device=logits.device)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    offsets = torch.stack((x, y), dim=-1).reshape(nms_radius**2, 2)
    return torch.einsum("bkn,kd->bnd", probabilities, offsets)


def _sample(feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    shape = shape_as_tensor(feature_map)
    scale = torch.stack((shape[-1], shape[-2])).to(keypoints) - 1
    grid = (2 * keypoints / scale - 1).unsqueeze(2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2)
    return sampled.squeeze(-1) if feature_map.shape[1] == 1 else sampled


def _chunked_topk(
    scores: torch.Tensor, count: int, chunk_size: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the exact global top-k through bounded, standard TopK operations.

    TensorRT's TopK latency grows sharply when both the input axis and K are
    large. Selecting K values from every disjoint chunk cannot discard a
    member of the global top-k, so a second TopK over their union is equivalent
    apart from the unspecified ordering of equal values. Keeping this as
    ordinary tensor operations preserves the direct TopK fallback and avoids a
    runtime-specific plugin.
    """
    if chunk_size is None:
        top = scores.topk(count)
        return top.values, top.indices
    if chunk_size < count:
        raise ValueError(f"topk_chunk_size ({chunk_size}) must be at least the candidate count ({count})")

    length = scores.shape[-1]
    padding = -length % chunk_size
    chunks = F.pad(scores, (0, padding), value=-torch.inf).reshape(scores.shape[0], -1, chunk_size)
    local_values, local_indices = chunks.topk(count, dim=-1, sorted=False)
    offsets = torch.arange(chunks.shape[1], device=scores.device, dtype=local_indices.dtype).reshape(1, -1, 1)
    local_indices = (local_indices + offsets * chunk_size).flatten(1)
    values, order = local_values.flatten(1).topk(count, dim=-1)
    return values, local_indices.gather(1, order)


class RaCo(nn.Module):
    weights_url = "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"

    def __init__(
        self,
        num_keypoints: int = 2048,
        candidate_multiplier: int = 2,
        max_num_candidates: int | None = 3840,
        nms_radius: int = 3,
        subpixel_sampling: bool = True,
        subpixel_temperature: float = 0.5,
        sort_by_ranker: bool = True,
        topk_chunk_size: int | None = 65536,
        weights: str | Path | None = weights_url,
    ) -> None:
        super().__init__()
        if nms_radius % 2 == 0:
            raise ValueError("nms_radius must be odd")
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if max_num_candidates is not None and max_num_candidates <= 0:
            raise ValueError("max_num_candidates must be positive or None")
        self.num_keypoints = num_keypoints
        # Rank a larger detector-selected pool, then retain only the requested
        # number of points. A 2x pool follows RaCo's 2048-candidate regime for
        # the common 1024-point setting without increasing descriptor or matcher
        # work. There is no benefit to over-extraction when ranking is disabled.
        multiplied_candidates = num_keypoints * candidate_multiplier
        if max_num_candidates is not None:
            multiplied_candidates = min(multiplied_candidates, max_num_candidates)
        self.num_candidates = max(num_keypoints, multiplied_candidates) if sort_by_ranker else num_keypoints
        self.nms_radius = nms_radius
        self.subpixel_sampling = subpixel_sampling
        self.subpixel_temperature = subpixel_temperature
        self.sort_by_ranker = sort_by_ranker
        self.topk_chunk_size = topk_chunk_size
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], persistent=False)

        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, 16)
        self.block2 = ResBlock(16, 32)
        self.block3 = ResBlock(32, 64)
        self.block4 = ResBlock(64, 128)
        self.conv1 = _conv1x1(16, 32)
        self.conv2 = _conv3x3(32, 32)
        self.conv3 = _conv3x3(64, 32)
        self.conv4 = _conv3x3(128, 32)
        self.score_head = nn.Sequential(
            _conv1x1(128, 8),
            nn.SELU(inplace=True),
            _conv3x3(8, 4),
            nn.SELU(inplace=True),
            _conv3x3(4, 4),
            nn.SELU(inplace=True),
            _conv3x3(4, 1),
        )
        ranker_layers: list[nn.Module] = [ResBlock(3, 12)]
        ranker_layers.extend(ResBlock(12, 12) for _ in range(8))
        ranker_layers.append(nn.Conv2d(12, 1, 5, padding=2, padding_mode="reflect"))
        self.ranker_head = nn.Sequential(*ranker_layers)

        if weights is not None:
            self._load_weights(weights)

    def _load_weights(self, weights: str | Path) -> None:
        if isinstance(weights, str) and weights.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(weights, map_location="cpu", weights_only=True)
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        covariance_keys = {key for key in state if key.startswith("covariance_estimator_head.")}
        filtered = {key: value for key, value in state.items() if key not in covariance_keys}
        self.load_state_dict(filtered, strict=True)
        if len(covariance_keys) != 5:
            raise RuntimeError(f"Expected five covariance-head tensors, found {len(covariance_keys)}")

    def fuse_batch_norm(self) -> None:
        """Fold inference BatchNorm parameters into their preceding convolutions."""
        if self.training:
            raise RuntimeError("BatchNorm folding requires RaCo.eval()")
        for module in self.modules():
            if isinstance(module, (ConvBlock, ResBlock)):
                module.fuse_batch_norm()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = (image - self.image_mean) / self.image_std

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
        logits = self.score_head(features)
        ranker_map = self.ranker_head(image)
        probabilities = F.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        nms = F.max_pool2d(probabilities, self.nms_radius, stride=1, padding=self.nms_radius // 2)
        probabilities_nms = probabilities * (probabilities == nms)
        _top_values, top_indices = _chunked_topk(
            probabilities_nms.flatten(1), self.num_candidates, self.topk_chunk_size
        )
        width = shape_as_tensor(probabilities)[-1]
        x = torch.remainder(top_indices, width)
        y = torch.div(top_indices, width, rounding_mode="floor")
        keypoints = torch.stack((x, y), dim=-1).to(probabilities.dtype)
        if self.subpixel_sampling:
            keypoints = keypoints + _subpixel_offsets(logits, top_indices, self.nms_radius, self.subpixel_temperature)
        detection_scores = _sample(probabilities, keypoints)
        ranker_scores = _sample(ranker_map, keypoints)
        keypoints = keypoints + 0.5

        if self.sort_by_ranker:
            order = ranker_scores.argsort(dim=1, descending=True)
            keypoints = keypoints.gather(1, order[..., None].expand(-1, -1, 2))
            detection_scores = detection_scores.gather(1, order)
            ranker_scores = ranker_scores.gather(1, order)
        return (
            keypoints[:, : self.num_keypoints],
            detection_scores[:, : self.num_keypoints],
            ranker_scores[:, : self.num_keypoints],
        )
