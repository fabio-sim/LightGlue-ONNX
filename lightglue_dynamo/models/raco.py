from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from ..ops.shape_utils import shape_as_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = nn.SELU()
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
        self.gate = nn.SELU()
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


def _gather_subpixel_offsets(
    logits: torch.Tensor, indices: torch.Tensor, nms_radius: int, temperature: float
) -> torch.Tensor:
    """Gather the same zero-padded neighborhoods as ``unfold`` without ONNX Pad."""
    batch = logits.shape[0]
    shape = shape_as_tensor(logits)
    height = shape[-2]
    width = shape[-1]
    center_x = torch.remainder(indices, width)
    center_y = torch.div(indices, width, rounding_mode="floor")
    coordinates = torch.arange(nms_radius, device=logits.device) - nms_radius // 2
    offset_y, offset_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    offset_x = offset_x.flatten()
    offset_y = offset_y.flatten()
    sample_x = center_x[..., None] + offset_x
    sample_y = center_y[..., None] + offset_y
    valid = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    sample_x = torch.minimum(torch.maximum(sample_x, torch.zeros_like(sample_x)), width - 1)
    sample_y = torch.minimum(torch.maximum(sample_y, torch.zeros_like(sample_y)), height - 1)
    linear = (sample_y * width + sample_x).flatten(1)
    patches = logits.flatten(1).gather(1, linear)
    patches = patches.reshape(batch, indices.shape[1], nms_radius**2)
    patches = torch.where(valid, patches, 0).transpose(1, 2)
    probabilities = F.softmax(patches / temperature, dim=1)
    offsets = torch.stack((offset_x, offset_y), dim=-1).to(logits)
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
    ranker_receptive_radius = 20
    ranker_patch_size = 42
    ranker_boundary_strip_size = 41
    boundary_reranked_count = 256
    boundary_window_count = 512

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
        ranker_scale: float = 1.0,
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
        if not 0 < ranker_scale <= 1:
            raise ValueError("ranker_scale must be in the interval (0, 1]")
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
        # Values below one are an explicit speed/rotation-coverage tradeoff;
        # preserve the checkpoint's native resolution by default.
        self.ranker_scale = ranker_scale
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], persistent=False)

        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.gate = nn.SELU()
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
            nn.SELU(),
            _conv3x3(8, 4),
            nn.SELU(),
            _conv3x3(4, 4),
            nn.SELU(),
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

    def _candidate_keypoints(
        self, image: torch.Tensor, count: int, *, gather_subpixel: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Spatial softmax is strictly monotonic, so local maxima and their
        # ordering can be selected directly from logits. Keep probability
        # computation after selection: standalone RaCo still returns detection
        # scores, while the matching pipeline can dead-code-eliminate that
        # otherwise unused full-resolution branch.
        nms = F.max_pool2d(logits, self.nms_radius, stride=1, padding=self.nms_radius // 2)
        logits_nms = torch.where(logits == nms, logits, -torch.inf)
        _top_values, top_indices = _chunked_topk(logits_nms.flatten(1), count, self.topk_chunk_size)
        width = shape_as_tensor(logits)[-1]
        x = torch.remainder(top_indices, width)
        y = torch.div(top_indices, width, rounding_mode="floor")
        keypoints = torch.stack((x, y), dim=-1).to(logits.dtype)
        if self.subpixel_sampling:
            subpixel = _gather_subpixel_offsets if gather_subpixel else _subpixel_offsets
            keypoints = keypoints + subpixel(logits, top_indices, self.nms_radius, self.subpixel_temperature)
        return image, keypoints, logits

    def extract_unranked(self, image: torch.Tensor) -> torch.Tensor:
        """Select detector keypoints directly for an experimental matching fast path."""
        _normalized_image, keypoints, _logits = self._candidate_keypoints(image, self.num_keypoints)
        return keypoints + 0.5

    def _ranker_scores(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        ranker_image = image
        ranker_keypoints = keypoints
        if self.ranker_scale != 1:
            ranker_image = F.interpolate(
                image,
                scale_factor=self.ranker_scale,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=False,
            )
            image_shape = shape_as_tensor(image)
            ranker_shape = shape_as_tensor(ranker_image)
            image_size = torch.stack((image_shape[-1], image_shape[-2])).to(keypoints) - 1
            ranker_size = torch.stack((ranker_shape[-1], ranker_shape[-2])).to(keypoints) - 1
            ranker_keypoints = keypoints * (ranker_size / image_size)
        return _sample(self.ranker_head(ranker_image), ranker_keypoints)

    def _candidate_patches(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Gather exact ranker receptive fields around candidate keypoints."""
        batch, channels, height, width = image.shape
        lower = keypoints.floor().long()
        radius = self.ranker_receptive_radius
        patch_size = self.ranker_patch_size
        x_corner = (lower[..., 0] - radius).clamp(0, width - patch_size)
        y_corner = (lower[..., 1] - radius).clamp(0, height - patch_size)
        offset = torch.arange(patch_size, device=image.device)
        y_offset, x_offset = torch.meshgrid(offset, offset, indexing="ij")
        x = x_corner[..., None, None] + x_offset
        y = y_corner[..., None, None] + y_offset
        linear_indices = (y * width + x).flatten(1)
        linear_indices = linear_indices[:, None].expand(-1, channels, -1)
        sampled = image.flatten(2).gather(2, linear_indices)
        patches = sampled.reshape(batch, channels, keypoints.shape[1], patch_size, patch_size)
        return patches.permute(0, 2, 1, 3, 4).flatten(0, 1)

    def _valid_ranker(self, patches: torch.Tensor) -> torch.Tensor:
        """Evaluate the ranker without padding on already haloed patches."""
        tensor = patches
        for block in self.ranker_head[:-1]:
            if not isinstance(block, ResBlock):
                raise RuntimeError("Candidate-local ranking requires the standard RaCo ranker")
            convolved = block.gate(block.bn1(F.conv2d(tensor, block.conv1.weight, block.conv1.bias)))
            convolved = block.bn2(F.conv2d(convolved, block.conv2.weight, block.conv2.bias))
            identity = block.match_dims(tensor[..., 2:-2, 2:-2])
            tensor = block.gate(convolved + identity)
        final = self.ranker_head[-1]
        if not isinstance(final, nn.Conv2d):
            raise RuntimeError("Candidate-local ranking requires the standard RaCo ranker")
        return F.conv2d(tensor, final.weight, final.bias)

    def _candidate_local_ranker_scores(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Evaluate exact full-resolution scores only at candidate locations."""
        if self.ranker_scale != 1:
            raise RuntimeError("Candidate-local ranking requires ranker_scale=1")
        batch = image.shape[0]
        shape = shape_as_tensor(image)
        height = shape[-2].to(keypoints)
        width = shape[-1].to(keypoints)
        lower = keypoints.floor()

        local_map = self._valid_ranker(self._candidate_patches(image, keypoints))
        local_map = local_map[:, 0].reshape(batch, keypoints.shape[1], 2, 2)
        fraction = keypoints - lower
        x_fraction = fraction[..., 0]
        y_fraction = fraction[..., 1]
        scores = (
            local_map[..., 0, 0] * (1 - x_fraction) * (1 - y_fraction)
            + local_map[..., 0, 1] * x_fraction * (1 - y_fraction)
            + local_map[..., 1, 0] * (1 - x_fraction) * y_fraction
            + local_map[..., 1, 1] * x_fraction * y_fraction
        )

        strip_size = self.ranker_boundary_strip_size
        horizontal = torch.cat((image[..., :strip_size, :], image[..., -strip_size:, :]), dim=0)
        horizontal_map = self.ranker_head(horizontal)
        horizontal_keypoints = torch.cat(
            (keypoints, keypoints - torch.stack((torch.zeros_like(height), height - strip_size))), dim=0
        )
        horizontal_scores = _sample(horizontal_map, horizontal_keypoints)
        top_scores = horizontal_scores[:batch]
        bottom_scores = horizontal_scores[batch:]

        vertical = torch.cat((image[..., :strip_size], image[..., -strip_size:]), dim=0)
        vertical_map = self.ranker_head(vertical)
        vertical_keypoints = torch.cat(
            (keypoints, keypoints - torch.stack((width - strip_size, torch.zeros_like(width)))), dim=0
        )
        vertical_scores = _sample(vertical_map, vertical_keypoints)
        left_scores = vertical_scores[:batch]
        right_scores = vertical_scores[batch:]

        radius = self.ranker_receptive_radius
        scores = torch.where(lower[..., 1] < radius, top_scores, scores)
        scores = torch.where(lower[..., 1] > height - radius - 2, bottom_scores, scores)
        scores = torch.where(lower[..., 0] < radius, left_scores, scores)
        return torch.where(lower[..., 0] > width - radius - 2, right_scores, scores)

    def extract_candidate_ranked(self, image: torch.Tensor) -> torch.Tensor:
        """Select the regular ranked set using sparse candidate-local evaluation."""
        image, keypoints, _logits = self._candidate_keypoints(
            image, self.num_candidates, gather_subpixel=True
        )
        ranker_scores = self._candidate_local_ranker_scores(image, keypoints)
        order = ranker_scores.topk(self.num_keypoints, dim=1).indices
        return keypoints.gather(1, order[..., None].expand(-1, -1, 2)) + 0.5

    def extract_boundary_ranked(
        self,
        image: torch.Tensor,
        *,
        reranked_count: int | None = None,
        window_count: int | None = None,
    ) -> torch.Tensor:
        """Rerank a fixed detector-order window around the output boundary."""
        if reranked_count is None:
            reranked_count = self.boundary_reranked_count
        if window_count is None:
            window_count = self.boundary_window_count
        if not 0 < reranked_count <= self.num_keypoints:
            raise ValueError("reranked_count must be in (0, num_keypoints]")
        if window_count < reranked_count:
            raise ValueError("window_count must be at least reranked_count")
        window_start = self.num_keypoints - reranked_count
        if window_start + window_count > self.num_candidates:
            raise ValueError("The reranking window exceeds the detector candidate pool")

        image, keypoints, _logits = self._candidate_keypoints(
            image, self.num_candidates, gather_subpixel=True
        )
        window_keypoints = keypoints[:, window_start : window_start + window_count]
        window_scores = self._candidate_local_ranker_scores(image, window_keypoints)
        window_order = window_scores.topk(reranked_count, dim=1).indices
        prefix = torch.arange(window_start, device=window_order.device, dtype=window_order.dtype)
        prefix = prefix[None].expand(window_order.shape[0], -1)
        selected = torch.cat((prefix, window_order + window_start), dim=1)
        return keypoints.gather(1, selected[..., None].expand(-1, -1, 2)) + 0.5

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, keypoints, logits = self._candidate_keypoints(image, self.num_candidates)
        probabilities = F.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        detection_scores = _sample(probabilities, keypoints)
        ranker_scores = self._ranker_scores(image, keypoints)
        keypoints = keypoints + 0.5

        if self.sort_by_ranker:
            ranker_scores, order = ranker_scores.topk(self.num_keypoints, dim=1)
            keypoints = keypoints.gather(1, order[..., None].expand(-1, -1, 2))
            detection_scores = detection_scores.gather(1, order)
            return keypoints, detection_scores, ranker_scores
        return (
            keypoints[:, : self.num_keypoints],
            detection_scores[:, : self.num_keypoints],
            ranker_scores[:, : self.num_keypoints],
        )
