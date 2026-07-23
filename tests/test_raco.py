import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from lightglue_dynamo.config import Extractor
from lightglue_dynamo.models import Pipeline
from lightglue_dynamo.models.aliked import ALIKEDDescriptor, DeformableConv2d, RaCoALIKED, SparseDescriptorHead
from lightglue_dynamo.models.lightglue import LightGlue
from lightglue_dynamo.models.raco import RaCo, _chunked_topk
from lightglue_dynamo.preprocessors import RaCoPreprocessor


class _ExtractorWithMetadata(torch.nn.Module):
    normalize_by_long_edge = True

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch = images.shape[0]
        keypoints = images.new_zeros(batch, 4, 2)
        scores = images.new_zeros(batch, 4)
        descriptors = images.new_zeros(batch, 4, 8)
        ranker_scores = images.new_zeros(batch, 4)
        return keypoints, scores, descriptors, ranker_scores


class _Matcher(torch.nn.Module):
    def forward(self, keypoints: torch.Tensor, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del keypoints, descriptors
        return torch.empty(0, 3, dtype=torch.int64), torch.empty(0)


def test_raco_checkpoint_filters_covariance_head(tmp_path: Path) -> None:
    reference = RaCo(num_keypoints=16, weights=None)
    state = dict(reference.state_dict())
    for index in range(5):
        state[f"covariance_estimator_head.unused_{index}"] = torch.zeros(1)
    checkpoint = tmp_path / "raco.pth"
    torch.save(state, checkpoint)

    loaded = RaCo(num_keypoints=16, weights=checkpoint)
    assert not hasattr(loaded, "covariance_estimator_head")
    assert all("covariance" not in name for name, _parameter in loaded.named_parameters())


def _randomize_batch_norm(model: torch.nn.Module) -> None:
    torch.manual_seed(10)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            with torch.no_grad():
                module.weight.copy_(torch.rand_like(module.weight) + 0.5)
                module.bias.copy_(torch.randn_like(module.bias))
                module.running_mean.copy_(torch.randn_like(module.running_mean))
                module.running_var.copy_(torch.rand_like(module.running_var) + 0.2)


def test_raco_batch_norm_folding_preserves_outputs() -> None:
    detector = RaCo(num_keypoints=128, weights=None).eval()
    _randomize_batch_norm(detector)
    fused = copy.deepcopy(detector)
    fused.fuse_batch_norm()
    images = torch.rand(2, 3, 64, 96)

    with torch.inference_mode():
        expected = detector(images)
        actual = fused(images)

    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in fused.modules())
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("portable", [False, True])
def test_aliked_batch_norm_folding_preserves_outputs(portable: bool) -> None:
    descriptor = ALIKEDDescriptor(weights=None, portable_deform_conv=portable).eval()
    _randomize_batch_norm(descriptor)
    fused = copy.deepcopy(descriptor)
    fused.fuse_batch_norm()
    images = torch.rand(2, 3, 64, 96)
    keypoints = torch.rand(2, 16, 2) * torch.tensor([95.0, 63.0])

    with torch.inference_mode():
        expected = descriptor(images, keypoints)
        actual = fused(images, keypoints)

    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in fused.modules())
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_raco_truncates_ranked_candidate_pool() -> None:
    torch.manual_seed(0)
    ranked_candidates = RaCo(num_keypoints=32, candidate_multiplier=1, weights=None).eval()
    selected = RaCo(num_keypoints=16, candidate_multiplier=2, weights=None).eval()
    selected.load_state_dict(ranked_candidates.state_dict())
    images = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        candidates = ranked_candidates(images)
        outputs = selected(images)

    assert selected.num_candidates == 32
    for output, candidate in zip(outputs, candidates, strict=True):
        torch.testing.assert_close(output, candidate[:, :16])


def test_raco_unranked_matching_path_does_not_execute_ranker() -> None:
    detector = RaCo(num_keypoints=16, sort_by_ranker=False, weights=None).eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.inference_mode():
        expected = detector(images)[0]

    class _UnexpectedRanker(torch.nn.Module):
        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            raise AssertionError("ranker should not execute")

    detector.ranker_head = _UnexpectedRanker()
    with torch.inference_mode():
        actual = detector.extract_unranked(images)
    torch.testing.assert_close(actual, expected)


def test_raco_aliked_ranker_bypass_is_explicit_and_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RaCo, "_load_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ALIKEDDescriptor, "_load_weights", lambda *_args, **_kwargs: None)

    assert not RaCoALIKED(num_keypoints=16).bypass_ranker
    assert RaCoALIKED(num_keypoints=16, bypass_ranker=True).bypass_ranker


def test_raco_caps_candidates_without_reducing_output_count() -> None:
    assert RaCo(num_keypoints=2048, weights=None).num_candidates == 3840
    assert RaCo(num_keypoints=4096, weights=None).num_candidates == 4096
    assert Extractor.raco_aliked.keypoint_candidate_count(2048) == 3840
    assert Extractor.raco_aliked.keypoint_candidate_count(4096) == 4096


@pytest.mark.parametrize("length", [65_536, 65_537, 131_071])
def test_chunked_topk_matches_direct_topk(length: int) -> None:
    torch.manual_seed(length)
    # Float64 random values make accidental ties vanishingly unlikely. TopK is
    # allowed to return any member/order within an equal-valued boundary.
    scores = torch.rand(2, length, dtype=torch.float64)
    expected = scores.topk(3840)
    values, indices = _chunked_topk(scores, 3840, 65_536)

    torch.testing.assert_close(values, expected.values, atol=0, rtol=0)
    torch.testing.assert_close(indices, expected.indices, atol=0, rtol=0)


def test_chunked_topk_has_direct_fallback() -> None:
    scores = torch.rand(2, 4096)
    expected = scores.topk(128)
    values, indices = _chunked_topk(scores, 128, None)

    torch.testing.assert_close(values, expected.values, atol=0, rtol=0)
    torch.testing.assert_close(indices, expected.indices, atol=0, rtol=0)


def test_logit_nms_matches_probability_nms() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 1, 64, 96)
    probabilities = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)
    probability_maxima = torch.nn.functional.max_pool2d(probabilities, 3, stride=1, padding=1)
    probability_scores = probabilities * (probabilities == probability_maxima)
    expected = _chunked_topk(probability_scores.flatten(1), 128, 1024)[1]

    logit_maxima = torch.nn.functional.max_pool2d(logits, 3, stride=1, padding=1)
    logit_scores = torch.where(logits == logit_maxima, logits, -torch.inf)
    actual = _chunked_topk(logit_scores.flatten(1), 128, 1024)[1]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_raco_preprocessor_is_rgb_float32() -> None:
    bgr = np.asarray([[[[0, 127, 255]]]], dtype=np.uint8)
    result = RaCoPreprocessor.preprocess(bgr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 0, 0], np.asarray([1, 127 / 255, 0], dtype=np.float32))


def test_raco_preprocessor_is_contiguous_and_matches_reference() -> None:
    rng = np.random.default_rng(13)
    bgr = rng.integers(0, 256, size=(2, 3, 17, 23, 3), dtype=np.uint8)
    expected = np.ascontiguousarray((bgr[..., ::-1].astype(np.float32) / np.float32(255)).transpose(0, 1, 4, 2, 3))
    result = RaCoPreprocessor.preprocess(bgr)
    assert result.flags.c_contiguous
    np.testing.assert_allclose(result, expected, atol=np.finfo(np.float32).eps, rtol=0)


def test_raco_pipeline_uses_universal_output_contract() -> None:
    assert Extractor.raco_aliked.value == "raco_aliked"
    outputs = Pipeline(_ExtractorWithMetadata(), _Matcher())(torch.zeros(2, 3, 32, 32))
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 4, 2)


def test_portable_deform_conv_matches_torchvision() -> None:
    torch.manual_seed(0)
    native = DeformableConv2d(3, 5).eval()
    portable = DeformableConv2d(3, 5, portable=True).eval()
    portable.load_state_dict(native.state_dict())
    tensor = torch.randn(2, 3, 8, 11)
    with torch.inference_mode():
        expected = native(tensor)
        actual = portable(tensor)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _reference_sparse_descriptors(
    head: SparseDescriptorHead, features: torch.Tensor, keypoints: torch.Tensor
) -> torch.Tensor:
    _batch, channels, height, width = features.shape
    scale = features.new_tensor([width - 1, height - 1])
    descriptors = []
    for index, points in enumerate(keypoints):
        pixel_points = (points / 2 + 0.5) * scale
        corners = (pixel_points.long() - head.kernel_size / 2 + 1).long()
        x_corner = corners[:, 0].clamp(0, width - 1 - head.kernel_size)
        y_corner = corners[:, 1].clamp(0, height - 1 - head.kernel_size)
        offset = torch.arange(head.kernel_size, device=features.device)
        y, x = torch.meshgrid(offset, offset, indexing="ij")
        x = x[..., None] + x_corner
        y = y[..., None] + y_corner
        patches = features[index, :, y, x].permute(3, 0, 1, 2)
        offsets = head.offset_conv(patches).clamp(-max(height, width) / 4.0, max(height, width) / 4.0)
        offsets = offsets[:, :, 0, 0].reshape(points.shape[0], 2, head.positions).transpose(1, 2)
        positions = pixel_points[:, None] + offsets
        positions = (2 * positions / scale - 1).reshape(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            features[index : index + 1], positions, mode="bilinear", align_corners=True
        )
        sampled = sampled.reshape(channels, points.shape[0], head.positions, 1).permute(1, 0, 2, 3)
        sampled = torch.nn.functional.selu(head.sf_conv(sampled)).squeeze(-1)
        descriptor = torch.einsum("ncp,pcd->nd", sampled, head.agg_weights)
        descriptors.append(torch.nn.functional.normalize(descriptor, p=2, dim=1))
    return torch.stack(descriptors)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_batched_sparse_descriptors_match_reference(batch: int) -> None:
    torch.manual_seed(1)
    head = SparseDescriptorHead(dimensions=8, positions=4).eval()
    features = torch.randn(batch, 8, 11, 13)
    keypoints = torch.rand(batch, 7, 2) * 2 - 1
    with torch.inference_mode():
        expected = _reference_sparse_descriptors(head, features, keypoints)
        actual = head(features, keypoints)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_descriptor_export_preserves_dynamic_batch() -> None:
    torch.manual_seed(2)
    head = SparseDescriptorHead(dimensions=8, positions=4).eval()
    features = torch.randn(2, 8, 11, 13)
    keypoints = torch.rand(2, 7, 2) * 2 - 1
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=4)
    width = torch.export.Dim("width", min=4)
    exported = torch.export.export(
        head, (features, keypoints), dynamic_shapes=({0: batch, 2: height, 3: width}, {0: batch})
    )

    dynamic_features = torch.randn(3, 8, 12, 15)
    dynamic_keypoints = torch.rand(3, 7, 2) * 2 - 1
    with torch.inference_mode():
        expected = head(dynamic_features, dynamic_keypoints)
        actual = exported.module()(dynamic_features, dynamic_keypoints)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_raco_export_preserves_dynamic_spatial_shapes() -> None:
    torch.manual_seed(3)
    detector = RaCo(num_keypoints=128, weights=None).eval()
    height_factor = torch.export.Dim("height_factor", min=2)
    width_factor = torch.export.Dim("width_factor", min=2)
    exported = torch.export.export(
        detector, (torch.rand(1, 3, 64, 96),), dynamic_shapes=({2: 32 * height_factor, 3: 32 * width_factor},)
    )

    images = torch.rand(1, 3, 96, 64)
    with torch.inference_mode():
        expected = detector(images)
        actual = exported.module()(images)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=1e-5)


def test_adaptive_depth_skips_unneeded_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: {})
    matcher = LightGlue(url="https://example.invalid/weights.pth", n_layers=3, depth_confidence=0.95).eval()
    for confidence in matcher.token_confidence:
        torch.nn.init.zeros_(confidence.token[0].weight)
        torch.nn.init.constant_(confidence.token[0].bias, 20)

    calls = [0, 0, 0]
    hooks = [
        layer.register_forward_hook(lambda _module, _args, _output, index=index: calls.__setitem__(index, 1))
        for index, layer in enumerate(matcher.transformers)
    ]
    keypoints = torch.rand(2, 8, 2) * 2 - 1
    descriptors = torch.rand(2, 8, 256)
    _matches, _scores, executed = matcher.forward_adaptive_depth(keypoints, descriptors)
    for hook in hooks:
        hook.remove()

    assert executed == 1
    assert calls == [1, 0, 0]
