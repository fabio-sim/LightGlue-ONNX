from pathlib import Path

import numpy as np
import pytest
import torch

from lightglue_dynamo.config import Extractor
from lightglue_dynamo.models import Pipeline
from lightglue_dynamo.models.aliked import DeformableConv2d
from lightglue_dynamo.models.lightglue import LightGlue
from lightglue_dynamo.models.raco import RaCo
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


def test_raco_preprocessor_is_rgb_float32() -> None:
    bgr = np.asarray([[[[0, 127, 255]]]], dtype=np.uint8)
    result = RaCoPreprocessor.preprocess(bgr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 0, 0], np.asarray([1, 127 / 255, 0], dtype=np.float32))


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
