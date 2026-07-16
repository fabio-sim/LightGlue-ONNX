from collections.abc import Callable

import pytest
import torch

from lightglue_dynamo.models.disk import DISK
from lightglue_dynamo.models.superpoint import SuperPoint


def _disable_weight_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: {"extractor": {}})
    monkeypatch.setattr(SuperPoint, "load_state_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(DISK, "load_state_dict", lambda *args, **kwargs: None)


@pytest.mark.parametrize("model_type,channels", [(SuperPoint, 1), (DISK, 3)])
def test_batched_extractors_match_independent_images(
    model_type: Callable[..., torch.nn.Module], channels: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_weight_loading(monkeypatch)
    torch.manual_seed(5)
    model = model_type(num_keypoints=16).eval()
    images = torch.rand(3, channels, 32, 48)

    with torch.inference_mode():
        actual = model(images)
        independent = [model(image[None]) for image in images]
        expected = tuple(torch.cat([outputs[index] for outputs in independent]) for index in range(3))

    for actual_output, expected_output in zip(actual, expected, strict=True):
        if actual_output.is_floating_point():
            torch.testing.assert_close(actual_output, expected_output, atol=1e-6, rtol=1e-6)
        else:
            torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)


@pytest.mark.parametrize("model_type,channels,divisor", [(SuperPoint, 1, 8), (DISK, 3, 16)])
def test_extractor_export_preserves_dynamic_batch_and_spatial_shapes(
    model_type: Callable[..., torch.nn.Module], channels: int, divisor: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_weight_loading(monkeypatch)
    torch.manual_seed(6)
    model = model_type(num_keypoints=16).eval()
    batch = torch.export.Dim("batch", min=1)
    height_factor = torch.export.Dim("height_factor", min=2)
    width_factor = torch.export.Dim("width_factor", min=2)
    exported = torch.export.export(
        model,
        (torch.rand(2, channels, 4 * divisor, 4 * divisor),),
        dynamic_shapes=({0: batch, 2: divisor * height_factor, 3: divisor * width_factor},),
    )

    images = torch.rand(3, channels, 4 * divisor, 6 * divisor)
    with torch.inference_mode():
        expected = model(images)
        actual = exported.module()(images)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=1e-6, rtol=1e-6)
