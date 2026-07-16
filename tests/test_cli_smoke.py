from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from lightglue_dynamo.cli import app


def _patch_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: {})

    from lightglue_dynamo.models.superpoint import SuperPoint

    def _load_state_dict(
        self: torch.nn.Module, state_dict: dict[str, torch.Tensor], *args: object, **kwargs: object
    ) -> object:
        return torch.nn.Module.load_state_dict(self, state_dict, strict=False)

    monkeypatch.setattr(SuperPoint, "load_state_dict", _load_state_dict)


def test_export_only_exposes_dynamo_exporter() -> None:
    result = CliRunner().invoke(app, ["export", "--help"])
    assert result.exit_code == 0, result.output
    assert "legacy-export" not in result.output
    assert "fuse-multi-head-attention" not in result.output


def test_export_and_infer_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_weights(monkeypatch)

    runner = CliRunner()
    model_path = tmp_path / "lightglue.onnx"
    output_path = tmp_path / "matches.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = runner.invoke(
        app,
        ["export", "superpoint", "-o", str(model_path), "-b", "2", "-h", "64", "-w", "64", "--num-keypoints", "128"],
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()
    assert not model_path.with_suffix(model_path.suffix + ".data").exists()

    result = runner.invoke(
        app,
        [
            "infer",
            str(model_path),
            str(repo_root / "assets/sacre_coeur1.jpg"),
            str(repo_root / "assets/sacre_coeur2.jpg"),
            "superpoint",
            "-h",
            "64",
            "-w",
            "64",
            "-d",
            "cpu",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
