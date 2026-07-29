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
    assert "bypass-ranker" in result.output
    assert "ranker-mode" in result.output
    assert "adaptive-depth" in result.output


@pytest.mark.parametrize("depth_confidence", ["-2", "-0.5", "0", "1", "1.1", "nan"])
def test_export_rejects_invalid_depth_confidence(depth_confidence: str) -> None:
    result = CliRunner().invoke(app, ["export", "superpoint", "--depth-confidence", depth_confidence])
    assert result.exit_code != 0
    assert "--depth-confidence must be -1" in result.output
    assert "strictly between 0 and 1" in result.output


def test_adaptive_export_rejects_multiple_pairs() -> None:
    result = CliRunner().invoke(app, ["export", "superpoint", "--depth-confidence", "0.95", "--batch-size", "4"])
    assert result.exit_code != 0
    assert "supports exactly one image" in result.output
    assert "use --batch-size 2" in result.output


def test_adaptive_export_rejects_converted_fp16() -> None:
    result = CliRunner().invoke(app, ["export", "superpoint", "--depth-confidence", "0.95", "--fp16"])
    assert result.exit_code != 0
    assert "Adaptive-depth ONNX must remain FP32" in result.output
    assert "--fp16" in result.output
    assert "TensorRT select FP16" in result.output


def test_export_rejects_ranker_bypass_for_other_extractors() -> None:
    result = CliRunner().invoke(app, ["export", "superpoint", "--bypass-ranker"])
    assert result.exit_code != 0
    assert "RaCo ranker options are only supported" in result.output
    assert "raco_aliked" in result.output


def test_export_and_infer_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx

    _patch_weights(monkeypatch)

    runner = CliRunner()
    model_path = tmp_path / "lightglue.onnx"
    output_path = tmp_path / "matches.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = runner.invoke(
        app,
        [
            "export",
            "superpoint",
            "-o",
            str(model_path),
            "-b",
            "2",
            "-h",
            "64",
            "-w",
            "64",
            "--num-keypoints",
            "128",
            "--fp16",
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()
    assert not model_path.with_suffix(model_path.suffix + ".data").exists()
    graph = onnx.load(model_path).graph
    assert all(node.op_type != "Floor" for node in graph.node)
    fp16_model = onnx.load(model_path.with_suffix(".fp16.onnx"))
    onnx.checker.check_model(fp16_model, full_check=True)

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


def test_adaptive_export_and_infer_preserve_output_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx

    from lightglue_dynamo.adaptive_depth import ADAPTIVE_DEPTH_METADATA_KEY, is_adaptive_depth_model

    _patch_weights(monkeypatch)

    runner = CliRunner()
    model_path = tmp_path / "lightglue_adaptive.onnx"
    repo_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        [
            "export",
            "superpoint",
            "-o",
            str(model_path),
            "-b",
            "2",
            "-h",
            "64",
            "-w",
            "64",
            "--num-keypoints",
            "128",
            "--depth-confidence",
            "0.95",
        ],
    )
    assert result.exit_code == 0, result.output

    model = onnx.load(model_path)
    assert [output.name for output in model.graph.output] == ["keypoints", "matches", "mscores"]
    assert {entry.key: entry.value for entry in model.metadata_props}[ADAPTIVE_DEPTH_METADATA_KEY] == "0.95"
    assert is_adaptive_depth_model(model_path)
    assert sum(node.op_type == "If" for node in model.graph.node) == 1

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
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Matches:" in result.output
