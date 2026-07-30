from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lightglue_dynamo.scripts.benchmark import _compare_records, _file_sha256


def _record(pair_index: int, latency: float, matches: int, precision: float | None) -> dict[str, float | int | None]:
    return {
        "pair_index": pair_index,
        "inference_wall_ms": latency,
        "match_count": matches,
        "epipolar_precision_1px": precision,
        "epipolar_precision_3px": precision,
        "epipolar_precision_5px": precision,
    }


def test_file_sha256_streams_complete_file(tmp_path: Path) -> None:
    content = b"benchmark provenance" * 100_000
    path = tmp_path / "model.onnx"
    path.write_bytes(content)

    assert _file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_compare_records_reports_weighted_geometry_and_failure_rates() -> None:
    reference = [
        _record(0, 10.0, 0, None),
        _record(1, 10.0, 10, 0.5),
        _record(2, 10.0, 20, 0.75),
        _record(3, 10.0, 4, 0.25),
    ]
    candidate = [
        _record(0, 9.0, 1, 1.0),
        _record(1, 11.0, 12, 0.5),
        _record(2, 10.0, 18, 0.5),
        _record(3, 10.0, 6, 0.5),
    ]

    result = _compare_records(reference, candidate, bootstrap_samples=0, bootstrap_seed=7)

    assert result["pairs"] == 4
    assert result["reference"]["zero_match_pairs"] == 1
    assert result["candidate"]["zero_match_pairs"] == 0
    assert result["reference"]["at_most_five_match_pairs"] == 2
    assert result["candidate"]["at_most_five_match_pairs"] == 1
    assert result["paired_deltas"]["mean_match_count"]["estimate"] == pytest.approx(0.75)
    assert result["paired_deltas"]["median_latency_ratio"]["estimate"] == pytest.approx(1.0)

    geometry = result["geometry"]["1px"]
    assert geometry["reference"]["correct_matches"] == pytest.approx(21.0)
    assert geometry["reference"]["weighted_precision"] == pytest.approx(21 / 34)
    assert geometry["candidate"]["correct_matches"] == pytest.approx(19.0)
    assert geometry["candidate"]["weighted_precision"] == pytest.approx(19 / 37)
    assert geometry["paired_deltas"]["correct_yield_per_pair"]["estimate"] == pytest.approx(-0.5)
    assert geometry["paired_deltas"]["weighted_precision"]["bootstrap_95_percent_interval"] is None
