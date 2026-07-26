from enum import StrEnum
from enum import auto as enum_auto
from typing import Any


class InferenceDevice(StrEnum):
    cpu = enum_auto()
    cuda = enum_auto()
    tensorrt = enum_auto()
    openvino = enum_auto()


class RankerMode(StrEnum):
    """RaCo keypoint-selection policy used by the matching pipeline."""

    auto = enum_auto()
    dense = enum_auto()
    candidate_local = "candidate-local"
    boundary = enum_auto()
    bypass = enum_auto()

    def resolve(self, num_keypoints: int, height: int | None, width: int | None) -> "RankerMode":
        """Resolve the measured deployment policy to one export-time branch.

        ONNX/TensorRT must receive a single branch so unselected ranker work is
        genuinely absent. Dynamic spatial exports therefore use the policy
        that is best over most measured resolutions, with explicit modes
        available when a deployment targets a different operating point.
        """
        if self is not RankerMode.auto:
            return self
        if num_keypoints >= 3072:
            return RankerMode.bypass
        if num_keypoints == 512:
            if height is not None and width is not None and min(height, width) >= 1280:
                return RankerMode.candidate_local
            return RankerMode.dense
        if num_keypoints == 2560:
            return RankerMode.boundary
        if 1024 <= num_keypoints < 2560 and (
            height is None or width is None or min(height, width) >= 768
        ):
            return RankerMode.boundary
        return RankerMode.dense


class Extractor(StrEnum):
    superpoint = enum_auto()
    disk = enum_auto()
    raco_aliked = enum_auto()

    @property
    def input_dim_divisor(self) -> int:
        match self:
            case Extractor.superpoint:
                return 8
            case Extractor.disk:
                return 16
            case Extractor.raco_aliked:
                return 32

    @property
    def input_channels(self) -> int:
        match self:
            case Extractor.superpoint:
                return 1
            case Extractor.disk:
                return 3
            case Extractor.raco_aliked:
                return 3

    @property
    def keypoint_candidate_multiplier(self) -> int:
        """Detector candidates considered for each public output keypoint."""
        return 2 if self is Extractor.raco_aliked else 1

    def keypoint_candidate_count(self, num_keypoints: int) -> int:
        """Return the statically exported detector pool for an output budget."""
        candidates = num_keypoints * self.keypoint_candidate_multiplier
        if self is Extractor.raco_aliked:
            candidates = min(candidates, 3840)
        return max(num_keypoints, candidates)

    @property
    def lightglue_config(self) -> dict[str, Any]:
        match self:
            case Extractor.superpoint:
                return {"url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"}
            case Extractor.disk:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
                }
            case Extractor.raco_aliked:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/raco_aliked_lightglue.pth",
                }
