from enum import StrEnum, auto
from typing import Any


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()
    tensorrt = auto()
    openvino = auto()


class Extractor(StrEnum):
    superpoint = auto()
    disk = auto()
    raco_aliked = auto()

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
