import torch

from lightglue_dynamo.ops.shape_utils import shape_as_tensor


class Pipeline(torch.nn.Module):
    def __init__(self, extractor: torch.nn.Module, matcher: torch.nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.matcher = matcher

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shape = shape_as_tensor(images)
        h = shape[-2]
        w = shape[-1]
        # Extract keypoints and features
        keypoints, _scores, descriptors = self.extractor(images)
        # Normalize keypoints
        size = torch.stack([w, h]).to(device=keypoints.device, dtype=keypoints.dtype)
        normalized_keypoints = 2 * keypoints / size - 1
        # Match keypoints
        matches, mscores = self.matcher(normalized_keypoints, descriptors)
        return keypoints, matches, mscores
