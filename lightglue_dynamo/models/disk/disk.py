import torch
import torch.nn.functional as F

from lightglue_dynamo.ops.shape_utils import shape_as_tensor

from .unet import Unet


def heatmap_to_keypoints(heatmap: torch.Tensor, n: int, window_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    # NMS
    b, _, h, w = heatmap.shape
    mask = F.max_pool2d(heatmap, kernel_size=window_size, stride=1, padding=window_size // 2)
    heatmap = torch.where(heatmap == mask, heatmap, torch.zeros_like(heatmap))

    # Select top-K
    top_scores, top_indices = heatmap.reshape(b, h * w).topk(n)
    shape = shape_as_tensor(heatmap).to(device=top_indices.device)
    h_i = shape[-2]
    w_i = shape[-1]
    one = h_i.new_tensor(1)
    denom = torch.stack([w_i, one])
    mod = torch.stack([h_i, w_i])
    top_indices = top_indices.unsqueeze(2).floor_divide(denom) % mod
    top_keypoints = top_indices.flip(2)

    return top_keypoints, top_scores


class DISK(torch.nn.Module):
    url = "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth"
    descriptor_dim: int
    nms_window_size: int
    num_keypoints: int

    def __init__(self, descriptor_dim: int = 128, nms_window_size: int = 5, num_keypoints: int = 1024) -> None:
        super().__init__()
        if nms_window_size % 2 != 1:
            raise ValueError(f"window_size has to be odd, got {nms_window_size}")

        self.descriptor_dim = descriptor_dim  # type: ignore[unresolved-attribute]
        self.nms_window_size = nms_window_size  # type: ignore[unresolved-attribute]
        self.num_keypoints = num_keypoints  # type: ignore[unresolved-attribute]

        self.unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, descriptor_dim + 1])

        self.load_state_dict(torch.hub.load_state_dict_from_url(self.url)["extractor"])

    def forward(
        self,
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = image.shape[0]

        unet_output: torch.Tensor = self.unet(image)
        descriptors = unet_output[:, : self.descriptor_dim]  # (B, D, H, W)
        heatmaps = unet_output[:, self.descriptor_dim :]  # (B, 1, H, W)

        keypoints, scores = heatmap_to_keypoints(heatmaps, n=self.num_keypoints, window_size=self.nms_window_size)

        descriptors = descriptors.permute(0, 2, 3, 1)
        batches = torch.arange(b, device=image.device)[:, None].expand(b, self.num_keypoints)
        descriptors = descriptors[(batches, keypoints[:, :, 1], keypoints[:, :, 0])]
        descriptors = F.normalize(descriptors, dim=-1)

        return (
            keypoints,  # (B, N, 2) with <X, Y>
            scores,  # (B, N)
            descriptors,  # (B, N, descriptor_dim)
        )
