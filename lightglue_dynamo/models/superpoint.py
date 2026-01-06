# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger
# Adapted by Fabio Milentiansen Sim


import torch
import torch.nn.functional as F
from torch import nn

from lightglue_dynamo.ops.shape_utils import shape_as_tensor


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Fast Non-maximum suppression to remove nearby points"""

    def max_pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    scores = scores[:, None]
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()).bool()
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)[:, 0]


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """

    weights_url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
    descriptor_dim: int
    nms_radius: int
    remove_borders: int | None
    num_keypoints: int

    def __init__(
        self, descriptor_dim: int = 256, nms_radius: int = 4, remove_borders: int | None = 4, num_keypoints: int = 1024
    ) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim  # type: ignore[unresolved-attribute]
        self.nms_radius = nms_radius  # type: ignore[unresolved-attribute]
        self.remove_borders = remove_borders  # type: ignore[unresolved-attribute]
        self.num_keypoints = num_keypoints  # type: ignore[unresolved-attribute]

        if self.remove_borders is not None and self.remove_borders <= 0:
            raise ValueError("remove_borders must be positive or None")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")

        # Layers
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        self.load_state_dict(torch.hub.load_state_dict_from_url(self.weights_url))

    def forward(
        self,
        image: torch.Tensor,  # (B, 1, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]  # 65 -> 64
        b, _, h, w = scores.shape  # C == 64
        shape = shape_as_tensor(scores).to(device=scores.device)
        h_i = shape[-2]
        w_i = shape[-1]
        s = 8  # scale factor (constant)
        scores = (
            scores.reshape(b, s, s, h, w)
            .permute(0, 3, 1, 4, 2)  # (B, H, S, W, S)
            .reshape(b, h * s, w * s)
        )

        scores = simple_nms(scores, self.nms_radius)  # (B, H, W)

        # Discard keypoints near the image borders
        if pad := self.remove_borders:
            scores[:, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, :pad] = -1
            scores[:, :, -pad:] = -1

        # Select top-K keypoints
        top_scores, top_indices = scores.reshape(b, h * s * w * s).topk(self.num_keypoints)
        s_i = h_i.new_tensor(s)
        one = h_i.new_tensor(1)
        denom = torch.stack([w_i * s_i, one])
        mod = torch.stack([h_i * s_i, w_i * s_i])
        top_indices = top_indices.unsqueeze(2).floor_divide(denom) % mod
        top_keypoints = top_indices.flip(2)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        top_descriptors = self.convDb(cDa)
        top_descriptors = F.normalize(top_descriptors, p=2, dim=1)

        # Extract descriptors at keypoint locations
        s_f = scores.new_tensor(float(s))
        divisor = torch.stack([w_i, h_i]).to(device=top_keypoints.device, dtype=scores.dtype)
        divisor = divisor * s_f - s_f / 2 - 0.5
        normalized_keypoints = 2 * (top_keypoints.to(dtype=scores.dtype) - (s_f / 2 - 0.5)) / divisor - 1
        top_descriptors = F.grid_sample(
            top_descriptors, normalized_keypoints[:, None], mode="bilinear", align_corners=True
        ).reshape(b, self.descriptor_dim, self.num_keypoints)
        top_descriptors = F.normalize(top_descriptors, p=2, dim=1).permute(0, 2, 1)

        return (
            top_keypoints,  # (B, N, 2) with <X, Y>
            top_scores,  # (B, N)
            top_descriptors,  # (B, N, descriptor_dim)
        )
