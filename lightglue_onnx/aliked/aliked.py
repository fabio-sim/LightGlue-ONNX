import os.path as osp
import time

import numpy as np
import torch

from torch import nn
from torchvision.models import resnet
from torchvision.transforms import ToTensor

from .soft_detect import DKD
from .padder import InputPadder
from .blocks import *


ALIKED_CFGS = {
    "aliked-t16": {
        "c1": 8,
        "c2": 16,
        "c3": 32,
        "c4": 64,
        "dim": 64,
        "K": 3,
        "M": 16,
    },
    "aliked-n16": {
        "c1": 16,
        "c2": 32,
        "c3": 64,
        "c4": 128,
        "dim": 128,
        "K": 3,
        "M": 16,
    },
    "aliked-n16rot": {
        "c1": 16,
        "c2": 32,
        "c3": 64,
        "c4": 128,
        "dim": 128,
        "K": 3,
        "M": 16,
    },
    "aliked-n32": {
        "c1": 16,
        "c2": 32,
        "c3": 64,
        "c4": 128,
        "dim": 128,
        "K": 3,
        "M": 32,
    },
}


class ALIKED(nn.Module):

    def __init__(
        self,
        model_name: str = "aliked-n32",
        device: str = "cuda",
        top_k: int = -1,  # -1 for threshold based mode, >0 for top K mode.
        scores_th: float = 0.2,
        n_limit: int = 5000,  # Maximum number of keypoints to be detected
        load_pretrained: bool = True,
    ):
        super().__init__()

        # get configurations
        c1, c2, c3, c4, dim, K, M = [
            v for _, v in ALIKED_CFGS[model_name].items()
        ]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2D = False
        mask = False
        self.device = device

        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(
            3, c1, self.gate, self.norm, conv_type=conv_types[0]
        )
        self.block2 = ResBlock(
            c1,
            c2,
            1,
            nn.Conv2d(c1, c2, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[1],
        )
        self.block3 = ResBlock(
            c2,
            c3,
            1,
            nn.Conv2d(c2, c3, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[2],
            mask=mask,
            device=self.device,
        )
        self.block4 = ResBlock(
            c3,
            c4,
            1,
            nn.Conv2d(c3, c4, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[3],
            mask=mask,
            device=self.device,
        )
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.score_head = nn.Sequential(
            resnet.conv1x1(dim, 8),
            self.gate,
            resnet.conv3x3(8, 4),
            self.gate,
            resnet.conv3x3(4, 4),
            self.gate,
            resnet.conv3x3(4, 1),
        )
        self.desc_head = SDDH(
            dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask, device=self.device
        )
        self.dkd = DKD(
            radius=2, top_k=top_k, scores_th=scores_th, n_limit=n_limit
        )

        # load pretrained
        if load_pretrained:
            url = f"https://raw.githubusercontent.com/ajuric/aliked-tensorrt/main/models/{model_name}.pth"
            print(f"loading {url}")
            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            self.to(device)
            self.eval()

    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32
        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        # ================================== score head
        score_map = torch.sigmoid(self.score_head(x1234))
        feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)

        # Unpads images
        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)

        return feature_map, score_map

    def forward(self, image):
        torch.cuda.synchronize()
        # t0 = time.time()
        feature_map, score_map = self.extract_dense_map(image)
        keypoints, kptscores, scoredispersitys = self.dkd(score_map)
        descriptors, offsets = self.desc_head(feature_map, keypoints)
        torch.cuda.synchronize()
        # t1 = time.time()

        # return {
        #     "keypoints": keypoints,  # B N 2
        #     "descriptors": descriptors,  # B N D
        #     "scores": kptscores,  # B N
        #     # 'score_dispersity': scoredispersitys,
        #     # 'score_map': score_map,  # Bx1xHxW
        #     # 'time': t1-t0,
        # }
        return keypoints, kptscores, descriptors

    def warmup(self, image: np.ndarray, num_iterations: int = 3) -> None:
        print("Starting warm-up ...")
        for _ in range(num_iterations):
            self.run(image)
        print("Warm-up done!")

    def run(self, img_rgb):

        img_tensor = ToTensor()(img_rgb)
        # img_tensor = img_tensor.to(self.device).unsqueeze_(0).half()
        img_tensor = img_tensor.to(self.device).unsqueeze_(0)

        with torch.no_grad():
            keypoints, descriptors, scores = self.forward(img_tensor)

        keypoints = keypoints[0]
        _, _, h, w = img_tensor.shape
        wh = torch.tensor([w - 1, h - 1], device=keypoints.device)
        keypoints = wh * (keypoints + 1) / 2

        return {
            "keypoints": keypoints.cpu().numpy(),  # N 2
            "scores": scores[0].cpu().numpy(),  # B N D
            "descriptors": descriptors[0].cpu().numpy(),  # N D
        }
