from torch import Tensor
import torch.nn.functional as F


class InputPadder(object):
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, h: int, w: int, divis_by: int=8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x: Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode='replicate')

    def unpad(self, x: Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]