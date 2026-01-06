import torch


def shape_as_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return dynamic shape as a tensor when exporting; otherwise use eager shape."""
    if torch.onnx.is_in_onnx_export() and hasattr(torch, "_shape_as_tensor"):
        return torch._shape_as_tensor(tensor)
    return torch.tensor(tensor.shape, device=tensor.device)
