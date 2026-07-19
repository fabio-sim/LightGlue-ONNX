import torch
import torch.nn.functional as F


def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (t.reshape((b, n, num_heads, head_dim)).transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape((b, n, d))
