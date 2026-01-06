from typing import Protocol

import torch
import torch.nn.functional as F
from torch.onnx import symbolic_helper


class _OnnxGraphContext(Protocol):
    def op(self, *args: object, **kwargs: object) -> torch._C.Value: ...


FUSE_MULTI_HEAD_ATTENTION = False
CUSTOM_OP_NAME = "fabiosim::multi_head_attention"


def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (t.reshape((b, n, num_heads, head_dim)).transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape((b, n, d))


fused_multi_head_attention = None


def multi_head_attention_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    if FUSE_MULTI_HEAD_ATTENTION and fused_multi_head_attention is not None:
        return fused_multi_head_attention(q, k, v, num_heads)
    else:
        return multi_head_attention(q, k, v, num_heads)


@symbolic_helper.parse_args("v", "v", "v", "i")
def symbolic_multi_head_attention(
    g: _OnnxGraphContext, q: torch._C.Value, k: torch._C.Value, v: torch._C.Value, num_heads_i: int
) -> torch._C.Value:
    return g.op("com.microsoft::MultiHeadAttention", q, k, v, num_heads_i=num_heads_i).setType(q.type())


def use_fused_multi_head_attention() -> None:
    global FUSE_MULTI_HEAD_ATTENTION, fused_multi_head_attention
    FUSE_MULTI_HEAD_ATTENTION = True
    fused_multi_head_attention = torch.library.custom_op(CUSTOM_OP_NAME, mutates_args=())(multi_head_attention)
    torch.onnx.register_custom_op_symbolic(CUSTOM_OP_NAME, symbolic_multi_head_attention, 9)
