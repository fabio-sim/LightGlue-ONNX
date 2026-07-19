import torch

from lightglue_dynamo.models.lightglue import CrossBlock, LearnableFourierPositionalEncoding
from lightglue_dynamo.ops import multi_head_attention


def _pairwise_cross_reference(block: CrossBlock, descriptors: torch.Tensor) -> torch.Tensor:
    outputs = []
    for pair in descriptors.reshape(-1, 2, *descriptors.shape[1:]):
        qk0, qk1 = block.to_qk(pair[0]), block.to_qk(pair[1])
        v0, v1 = block.to_v(pair[0]), block.to_v(pair[1])
        messages = []
        for query, key, value in ((qk0, qk1, v1), (qk1, qk0, v0)):
            batch_query = query.unsqueeze(0)
            batch_key = key.unsqueeze(0)
            batch_value = value.unsqueeze(0)
            message = multi_head_attention(batch_query, batch_key, batch_value, block.num_heads).squeeze(0)
            messages.append(block.to_out(message))
        outputs.extend(pair[index] + block.ffn(torch.cat((pair[index], messages[index]), dim=-1)) for index in range(2))
    return torch.stack(outputs)


def test_fourier_encoding_broadcasts_over_attention_heads() -> None:
    encoding = LearnableFourierPositionalEncoding(2, descriptor_dim=256, num_heads=4)(torch.rand(6, 32, 2))
    assert encoding.shape == (2, 6, 1, 32, 64)


def test_batched_cross_attention_matches_pairwise_reference() -> None:
    torch.manual_seed(4)
    block = CrossBlock(embed_dim=16, num_heads=4).eval()
    descriptors = torch.randn(6, 8, 16)
    with torch.inference_mode():
        expected = _pairwise_cross_reference(block, descriptors)
        actual = block(descriptors)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
