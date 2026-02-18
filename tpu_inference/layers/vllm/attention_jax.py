# SPDX-License-Identifier: Apache-2.0

import functools
from typing import final, override

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
    BlockSizes, SegmentIds, make_splash_mha_single_device)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    FullMask, MultiHeadMask)
from torchax.interop import jax_view, torch_view
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

from tpu_inference.layers.common.attention_metadata import AttentionMetadata


@final
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class JaxEncoderOnlyAttentionBackend(AttentionBackend):
    """JaxEncoderOnlyAttentionBackend is a bridge to jax attention library.
    
    Here we using splash attention from jax to support encoder-only attention.
    """

    @override
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @override
    @staticmethod
    def get_impl_cls() -> type["JaxEncoderOnlyAttentionBackendImpl"]:
        return JaxEncoderOnlyAttentionBackendImpl

    @override
    @staticmethod
    def get_builder_cls():
        # Just a dummy class to make EncoderOnlyAttention happy.
        # Our attention metadata building flow is entirely different from
        # the original design from vLLM code base.
        return object


@final
class JaxEncoderOnlyAttentionBackendImpl(AttentionImpl[AttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **extra_kwargs,
    ) -> None:

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        assert alibi_slopes is None  # Alibi slopes is not supported
        assert sliding_window is None  # Sliding window is not supported yet.
        assert num_kv_heads is not None and num_kv_heads == num_heads

        if attn_type != AttentionType.ENCODER_ONLY:
            raise NotImplementedError(
                f"Attention type {attn_type} is not supported.")

    @override
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = jax_view(query), jax_view(key), jax_view(value)
        out = _jax_encoder_attn_func(
            query,
            key,
            value,
            attn_metadata,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
        )
        return torch_view(out)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass


def _ceiling_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
    return _ceiling_div(x, a) * a


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_size",
        "num_heads",
        "num_kv_heads",
    ),
)
def _jax_encoder_attn_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attn_metadata: AttentionMetadata,
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    scale: int,
) -> jax.Array:
    # Get shapes from input tensors
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads

    assert q_len == k_len, "For encoder_only, token lengths should be the same"

    # Convert the shapes from vLLM's convention to what the attention function expects
    q_thd = q.reshape(q_len, num_heads, head_size)
    k_thd = k.reshape(k_len, num_kv_heads, head_size)
    v_thd = v.reshape(k_len, num_kv_heads, head_size)

    # vLLM does pre-compute the scale (=1/sqrt(head_size)) for our convenience.
    # (q * scale) @ k == (q @ k) * scale in the torch
    q_thd = q_thd * (scale)

    # TODO: Give some heuristic block size. Need to tune for best performance.
    block_q = min(q_len, 1024)
    # Align BKV to 128 per kernel request
    block_kv = min(k_len, 1024)
    block_kv = _align_to(block_kv, 128)
    block_kv_compute = min(block_kv, 256)
    # Align block_kv to block_kv_compute
    block_kv = _align_to(block_kv, block_kv_compute)
    assert block_kv % block_kv_compute == 0

    # Swap axes to head-first per kernel limit
    q_htd = q_thd.swapaxes(0, 1)
    k_htd = k_thd.swapaxes(0, 1)
    v_htd = v_thd.swapaxes(0, 1)

    def pad_token(t: jax.Array, size) -> jax.Array:
        # tensor is [num_head, token, head_dim]
        result = jnp.pad(t, ((0, 0), (0, size), (0, 0)), constant_values=0)
        return result

    # Pad q, k, v sequence, align to block sizes
    q_pad_htd = pad_token(q_htd, _align_to(q_len, block_q) - q_len)
    k_pad_htd = pad_token(k_htd, _align_to(k_len, block_kv) - k_len)
    v_pad_htd = pad_token(v_htd, _align_to(k_len, block_kv) - k_len)
    assert k_pad_htd.shape == v_pad_htd.shape

    def build_segment_ids() -> SegmentIds:
        # Create segment IDs since the sequence may contain many requests
        max_num_seqs = attn_metadata.seq_lens.shape[0]
        # Add max_num_seqs (fake ID) as the invalid padding value of segment ID
        zero_2_max_num_seqs = jnp.arange(max_num_seqs + 1, dtype=jnp.int32)
        seq_lens_concat_zero = jnp.concatenate([
            attn_metadata.seq_lens,
            jnp.array([0], dtype=attn_metadata.seq_lens.dtype),
        ])
        # When longer than total_repeat_length, remaining values will be discarded.
        # When shorter than total_repeat_length, the final value will be repeated.
        # With additional invalid segment ID at the end, we could make sure
        # the repeated value is not the same as the valid segment IDs.
        qkv_segment_ids = jnp.repeat(
            zero_2_max_num_seqs,
            seq_lens_concat_zero,
            total_repeat_length=q_len,
        )

        def build_padded_segment(size: int) -> jax.Array:
            padding_segment_id = max_num_seqs
            result = jnp.pad(
                qkv_segment_ids,
                (0, size),
                constant_values=padding_segment_id,
            )
            return result

        # Create segment IDs for the padded sequences
        segment_ids = SegmentIds(
            q=build_padded_segment(q_pad_seq_len - q_len),
            kv=build_padded_segment(k_pad_seq_len - k_len),
        )
        return segment_ids

    # Create attention mask
    # The mask should be applied to the padded sequence length
    q_pad_seq_len = q_pad_htd.shape[1]
    k_pad_seq_len = k_pad_htd.shape[1]
    assert q_pad_seq_len % block_q == 0
    assert k_pad_seq_len % block_kv == 0
    layer_mask = FullMask((q_pad_seq_len, k_pad_seq_len))
    mask = MultiHeadMask([layer_mask for _ in range(num_heads)])

    # Create BlockSizes
    block_sizes = BlockSizes(
        block_q=block_q,
        block_kv=block_kv_compute,
        block_kv_compute=block_kv_compute,
    )
    # Make and run the kernel
    kernel = make_splash_mha_single_device(
        mask,
        block_sizes=block_sizes,
    )
    output_htd = kernel(
        q_pad_htd,
        k_pad_htd,
        v_pad_htd,
        build_segment_ids(),
    )
    assert isinstance(output_htd, jax.Array), (
        f"With save_residual=False, expect jax.Array, but got {type(output_htd)}"
    )

    # Unpad and transpose back to vLLM's shape convention
    output = output_htd[:, :q_len, :].swapaxes(0, 1)
    return output.reshape(q_len, q_compute_dim).astype(q.dtype)
