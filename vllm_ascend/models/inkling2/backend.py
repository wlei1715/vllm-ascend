# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal v1 attention backend for the Inkling2 NPU branch (round 4).

The rel-attention math runs inside `Inkling2NpuAttention` as plain torch
over self-managed per-pass KV caches (validated to Level 4 in round 3:
LEVEL4-ROUND3-PASS, incremental decode + real-tokenizer generation).
What the v1 engine needs from a backend here is only bookkeeping:

* every attention-like layer registers itself in
  ``compilation_config.static_forward_context`` and answers
  ``get_kv_cache_config()``/``get_attn_backend()`` so the v1 allocator
  sizes, pools and binds a paged KV cache for it;
* the metadata builder returns the common attention metadata as-is —
  the layers do not consume it yet.

The paged cache the engine allocates is therefore sized but not yet
read by the attention (the per-pass tensor caches of round 3 are still
in charge).  Gathering K/V through the block tables into the torch
attention — retiring the plain tensor caches — is the next round, as is
an Ascend attention kernel (this path materializes the
``(heads, T, S)`` score matrix).
"""

from __future__ import annotations

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)


class Inkling2NpuAttentionImpl(AttentionImpl):
    """Never invoked: the layer computes attention inline (see module doc)."""

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
        attn_type: object = None,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

    def forward(  # type: ignore[override]
        self,
        layer,
        query: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        kv_cache: "torch.Tensor",
        attn_metadata,
        output: "torch.Tensor",
        output_scale: "torch.Tensor | None" = None,
        output_block_scale: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        raise NotImplementedError(
            "Inkling2NpuAttention computes inline; the backend impl class "
            "exists only to satisfy the v1 backend contract."
        )


class Inkling2NpuAttentionMetadataBuilder(
    AttentionMetadataBuilder[CommonAttentionMetadata]
):
    """Pass-through builder: layers read no per-backend metadata this round."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CommonAttentionMetadata:
        return common_attn_metadata


class Inkling2NpuAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING2_NPU"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> "tuple[int, ...]":
        # vllm-ascend's v1 bind path (worker/v2/attn_utils.py) asks each
        # backend for the paged-cache view shape.  This round the attention
        # does not read the paged cache (see module doc); the shape only
        # drives allocation, so the standard K/V-stack layout suffices.
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> "list[int]":
        return [128]

    @staticmethod
    def get_impl_cls() -> type[Inkling2NpuAttentionImpl]:
        return Inkling2NpuAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Inkling2NpuAttentionMetadataBuilder]:
        return Inkling2NpuAttentionMetadataBuilder


__all__ = [
    "Inkling2NpuAttentionBackend",
    "Inkling2NpuAttentionImpl",
    "Inkling2NpuAttentionMetadataBuilder",
]