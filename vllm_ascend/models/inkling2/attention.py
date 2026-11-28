# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 rel-attention + short convs, NPU branch (correctness first).

The production NVIDIA path runs a fused Triton prologue (q/r clipping, K/V
conv + norm, KV-cache insert, sheared relative-logit precompute) and FA4.
Here the same math runs as torch ops on Ascend (see `ops.py` for the
formulas and their source references).

Correctness-round scope: a single sequence per forward, positions
``0..T-1``. The KV and conv-state caches are plain per-pass torch tensors
(``{pass_idx: tensor}``); hooking them into the vLLM-ascend block manager is
the next round's work, as is a fused Ascend attention kernel (this one
materializes the ``(heads, T, S)`` score matrix).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from .ops import (
    current_slot,
    feature_rms,
    gate_sink,
    l2_clip,
    rel_bias,
    short_conv_dynamic,
    short_conv_static,
)


class Inkling2NpuShortConv(nn.Module):
    """`Inkling2ShortConv` (nvidia/sconv.py) as plain torch.

    Static: `weight` `(dim, 1, W)` (checkpoint layout), column `j` pairs with
    lag `W-1-j`. Dynamic (full layers' K/V): per-token tap deltas
    `static_w (W, dim) + weight_proj(x) -> (T, heads, W)`, row `k` pairs with
    lag `k`. Conv state is kept per stream **and per loop pass**.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dynamic = dynamic_input_size is not None
        if self.dynamic:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight_proj = nn.Parameter(
                torch.empty(
                    num_heads * kernel_size,
                    dynamic_input_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
        # pass_idx -> (W - 1, dim) of the inputs before the current batch.
        self._past: dict[int, torch.Tensor] = {}
        # Graph mode (6c): fixed-address conv state for NPU graph capture.
        # `forward` updates the buffer IN PLACE (rolling shift) so a captured
        # replay advances the state, unlike the `torch.cat` path whose replay
        # would rewrite the same frozen tensor.
        self._graph_mode = False
        self._gbuf: dict[tuple, torch.Tensor] = {}
        # 6e: per-key (W, dim) scratch window — `[state; x]` assembled with
        # two row copies instead of a cat (cat ~10 µs vs copy ~5 µs measured
        # in-graph; values identical).
        self._gwin: dict[tuple, torch.Tensor] = {}
        # 6e: lazily cached fp32 copies of the constant conv weights (kills
        # a per-call cast of a parameter).  Cache attribute `_w32buf`;
        # builder `_w32()`.
        self._w32buf: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._past.clear()
        self._gbuf.clear()
        self._gwin.clear()

    def reset_slot(self, slot: int) -> None:
        for key in [k for k in self._past if k[0] == slot]:
            self._past.pop(key, None)
        for key in [k for k in self._gbuf if k[0] == slot]:
            self._gbuf.pop(key, None)
        for key in [k for k in self._gwin if k[0] == slot]:
            self._gwin.pop(key, None)

    def enable_graph_mode(self) -> None:
        self._graph_mode = True

    def _w32(self) -> torch.Tensor:
        """fp32 copy of the constant conv weight(s), built lazily."""
        w = self._w32buf
        if w is None or w.device != (
            self.static_w.device if self.dynamic else self.weight.device
        ):
            if self.dynamic:
                w = self.static_w.detach().float()
            else:
                w = self.weight.reshape(self.dim, -1).detach().float()
            self._w32buf = w
        return w

    def _graph_forward(
        self,
        x: torch.Tensor,
        buf: torch.Tensor,
        dyn: torch.Tensor | None,
        win: torch.Tensor,
    ) -> torch.Tensor:
        """T=1 in-place-state variant of the decode fast paths (`ops.py`)."""
        W = self.kernel_size
        head_dim = self.dim // max(self.num_heads, 1)
        # Assemble the (W, C) window `[state; x]` into the scratch buffer:
        # two row copies instead of a cat (6e; identical values).
        win[: W - 1].copy_(buf)
        win[W - 1:].copy_(x)
        xt = win.float()  # (W, C)
        if not self.dynamic:
            w = self._w32()
            acc = (w.t() * xt).sum(0)
            y = (acc + x.squeeze(0).float()) * (W + 1) ** -0.5
        else:
            assert dyn is not None
            taps = self._w32() + dyn[0].repeat_interleave(
                head_dim, dim=0
            ).t()
            acc = (taps * xt.flip(0)).sum(0)
            y = (acc + x.squeeze(0).float()) * (2 * W + 1) ** -0.5
        # Rolling in-place state update == `cat(past[1:], x)` of the cat path
        # (win[1:] == [buf[1:], x], non-overlapping with buf).
        buf.copy_(win[1:])
        return y.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_dynamic: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """`x` bf16 `(T, dim)` -> fp32 conv output `(T, dim)`."""
        key = (current_slot(), pass_idx)
        if self._graph_mode and x.shape[0] == 1:
            buf = self._gbuf.get(key)
            if buf is None:
                past = self._past.get(key)
                buf = x.new_zeros((self.kernel_size - 1, self.dim))
                if past is not None:
                    buf.copy_(past.to(x.dtype))
                self._gbuf[key] = buf
            win = self._gwin.get(key)
            if win is None:
                win = x.new_zeros((self.kernel_size, self.dim))
                self._gwin[key] = win
            dyn = None
            if self.dynamic:
                assert x_dynamic is not None
                dyn = torch.nn.functional.linear(x_dynamic, self.weight_proj)
                dyn = dyn.view(x.shape[0], self.num_heads, self.kernel_size)
            return self._graph_forward(x, buf, dyn, win)
        past = self._past.get(key)
        if self.dynamic:
            assert x_dynamic is not None
            dyn = torch.nn.functional.linear(x_dynamic, self.weight_proj)
            dyn = dyn.view(x.shape[0], self.num_heads, self.kernel_size)
            y, past = short_conv_dynamic(
                x, self.static_w, dyn, self.dim // self.num_heads, past
            )
        else:
            y, past = short_conv_static(x, self.weight, past)
        self._past[key] = past
        # A T>1 continuation (e.g. later chunked-prefill chunks) advanced the
        # cat-based state: refresh the in-place graph buffer so the next
        # captured replay continues from the new state.
        gbuf = self._gbuf.get(key)
        if gbuf is not None:
            gbuf.copy_(past.to(gbuf.dtype))
        return y


class Inkling2NpuAttention(nn.Module, AttentionLayerBase):
    """`Inkling2Attention` (nvidia/attention.py) as plain torch.

    With a `prefix` (engine mode) the layer registers itself in the
    static forward context and answers the v1 engine's per-layer KV
    cache queries (`get_kv_cache_spec` / `get_attn_backend`), so the
    v1 allocator sizes and binds a paged cache for it.  This round the
    compute still runs over the self-managed per-pass tensor caches
    (`_kv`/`_kpos`, validated in round 3); reading the engine's paged
    cache is the next round.
    """

    def __init__(self, config, is_local: bool, prefix: str | None = None) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.d_rel = config.d_rel
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.window = config.sliding_window_size if is_local else None
        self.rel_extent = (
            config.sliding_window_size if is_local else config.rel_extent
        )
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        self.kernel_size = config.sconv_kernel_size

        hidden = config.hidden_size
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        rel_size = self.num_heads * self.d_rel
        bf16 = torch.bfloat16
        # Checkpoint parameter names are kept (no fused qkvr / TP sharding):
        # wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv, wo_ud.
        self.wq_du = nn.Linear(hidden, q_size, bias=False, dtype=bf16)
        self.wr_du = nn.Linear(hidden, rel_size, bias=False, dtype=bf16)
        self.gate_attn = nn.Linear(hidden, q_size, bias=False, dtype=bf16)
        self.attn_sink_proj = nn.Linear(hidden, self.num_heads, bias=False, dtype=bf16)
        self.wk_dv = nn.Linear(hidden, kv_size, bias=False, dtype=bf16)
        self.wv_dv = nn.Linear(hidden, kv_size, bias=False, dtype=bf16)
        self.wo_ud = nn.Linear(q_size, hidden, bias=False, dtype=bf16)
        # fp32 copies of the (fixed) clip scales: avoids a per-call cast.
        self.register_buffer(
            "_q_scale_f", torch.empty(self.num_heads, self.head_dim), persistent=False
        )
        self.register_buffer(
            "_r_scale_f", torch.empty(self.num_heads, self.d_rel), persistent=False
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=bf16),
            requires_grad=False,
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=bf16),
            requires_grad=False,
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel, dtype=bf16),
            requires_grad=False,
        )
        dynamic_input = None if is_local else hidden
        self.k_sconv = Inkling2NpuShortConv(
            kv_size,
            self.kernel_size,
            dynamic_input_size=dynamic_input,
            num_heads=self.num_kv_heads,
        )
        self.v_sconv = Inkling2NpuShortConv(
            kv_size,
            self.kernel_size,
            dynamic_input_size=dynamic_input,
            num_heads=self.num_kv_heads,
        )
        # pass_idx -> (k, v) of the tokens seen so far, and their positions.
        # A decode step appends only the new tokens' K/V (the caches persist
        # across calls until `reset_state`), so prefill + step-by-step
        # decoding reuses the same tensors a single full-sequence forward
        # would build.
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._kpos: dict[int, torch.Tensor] = {}
        # Graph mode (6c): fixed-size capture-friendly KV cache (see
        # `enable_graph_mode`); inactive until explicitly enabled.
        self._graph_mode = False
        self._s_pad: int | None = None
        self._gcache: dict = {}
        self._scales_synced = False
        # 6e: projection fusion (6 prologue GEMMs on one input -> 1 GEMM on a
        # concatenated weight, 41.8 -> 18.6 us/visit in isolation) was
        # EVALUATED AND REMOVED.  Root cause of its model-level mismatch:
        # on looped visits the local layers receive THREE DISTINCT blended
        # inputs (attn/k/v streams, `loop.py:_run_layer`), so "local layers
        # have x_q == x_k == x_v" holds only on non-blended visits; the
        # fused path computed k/v from x_q and changed the token stream
        # (same-process A/B: 24-token logits maxdiff 76, tokens 5/24).
        # It also gave no measurable whole-step gain, so it is not worth
        # re-introducing with a guard.
        # 6d: fused attention core (torch_npu.npu_fusion_attention), opt-in
        # via INKLING2_NPU_FIA=1 (default off -- keeps the torch composite as
        # the A/B reference). Empirically confirmed (probe_fia_pse.py +
        # 6d_probe_fia_deep.py): `pse` is a PRE-scale additive bias
        # (softmax(scale*(pse + qk))), so rel-bias/-inf masking folds in as
        # pse = rel/scale; the returned softmax_max/softmax_sum recover lse
        # (max + log(sum)) for the sink/gate epilogue; non-contiguous BNSD
        # views are accepted (no layout copies needed).
        self._fia = os.environ.get("INKLING2_NPU_FIA", "0") == "1"
        if prefix is not None:
            self._register_in_forward_context(prefix)

    # ----- v1 engine layer interface (AttentionLayerBase) -----

    def _register_in_forward_context(self, prefix: str) -> None:
        from vllm.config import get_current_vllm_config

        try:
            cfg = get_current_vllm_config()
        except AssertionError:
            # Standalone use (no engine): the v1 layer interface is not
            # consulted; the engine always builds models inside a config
            # context, so there registration is never skipped.
            return
        forward_context = cfg.compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type:
        from .backend import Inkling2NpuAttentionBackend

        return Inkling2NpuAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        # r198 baseline pairs vllm v0.34.0 (pre-#61452 layout): the engine
        # queries layers via ``get_kv_cache_spec`` returning a KVCacheSpec,
        # not the post-#61452 ``get_kv_cache_config``/KVCacheLayerConfig API
        # the original port was written against.
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        common = dict(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=kv_cache_dtype_str_to_dtype(
                vllm_config.cache_config.cache_dtype,
                vllm_config.model_config,
            ),
        )
        if self.window is not None:
            return SlidingWindowSpec(sliding_window=self.window, **common)
        return FullAttentionSpec(**common)

    def reset_state(self) -> None:
        self._kv.clear()
        self._kpos.clear()
        self._gcache.clear()
        self.k_sconv.reset_state()
        self.v_sconv.reset_state()

    def reset_slot(self, slot: int) -> None:
        """Drop one request slot's per-pass caches (request finished/reused)."""
        keys = [key for key in self._kv if key[0] == slot]
        for key in keys:
            self._kv.pop(key, None)
            self._kpos.pop(key, None)
        for key in [k for k in self._gcache if k[0] == slot]:
            self._gcache.pop(key, None)
        self.k_sconv.reset_slot(slot)
        self.v_sconv.reset_slot(slot)

    def enable_graph_mode(self, s_pad: int) -> None:
        """Switch T=1 decode to the fixed-shape capture-friendly cache.

        The cat-based `_kv` cache reallocates every step (growing S, moving
        addresses), which an NPU graph replay cannot advance.  Graph mode
        instead preallocates `s_pad` K/V rows plus a device write cursor and
        appends in place; unused rows hold a +inf position sentinel so the
        relative-bias -inf masks them out of the softmax (the sentinel
        distance is negative / out of extent exactly like the real mask).
        """
        self._graph_mode = True
        self._s_pad = s_pad
        self.k_sconv.enable_graph_mode()
        self.v_sconv.enable_graph_mode()

    def _graph_cache(self, key, s_pad_override: int | None = None):
        """(Re)create the fixed-size cache entry, seeded from `_kv` if any."""
        dev = self.wq_du.weight.device
        s_pad = self._s_pad if s_pad_override is None else s_pad_override
        cache = {
            # Zero-filled: unused rows must contribute exactly 0 (bf16
            # garbage could be inf/NaN and poison the -inf-masked softmax).
            "k": torch.zeros(
                s_pad, self.num_kv_heads, self.head_dim,
                dtype=torch.bfloat16, device=dev,
            ),
            "v": torch.zeros(
                s_pad, self.num_kv_heads, self.head_dim,
                dtype=torch.bfloat16, device=dev,
            ),
            # Position sentinel: masked out by the relative-bias -inf.
            "kpos": torch.full(
                (s_pad,), 2**31 - 1, dtype=torch.int64, device=dev
            ),
            "wptr": torch.zeros((), dtype=torch.int64, device=dev),
        }
        prev = self._kv.get(key)
        if prev is not None:
            n = prev[0].shape[0]
            assert n <= self._s_pad, (key, n, self._s_pad)
            cache["k"][:n].copy_(prev[0])
            cache["v"][:n].copy_(prev[1])
            cache["kpos"][:n].copy_(self._kpos[key])
            cache["wptr"].fill_(n)
        return cache

    def resize_graph_caches(self, slot: int, s_pad: int) -> None:
        """Re-allocate one slot's fixed KV caches at a new bucket size (6e).

        Used when a decode request outgrows its s_pad bucket: the captured
        graph is dropped and re-captured, and this carries the live K/V
        rows over (the graph-mode path appends to `_gcache` only, so the
        cat-path `_kv` is NOT authoritative here).  Non-destructive for
        other slots.
        """
        self._s_pad = s_pad
        for key in [k for k in self._gcache if k[0] == slot]:
            old = self._gcache[key]
            n = int(old["wptr"].item())  # host read, outside capture
            assert n <= s_pad, (key, n, s_pad)
            new = self._graph_cache(key, s_pad_override=s_pad)
            # `_graph_cache` seeds from `_kv` (prefill-era); overwrite with
            # the live graph-mode rows, which include decoded tokens.
            new["k"].zero_()
            new["v"].zero_()
            new["kpos"].fill_(2**31 - 1)
            new["k"][:n].copy_(old["k"][:n])
            new["v"][:n].copy_(old["v"][:n])
            new["kpos"][:n].copy_(old["kpos"][:n])
            new["wptr"].fill_(n)
            self._gcache[key] = new

    def _graph_forward(self, positions, k, v, pass_idx: int):
        """T=1 attention over the fixed-size cache (capture-friendly).

        Same math as the T=1 branch of `forward`, but K/V are appended in
        place at the device write cursor and attention runs over the full
        padded cache (sentinel rows get -inf relative bias -> zero probs).
        """
        key = (current_slot(), pass_idx)
        cache = self._gcache.get(key)
        if cache is None:
            cache = self._graph_cache(key)
            self._gcache[key] = cache
        kc, vc, kpos, wptr = cache["k"], cache["v"], cache["kpos"], cache["wptr"]
        # Append this step's K/V and position at the cursor, then advance.
        kc.index_copy_(0, wptr.view(1), k)
        vc.index_copy_(0, wptr.view(1), v)
        kpos.index_copy_(0, wptr.view(1), positions)
        wptr.add_(1)
        return kc, vc, kpos

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        T = x_q.shape[0]
        H, D, Hkv = self.num_heads, self.head_dim, self.num_kv_heads
        if not self._scales_synced:
            self._q_scale_f.copy_(self.q_scale.float())
            self._r_scale_f.copy_(self.r_scale.float())
            self._scales_synced = True

        q = self.wq_du(x_q).view(T, H, D)
        r = self.wr_du(x_q).view(T, H, self.d_rel)
        gate = self.gate_attn(x_q).view(T, H, D)
        sinks = self.attn_sink_proj(x_q).view(T, H)
        k = self.wk_dv(x_k).view(T, Hkv, D)
        v = self.wv_dv(x_v).view(T, Hkv, D)

        # Prologue: clip q/r (round then scale by tau), K/V convs + norms.
        q_clip = l2_clip(q, self._q_scale_f, self.qk_clip, tau=tau).to(torch.bfloat16)
        r_clip = l2_clip(r, self._r_scale_f, self.rel_clip, tau=tau).to(torch.bfloat16)
        # The kernels round the conv output to the projection dtype before
        # the K clip / V norm (nvidia/ops/sconv.py:795).
        k = self.k_sconv(
            k.reshape(T, Hkv * D),
            x_dynamic=x_k,
            pass_idx=pass_idx,
        ).view(T, Hkv, D).to(torch.bfloat16)
        v = self.v_sconv(
            v.reshape(T, Hkv * D),
            x_dynamic=x_v,
            pass_idx=pass_idx,
        ).view(T, Hkv, D).to(torch.bfloat16)
        k = l2_clip(k, None, self.qk_clip).to(torch.bfloat16)
        v = feature_rms(v, self.eps).to(torch.bfloat16)

        # Present K/V of this visit into the pass's cache, attend over all.
        state_key = (current_slot(), pass_idx)
        if self._graph_mode and T == 1:
            # Capture-friendly fixed-size cache: in-place append at the
            # device cursor, attention over the padded buffer (sentinel
            # positions get -inf relative bias -> zero probability).
            k_all, v_all, kpos = self._graph_forward(positions, k, v, pass_idx)
        else:
            prev = self._kv.get(state_key)
            if prev is not None:
                k_all = torch.cat((prev[0], k), 0)
                v_all = torch.cat((prev[1], v), 0)
                kpos = torch.cat((self._kpos[state_key], positions), 0)
            else:
                k_all, v_all = k, v
                # `positions` may be a VIEW into an engine-owned buffer that
                # the engine overwrites on the next step (allocator reuse),
                # silently corrupting the stored position index of the KV
                # cache.  Store a copy, never the alias.
                kpos = positions.contiguous().clone()
            self._kv[state_key] = (k_all, v_all)
            self._kpos[state_key] = kpos
            # A T>1 forward (prefill / later chunked-prefill chunks) grew the
            # cat-based cache: re-seed the fixed-size graph cache in place so
            # the next captured replay continues from the new state (the
            # captured graph keeps pointing at the same buffers).
            cache = self._gcache.get(state_key)
            if cache is not None:
                n2 = k_all.shape[0]
                # Buffer size, not the global `_s_pad`: with s_pad buckets
                # (6e) another slot's cache may use a different bucket.
                assert n2 <= cache["k"].shape[0], (
                    state_key, n2, cache["k"].shape[0]
                )
                cache["k"].zero_()
                cache["v"].zero_()
                cache["k"][:n2].copy_(k_all)
                cache["v"][:n2].copy_(v_all)
                cache["kpos"].fill_(2**31 - 1)
                cache["kpos"][:n2].copy_(kpos)
                cache["wptr"].fill_(n2)

        import os

        attn_dump = os.getenv("INKLING2_DUMP_ATTN")
        if attn_dump and (self.prefix or "").endswith("layers.0.attn"):
            prev_s = 0 if prev is None else int(prev[0].shape[0])
            ksum = float(k_all[: max(prev_s, 1)].float().abs().sum())
            with open(attn_dump, "a") as f:
                f.write(
                    f"L0 slot={state_key[0]} pass={pass_idx} T={T} "
                    f"S_prev={prev_s} S_now={int(k_all.shape[0])} "
                    f"kpos_head={kpos[:8].tolist()} "
                    f"kpos_tail={kpos[-4:].tolist()} ksum={ksum:.4f}\n"
                )

        S = k_all.shape[0]
        group = H // Hkv

        # Grouped (GQA) scores without materializing the repeated K/V:
        # q (Hkv, g, D) x k^T (Hkv, D, S) -> (Hkv, g, S) == (H, S) at T=1.
        # The relative bias already carries -inf for every position the
        # causal/sliding mask would block when `window == rel_extent` (the
        # real local config) and for negative distances (full layers), so
        # the explicit mask only fires when the two disagree.
        need_mask = self.window is not None and self.window != self.rel_extent
        if self._fia:
            out, lse = self._fia_forward(
                q_clip, r_clip, k_all, v_all, positions, kpos, need_mask, T, H
            )
            out = gate_sink(out, gate, lse, sinks)
            return self.wo_ud(out.reshape(T, H * D).to(torch.bfloat16))
        kf = k_all.float() if T == 1 else None
        rel = self._rel_scores(r_clip, positions, kpos, H)  # (H, T, S)
        if T == 1:
            qg = q_clip.float().view(Hkv, group, D)
            scores = torch.matmul(qg, kf.permute(1, 2, 0))  # (Hkv,g,S)
            scores = scores * self.scaling + rel.view(Hkv, group, S)
            if need_mask:
                scores = scores.masked_fill(
                    self._mask(positions, kpos).unsqueeze(0), float("-inf")
                )
            lse = torch.logsumexp(
                scores.reshape(H, S), dim=-1, keepdim=True
            )  # (H, 1) == (H, T)
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v_all.float().permute(1, 0, 2))  # (Hkv,g,D)
            out = out.reshape(1, H, D)
        else:
            kk = k_all.repeat_interleave(group, dim=1)  # (S, H, D)
            vv = v_all.repeat_interleave(group, dim=1)
            scores = torch.einsum("thd,shd->hts", q_clip.float(), kk.float())
            scores = scores * self.scaling
            scores = scores + rel
            if need_mask:
                scores = scores.masked_fill(
                    self._mask(positions, kpos).unsqueeze(0), float("-inf")
                )
            lse = torch.logsumexp(scores, dim=-1)  # (H, T)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hts,shd->thd", probs, vv.float())
        out = gate_sink(out, gate, lse, sinks)
        return self.wo_ud(out.reshape(T, H * D).to(torch.bfloat16))

    def _fia_forward(
        self,
        q_clip: torch.Tensor,
        r_clip: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        positions: torch.Tensor,
        kpos: torch.Tensor,
        need_mask: bool,
        T: int,
        H: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused attention core: `npu_fusion_attention` + rel-bias via `pse`.

        FIA computes `softmax(scale * (pse + qk))` with a PRE-scale additive
        `pse` (probe-verified, 6b/6d), so the rel-bias (already carrying -inf
        for negative distances, out-of-extent keys and sentinel padding rows)
        folds in as `pse = rel / scaling`; tau is already baked into
        `q_clip`/`r_clip` by `l2_clip`, so the standard `head_dim**-0.5`
        scale is exact. The returned fp32 softmax stats recover the lse the
        sink/gate epilogue needs: `lse = softmax_max + log(softmax_sum)`
        (the stats carry a trailing block dim; index 0 is the value).
        Non-contiguous BNSD views are accepted by the op, so no layout
        copies are made. Capture-safe: no host branch depends on tensor
        values (shapes are static in graph mode).
        """
        import torch_npu

        rel = self._rel_scores(r_clip, positions, kpos, H)  # (H, T, S)
        pse = rel / self.scaling
        if need_mask:
            pse = pse.masked_fill(
                self._mask(positions, kpos).unsqueeze(0), float("-inf")
            )
        q_b = q_clip.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)  # (1,H,T,D)
        k_b = k_all.permute(1, 0, 2).unsqueeze(0)  # (1, Hkv, S, D)
        v_b = v_all.permute(1, 0, 2).unsqueeze(0)
        # `sync=True` is the ONLY invocation that NPU-graph capture accepts:
        # FIA internally launches on a sub-stream whose last task is not an
        # event record, so aclmdlRICaptureEnd fails (107025) unless the op
        # synchronizes that sub-stream itself (probed across pse/no-pse,
        # gen_mask_parallel/sync combinations). It is only needed while
        # capturing/executing the graph (T=1 graph mode); the eager T>1
        # prefill path stays sync=False to avoid a host sync per visit.
        ret = torch_npu.npu_fusion_attention(
            q_b, k_b, v_b, H, "BNSD",
            pse=pse.bfloat16().unsqueeze(0),  # (1, H, T, S), dtype == qkv
            scale=self.scaling,
            sync=self._graph_mode and T == 1,
        )
        out = ret[0].squeeze(0).permute(1, 0, 2).float()  # (T, H, D)
        lse = (ret[1][..., 0] + ret[2][..., 0].log()).squeeze(0)  # (H, T)
        return out, lse

    def _rel_scores(
        self,
        r_clip: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """Relative logits `(H, T, S)` indexed by key distance.

        `k_pos` are the positions of the cached keys (prefill + decoded so
        far); `q_pos` the current step's query positions.
        """
        # 6e: fp32 copy of the constant rel projection (built lazily);
        # `rel_bias`'s `rel_proj.float()` is a no-op on it, values identical.
        rp = getattr(self, "_rel_proj_f", None)
        if rp is None or rp.device != self.rel_logits_proj.device:
            rp = self.rel_logits_proj.detach().float()
            self._rel_proj_f = rp
        rel = rel_bias(r_clip, rp, self.rel_extent)  # (T,H,E)
        dist = q_pos[:, None] - k_pos[None, :]  # (T, S)
        idx = dist.clamp(0, self.rel_extent - 1)
        bias = rel.permute(1, 0, 2).gather(
            2, idx.unsqueeze(0).expand(H, -1, -1)
        )
        # Out of range: -inf negative distances; window layers also mask the
        # distances past the extent (nvidia/ops/sconv.py:486 ``shift_rel_proj``),
        # full layers add 0.
        bad = dist < 0
        if self.is_local:
            bad = bad | (dist >= self.rel_extent)
        return bias.masked_fill(bad.unsqueeze(0), float("-inf"))

    def _mask(self, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor | None:
        dist = q_pos[:, None] - k_pos[None, :]
        bad = dist < 0
        if self.window is not None:
            bad = bad | (dist >= self.window)
        return bad


__all__ = ["Inkling2NpuAttention", "Inkling2NpuShortConv"]
