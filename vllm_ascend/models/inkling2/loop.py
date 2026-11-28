# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2, NPU platform branch (Ascend 910B) — correctness-first.

Round 4 scope: the **vLLM v1 engine interface** — the model class answers
the v1 runner's calls (`forward` returns hidden states, `compute_logits`
projects them for the sampler), each attention layer registers as a
`AttentionLayerBase` with a declared `AttentionSpec` (sliding-window /
full) so the v1 allocator sizes and pools a paged KV cache for it, and a
minimal custom backend (`backend.py`) supplies the pass-through metadata
builder.  Constraint this round: one request per batch
(`max_num_seqs=1`); the compute still runs over the round-3 self-managed
per-pass caches, so the paged cache is sized but not yet read.

Round 2 scope: the **full text stack** in torch, mirroring
`nvidia/loop.py` + `nvidia/model.py`:

* block attention residual (``attn_res``: softmax mix of the completed
  residual blocks + the running partial block per pseudo-query);
* rel-attention with q/r L2 clipping, dynamic/static K/V short convs,
  sliding-window / causal masking, attention sink + output gate;
* hidden-stream short convs with the fused residual update;
* the loop schedule: prelude (L0..5), ``num_passes`` x core (L6..17) with
  per-pass KV + conv states and per-sublayer message blending, coda (L18..23);
* the routed MoE (`moe.py`, validated to Level 3).

Still out of scope (documented TODOs): TP > 1, MTP, vision, the vLLM v1
engine plumbing (attention metadata / KV-cache block manager — caches are
plain torch tensors here), and fused Ascend kernels (performance round).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.models.inkling2.configs import Inkling2ModelConfig
from .attention import Inkling2NpuAttention, Inkling2NpuShortConv
from .moe import Inkling2NpuMoE
from . import ops
from .ops import (
    attn_res,
    attn_res_view,
    blend,
    blend_fast,
    log_scaling_tau,
    rms_norm,
)

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class NpuRMSNorm(nn.Module):
    """`Inkling2RMSNorm`: `num_inputs > 1` stacks per-input weights."""

    def __init__(
        self, hidden_size: int, eps: float, num_inputs: int = 1
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eps = eps
        # 6e: fp32 copy of the (constant) weight, built lazily — kills a
        # per-call select+cast; `rms_norm`'s `weight.float()` is a no-op on it
        # and the values are identical.
        self._w32: torch.Tensor | None = None

    def w32(self) -> torch.Tensor:
        w = self._w32
        if w is None or w.device != self.weight.device:
            self._w32 = self.weight.detach().float()
            w = self._w32
        return w

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.w32()[index], self.eps)


class NpuAttnRes(nn.Module):
    """`Inkling2AttnRes`: fp32 pseudo-queries pre-scaled by `D**-0.5`."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        self.scale = hidden**-0.5


class NpuMessageController(nn.Module):
    """One loop stream's blend weights (`controller_{down,gain,forget}`)."""

    def __init__(self, rank: int, hidden_size: int) -> None:
        super().__init__()
        bf16 = torch.bfloat16
        self.down_weight = nn.Parameter(
            torch.empty(rank, 2 * hidden_size, dtype=bf16), requires_grad=False
        )
        self.gain_weight = nn.Parameter(
            torch.empty(hidden_size, rank, dtype=bf16), requires_grad=False
        )
        self.forget_weight = nn.Parameter(
            torch.empty(hidden_size, rank, dtype=bf16), requires_grad=False
        )


class NpuDecoderLayer(nn.Module):
    """One decoder layer (loop core layers carry the stream controllers)."""

    def __init__(
        self, config: Inkling2ModelConfig, layer_id: int, prefix: str = ""
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        # Engine mode (`prefix` given): the attention layer registers as a
        # v1 AttentionLayerBase under "<prefix>layers.<i>.attn" so the v1
        # allocator enumerates and sizes its KV cache.  Standalone mode
        # (prefix ""): no registration, self-managed caches only.
        self.attn = Inkling2NpuAttention(
            config, self.is_local,
            prefix=f"{prefix}layers.{layer_id}.attn" if prefix else None,
        )
        self.attn_norm = NpuRMSNorm(hidden, eps, num_inputs=self.num_attn_inputs)
        self.mlp_norm = NpuRMSNorm(hidden, eps)
        self.mlp = Inkling2NpuMoE(config)
        self.attn_sconv = Inkling2NpuShortConv(hidden, config.sconv_kernel_size)
        self.mlp_sconv = Inkling2NpuShortConv(hidden, config.sconv_kernel_size)
        if layer_id in config.loop_core_layers:
            rank = config.loop_config["gate_rank"]
            self.controllers = nn.ModuleDict(
                {s: NpuMessageController(rank, hidden) for s in LOOP_STREAMS}
            )
            # 6e: pre-stacked / pre-transposed constant controller weights
            # for the two `blend` call patterns (see `blend`); built lazily
            # so weight loading / `.to(device)` order does not matter.
            self._blend_pre: tuple | None = None

    def attention(
        self,
        positions: torch.Tensor,
        inputs,
        tau: torch.Tensor | None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """This rank's attention output (TP=1: the full one)."""
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], pass_idx=pass_idx)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            pass_idx=pass_idx,
        )

    def _blend_precomputed(self) -> tuple:
        """Pre-transposed constant weights for the ("attn","k","v") and
        ("mlp",) `blend` calls (6e): stacking/transposing the constant
        controller weights per call cost ~90 µs/call in the graph (measured,
        6e_opt_micro.py); the precomputed layout is bit-identical."""
        pre = self._blend_pre
        if pre is None:
            down4 = torch.stack(
                [self.controllers[s].down_weight for s in LOOP_STREAMS]
            )
            up4 = torch.stack(
                [
                    torch.cat(
                        [
                            self.controllers[s].gain_weight,
                            self.controllers[s].forget_weight,
                        ],
                        0,
                    )
                    for s in LOOP_STREAMS
                ]
            )
            w_attn = self.attn_norm.w32()
            if w_attn.shape[0] == 1:
                w_attn = w_attn.expand(3, -1)
            pre = (
                (down4[:3].transpose(1, 2).contiguous(),
                 up4[:3].transpose(1, 2).contiguous().float(),
                 w_attn),
                (down4[3:].transpose(1, 2).contiguous(),
                 up4[3:].transpose(1, 2).contiguous().float(),
                 self.mlp_norm.w32()),
            )
            self._blend_pre = pre
        return pre

    def blend(
        self,
        streams: tuple[str, ...],
        dest: torch.Tensor,
        message: torch.Tensor,
        norm: NpuRMSNorm,
        norm_index: int | None = None,
    ) -> torch.Tensor:
        if norm_index is None and streams in (("attn", "k", "v"), ("mlp",)) and (
            (len(streams) == 3 and norm is self.attn_norm)
            or (len(streams) == 1 and norm is self.mlp_norm)
        ):
            down_t, up_t32, w32 = self._blend_precomputed()[
                0 if len(streams) == 3 else 1
            ]
            if w32.shape[0] == 1 and len(streams) == 3:
                w32 = w32.expand(3, -1)
            return blend_fast(dest, message, down_t, up_t32, w32, norm.eps)
        down = torch.stack([self.controllers[s].down_weight for s in streams])
        up = torch.stack(
            [
                torch.cat(
                    [self.controllers[s].gain_weight, self.controllers[s].forget_weight],
                    0,
                )
                for s in streams
            ]
        )
        weight = norm.weight if norm_index is None else norm.weight[norm_index]
        return blend(dest, message, down, up, weight.expand(len(streams), -1), norm.eps)

    def reset_state(self) -> None:
        self.attn.reset_state()
        self.attn_sconv.reset_state()
        self.mlp_sconv.reset_state()

    def reset_slot(self, slot: int) -> None:
        self.attn.reset_slot(slot)
        self.attn_sconv.reset_slot(slot)
        self.mlp_sconv.reset_slot(slot)


class NpuLoopModel(nn.Module):
    """`Inkling2LoopModel`: block stack + the loop visit schedule."""

    def __init__(self, config: Inkling2ModelConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.padded_vocab_size or config.vocab_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        )
        self.embed_norm = NpuRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            NpuDecoderLayer(config, i, prefix=prefix)
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = NpuAttnRes(config)
        self.norm = NpuRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.loop_core = config.loop_core_layers
        core = len(self.loop_core)
        shape = (2 * core, self.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(core, len(LOOP_STREAMS), self.hidden_size),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(
            -1, 2, self.hidden_size
        )
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def reset_slot(self, slot: int) -> None:
        for layer in self.layers:
            layer.reset_slot(slot)

    def enable_graph_mode(self, s_pad: int) -> None:
        """Switch every layer's decode path to fixed-shape capture-friendly
        state (see `Inkling2NpuAttention.enable_graph_mode`); T>1 forwards
        (prefill) keep the eager cat-based caches."""
        for layer in self.layers:
            layer.attn.enable_graph_mode(s_pad)
            layer.attn_sconv.enable_graph_mode()
            layer.mlp_sconv.enable_graph_mode()

    def _max_loop_blocks(self) -> int:
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: NpuRMSNorm | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        """`nvidia/loop.py:241` for TP=1 (no all-reduce / two-shot paths)."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        messages = None
        if bank is not None:
            # Four streams' messages: a mix of the bank's two blocks.
            # (bank is a contiguous clone: both sources + last are its rows.)
            messages = attn_res_view(
                bank,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            dests = inputs
            if layer.is_local:
                dests = dests.expand(3, *dests.shape[1:])
            attn_inputs = layer.blend(("attn", "k", "v"), dests, messages[:3], layer.attn_norm)
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = layer.attn_sconv(attn_partial, pass_idx=pass_idx)  # fp32
        if first:
            # sources = stack[:nb+1], last = attn_out: the contiguous view
            # stack[:nb+2] covers both (its last row is overwritten by the
            # fp32 `tail`); `num_blocks + 2 <= stack rows` because a
            # block-starting visit has num_blocks < max blocks.
            mlp_in = attn_res_view(
                stack[: num_blocks + 2],
                self.attn_res.mlp_queries[i],
                None if messages is not None else layer.mlp_norm.w32(),
                layer.mlp_norm.eps if messages is None else 0.0,
                self.normalize_values,
                tail=attn_out,
            )
        else:
            # sources = stack[:nb], last+addend = partial + attn_out: the
            # view stack[:nb+1] with the fp32 addend added onto its last row.
            mlp_in = attn_res_view(
                stack[: num_blocks + 1],
                self.attn_res.mlp_queries[i],
                None if messages is not None else layer.mlp_norm.w32(),
                layer.mlp_norm.eps if messages is None else 0.0,
                self.normalize_values,
                tail_add=attn_out,
            )
        if messages is not None:
            mlp_in = layer.blend(("mlp",), mlp_in, messages[3:], layer.mlp_norm)

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        moe_out = layer.mlp(mlp_in[0].to(torch.bfloat16))
        # The conv result is rounded to bf16 before the residual add, then
        # the sum is rounded once more (nvidia/ops/sconv.py `_sconv_kernel`).
        y = layer.mlp_sconv(moe_out, pass_idx=pass_idx).to(torch.bfloat16)
        acc = attn_out if residual is None else residual.float() + attn_out
        stack[num_blocks].copy_((acc + y.float()).to(stack.dtype))

        # sources = stack[:nb], last = stack[nb]: contiguous stack[:nb+1].
        outputs = attn_res_view(
            stack[: num_blocks + 1],
            next_query,
            None if next_norm is None else next_norm.w32(),
            0.0 if next_norm is None else next_norm.eps,
            self.normalize_values,
        )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty((self._max_loop_blocks() + 1, num_tokens, hidden))
        stack[0].copy_(inputs_embeds)
        tau = log_scaling_tau(
            positions,
            self.config.log_scaling_n_floor,
            self.config.log_scaling_alpha,
        )
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) as the bank.
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        self.attn_res.attn_queries[i - 1],
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
        assert inputs is not None
        return inputs[0]


class Inkling2LoopForConditionalGeneration(nn.Module):
    """`Inkling2LoopForConditionalGeneration` for the NPU platform."""

    # Checkpoint name substitutions (the fused qkvr / sharded norms of the
    # NVIDIA branch are not fused here — TP=1 keeps the checkpoint names).
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    # Full layers hold q/k/v input norms; the checkpoint has three tensors.
    _shards = {
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }
    _scaled_queries = ("attn_queries", "mlp_queries", "output_query")

    def __init__(self, *, vllm_config=None, config=None, prefix: str = "") -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        assert text_config.loop_config is not None, "loop checkpoint required"
        self.config = text_config
        self.vllm_config = vllm_config
        # Engine-mode state-slot bookkeeping (see `forward` / `_assign_slots`).
        self._slot_seq = 0
        self._live: dict[int, tuple[int, int]] = {}
        # NPU graph capture for T=1 engine decode steps (6c): per-slot
        # {graph, static input buffers, static output}.  Only used when the
        # engine opts in via `enable_graph_mode`; capture is lazy (first
        # decode step of a request) and invalidated with the slot's state.
        self._graphs: dict[int, dict] = {}
        self._graph_s_pad: int | None = None
        # 6e: max capture pad (from INKLING2_NPU_GRAPH_S_PAD); rows above it
        # are an error in graph mode, so it must cover max_model_len.
        self._graph_cap: int | None = None
        self.model = NpuLoopModel(text_config, prefix=f"{prefix}model.")
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.padded_vocab_size or text_config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def reset_state(self) -> None:
        self.model.reset_state()
        self._graphs.clear()

    def load_weights(self, weights) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        scale = self.model.attn_res.scale
        for name, tensor in weights:
            if name.startswith(
                ("model.visual.", "layers.", "markov_head.", "mask_embed", "fuse_weight")
            ):
                # Vision encoder and MTP tensors: unsupported this round.
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard = None
            for old, (new, idx) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard = idx
                    break
            if name not in params:
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if any(k in name for k in self._scaled_queries):
                param.data.copy_(
                    tensor.view(param.shape).to(torch.float32).mul_(scale)
                )
            elif shard is not None or (
                param.dim() == 2
                and tensor.dim() == 1
                and param.shape[1] == tensor.shape[0]
            ):
                # Stacked norms: one row per sublayer input (q/k/v), local
                # layers and the single-input norms use row 0.
                idx = 0 if shard is None else shard
                assert param.data[idx].shape == tensor.shape, name
                param.data[idx].copy_(tensor.to(param.dtype))
            elif name.endswith(("experts.w13_weight", "experts.w2_weight")):
                # Expert weights are stored `[E, K, N]` for grouped GEMM; the
                # checkpoint has `[E, N, K]`.  Transposing on load is memory
                # neutral (the parameter replaces the shard).
                expected = tensor.shape[:1] + (tensor.shape[2], tensor.shape[1])
                assert param.shape == expected, (name, param.shape, tensor.shape)
                param.data.copy_(tensor.transpose(1, 2).to(param.dtype))
            else:
                assert param.shape == tensor.shape, (name, param.shape, tensor.shape)
                param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {missing[:8]}")
        self.model.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors=None,
        **kwargs,
    ) -> torch.Tensor:
        """v1 engine forward: hidden states of all scheduled tokens.

        Text-only, single request per batch this round (scheduler
        `max_num_seqs=1`): a batch whose first token sits at position 0
        starts a new request, so the per-pass KV / conv-state caches are
        reset; a batch continuing at positions > 0 (decode steps, later
        chunks of a chunked prefill) reuses them — the round-3 A1 test
        showed chunked continuation is bit-exact with same-chunking
        recompute.
        """
        assert input_ids is not None, "text-only round: input_ids required"
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        import os

        fwd_dump = os.getenv("INKLING2_DUMP_INPUTS")
        if fwd_dump:
            with open(fwd_dump, "a") as f:
                f.write(
                    f"T={input_ids.shape[0]} pos={positions.tolist()[:8]} "
                    f"ids={input_ids.tolist()[:8]} "
                    f"embeds={'yes' if inputs_embeds is not None else 'no'}\n"
                )
        reqs = self._batch_requests(positions)
        if reqs is None:
            # Standalone / no engine metadata: single request, slot 0.
            # (The v1 engine's profile run also lands here: a full state
            # reset clears any engine-path leftovers.)
            ops.set_current_slot(0)
            if bool(positions[0] == 0):
                self.reset_state()
                self._live = {}
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_norm(
                    self.model.embed_tokens(input_ids)
                )
            return self.model(positions, inputs_embeds)

        # Engine path: the batch holds tokens of several requests
        # (adjacent `query_start_loc` segments).  State slots are assigned
        # by POSITION CONTINUITY, not by the metadata's `req_idx`: the
        # engine's `req_idx` is the batch ROW index (`idx_mapping_np` is
        # `arange`, sliced to the live rows), which SHIFTS whenever a
        # request finishes (v1 compacts the batch), silently re-keying
        # live requests onto finished requests' state.  Instead: a segment
        # starting at position 0 (prefill / first chunked-prefill chunk)
        # gets a fresh monotonic slot; a continuation segment reuses the
        # live slot whose expected next position equals its first position
        # (tie-broken by row proximity -- rows shift by small deltas on
        # compaction).  Slots no segment claims are finished: freed.  The
        # segments run sequentially in Python -- the module-global slot
        # (`ops.set_current_slot`) is therefore race-free under TP=1.
        import os

        segs = [(lo, hi) for lo, hi in reqs]
        # Auto-opt engine decode into NPU graph capture (6c).  Env switches:
        # INKLING2_NPU_GRAPH=0 keeps the eager path (A/B measurement);
        # INKLING2_NPU_GRAPH_S_PAD sizes the padded capture cache (must
        # cover the request context; the checkpoint's
        # max_position_embeddings is 1M and would OOM, so default 2048 =
        # the engine's max_model_len this round).
        if self._graph_s_pad is None and os.getenv("INKLING2_NPU_GRAPH", "1") != "0":
            s_pad = int(os.getenv("INKLING2_NPU_GRAPH_S_PAD", "2048"))
            self._graph_cap = s_pad
            self.enable_graph_mode(s_pad)
        assigned = self._assign_slots(segs, positions)
        dump = os.getenv("INKLING2_DUMP_REQS")
        if dump:
            with open(dump, "a") as f:
                f.write(
                    f"T={input_ids.shape[0]} n_reqs={len(segs)} "
                    f"slots={[s for s, _, _ in assigned]} "
                    f"pos0={[int(positions[lo]) for _, lo, _ in assigned]} "
                    f"bounds={[(lo, hi) for _, lo, hi in assigned]} "
                    f"ids0={[int(input_ids[lo]) for _, lo, _ in assigned]}\n"
                )
        hidden_rows = []
        # Mixed batches (any multi-token eager segment present) must not
        # touch the NPU graphs at all: capture-then-eager in one forward
        # corrupts the captured graph (replay later hits an invalid GM
        # address) and replay-then-eager in one forward crashes outright
        # (probe experiments E1/E4/E8 vs E2/E6/E9).  Cross-forward eager
        # work leaves captured graphs valid (E6), so only this forward's
        # T=1 segments fall back to the eager path; the next all-decode
        # forward replays the graphs as usual.
        eager_mix = any(hi - lo > 1 for _, lo, hi in assigned)
        for slot, lo, hi in assigned:
            if (
                hi - lo == 1
                and self._graph_s_pad is not None
                and not eager_mix
            ):
                # T=1 engine decode: fixed-shape NPU graph capture/replay
                # (6c).  The first step of a request runs eagerly (seeding
                # the fixed caches from the prefill's cat-path state) and
                # captures; later steps replay.
                hidden_rows.append(
                    self._engine_graph_step(
                        slot, input_ids[lo:hi], positions[lo:hi]
                    )
                )
                continue
            ops.set_current_slot(slot)
            seg_ids = input_ids[lo:hi]
            seg_pos = positions[lo:hi]
            if inputs_embeds is not None:
                seg_embeds = inputs_embeds[lo:hi]
            else:
                seg_embeds = self.model.embed_norm(
                    self.model.embed_tokens(seg_ids)
                )
            hidden_rows.append(self.model(seg_pos, seg_embeds))
        return torch.cat(hidden_rows, 0)

    def _assign_slots(
        self,
        segs: list[tuple[int, int]],
        positions: torch.Tensor,
    ) -> list[tuple[int, int, int]]:
        """Map `(lo, hi)` segments to stable state slots (see forward)."""
        out: list[tuple[int, int, int]] = []
        used: set[int] = set()
        new_live: dict[int, tuple[int, int]] = {}
        for order, (lo, hi) in enumerate(segs):
            first = int(positions[lo])
            last = int(positions[hi - 1])
            if first == 0:
                slot = self._slot_seq
                self._slot_seq += 1
                self.model.reset_slot(slot)
                self._graphs.pop(slot, None)
            else:
                cands = [
                    s
                    for s, (nxt, _row) in self._live.items()
                    if nxt == first and s not in used
                ]
                if cands:
                    # Prefer the live slot whose row was closest: batch
                    # compaction shifts rows by small deltas.
                    cands.sort(key=lambda s: abs(self._live[s][1] - order))
                    slot = cands[0]
                else:
                    # Unknown continuation (no matching live slot): treat
                    # as a fresh request to never read foreign state.
                    slot = self._slot_seq
                    self._slot_seq += 1
                    self.model.reset_slot(slot)
                    self._graphs.pop(slot, None)
            used.add(slot)
            new_live[slot] = (last + 1, order)
            out.append((slot, lo, hi))
        for s in self._live:
            if s not in used:
                self.model.reset_slot(s)
                self._graphs.pop(s, None)
        self._live = new_live
        return out

    def _batch_requests(self, positions: torch.Tensor):
        """`(lo, hi)` per non-empty request segment, or None if unavailable.

        Segments come from the common attention metadata's
        `query_start_loc` (the engine token layout is grouped by request,
        adjacent).  Returns None outside the engine (standalone parity
        runs) or when the metadata is missing.
        """
        try:
            from vllm.forward_context import get_forward_context

            fc = get_forward_context()
        except Exception:
            return None
        raw = getattr(fc, "attn_metadata", None)
        if isinstance(raw, dict):
            md = next(iter(raw.values()))
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            md = next(iter(raw[0].values()))
        else:
            md = raw
        if md is None:
            return None
        try:
            qsl = md.query_start_loc_cpu
        except AttributeError:
            return None
        if qsl is None:
            return None
        bounds = qsl.tolist() if not isinstance(qsl, list) else qsl
        num_reqs = len(bounds) - 1
        if num_reqs < 1:
            return None
        segs = []
        for b in range(num_reqs):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            if lo == hi:
                continue  # padded row
            segs.append((lo, hi))
        return segs or None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`VllmModel` protocol: text embeddings (norm applied)."""
        return self.model.embed_norm(self.model.embed_tokens(input_ids))

    def enable_graph_mode(self, s_pad: int) -> None:
        """Opt the decode path into NPU-graph capture mode (6c)."""
        self.model.enable_graph_mode(s_pad)
        self._graph_s_pad = s_pad

    def _graph_buckets(self) -> list[int]:
        """s_pad bucket candidates (6e), ascending, capped at the max."""
        raw = os.environ.get(
            "INKLING2_NPU_GRAPH_BUCKETS", "256,512,1024,2048"
        )
        cap = self._graph_cap or self._graph_s_pad or 2048
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            b = int(tok)
            if 0 < b <= cap:
                out.append(b)
        if not out or out[-1] != cap:
            out.append(cap)
        return sorted(set(out))

    def _bucket_for(self, need: int) -> int | None:
        """Smallest bucket that covers `need` rows, or None if too long."""
        buckets = self._graph_buckets()
        for b in buckets:
            if b >= need:
                return b
        return None

    def _grow_graph_bucket(self, slot: int, need: int) -> bool:
        """Re-size a slot's capture bucket after its context outgrew it.

        Moves the live graph-mode K/V rows (including decoded tokens) to a
        bigger bucket and drops the slot's graph so the next decode step
        re-captures.  Returns False when even the max bucket is too small
        (the caller then stays on the eager path).
        """
        b = self._bucket_for(need)
        if b is None:
            return False
        for layer in self.model.layers:
            layer.attn.resize_graph_caches(slot, b)
        self.model.enable_graph_mode(b)
        self._graph_s_pad = b
        self._graphs.pop(slot, None)
        return True

    def _engine_graph_step(
        self, slot: int, ids_t: torch.Tensor, pos_t: torch.Tensor
    ) -> torch.Tensor:
        """One T=1 engine decode step via NPU graph capture/replay.

        First step of a request: eager (seeds the fixed caches from the
        prefill's cat-path state, warms every kernel), then captures the
        identical-shape graph.  Ascend capture only RECORDS (verified: the
        write cursor does not advance), so the graph is good for the next
        steps.  Later steps: copy the new token/position into the static
        buffers and replay.
        """
        st = self._graphs.get(slot)
        # 6e s_pad bucketing: capture at the smallest bucket that covers
        # this request's context (short chats then pay the 256-row padded
        # softmax, not the 2048-row one).  On growth, the slot's live K/V is
        # moved up a bucket and the graph re-captured.
        need = int(pos_t[0]) + 1
        if st is not None and need > st["s_pad"]:
            if not self._grow_graph_bucket(slot, need):
                raise RuntimeError(
                    f"context {need} exceeds the max graph s_pad "
                    f"({self._graph_s_pad}); raise INKLING2_NPU_GRAPH_S_PAD "
                    "to at least max_model_len"
                )
            st = None
        if st is None:
            b = self._bucket_for(need)
            if b is None:
                raise RuntimeError(
                    f"context {need} exceeds the max graph s_pad "
                    f"({self._graph_s_pad}); raise INKLING2_NPU_GRAPH_S_PAD "
                    "to at least max_model_len"
                )
            if self._graph_s_pad != b:
                # Only new cache allocations use the new bucket; other
                # slots keep their buffers and their captured graphs
                # replay unchanged.
                self.model.enable_graph_mode(b)
                self._graph_s_pad = b
            ops.set_current_slot(slot)
            embeds = self.model.embed_norm(self.model.embed_tokens(ids_t))
            hidden = self.model(pos_t, embeds)
            g = torch.npu.NPUGraph()
            ids_buf = ids_t.clone()
            pos_buf = pos_t.clone()
            with torch.npu.graph(g):
                hidden_s = self.decode_step(ids_buf, pos_buf, slot)
            self._graphs[slot] = {
                "g": g,
                "ids": ids_buf,
                "pos": pos_buf,
                "hidden": hidden_s,
                "s_pad": b,
            }
            return hidden
        st["ids"].copy_(ids_t)
        st["pos"].copy_(pos_t)
        st["g"].replay()
        return st["hidden"]

    def decode_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot: int = 0,
    ) -> torch.Tensor:
        """Sync-free T=1 decode forward, the graph-capture entry point.

        Embeddings + loop core only -- no reset / metadata / batching
        logic, no host reads of device tensors, so the whole call can be
        recorded by `torch.npu.NPUGraph` and replayed per decode step.
        """
        ops.set_current_slot(slot)
        embeds = self.model.embed_norm(self.model.embed_tokens(input_ids))
        return self.model(positions, embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The sampler's logits: fp32 lm_head (see `forward_logits`)."""
        w = self.lm_head.weight
        fp32 = getattr(self, "_lm_fp32", None)
        if fp32 is None or fp32.device != w.device:
            self._lm_fp32 = w.detach().float()
        out = torch.nn.functional.linear(hidden_states.float(), self._lm_fp32)
        import os

        dump = os.getenv("INKLING2_DUMP_LOGITS")
        if dump:
            top = torch.topk(out[-1].float(), 5)
            with open(dump, "a") as f:
                f.write(
                    f"T={hidden_states.shape[0]} "
                    + repr([int(t) for t in top.indices.tolist()])
                    + " "
                    + repr([round(v, 4) for v in top.values.tolist()])
                    + "\n"
                )
        return out

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        incremental: bool = False,
    ) -> torch.Tensor:
        """Standalone single-sequence path (offline parity runs).

        `incremental=False` (default): one independent single-sequence run —
        the KV / conv-state caches are reset first. `incremental=True`:
        a decode continuation step — the caches (keyed by loop pass) persist,
        so calling with the prefill batch once and then single tokens
        reproduces the full-sequence forward token by token.
        """
        ops.set_current_slot(0)
        if not incremental:
            self.reset_state()
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        embeds = self.model.embed_norm(self.model.embed_tokens(input_ids))
        hidden = self.model(positions, embeds)
        return self.compute_logits(hidden)


__all__ = ["Inkling2LoopForConditionalGeneration"]
