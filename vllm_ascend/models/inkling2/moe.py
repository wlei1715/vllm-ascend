# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE, NPU correctness version (Ascend 910B).

Reference math is `tests/models/inkling2/test_moe.py` (authoritative):

* Routing (``_torch_select_experts``):
    logits = x @ gate.weight.T          (bf16 GEMM, fp32 accumulation)
    ids    = topk(sigmoid(logits) + gate.bias, topk, sorted=True)
    w      = softmax(logsigmoid(logits)[ids]) * route_scale
  No softmax over the full expert set — renormalized *sigmoids* of the
  selected experts only.

* Expert FFN (``_moe_reference``):  w13 rows are INTERLEAVED —
    gate_up = x @ w13.T               # [.., 2*inter]
    gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
  i.e. w13 row 2i is gate_i and row 2i+1 is up_i (NOT a contiguous
  first-half/second-half split).  Activation with softcaps:
    act = gate_cap*tanh(gate/gate_cap)*sigmoid(gate)
        * up_cap*tanh(up/up_cap)
  The routing weight is folded in before the down projection:
    out += (act*w) @ w2.T

This module runs the BF16 checkpoint (offline-dequantized); quantized
checkpoints are rejected with a clear error.
"""

from __future__ import annotations

import torch
from torch import nn


class Inkling2NpuGate(nn.Module):
    """Top-k router: renormalized sigmoid probabilities scaled by
    `route_scale`."""

    def __init__(self, config) -> None:
        super().__init__()
        num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.use_bias = getattr(config, "use_gate_bias", True)
        self.weight = nn.Parameter(
            torch.empty(num_experts, config.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32), requires_grad=False
            )
            if self.use_bias
            else None
        )

    @staticmethod
    def _load_weight(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype
        param.data.copy_(loaded)

    @staticmethod
    def _load_bias(param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.to(torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """bf16 `(T, H)` -> fp32 routing weights `(T, topk)`, ids `(T, topk)`."""
        logits = torch.nn.functional.linear(x, self.weight).float()
        keys = logits.sigmoid()
        if self.bias is not None:
            keys = keys + self.bias
        # torch.topk returns lower indices first on ties, matching the
        # "breaks ties low" contract of the Triton selection.
        ids = torch.topk(keys, self.topk, dim=-1, sorted=True).indices
        selected = torch.nn.functional.logsigmoid(logits.gather(1, ids))
        weights = selected.softmax(-1) * self.route_scale
        return weights, ids


class Inkling2NpuExperts(nn.Module):
    """BF16 experts on Ascend grouped GEMM.

    Weights are held in `[E, K, N]` (in, out) orientation because
    `torch_npu.npu_grouped_matmul` requires that layout — the checkpoint
    stores `[E, N, K]`, so `load_weights` transposes on the way in (memory
    neutral: the parameter replaces the shard rather than adding a copy).

    One grouped GEMM covers all active experts per projection, using a
    per-expert `group_list` of token counts (0 for inactive experts), so no
    weight gather/copy per call is needed.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts, config.hidden_size, 2 * config.intermediate_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_experts, config.intermediate_size, config.hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        # Per-expert token counts are reused every call; `_group_list` is
        # filled in place to avoid an allocation + H2D copy per step.
        self._group_list: torch.Tensor | None = None
        self._gmm_available: bool | None = None
        # 6e: cached constant device buffers — `arange` and `ones` launch a
        # kernel per call otherwise (values never change for a given shape).
        self._flat_tok: torch.Tensor | None = None
        self._ones: torch.Tensor | None = None

    @staticmethod
    def _load_weight(param: nn.Parameter, loaded: torch.Tensor) -> None:
        """Checkpoint stores `[E, N, K]`; the parameter holds `[E, K, N]`."""
        expected = loaded.shape[:1] + (loaded.shape[2], loaded.shape[1])
        assert param.shape == expected and loaded.dtype == param.dtype, (
            f"{loaded.shape} -> {expected} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded.transpose(1, 2))

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        """bf16 `(T, H)` x, fp32 `(T, topk)` weights, `(T, topk)` ids ->
        bf16 `(T, H)`."""
        T, K = ids.shape
        flat_ids = ids.reshape(-1)
        flat_tok = self._flat_tok
        if flat_tok is None or flat_tok.shape[0] != T * K:
            flat_tok = (
                torch.arange(T, device=x.device).unsqueeze(1).expand(T, K).reshape(-1)
            )
            self._flat_tok = flat_tok
        order = torch.argsort(flat_ids, stable=True)
        s_ids = flat_ids[order]
        s_tok = flat_tok[order]
        s_w = weights.reshape(-1)[order]
        n = s_ids.numel()
        if n == 0:
            return torch.zeros_like(x)

        if self._gmm_available is None:
            # NOTE (6c): `npu_grouped_matmul` lives at the `torch_npu` TOP
            # LEVEL, and `torch.npu` IS `torch_npu.npu` -- so probing
            # `hasattr(torch.npu, "npu_grouped_matmul")` is always False and
            # every run silently took the slow per-expert fallback (48
            # nonzero/tolist host syncs per decode step).  Probe the module
            # that actually owns the symbol.
            try:
                import torch_npu

                self._gmm_available = hasattr(
                    torch_npu, "npu_grouped_matmul"
                )
            except ImportError:
                self._gmm_available = False

        if self._gmm_available:
            xp = x[s_tok].contiguous()
            gl = self._group_list
            if gl is None or gl.numel() != self.num_experts:
                gl = torch.zeros(
                    self.num_experts, dtype=torch.int64, device=x.device
                )
                self._group_list = gl
            gl.zero_()
            ones = self._ones
            if ones is None or ones.shape[0] != n or ones.device != s_ids.device:
                ones = torch.ones(n, dtype=s_ids.dtype, device=s_ids.device)
                self._ones = ones
            gl.index_add_(0, s_ids, ones)
            from torch_npu import npu_grouped_matmul

            gate_up = npu_grouped_matmul(
                [xp],
                weight=[self.w13_weight],
                group_list=gl,
                group_list_type=1,
                split_item=2,
                group_type=0,
            )[0]
            token = s_tok
            sw = s_w
        else:
            bounds = (s_ids[1:] != s_ids[:-1]).nonzero().flatten().tolist()
            starts = [0] + [b + 1 for b in bounds]
            ends = [b + 1 for b in bounds] + [n]
            chunks, toks, ws = [], [], []
            for s, e in zip(starts, ends):
                expert = int(s_ids[s])
                tok = s_tok[s:e]
                gu = torch.nn.functional.linear(
                    x[tok], self.w13_weight[expert].t()
                )
                chunks.append(gu)
                toks.append(tok)
                ws.append(s_w[s:e])
            if not chunks:
                return torch.zeros_like(x)
            gate_up = torch.cat(chunks, 0)
            token = torch.cat(toks, 0)
            sw = torch.cat(ws, 0)

        # w13 rows are interleaved: even rows gate, odd rows up.  One fp32 cast of
        # the whole gate_up (6e) instead of four strided per-half casts —
        # elementwise casts, values identical.
        gu32 = gate_up.float()
        gate, up = gu32[:, 0::2], gu32[:, 1::2]
        act = (
            self.gate_cap * torch.tanh(gate / self.gate_cap)
            * torch.sigmoid(gate)
            * self.up_cap * torch.tanh(up / self.up_cap)
        )
        act = (act * sw[:, None]).to(torch.bfloat16)

        if self._gmm_available:
            from torch_npu import npu_grouped_matmul

            down = npu_grouped_matmul(
                [act.contiguous()],
                weight=[self.w2_weight],
                group_list=gl,
                group_list_type=1,
                split_item=2,
                group_type=0,
            )[0]
            out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
            out.index_add_(0, token, down.float())
            return out.to(x.dtype)

        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for s, e in zip(starts, ends):
            expert = int(s_ids[s])
            down = torch.nn.functional.linear(
                act[s:e], self.w2_weight[expert].t()
            )
            out.index_add_(0, s_tok[s:e], down.float())
        return out.to(x.dtype)


class Inkling2NpuMoE(nn.Module):
    """`Inkling2MoE` counterpart for the NPU branch: gate + BF16 experts."""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2NpuGate(config)
        self.experts = Inkling2NpuExperts(config)

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 `(T, H)` -> bf16 `(T, H)`."""
        weights, ids = self.gate(x)
        return self.experts(x.contiguous(), weights, ids)