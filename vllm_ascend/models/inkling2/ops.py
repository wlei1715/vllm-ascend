# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference math of the Inkling2 text stack, NPU branch.

Every function is a faithful port of the NVIDIA kernels, written as plain
torch so it runs on Ascend (910B) without triton/CUDA. Source references:

* RMSNorm / q,r L2 clipping: ``nvidia/ops/norm.py`` (``_rms_norm_kernel``,
  ``_l2_clip``, ``qr_clip_rel_tokens``).
* Short convs: ``nvidia/ops/sconv.py`` (``_conv`` / ``_sconv_kernel`` /
  ``_kv_conv_tile``).
* Block attention residual: ``nvidia/ops/attn_res.py`` (``attn_res``).
* Loop message blend: ``nvidia/ops/loop.py`` (``pair_rms_norm`` /
  ``blend_rms_norm``).
* Output gate + attention sink: ``nvidia/ops/attention.py`` (``gate_sink``).

Exact formulas (fp32 unless noted), with W = sconv_kernel_size = 4:

RMSNorm           y = x * rsqrt(mean(x^2) + eps) * w            -> bf16
L2 clip           y = x*s ; y *= min(threshold / (||y||_2 + 1e-6), 1)
                  with tau: y = bf16(y) * tau (fp32), stored bf16
static short conv y_t = (sum_j w[c, j] * x[(t - (W-1-j))] + x_t) * (W+1)^-0.5
dynamic short conv y_t = (sum_k (sw[k,c] + dyn[t,head,k]) * x[t-k] + x_t)
                          * (2W+1)^-0.5
K after conv      y *= min(qk_clip / (||y||_2 + 1e-6), 1)        (per head)
V after conv      y *= rsqrt(mean(y^2 over head_dim) + eps)       (no gamma)
rel bias          rel[t,h,d] = r_clip[t,h,:] @ rel_proj[:, d],  d = pos_t - pos_s
                  (-inf where d < 0, and for window layers d >= extent)
scores            q_clip[t,h,:] . k[s,hkv] * head_dim^-0.5 + rel[t,h,dist]
out               softmax(scores) @ v ; lse = logsumexp(scores)
gate/sink         out *= sigmoid(gate[t,h,d]) * sigmoid(lse[t,h] - sink[t,h])
hidden short conv out_t = (residual + addend) + conv(moe_out)     (fp32 sum)
attn_res          per query q (pre-scaled by D^-0.5):
                  logit_n = (q . x_n) * rsqrt(mean(x_n^2) + 1e-6)
                  p_n = exp(logit_n - max) / sum ; with normalize_values
                  p_n *= rsqrt(mean(x_n^2) + 1e-6)
                  mix = (sum_n p_n x_n) / sum_n p_n
                  with a norm weight: mix *= rsqrt(mean(mix^2) + eps) * w
blend             pair = bf16([dest*rms(dest), msg*rms(msg)])   (eps 1e-6)
                  h = bf16(pair @ down^T) ; [gain, forget] = h @ up^T (fp32)
                  x = (1 - tanh(forget)) * dest + tanh(gain) * msg
                  out = bf16(x * rsqrt(mean(x^2) + eps) * w)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLIP_EPS = 1e-6
ATTN_RES_EPS = 1e-6


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out_dtype=torch.bfloat16
) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(out_dtype)


def l2_clip(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    threshold: float,
    eps: float = CLIP_EPS,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x * scale` clipped to an L2 norm of `threshold`, then `* tau`.

    `x` is `(T, H, D)` bf16, `scale` `(H, D)` fp32/bf16 or None (no scale);
    `tau` fp32 `(T,)` or None.  Returns fp32; the kernels round to bf16 on
    store (callers do).  Pass `scale` pre-converted to fp32 to skip a
    per-call cast.
    """
    y = x.float() if scale is None else x.float() * scale
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    y = y * (threshold / (norm + eps)).clamp(max=1.0)
    if tau is not None:
        # The kernel rounds the clipped value, then scales by fp32 tau.
        y = y.to(torch.bfloat16).float() * tau.view(-1, 1, 1)
    return y


def feature_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS normalization over the last dim (the V conv's norm)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


def short_conv_static(
    x: torch.Tensor, weight: torch.Tensor, past: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv (+ identity) of `x` `(T, C)` bf16.

    `weight` is `(C, W)` (column `j` pairs with lag `W-1-j`, the checkpoint's
    `(C, 1, W)` layout); `past` holds the `W-1` inputs before `x`. Returns
    fp32 `y` and the new `past`.
    """
    w = weight.reshape(weight.shape[0], -1).float()
    W = w.shape[1]
    T, C = x.shape
    if T == 0:
        return x.float(), past if past is not None else x.new_zeros((W - 1, C))
    if past is None:
        past = x.new_zeros((W - 1, C))
    if T == 1:
        # Decode fast path: one elementwise dot over the W-deep window
        # (mathematically identical to the loop below, fewer kernels).
        xt = torch.cat((past.to(x.dtype), x), 0).float()  # (W, C)
        acc = (w.t() * xt).sum(0)
        y = (acc + x.squeeze(0).float()) * (W + 1) ** -0.5
        return y.unsqueeze(0), xt[-(W - 1):].to(x.dtype)
    xt = torch.cat((past.to(x.dtype), x), 0).float()
    acc = torch.zeros(T, C, dtype=torch.float32, device=x.device)
    for j in range(W):
        lag = W - 1 - j
        acc = acc + xt[W - 1 - lag : W - 1 - lag + T] * w[:, j]
    y = (acc + x.float()) * (W + 1) ** -0.5
    return y, xt[-(W - 1) :].to(x.dtype)


def short_conv_dynamic(
    x: torch.Tensor,
    static_w: torch.Tensor,
    dyn: torch.Tensor,
    head_dim: int,
    past: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal conv with per-token taps: `dyn` `(T, heads, W)` deltas on the
    `(W, C)` static part; row `k` pairs with lag `k` (the dynamic layout
    differs from the static one, see `nvidia/ops/sconv.py:131`).
    """
    W, C = static_w.shape
    T = x.shape[0]
    if T == 0:
        return x.float(), past if past is not None else x.new_zeros((W - 1, C))
    if past is None:
        past = x.new_zeros((W - 1, C))
    if T == 1:
        # Decode fast path: row k of the taps pairs with lag k, i.e. the
        # reversed window (mathematically identical to the loop below).
        xt = torch.cat((past.to(x.dtype), x), 0).float()  # (W, C)
        taps = static_w.float() + dyn[0].repeat_interleave(head_dim, dim=0).t()
        acc = (taps * xt.flip(0)).sum(0)
        y = (acc + x.squeeze(0).float()) * (2 * W + 1) ** -0.5
        return y.unsqueeze(0), xt[-(W - 1):].to(x.dtype)
    xt = torch.cat((past.to(x.dtype), x), 0).float()
    dyn_exp = dyn.float().repeat_interleave(head_dim, dim=1)  # (T, C, W)
    acc = torch.zeros(T, C, dtype=torch.float32, device=x.device)
    for k in range(W):
        acc = acc + xt[W - 1 - k : W - 1 - k + T] * (
            static_w[k].float() + dyn_exp[..., k]
        )
    y = (acc + x.float()) * (2 * W + 1) ** -0.5
    return y, xt[-(W - 1) :].to(x.dtype)


def rel_bias(
    r_clip: torch.Tensor, rel_proj: torch.Tensor, extent: int
) -> torch.Tensor:
    """`(T, H, d_rel)` clipped r x `(d_rel, extent)` projection -> `(T, H,
    extent)` fp32 relative logits indexed by key distance."""
    return torch.einsum("thc,ce->the", r_clip.float(), rel_proj.float())


def gate_sink(
    out: torch.Tensor, gate: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """`(T,H,D)` attention output * sigmoid(gate) * sigmoid(lse - sink)."""
    return out * torch.sigmoid(gate.float()) * torch.sigmoid(
        (lse.transpose(0, 1) - sink).unsqueeze(-1).float()
    )


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = ATTN_RES_EPS,
) -> torch.Tensor:
    """Softmax mix of `[*stack, last (+ addend)]` per pseudo-query.

    `queries` are fp32 `(Q, D)` (pre-scaled by `D**-0.5`). Returns `(Q, T,
    D)` bf16 with `norm_weight` `(Q, D)`, else the fp32 mixes.
    """
    sources = [s.float() for s in stack]
    tail = last.float() if addend is None else last.float() + addend.float()
    x = torch.stack([*sources, tail], 0)  # (N, T, D)
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)  # (N, T, 1)
    rstd_t = rstd.squeeze(-1).unsqueeze(1)  # (N, 1, T) for the (N, Q, T) probs
    q = queries.float()
    logits = torch.einsum("qd,ntd->nqt", q, x) * rstd_t
    probs = torch.exp(logits - logits.amax(0, keepdim=True))
    inv_sum = 1.0 / probs.sum(0)  # (Q, T)
    if normalize_values:
        probs = probs * rstd_t
    mix = torch.einsum("nqt,ntd->qtd", probs, x) * inv_sum.unsqueeze(-1)
    assert mix.shape == (q.shape[0], x.shape[1], x.shape[2]), mix.shape
    if norm_weight is None:
        return mix
    mix = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + norm_eps)
    return (mix * norm_weight.float().unsqueeze(1)).to(torch.bfloat16)


def attn_res_view(
    x_view: torch.Tensor,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    tail: torch.Tensor | None = None,
    tail_add: torch.Tensor | None = None,
    eps: float = ATTN_RES_EPS,
) -> torch.Tensor:
    """`attn_res` on a CONTIGUOUS bf16 view holding every source and the
    last row (6e op-count optimization; values are bit-identical to
    `attn_res([*x_view[:-1]], x_view[-1], ...)`: same casts, same ops).

    `x_view` is `(N, T, D)` bf16 — e.g. the loop `stack[:n]` slice whose
    rows are the attention-residual sources and whose last row is `last`.
    `tail` REPLACES the cast copy of the last row (the caller has it in
    fp32 already); `tail_add` is ADDED to it (the `last + addend` case).
    """
    x = x_view.float()
    if tail is not None:
        x[-1] = tail
    elif tail_add is not None:
        x[-1] += tail_add
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)  # (N, T, 1)
    rstd_t = rstd.squeeze(-1).unsqueeze(1)  # (N, 1, T)
    q = queries.float()
    logits = torch.einsum("qd,ntd->nqt", q, x) * rstd_t
    probs = torch.exp(logits - logits.amax(0, keepdim=True))
    inv_sum = 1.0 / probs.sum(0)  # (Q, T)
    if normalize_values:
        probs = probs * rstd_t
    mix = torch.einsum("nqt,ntd->qtd", probs, x) * inv_sum.unsqueeze(-1)
    if norm_weight is None:
        return mix
    mix = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + norm_eps)
    return (mix * norm_weight.float().unsqueeze(1)).to(torch.bfloat16)


def blend_fast(
    dest: torch.Tensor,
    message: torch.Tensor,
    down_t: torch.Tensor,
    up_t32: torch.Tensor,
    norm_weight32: torch.Tensor,
    norm_eps: float,
    eps: float = ATTN_RES_EPS,
) -> torch.Tensor:
    """`blend` on PRE-TRANSPOSED constant weights (6e): `down_t` is
    `(S, 2D, r)` bf16 (`down.transpose(1, 2)` made contiguous at load) and
    `up_t32` is `(S, r, 2D)` fp32 (`up.transpose(1, 2).contiguous().float()`
    at load), `norm_weight32` fp32.  Kills the per-call stack/transpose/cast
    of constant controller weights; the arithmetic is bit-identical.
    """
    d_rstd = torch.rsqrt(dest.square().mean(-1, keepdim=True) + eps)
    m_rstd = torch.rsqrt(message.square().mean(-1, keepdim=True) + eps)
    pair = torch.cat((dest * d_rstd, message * m_rstd), -1).to(torch.bfloat16)
    hidden = torch.bmm(pair, down_t)  # bf16 (S, T, r)
    gf = torch.bmm(hidden.float(), up_t32)  # fp32 (S, T, 2D)
    d = dest.shape[-1]
    gain, forget = gf[..., :d], gf[..., d:]
    x = (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    return (x * norm_weight32.unsqueeze(1)).to(torch.bfloat16)


def blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    eps: float = ATTN_RES_EPS,
) -> torch.Tensor:
    """Gated low-rank blend of fp32 `(S, T, D)` `dest`/`message`.

    `down` `(S, r, 2D)`, `up` `(S, 2D, r)` = `[gain; forget]` rows, `norm_weight`
    `(S, D)`. Returns bf16 `(S, T, D)`.
    """
    d_rstd = torch.rsqrt(dest.square().mean(-1, keepdim=True) + eps)
    m_rstd = torch.rsqrt(message.square().mean(-1, keepdim=True) + eps)
    pair = torch.cat((dest * d_rstd, message * m_rstd), -1).to(torch.bfloat16)
    hidden = torch.bmm(pair, down.transpose(1, 2))  # bf16 (S, T, r)
    gf = torch.bmm(hidden.float(), up.transpose(1, 2).float())  # fp32 (S, T, 2D)
    d = dest.shape[-1]
    gain, forget = gf[..., :d], gf[..., d:]
    x = (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    return (x * norm_weight.float().unsqueeze(1)).to(torch.bfloat16)


def log_scaling_tau(
    positions: torch.Tensor, n_floor: int | None, alpha: float
) -> torch.Tensor | None:
    """`nvidia/ops/attention.py:356`: `1 + alpha * log(max((p+1)/n_floor, 1))`."""
    if n_floor is None:
        return None
    ratio = ((positions.float() + 1.0) / n_floor).clamp(min=1.0)
    return 1.0 + alpha * torch.log(ratio)


def expert_activation(
    gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """SwiGLU with both softcaps (fp32 in, fp32 out)."""
    return (
        gate_cap * torch.tanh(gate / gate_cap) * torch.sigmoid(gate)
        * up_cap * torch.tanh(up / up_cap)
    )


# ---------------------------------------------------------------------------
# Per-request state slot.
#
# The engine batches tokens of several requests into one forward; the
# self-managed per-pass KV / conv caches must therefore be keyed per
# request too.  The slot is the v1 request-table index (`req_idx` from the
# common attention metadata): stable for a request's lifetime, reused only
# after the request finished (and every reuse starts with a prefill at
# position 0, which resets the slot -- see loop.py).  Standalone use keeps
# slot 0.  TP=1 executes the request segments sequentially in Python, so a
# module-global "current slot" is race-free.

_current_slot = 0


def set_current_slot(slot: int) -> None:
    global _current_slot
    _current_slot = int(slot)


def current_slot() -> int:
    return _current_slot


__all__ = [
    "attn_res",
    "attn_res_view",
    "blend",
    "blend_fast",
    "expert_activation",
    "feature_rms",
    "gate_sink",
    "l2_clip",
    "log_scaling_tau",
    "rel_bias",
    "current_slot",
    "rms_norm",
    "set_current_slot",
    "short_conv_dynamic",
    "short_conv_static",
]
