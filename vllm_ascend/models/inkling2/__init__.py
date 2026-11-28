# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NPU (Ascend 910B) platform implementation, hosted in vllm-ascend.

Migrated from the vllm fork's `vllm/models/inkling2/npu/` (adaptor branch,
commits 11603b1a7d + 6621c62bed + f0a63c1cbd) per the upstream convention
that platform implementations live in the platform plugin (vllm-ascend),
not in vllm core.  Registered as the Ascend override for the
``Inkling2LoopForConditionalGeneration`` architecture via
``vllm_ascend/models/__init__.py::register_model``.

Correctness-first torch composition: text-only, TP=1, no MTP, no vision.
Expert weights are the offline-dequantized BF16 checkpoint (the Ascend cube
unit has no FP4/FP8 data paths, see A1 of the adaptation notes).  MoE runs
as a per-expert BF16 GEMM loop with the exact routing / SwiGLU math of the
NVIDIA reference (`tests/models/inkling2/test_moe.py::_moe_reference`).
"""

from .loop import Inkling2LoopForConditionalGeneration
from .moe import Inkling2NpuMoE

__all__ = ["Inkling2LoopForConditionalGeneration", "Inkling2NpuMoE"]