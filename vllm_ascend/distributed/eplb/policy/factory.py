# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.distributed.eplb.policy import AbstractEplbPolicy, DefaultEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy


def create_eplb_policy(policy: str | None, stair_config: StairConfig) -> AbstractEplbPolicy:
    """Create the configured Ascend EPLB policy.

    ``None`` selects the Ascend default ("stair"): vllm main after the EPLB
    policy refactor leaves the platform default unset (None) in the config
    and validates it on the platform instead, so resolve it here.
    """
    if policy is None:
        policy = "stair"
    if policy == "default":
        return DefaultEplbPolicy()
    if policy == "stair":
        return StairEplbPolicy(stair_config)
    raise ValueError(f"Unsupported EPLB policy: {policy!r}")
