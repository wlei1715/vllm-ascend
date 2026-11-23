# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from collections.abc import Mapping
from typing import Any

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import KVCacheGroup
from vllm_ascend.utils import vllm_version_is


_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


def make_kimi_cache_groups(
    full_history_specs: Mapping[str, Any],
    recurrent_specs: Mapping[str, Any],
    *,
    recurrent_group_size: int,
    uniform_recurrent: bool,
) -> list[KVCacheGroup] | None:
    """Share full-history ownership and balance recurrent groups by slot budget.

    Callers validate roles and geometry, and supply specs in layer order with
    target attention first for scheduler conversion. The recurrent slot budget
    is explicit: K3 uses target plus draft pages, whereas K3.1 overlays only
    target MLA pages, excluding both indexer and draft storage.

    K3 retains per-layer recurrent specs for its shared-tuple allocation;
    K3.1 keeps native merged Mamba specs for its paired-page allocation.
    """
    names = list(recurrent_specs)
    count = cdiv(len(names), recurrent_group_size)
    recurrent_layers = [names[i::count] for i in range(count)]
    if vllm_version_is("0.33.0"):
        from vllm.v1.core.kv_cache_planning import create_kv_cache_group_specs
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

        full_history = UniformTypeKVCacheSpecs.from_specs(dict(full_history_specs))
        if full_history is None:
            return None
        groups = [KVCacheGroupSpec(list(full_history_specs), full_history)]
        if uniform_recurrent:
            for layer_names in recurrent_layers:
                group_specs = {name: recurrent_specs[name] for name in layer_names}
                uniform = UniformTypeKVCacheSpecs.from_specs(group_specs)
                assert uniform is not None
                groups.append(KVCacheGroupSpec(layer_names, uniform))
        elif recurrent_layers:
            groups.extend(create_kv_cache_group_specs(dict(recurrent_specs), recurrent_layers))
        return groups

    first = next(iter(full_history_specs.values()))
    if any(
        lc.manager_config != first.manager_config
        or lc.spec.block_size != first.spec.block_size
        or lc.spec.dcp_sharded != first.spec.dcp_sharded
        for lc in full_history_specs.values()
    ):
        return None
    groups = [KVCacheGroup.from_layers(full_history_specs)]
    for layer_names in recurrent_layers:
        groups.append(KVCacheGroup.from_layers({name: recurrent_specs[name] for name in layer_names}))
    return groups


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: Mapping[str, Any],
) -> list[KVCacheGroup] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one group lets them share one scheduler
    block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    if vllm_version_is("0.33.0"):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        def is_full_attn(entry) -> bool:
            return isinstance(entry, FullAttentionSpec)

        def is_mamba(entry) -> bool:
            return isinstance(entry, MambaSpec)

        def get_spec(entry):
            return entry
    else:

        def is_full_attn(entry) -> bool:
            return issubclass(entry.manager_cls, FullAttentionManager)

        def is_mamba(entry) -> bool:
            return isinstance(entry.spec, MambaSpec)

        def get_spec(entry):
            return entry.spec

    target_attention_specs = {
        name: entry
        for name, entry in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and is_full_attn(entry)
    }
    draft_attention_specs = {
        name: entry
        for name, entry in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and is_full_attn(entry)
    }
    mamba_specs = {
        name: entry
        for name, entry in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and is_mamba(entry)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [get_spec(e) for e in (*target_attention_specs.values(), *draft_attention_specs.values())]
    all_specs = [*attention_specs, *(get_spec(e) for e in mamba_specs.values())]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.dcp_sharded for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_entry = next(iter(mamba_specs.values()))
    if any(entry != first_mamba_entry for entry in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    groups = make_kimi_cache_groups(
        mixed_attention_specs,
        mamba_specs,
        recurrent_group_size=len(mixed_attention_specs),
        uniform_recurrent=True,
    )
    if groups is None:
        return None

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups
