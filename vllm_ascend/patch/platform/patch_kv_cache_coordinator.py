# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from math import lcm

import vllm
import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheSpec,
    MambaSpec,
)

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from vllm.v1.core.kv_cache_coordinator import SpecGroup
    from vllm.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
else:
    from vllm.v1.core.kv_cache_coordinator import eagle_proof_margin
    from vllm.v1.core.single_type_kv_cache_manager import create_single_type_manager
    from vllm.v1.kv_cache_interface import PrefixLookupGroup, prefix_lookup_groups

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _skips_eagle_block_drop(kv_transfer_config) -> bool:
    """Whether the EAGLE last-block drop must be suppressed on this process.

    Suppressed for a pure PD prefill producer and for a standalone instance.
    Consumers and ``kv_both`` instances keep upstream behavior.
    """
    return kv_transfer_config is None or (
        getattr(kv_transfer_config, "is_kv_producer", False)
        and not getattr(kv_transfer_config, "is_kv_consumer", False)
    )


def _select_kv_token_budget(max_model_len: int, max_in_flight_tokens: int | None) -> int:
    return max_in_flight_tokens if max_in_flight_tokens is not None else max_model_len


def _is_deepseek_v4_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    if getattr(kv_cache_spec, "model_version", None) in {"deepseek_v4", "deepseek_v41"}:
        return True

    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if nested_specs is None:
        return False

    if isinstance(nested_specs, Mapping):
        nested_specs = nested_specs.values()
    elif not isinstance(nested_specs, (list, tuple, set)):
        return False

    return any(getattr(spec, "model_version", None) in {"deepseek_v4", "deepseek_v41"} for spec in nested_specs)


def _is_deepseek_v4_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    return any(
        any(_is_deepseek_v4_kv_cache_spec(spec) for spec in group.distinct_specs)
        for group in kv_cache_config.kv_cache_groups
    )


def _manager_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if vllm_version_is("0.33.0") and isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """KV cache coordinator for hybrid models on Ascend."""

    def __init__(  # type: ignore[misc]
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_in_flight_tokens: int | None = None,
        scheduler_block_size: int | None = None,
        num_prefill_lookahead: int = 0,
        num_hash_lookahead_tokens: int = 0,
        allow_partial_hash_hits: bool = True,
    ):
        del pcp_world_size, eagle_attn_layer_names
        self.num_reprefillable_tokens = max(0, (num_prefill_lookahead or 0) - 1)
        self.dcp_world_size = dcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.allow_partial_hash_hits = allow_partial_hash_hits
        token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens)
        self.max_in_flight_tokens = token_budget
        self.retention_interval = kv_cache_config.prefix_cache_retention_interval
        validate_retention_interval = getattr(
            vllm_kv_cache_coordinator,
            "_validate_prefix_cache_retention_interval",
            None,
        )
        if self.retention_interval is not None and validate_retention_interval is not None:
            validate_retention_interval(
                self.retention_interval,
                self.scheduler_block_size,
                kv_cache_config,
            )
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {  # type: ignore[no-redef]
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = {
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if issubclass(g.manager_cls, FullAttentionManager) and g.prefix_cacheable
            }
        self.drop_eagle_group_ids = self.eagle_group_ids if use_eagle else set()

        if vllm_version_is("0.33.0"):
            extra_mgr_kwargs: dict = {
                "scheduler_block_size": scheduler_block_size,
                "needs_kv_cache_zeroing": kv_cache_config.needs_kv_cache_zeroing,
            }
            self.single_type_managers = tuple(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=_manager_spec(kv_cache_group.kv_cache_spec),
                    block_pool=self.block_pool,
                    enable_caching=enable_caching,
                    kv_cache_group_id=i,
                    dcp_world_size=(
                        1 if isinstance(_manager_spec(kv_cache_group.kv_cache_spec), MambaSpec) else dcp_world_size
                    ),
                    pcp_world_size=1,
                    max_in_flight_tokens=token_budget,
                    max_model_len=max_model_len,
                    **extra_mgr_kwargs,
                )
                for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
            )
        else:
            self.single_type_managers = tuple(
                create_single_type_manager(
                    kv_cache_group,
                    max_in_flight_tokens=token_budget,
                    max_model_len=max_model_len,
                    block_pool=self.block_pool,
                    enable_caching=enable_caching,
                    kv_cache_group_id=i,
                    dcp_world_size=1 if issubclass(kv_cache_group.manager_cls, MambaManager) else dcp_world_size,
                    pcp_world_size=1,
                    scheduler_block_size=scheduler_block_size,
                    needs_kv_cache_zeroing=kv_cache_config.needs_kv_cache_zeroing,
                )
                for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
            )
            self.group_block_sizes = tuple(manager.block_size for manager in self.single_type_managers)

        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                self._get_effective_block_size(g) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
                if g.prefix_cacheable
            ), "block_size must be divisible by hash_block_size"
        self.enable_partial_hash_hits = allow_partial_hash_hits and any(
            issubclass(g.manager_cls, MambaManager)
            and g.manager_config.params.mamba_cache_mode == "align"
            and (g.block_size > hash_block_size or (dcp_world_size > 1 and g.block_size == hash_block_size))
            for g in kv_cache_config.kv_cache_groups
        )
        if self.enable_partial_hash_hits:
            self.enable_partial_hash_hits = all(
                not g.prefix_cacheable
                or getattr(manager, "block_size", self._get_effective_block_size(g)) == hash_block_size
                or getattr(manager, "supports_fine_grained_hash_lookup", False)
                for manager, g in zip(self.single_type_managers, kv_cache_config.kv_cache_groups)
            )
        self.verify_and_split_kv_cache_groups()

        for mgr in self.single_type_managers:
            mgr.cache_hit_alignment_tokens = self._cache_hit_alignment_tokens

        self.use_eagle = use_eagle
        kv_transfer_config = getattr(kv_cache_config, "kv_transfer_config", None)
        self.skips_eagle_block_drop = _skips_eagle_block_drop(kv_transfer_config)
        # Keep Mamba checkpoint writes on the same boundary as prefill split
        # and cache lookup. Only roles using EAGLE replay need the block drop.
        if use_eagle and not self.skips_eagle_block_drop:
            for group_id, manager in enumerate(self.single_type_managers):
                if isinstance(manager, MambaManager) and group_id in self.eagle_group_ids:
                    manager.drop_eagle_checkpoint_block = True

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        tail_alignment = getattr(self, "tail_pool_alignment", 1)
        if self.enable_partial_hash_hits:
            return lcm(self.hash_block_size, tail_alignment)
        return lcm(self.scheduler_block_size or self.lcm_block_size, tail_alignment)

    def _get_effective_block_size(self, group_or_spec) -> int:
        block_size = group_or_spec.block_size
        if isinstance(group_or_spec, MambaSpec) or (
            hasattr(group_or_spec, "manager_cls") and issubclass(group_or_spec.manager_cls, MambaManager)
        ):
            return block_size
        if group_or_spec.dcp_sharded:
            block_size *= self.dcp_world_size
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        """Group KV cache groups by allocation policy for cache hit lookup."""
        self.tail_pool_alignment = 1
        for g in self.kv_cache_config.kv_cache_groups:
            if not g.prefix_cacheable:
                for tail_spec in g.distinct_specs:
                    self.tail_pool_alignment = lcm(self.tail_pool_alignment, getattr(tail_spec, "compress_ratio", 1))

        if not vllm_version_is("0.33.0"):
            groups = self.kv_cache_config.kv_cache_groups
            self.attention_groups: list[PrefixLookupGroup] = [  # type: ignore[assignment]
                g._replace(use_eagle=any(i in self.drop_eagle_group_ids for i in g.group_ids))
                for g in prefix_lookup_groups(groups)
            ]
            assert self.attention_groups, "Prefix caching requires at least one cacheable KV cache group."
            self.attention_groups.sort(key=lambda g: not issubclass(g.manager_cls, FullAttentionManager))
            first = self.attention_groups[0]
            self.full_attention_group_id: int | None = (
                first.group_ids[0] if issubclass(first.manager_cls, FullAttentionManager) else None
            )
            for group in self.attention_groups:
                if group.use_eagle:
                    for gid in group.group_ids:
                        self.single_type_managers[gid].use_eagle = True
            block_sizes = [self.single_type_managers[group.group_ids[0]].block_size for group in self.attention_groups]
            self.lcm_block_size = lcm(*block_sizes)
            return

        self.attention_groups: list[SpecGroup] = []  # type: ignore[no-redef]
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if not g.prefix_cacheable:
                continue
            manager_cls = self.single_type_managers[i].__class__
            spec = _manager_spec(g.kv_cache_spec)
            use_eagle = i in self.drop_eagle_group_ids

            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls, "Expected same manager class for identical KV cache specs."
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(SpecGroup(spec, [i], manager_cls, use_eagle))

        assert self.attention_groups, "Prefix caching requires at least one cacheable KV cache group."
        self.attention_groups.sort(key=lambda group: not isinstance(group.spec, FullAttentionSpec))
        first = self.attention_groups[0]
        self.full_attention_group_id = first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
        block_sizes = [self._get_effective_block_size(group.spec) for group in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | tuple[tuple[list[KVCacheBlock], ...], int, int]:
        if not vllm_version_is("0.33.0"):
            return self._find_longest_cache_hit_v2(block_hashes, max_cache_hit_length)
        return self._find_longest_cache_hit_v033(block_hashes, max_cache_hit_length)

    def _find_longest_cache_hit_v2(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and issubclass(
            self.attention_groups[0].manager_cls, FullAttentionManager
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (manager_config, block_size, group_ids, use_eagle) in enumerate(self.attention_groups):
                manager_cls = manager_config.manager_cls
                first_group_id = group_ids[0]
                manager = self.single_type_managers[first_group_id]
                group_block_size = manager.block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if issubclass(manager_cls, FullAttentionManager) and cached_blocks is not None:
                    curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified and not self.skips_eagle_block_drop

                _max_length = curr_hit_length
                if drop_eagle_block and not issubclass(manager_cls, MambaManager):
                    eagle_margin = eagle_proof_margin(
                        group_block_size,
                        self.hash_block_size,
                        self.enable_partial_hash_hits and manager_cls.supports_fine_grained_hash_lookup,
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin,
                        len(block_hashes) * self.hash_block_size,
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=manager.block_pool,
                    block_size=block_size,
                    manager_config=manager_config,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=manager.dcp_world_size,
                    pcp_world_size=manager.pcp_world_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        for group in self.attention_groups:
            if not issubclass(group.manager_cls, FullAttentionManager):
                continue
            manager = self.single_type_managers[group.group_ids[0]]
            if getattr(manager, "retains_longer_hit", False):
                continue
            num_blocks = cdiv(hit_length, manager.block_size)
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        cache_hit_blocks = tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
        return cache_hit_blocks, hit_length, longest_hit_length - hit_length

    def _find_longest_cache_hit_v033(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            return block_hashes

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(self.attention_groups):
                group_block_size = self._get_effective_block_size(spec)
                first_group_id = group_ids[0]
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified and not self.skips_eagle_block_drop

                _max_length = curr_hit_length
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.dcp_world_size if spec.dcp_sharded else 1,
                    pcp_world_size=1,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(hit_length, self._get_effective_block_size(group.spec))
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        cache_hit_blocks = tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
        return cache_hit_blocks, hit_length, longest_hit_length - hit_length

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups
        if not vllm_version_is("0.33.0"):
            for manager_config, block_size, group_ids, use_eagle in self.attention_groups:
                manager_cls = manager_config.manager_cls
                manager = self.single_type_managers[group_ids[0]]
                drop_eagle = use_eagle and not self.skips_eagle_block_drop
                lookup_length = max_cache_hit_length
                if drop_eagle and not issubclass(manager_cls, MambaManager):
                    eagle_margin = eagle_proof_margin(
                        manager.block_size,
                        self.hash_block_size,
                        self.enable_partial_hash_hits and manager_cls.supports_fine_grained_hash_lookup,
                    )
                    lookup_length = min(
                        max_cache_hit_length + eagle_margin,
                        len(block_hashes) * self.hash_block_size,
                    )
                blocks, group_hit = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=lookup_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=manager.block_pool,
                    block_size=block_size,
                    manager_config=manager_config,
                    drop_eagle_block=drop_eagle,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=manager.dcp_world_size,
                    pcp_world_size=manager.pcp_world_size,
                )
                for gid, blks in zip(group_ids, blocks):
                    hit_blocks[gid] = blks
                    hit_lengths[gid] = group_hit
            return tuple(hit_blocks), tuple(hit_lengths)

        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle and not self.skips_eagle_block_drop,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=self.dcp_world_size if spec.dcp_sharded else 1,
                pcp_world_size=1,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit
        return tuple(hit_blocks), tuple(hit_lengths)


def get_kv_cache_coordinator(  # type: ignore[misc]
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int | None = None,
    use_eagle: bool = False,
    enable_caching: bool = True,
    enable_kv_cache_events: bool = False,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    scheduler_block_size: int | None = None,
    hash_block_size: int = 0,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
    num_hash_lookahead_tokens: int = 0,
    num_prefill_lookahead: int = 0,
    allow_partial_hash_hits: bool = True,
) -> KVCacheCoordinator:
    del pcp_world_size
    token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens)
    hybrid_kwargs = dict(
        kv_cache_config=kv_cache_config,
        max_model_len=max_model_len,
        use_eagle=use_eagle,
        enable_caching=enable_caching,
        enable_kv_cache_events=enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=1,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_in_flight_tokens=token_budget,
        scheduler_block_size=scheduler_block_size,
        num_prefill_lookahead=num_prefill_lookahead,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        allow_partial_hash_hits=allow_partial_hash_hits,
    )
    if any(group.role == KVCacheGroupRole.HISPARSE_SOURCE for group in kv_cache_config.kv_cache_groups):
        return _orig_get_kv_cache_coordinator(
            **{name: value for name, value in hybrid_kwargs.items() if name != "eagle_attn_layer_names"}
        )
    num_cacheable_groups = sum(group.prefix_cacheable for group in kv_cache_config.kv_cache_groups)
    has_private_groups = 0 < num_cacheable_groups < len(kv_cache_config.kv_cache_groups)
    if num_cacheable_groups and (_is_deepseek_v4_kv_cache_config(kv_cache_config) or has_private_groups):
        return AscendHybridKVCacheCoordinator(**hybrid_kwargs)  # type: ignore[call-arg]

    if num_cacheable_groups == 0 or len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        if num_cacheable_groups == 0 and vllm_version_is("0.33.0"):
            kv_cache_config = copy(kv_cache_config)
            kv_cache_config.kv_cache_groups = [
                replace(group, kv_cache_spec=_manager_spec(group.kv_cache_spec))
                for group in kv_cache_config.kv_cache_groups
            ]
        orig_kwargs = dict(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            enable_caching=enable_caching and num_cacheable_groups > 0,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            max_in_flight_tokens=token_budget,
            scheduler_block_size=scheduler_block_size,
            num_prefill_lookahead=num_prefill_lookahead,
            allow_partial_hash_hits=allow_partial_hash_hits,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        )
        return _orig_get_kv_cache_coordinator(**orig_kwargs)

    return AscendHybridKVCacheCoordinator(**hybrid_kwargs)  # type: ignore[call-arg]


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
