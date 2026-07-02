from __future__ import annotations

from streaming.backend.selector.cache import (
    build_seen_view_pool,
    build_selected_runtime_view_spec,
    make_view_condition_cache_state,
    to_jsonable,
    tree_shape_spec,
)
from streaming.backend.selector.selector import (
    Stage1SelectionResult,
    Stage2SelectionResult,
    ViewConditionCacheConfig,
    ViewConditionCacheState,
    ViewConditionSelectionMode,
    ViewConditionSelector,
)


def build_view_condition_selector(
    config: ViewConditionCacheConfig,
    state: ViewConditionCacheState,
) -> ViewConditionSelector:
    from streaming.backend.selector.selector_vote import TokenVoteSelector

    if config.selection_mode is ViewConditionSelectionMode.TOKEN_VOTE:
        return TokenVoteSelector(config, state)
    raise ValueError(
        f"Unsupported view-condition selection mode: {config.selection_mode}"
    )


__all__ = [
    "Stage2SelectionResult",
    "Stage1SelectionResult",
    "ViewConditionSelector",
    "build_view_condition_selector",
    "build_seen_view_pool",
    "build_selected_runtime_view_spec",
    "make_view_condition_cache_state",
    "to_jsonable",
    "tree_shape_spec",
]
