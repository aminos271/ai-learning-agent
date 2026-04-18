from typing import Any


def normalize_search_filter(
    concept: str | None = None,
    note_type: str | None = None,
    save_mode: str | None = None,
    source: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """初始化filter"""
    normalized = dict(metadata_filter or {})
    if concept:
        normalized["concept"] = concept
    if note_type:
        normalized["note_type"] = note_type
    if save_mode:
        normalized["save_mode"] = save_mode
    if source:
        normalized["source"] = source
    return normalized


def split_search_filters(
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    拆成两个，其中对于vector_filter去除了concept，
    防止查找时用了还没仔细规划过的定义而导致的查找失败
    """
    rerank_filter = dict(metadata_filter or {})
    vector_filter = dict(rerank_filter)
    vector_filter.pop("concept", None)
    return vector_filter, rerank_filter

