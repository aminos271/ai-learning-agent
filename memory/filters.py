from typing import Any, Dict, Optional, Tuple


def normalize_search_filter(
    concept: Optional[str] = None,
    note_type: Optional[str] = None,
    save_mode: Optional[str] = None,
    source: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rerank_filter = dict(metadata_filter or {})
    vector_filter = dict(rerank_filter)
    vector_filter.pop("concept", None)
    return vector_filter, rerank_filter


def build_rerank_metadata_filter(
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not metadata_filter:
        return {}
    return dict(metadata_filter)


__all__ = [
    "normalize_search_filter",
    "split_search_filters",
    "build_rerank_metadata_filter",
]
