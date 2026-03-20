import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.base_retriever import BaseQdrantStore

from .filters import build_rerank_metadata_filter
from .schemas import MemoryScoreBreakdown, RetrievedItem


GENERIC_CONCEPTS = {
    "",
    "general",
    "default",
    "note",
    "notes",
    "memo",
    "memory",
    "记忆",
    "笔记",
    "记录",
    "知识",
    "内容",
}


KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "note",
    "notes",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "with",
    "一下",
    "一个",
    "一些",
    "上次",
    "之前",
    "关于",
    "刚刚",
    "刚才",
    "可以",
    "可能",
    "告诉",
    "如何",
    "怎么",
    "我们",
    "我的",
    "找到",
    "有关",
    "最近",
    "有什么",
    "笔记",
    "记录",
    "这次",
    "这个",
    "那个",
}


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_keywords(text: Optional[str]) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    keywords: List[str] = []

    for token in re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", normalized):
        if token not in KEYWORD_STOPWORDS:
            keywords.append(token)

    for block in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(block) < 2:
            continue

        if len(block) <= 8 and block not in KEYWORD_STOPWORDS:
            keywords.append(block)

        for size in (2, 3):
            if len(block) < size:
                continue
            for start in range(len(block) - size + 1):
                token = block[start : start + size]
                if token not in KEYWORD_STOPWORDS:
                    keywords.append(token)

    return list(dict.fromkeys(keywords))


def compute_keyword_score(query: str, content: str) -> float:
    keywords = extract_keywords(query)
    if not keywords:
        return 0.0

    normalized_content = normalize_text(content)
    if not normalized_content:
        return 0.0

    matched_keywords = 0
    total_hits = 0
    for keyword in keywords:
        hit_count = normalized_content.count(keyword)
        if hit_count <= 0:
            continue
        matched_keywords += 1
        total_hits += min(hit_count, 3)

    if matched_keywords <= 0:
        return 0.0

    coverage = matched_keywords / len(keywords)
    density = min(total_hits / len(keywords), 1.0)
    return max(0.0, min(coverage * 0.7 + density * 0.3, 1.0))


def is_high_confidence_concept(concept: Optional[str]) -> bool:
    candidate = (concept or "").strip()
    if not candidate:
        return False

    normalized = candidate.lower()
    if normalized in GENERIC_CONCEPTS:
        return False

    if len(candidate) < 2 or len(candidate) > 32:
        return False

    keyword_count = len(extract_keywords(candidate))
    return keyword_count > 0


def maybe_expand_query_with_concept(
    query: str,
    metadata_filter: Optional[Dict[str, object]] = None,
) -> Tuple[str, Optional[str]]:
    rerank_filter = build_rerank_metadata_filter(metadata_filter)
    concept = str(rerank_filter.get("concept") or "").strip()
    if not is_high_confidence_concept(concept):
        return query, None

    normalized_query = normalize_text(query)
    normalized_concept = normalize_text(concept)
    if not normalized_concept or normalized_concept in normalized_query:
        return query, None

    query_keywords = set(extract_keywords(query))
    concept_keywords = set(extract_keywords(concept))
    overlap = query_keywords & concept_keywords
    is_short_query = len(query.strip()) <= 18
    has_recall_marker = any(marker in query for marker in ("这个", "那个", "它", "相关", "之前", "上次"))

    if overlap or is_short_query or has_recall_marker:
        return f"{query} {concept}".strip(), concept

    return query, None


def resolve_candidate_k(top_k: int) -> int:
    return max(top_k * 4, 12)


def parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None

    try:
        if timestamp.endswith("Z"):
            timestamp = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return None


def compute_recency_score(timestamp: Optional[str]) -> float:
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return 0.0

    now = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    age_seconds = max((now - parsed).total_seconds(), 0.0)
    one_day = 24 * 60 * 60
    return 1.0 / (1.0 + age_seconds / one_day)


def compute_memory_score(
    store: BaseQdrantStore,
    question: str,
    question_vector: List[float],
    item: RetrievedItem,
) -> MemoryScoreBreakdown:
    semantic_score = float(item.retrieval_meta.get("similarity", 0.0) or 0.0)
    if semantic_score <= 0 and item.retrieval_meta.get("distance") is not None:
        semantic_score = 1.0 - float(item.retrieval_meta["distance"])

    if semantic_score <= 0:
        doc_vector = store.embeddings.embed_documents([item.content])[0]
        semantic_score = store._cosine_similarity(question_vector, doc_vector)
    semantic_score = max(0.0, min(semantic_score, 1.0))

    keyword_score = compute_keyword_score(question, item.content)
    importance = float(item.metadata.get("importance", 0.0) or 0.0)
    importance_score = max(0.0, min(importance, 1.0))
    recency_score = compute_recency_score(item.metadata.get("timestamp"))

    final_score = (
        semantic_score * 0.45
        + keyword_score * 0.25
        + importance_score * 0.2
        + recency_score * 0.1
    )

    print("SCORE:", semantic_score, keyword_score, importance_score, recency_score)

    return MemoryScoreBreakdown(
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        importance_score=importance_score,
        recency_score=recency_score,
        final_score=final_score,
    )


def debug_rerank_result(ranked: List[Dict[str, object]], top_k: int):
    if not ranked:
        print("📊 Memory rerank: 没有候选结果。")
        return

    print("📊 Memory rerank 结果:")
    for index, item in enumerate(ranked[:top_k], start=1):
        print(
            f"  {index}. note='{item['summary']}' | "
            f"semantic={item['semantic_score']:.4f} | "
            f"keyword={item['keyword_score']:.4f} | "
            f"importance={item['importance_score']:.4f} | "
            f"recency={item['recency_score']:.4f} | "
            f"final={item['final_score']:.4f}"
        )


__all__ = [
    "GENERIC_CONCEPTS",
    "KEYWORD_STOPWORDS",
    "compute_keyword_score",
    "compute_memory_score",
    "compute_recency_score",
    "debug_rerank_result",
    "extract_keywords",
    "is_high_confidence_concept",
    "maybe_expand_query_with_concept",
    "normalize_text",
    "parse_timestamp",
    "resolve_candidate_k",
]
