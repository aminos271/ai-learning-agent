import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.base_retriever import BaseQdrantStore

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


def extract_keywords(text: Optional[str]) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not text:
        return []

    keywords: List[str] = []

    for token in re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text):
        if token not in KEYWORD_STOPWORDS:
            keywords.append(token)

    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(block) < 2:
            continue

        if len(block) <= 8 and block not in KEYWORD_STOPWORDS:
            keywords.append(block)

        for size in (2, 3):
            if len(block) >= size:
                for i in range(len(block) - size + 1):
                    token = block[i : i + size]
                    if token not in KEYWORD_STOPWORDS:
                        keywords.append(token)

    return list(dict.fromkeys(keywords))


def expand_query_with_concept(
    query: str,
    metadata_filter: Optional[Dict[str, object]] = None,
) -> Tuple[str, Optional[str]]:
    concept = str((metadata_filter or {}).get("concept") or "").strip()
    if not concept:
        return query, None

    concept_lower = concept.lower()
    if concept_lower in GENERIC_CONCEPTS:
        return query, None

    if len(concept) < 2 or len(concept) > 32:
        return query, None

    query_lower = query.strip().lower()
    concept_lower = concept.strip().lower()
    if concept_lower in query_lower:
        return query, None

    query_keywords = set(extract_keywords(query))
    concept_keywords = set(extract_keywords(concept))
    overlap = query_keywords & concept_keywords

    is_short_query = len(query.strip()) <= 18
    has_recall_marker = any(x in query for x in ("这个", "那个", "它", "相关", "之前", "上次"))

    if overlap or is_short_query or has_recall_marker:
        return f"{query} {concept}".strip(), concept

    return query, None


def resolve_candidate_k(top_k: int) -> int:
    return max(top_k * 4, 12)


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

    keywords = extract_keywords(question)
    content = re.sub(r"\s+", " ", (item.content or "").strip().lower())

    if not keywords or not content:
        keyword_score = 0.0
    else:
        matched_keywords = 0
        total_hits = 0
        for keyword in keywords:
            hit_count = content.count(keyword)
            if hit_count > 0:
                matched_keywords += 1
                total_hits += min(hit_count, 3)

        if matched_keywords == 0:
            keyword_score = 0.0
        else:
            coverage = matched_keywords / len(keywords)
            density = min(total_hits / len(keywords), 1.0)
            keyword_score = max(0.0, min(coverage * 0.7 + density * 0.3, 1.0))

    importance = float(item.metadata.get("importance", 0.0) or 0.0)
    importance_score = max(0.0, min(importance, 1.0))

    timestamp = item.metadata.get("timestamp")
    recency_score = 0.0
    if timestamp:
        try:
            if timestamp.endswith("Z"):
                timestamp = timestamp.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_seconds = max((now - parsed).total_seconds(), 0.0)
            recency_score = 1.0 / (1.0 + age_seconds / (24 * 60 * 60))
        except ValueError:
            recency_score = 0.0

    final_score = (
        semantic_score * 0.45
        + keyword_score * 0.25
        + importance_score * 0.2
        + recency_score * 0.1
    )

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