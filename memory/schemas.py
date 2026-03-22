from dataclasses import dataclass

from core.base_retriever import RetrievedItem


@dataclass(frozen=True)
class MemoryScoreBreakdown:
    """Memory rerank 的分数拆解。"""

    semantic_score: float
    keyword_score: float
    importance_score: float
    recency_score: float
    final_score: float



