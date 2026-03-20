from typing import Any, Dict, List, Optional

from core.base_retriever import BaseRetriever
from core.config import Config

from .filters import build_rerank_metadata_filter, normalize_search_filter, split_search_filters
from .scoring import (
    GENERIC_CONCEPTS,
    KEYWORD_STOPWORDS,
    compute_keyword_score,
    compute_memory_score,
    compute_recency_score,
    debug_rerank_result,
    extract_keywords,
    is_high_confidence_concept,
    maybe_expand_query_with_concept,
    normalize_text,
    parse_timestamp,
    resolve_candidate_k,
)
from .schemas import RetrievedItem


class MemoryRetriever(BaseRetriever):
    """长期记忆检索器。"""

    GENERIC_CONCEPTS = GENERIC_CONCEPTS
    KEYWORD_STOPWORDS = KEYWORD_STOPWORDS

    def __init__(self):
        super().__init__(
            collection_name=Config.MEMORY_COLLECTION_NAME,
            collection_label="记忆集合",
        )

    def _normalize_search_filter(
        self,
        concept: Optional[str] = None,
        note_type: Optional[str] = None,
        save_mode: Optional[str] = None,
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return normalize_search_filter(
            concept=concept,
            note_type=note_type,
            save_mode=save_mode,
            source=source,
            metadata_filter=metadata_filter,
        )

    def _split_search_filters(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        return split_search_filters(metadata_filter)

    def _normalize_text(self, text: Optional[str]) -> str:
        return normalize_text(text)

    def _extract_keywords(self, text: Optional[str]) -> List[str]:
        return extract_keywords(text)

    def _compute_keyword_score(self, query: str, content: str) -> float:
        return compute_keyword_score(query, content)

    def _is_high_confidence_concept(self, concept: Optional[str]) -> bool:
        return is_high_confidence_concept(concept)

    def _maybe_expand_query_with_concept(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[str]]:
        return maybe_expand_query_with_concept(query, metadata_filter)

    def _resolve_candidate_k(self, top_k: int) -> int:
        return resolve_candidate_k(top_k)

    def _parse_timestamp(self, timestamp: Optional[str]):
        return parse_timestamp(timestamp)

    def _compute_recency_score(self, timestamp: Optional[str]) -> float:
        return compute_recency_score(timestamp)

    def _build_rerank_metadata_filter(
        self, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return build_rerank_metadata_filter(metadata_filter)

    def _debug_rerank_result(self, ranked: List[Dict[str, Any]], top_k: int):
        return debug_rerank_result(ranked, top_k)

    def _rerank_documents(
        self,
        question: str,
        items: List[RetrievedItem],
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        """Memory 检索策略：semantic + keyword + importance + recency。"""
        if not items:
            return []

        question_vector = self.embeddings.embed_query(question)
        ranked = []

        for item in items:
            score = compute_memory_score(
                store=self,
                question=question,
                question_vector=question_vector,
                item=item,
            )

            ranked.append(
                {
                    "final_score": score.final_score,
                    "semantic_score": score.semantic_score,
                    "keyword_score": score.keyword_score,
                    "importance_score": score.importance_score,
                    "recency_score": score.recency_score,
                    "summary": item.content[:60].replace("\n", " "),
                    "item": RetrievedItem(
                        content=item.content,
                        metadata=dict(item.metadata),
                        retrieval_meta={
                            **item.retrieval_meta,
                            "similarity": score.semantic_score,
                            "keyword_score": score.keyword_score,
                            "importance_score": score.importance_score,
                            "recency_score": score.recency_score,
                            "rerank_score": score.final_score,
                        },
                    ),
                }
            )

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        self._debug_rerank_result(ranked, top_k)
        return [item["item"] for item in ranked[:top_k]]

    def retrieve(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """兼容 BaseRetriever 接口，但不再把 concept 作为硬过滤条件。"""



        vector_filter, rerank_filter = self._split_search_filters(metadata_filter)
        effective_query, concept_hint = self._maybe_expand_query_with_concept(
            question,
            rerank_filter,
        )
        candidate_k = self._resolve_candidate_k(k)

        items = self._similarity_search_items(
            effective_query,
            k=candidate_k,
            metadata_filter=vector_filter,
        )

        reranked_items = self._rerank_documents(
            effective_query,
            items,
            metadata_filter=rerank_filter,
            top_k=k,
        )

        if concept_hint:
            for item in reranked_items:
                item.retrieval_meta["concept_hint_used"] = concept_hint

        return reranked_items

    def search_notes(
        self,
        query: str,
        k: int = 3,
        concept: Optional[str] = None,
        note_type: Optional[str] = None,
        save_mode: Optional[str] = None,
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """按 Memory 策略检索历史笔记。"""
        try:
            search_filter = self._normalize_search_filter(
                concept=concept,
                note_type=note_type,
                save_mode=save_mode,
                source=source,
                metadata_filter=metadata_filter,
            )
            vector_filter, _ = self._split_search_filters(search_filter)
            effective_query, concept_hint = self._maybe_expand_query_with_concept(query, search_filter)
            print(
                f"🧠 正在翻阅历史记忆: '{query}' "
                f"vector_filters={vector_filter} "
                f"concept_hint={concept_hint or '-'} "
                f"effective_query='{effective_query}'"
            )
            return self.retrieve(query, k=k, metadata_filter=search_filter)
        except Exception as e:
            print(f"❌ MemoryRetriever.search_notes 失败：{e}")
            return []

    def search(
        self,
        query: str,
        k: int = 3,
        concept: Optional[str] = None,
        note_type: Optional[str] = None,
        save_mode: Optional[str] = None,
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """兼容旧接口，返回纯文本结果列表。"""
        try:
            items = self.search_notes(
                query,
                k=k,
                concept=concept,
                note_type=note_type,
                save_mode=save_mode,
                source=source,
                metadata_filter=metadata_filter,
            )
            results = [item.content for item in items]
            return results if results else []
        except Exception as e:
            print(f"❌ MemoryRetriever.search 失败：{e}")
            return []


__all__ = ["MemoryRetriever"]
