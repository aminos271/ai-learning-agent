from typing import Any, Dict, List, Optional

from core.base_retriever import BaseRetriever
from core.config import Config

from .filters import normalize_search_filter, split_search_filters
from .scoring import (
    GENERIC_CONCEPTS,
    KEYWORD_STOPWORDS,
    compute_memory_score,
    debug_rerank_result,
    maybe_expand_query_with_concept,
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
        debug_rerank_result(ranked, top_k)
        return [item["item"] for item in ranked[:top_k]]

    def retrieve(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """记忆检索主流程：拆过滤条件 -> 扩展query -> 放大召回 -> rerank"""
    


        vector_filter, rerank_filter = split_search_filters(metadata_filter)
        effective_query, concept_hint = maybe_expand_query_with_concept(
            question,
            rerank_filter,
        )
       
        candidate_k = resolve_candidate_k(k)

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
            # search_filter = self._normalize_search_filter(
            #     concept=concept,
            #     note_type=note_type,
            #     save_mode=save_mode,
            #     source=source,
            #     metadata_filter=metadata_filter,
            # )
            # vector_filter, _ = split_search_filters(search_filter)
            # effective_query, concept_hint = maybe_expand_query_with_concept(query, search_filter)
            # print(
            #     f"🧠 正在翻阅历史记忆: '{query}' "
            #     f"vector_filters={vector_filter} "
            #     f"concept_hint={concept_hint or '-'} "
            #     f"effective_query='{effective_query}'"
            # )
            # 现在还没真正实现filter功能，只保留接口
            metadata_filter = {}
            return self.retrieve(query, k=k, metadata_filter=metadata_filter)
        except Exception as e:
            print(f"❌ MemoryRetriever.search_notes 失败：{e}")
            return []



__all__ = ["MemoryRetriever"]
