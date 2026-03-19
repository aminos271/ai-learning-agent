from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from core.base_retriever import BaseQdrantStore, BaseRetriever, RetrievedItem
from core.config import Config


class MemoryWriter(BaseQdrantStore):
    """长期记忆写入器。"""

    def __init__(self):
        super().__init__(
            collection_name=Config.MEMORY_COLLECTION_NAME,
            collection_label="记忆集合",
        )

    def add_note(
        self,
        content: str,
        concept: Optional[str] = None,
        note_type: str = "general",
        save_mode: str = "raw_note",
        importance: Optional[float] = None,
        timestamp: Optional[str] = None,
        source: str = "user",
    ):
        """添加一条学习笔记，并补全结构化 metadata。"""
        try:
            vector_store = self._get_vector_store()
            resolved_concept = self._resolve_concept(content, concept)
            resolved_importance = self._resolve_importance(
                content=content,
                note_type=note_type,
                save_mode=save_mode,
                importance=importance,
            )
            resolved_timestamp = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            print(f"📝 正在将笔记存入大脑: [{resolved_concept}] ({note_type}) {content[:20]}...")

            doc = Document(
                page_content=content,
                metadata={
                    "memory_type": "note",
                    "note_type": note_type,
                    "save_mode": save_mode,
                    "concept": resolved_concept,
                    "importance": resolved_importance,
                    "timestamp": resolved_timestamp,
                    "source": source,
                },
            )

            vector_store.add_documents([doc])
            print("✅ 笔记保存成功！")

        except Exception as e:
            print(f"❌ MemoryWriter.add_note 失败：{e}")
            raise RuntimeError(f"MemoryWriter.add_note 失败：{e}") from e

    def _resolve_concept(self, content: str, concept: Optional[str]) -> str:
        candidate = (concept or "").strip()
        if candidate and candidate.lower() != "general":
            return candidate

        text = " ".join((content or "").strip().split())
        if not text:
            return "general"

        for separator in ("：", ":", "|", "｜", "-", "，", ",", "。", "."):
            if separator not in text:
                continue
            head = text.split(separator, 1)[0].strip()
            if 2 <= len(head) <= 24:
                return head

        if " " in text:
            words = " ".join(text.split()[:3]).strip()
            if words:
                return words[:24]

        return text[:12] or "general"

    def _resolve_importance(
        self,
        content: str,
        note_type: str,
        save_mode: str,
        importance: Optional[float],
    ) -> float:
        if importance is not None:
            return max(0.0, min(float(importance), 1.0))

        score = 0.55
        if save_mode == "correction_note":
            score += 0.2
        elif save_mode == "summary_note":
            score += 0.1

        if (note_type or "").strip() and note_type != "general":
            score += 0.05

        text_length = len((content or "").strip())
        score += min(text_length / 400.0, 0.15)
        return max(0.0, min(score, 1.0))


class MemoryRetriever(BaseRetriever):
    """长期记忆检索器。"""

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

    def _parse_timestamp(self, timestamp: Optional[str]) -> Optional[datetime]:
        if not timestamp:
            return None

        try:
            if timestamp.endswith("Z"):
                timestamp = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None

    def _compute_recency_score(self, timestamp: Optional[str]) -> float:
        parsed = self._parse_timestamp(timestamp)
        if parsed is None:
            return 0.0

        now = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        age_seconds = max((now - parsed).total_seconds(), 0.0)
        one_day = 24 * 60 * 60
        return 1.0 / (1.0 + age_seconds / one_day)

    def _build_rerank_metadata_filter(
        self, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not metadata_filter:
            return {}
        return dict(metadata_filter)

    def _debug_rerank_result(self, ranked: List[Dict], top_k: int):
        if not ranked:
            print("📊 Memory rerank: 没有候选结果。")
            return

        print("📊 Memory rerank 结果:")
        for index, item in enumerate(ranked[:top_k], start=1):
            print(
                f"  {index}. note='{item['summary']}' | "
                f"semantic={item['semantic_score']:.4f} | "
                f"importance={item['importance_score']:.4f} | "
                f"recency={item['recency_score']:.4f} | "
                f"concept_match={item['concept_score']:.4f} | "
                f"final={item['final_score']:.4f}"
            )

    def _rerank_documents(
        self,
        question: str,
        items: List[RetrievedItem],
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        """Memory 检索策略：语义相关性 + importance + recency + concept 精确匹配。"""
        if not items:
            return []

        question_vector = self.embeddings.embed_query(question)
        rerank_filter = self._build_rerank_metadata_filter(metadata_filter)
        requested_concept = (rerank_filter.get("concept") or "").strip().lower()
        ranked = []

        for item in items:
            semantic_score = float(item.retrieval_meta.get("similarity", 0.0) or 0.0)
            if semantic_score <= 0 and item.retrieval_meta.get("distance") is not None:
                semantic_score = 1.0 - float(item.retrieval_meta["distance"])

            if semantic_score <= 0:
                doc_vector = self.embeddings.embed_documents([item.content])[0]
                semantic_score = self._cosine_similarity(question_vector, doc_vector)

            importance = float(item.metadata.get("importance", 0.0) or 0.0)
            importance_score = max(0.0, min(importance, 1.0))
            recency_score = self._compute_recency_score(item.metadata.get("timestamp"))

            concept_score = 0.0
            doc_concept = str(item.metadata.get("concept", "")).strip().lower()
            if requested_concept and doc_concept == requested_concept:
                concept_score = 1.0

            final_score = (
                semantic_score * 0.55
                + importance_score * 0.2
                + recency_score * 0.15
                + concept_score * 0.1
            )

            ranked.append(
                {
                    "final_score": final_score,
                    "semantic_score": semantic_score,
                    "importance_score": importance_score,
                    "recency_score": recency_score,
                    "concept_score": concept_score,
                    "summary": item.content[:60].replace("\n", " "),
                    "item": RetrievedItem(
                        content=item.content,
                        metadata=dict(item.metadata),
                        retrieval_meta={
                            **item.retrieval_meta,
                            "similarity": semantic_score,
                            "importance_score": importance_score,
                            "recency_score": recency_score,
                            "concept_score": concept_score,
                            "rerank_score": final_score,
                        },
                    ),
                }
            )

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        self._debug_rerank_result(ranked, top_k)
        return [item["item"] for item in ranked[:top_k]]

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
            print(f"🧠 正在翻阅历史记忆: '{query}' filters={search_filter}")
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
