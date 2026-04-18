from datetime import datetime, timezone

from langchain_core.documents import Document

from core.base_retriever import BaseQdrantStore
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
        concept: str | None = None,
        note_type: str = "general",
        save_mode: str = "raw_note",
        importance: float | None = None,
        timestamp: str | None = None,
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

    def _resolve_concept(self, content: str, concept: str | None) -> str:
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
        importance: float | None,
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



