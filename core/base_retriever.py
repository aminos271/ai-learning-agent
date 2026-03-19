import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config import Config


@dataclass
class RetrievedItem:
    """统一的检索结果结构。"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_meta: Dict[str, Any] = field(default_factory=dict)


class BaseQdrantStore:
    """Qdrant 公共基础设施。"""

    DYNAMIC_RETRIEVAL_KEYS = {
        "score",
        "similarity",
        "distance",
        "rerank_score",
        "recency_score",
        "importance_score",
        "concept_score",
        "metadata_match_score",
    }

    def __init__(self, collection_name: str, collection_label: str = "集合"):
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.EMBEDDING_URL,
        )
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self.collection_name = collection_name
        self.collection_label = collection_label
        self.vector_store = None
        self.metadata_payload_key = "metadata"

    def _get_vector_store(self):
        """懒加载 vector store，必要时自动创建集合。"""
        if self.vector_store is None:
            try:
                exists = self.client.collection_exists(self.collection_name)
            except Exception as e:
                raise RuntimeError(
                    f"Qdrant 连接失败，无法检查{self.collection_label} '{self.collection_name}' 是否存在。原始错误: {e}"
                ) from e

            if not exists:
                print(f"📦 {self.collection_label} '{self.collection_name}' 不存在，正在创建...")
                vector_dim = len(self.embeddings.embed_query("test"))
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"✅ {self.collection_label}创建成功，向量维度: {vector_dim}")

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            self.metadata_payload_key = (
                getattr(self.vector_store, "metadata_payload_key", None) or "metadata"
            )

        return self.vector_store

    def _make_qdrant_filter(self, metadata_filter: Optional[Dict[str, Any]] = None):
        """把 metadata dict 转为 Qdrant Filter 对象。"""
        if not metadata_filter:
            return None

        must_conditions = []
        for key, value in metadata_filter.items():
            if value is None:
                continue
            must_conditions.append(
                models.FieldCondition(
                    key=self._build_metadata_filter_key(key),
                    match=models.MatchValue(value=value),
                )
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def _build_metadata_filter_key(self, key: str) -> str:
        if not key:
            return key

        prefix = f"{self.metadata_payload_key}."
        return key if key.startswith(prefix) else f"{prefix}{key}"

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _split_static_and_dynamic_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        static_metadata = dict(metadata or {})
        retrieval_meta: Dict[str, Any] = {}

        raw_score = static_metadata.pop("score", None)
        if raw_score is not None:
            retrieval_meta["similarity"] = raw_score

        for key in self.DYNAMIC_RETRIEVAL_KEYS - {"score"}:
            if key in static_metadata:
                retrieval_meta[key] = static_metadata.pop(key)

        return static_metadata, retrieval_meta

    def _make_retrieved_item(
        self,
        doc: Document,
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> RetrievedItem:
        static_metadata, dynamic_from_doc = self._split_static_and_dynamic_metadata(
            doc.metadata
        )
        merged_retrieval_meta = {
            **dynamic_from_doc,
            **dict(retrieval_meta or {}),
        }
        return RetrievedItem(
            content=doc.page_content,
            metadata=static_metadata,
            retrieval_meta=merged_retrieval_meta,
        )

    def _similarity_search_items(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        vector_store = self._get_vector_store()
        qdrant_filter = self._make_qdrant_filter(metadata_filter)

        try:
            docs_with_scores = vector_store.similarity_search_with_score(
                question,
                k=k,
                filter=qdrant_filter,
            )
            items = []
            for doc, score in docs_with_scores:
                retrieval_meta: Dict[str, Any] = {}
                if score is not None:
                    retrieval_meta["similarity"] = float(score)
                items.append(self._make_retrieved_item(doc, retrieval_meta))
            return items
        except (AttributeError, TypeError):
            search_kwargs = {"k": k}
            if qdrant_filter:
                search_kwargs["filter"] = qdrant_filter

            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(question)
            return [self._make_retrieved_item(doc) for doc in docs]


class BaseRetriever(BaseQdrantStore, ABC):
    """公共检索能力底座，具体 rerank 策略交给子类实现。"""

    @abstractmethod
    def _rerank_documents(
        self,
        question: str,
        items: List[RetrievedItem],
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        raise NotImplementedError

    def retrieve(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """基础检索接口，提供统一的 metadata filter 和 rerank 调用。"""
        items = self._similarity_search_items(
            question,
            k=k,
            metadata_filter=metadata_filter,
        )
        return self._rerank_documents(question, items, metadata_filter, top_k=k)
