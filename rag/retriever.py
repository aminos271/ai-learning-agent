from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from typing import Any, Optional
from pydantic import BaseModel, Field
from core.config import Config
from core.base_retriever import BaseRetriever, RetrievedItem
from graph.prompts import rag_muti_retriever_prompt
from rag.rerank import rerank_documents
    
class MultiQueries(BaseModel):
    """用于强制大模型输出问题列表"""
    queries: list[str] = Field(description="包含 3 个不同表述的改写问题列表")
    
class QdrantRetriever(BaseRetriever):
    """基于 Qdrant 的向量检索器"""

    def __init__(self):
        super().__init__(
            collection_name=Config.COLLECTION_NAME,
            collection_label="知识库集合",
        )
        self.supported_metadata_filter_keys = {"source", "section_path", "h1", "h2"}
        print("初始化 embedding 模型成功")
        print("初始化 Qdrant 客户端连接成功")

    def _normalize_metadata_filter(
        self,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not metadata_filter:
            return {}

        return {
            key: value
            for key, value in metadata_filter.items()
            if key in self.supported_metadata_filter_keys and value is not None
        }

    def multi_query_search(
        self,
        question: str,
        llm,
        k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        """多问法检索 + 去重 + metadata aware rerank。"""
        parser = JsonOutputParser(pydantic_object=MultiQueries)
        active_filter = self._normalize_metadata_filter(metadata_filter)

        prompt = rag_muti_retriever_prompt.partial(
            format_instructions=parser.get_format_instructions()
        ) 

        print(f"🌀 启动原生 Multi-Query 引擎，正在裂变问题: '{question}'")

        chain = prompt | llm | parser
        response = chain.invoke({"question": question})
        generated_queries = response.get("queries", [])

        print(f"改写后的问题：\n{generated_queries}")

        unique_items: dict[str, RetrievedItem] = {}
        all_queries = [question] + generated_queries

        for q in all_queries:
            items = self._similarity_search_items(q, k=k, metadata_filter=active_filter)
            for item in items:
                source = item.metadata.get("source")
                chunk_id = item.metadata.get("chunk_id")
                # 生成标识，查看是否重复出现
                item_key = (
                    f"{source}:{chunk_id}"
                    if source is not None or chunk_id is not None
                    else item.content
                )
                existing = unique_items.get(item_key)

                # matched_queries相当于贴标签，标记文档是从哪些query查询获得的
                matched_queries = set(item.retrieval_meta.get("matched_queries", []))
                matched_queries.add(q)

                # 如果没重复，处理新chunk
                if existing is None:
                    # sorted返回的是list
                    item.retrieval_meta["matched_queries"] = sorted(matched_queries)
                    unique_items[item_key] = item
                    continue
                
                # 如果重复，通过similarity进行比较，保留相似度大的，同时记录所有查询这个chunk的query
                existing_queries = set(existing.retrieval_meta.get("matched_queries", []))
                combined_queries = sorted(existing_queries | matched_queries)

                existing_similarity = float(
                    existing.retrieval_meta.get("similarity", 0.0) or 0.0
                )
                current_similarity = float(
                    item.retrieval_meta.get("similarity", 0.0) or 0.0
                )

                if current_similarity > existing_similarity:
                    item.retrieval_meta["matched_queries"] = combined_queries
                    unique_items[item_key] = item
                else:
                    existing.retrieval_meta["matched_queries"] = combined_queries

        items_list = list(unique_items.values())
        print(f"✅ 裂变检索完成，共合并去重得到 {len(items_list)} 个独立文档块。")

        reranked_docs = rerank_documents(
            question,
            items_list,
            active_filter,
            top_k=k,
        )

        return reranked_docs
    
    def add_documents(self, docs: list[Document]):
        """将文本列表批量向量化并存入 Qdrant"""
        vector_store = self._get_vector_store()
        vector_store.add_documents(docs)
        print(f"✅ 成功将 {len(docs)} 个知识块存入 Qdrant 集合: {self.collection_name}")
