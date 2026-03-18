from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from core.config import Config
import math

    
class MultiQueries(BaseModel):
    """用于强制大模型输出问题列表"""
    queries: List[str] = Field(description="包含 3 个不同表述的改写问题列表")
    
class QdrantRetriever:
    """基于 Qdrant 的向量检索器"""

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.EMBEDDING_URL
        )
        print("初始化 embedding 模型成功")

        self.client = QdrantClient(url=Config.QDRANT_URL)
        self.collection_name = Config.COLLECTION_NAME
        self.vector_store = None
        print("初始化 Qdrant 客户端连接成功")

    def _get_vector_store(self):
        """懒加载 vector store，自动创建不存在的集合"""
        if not self.vector_store:
            try:
                # 尝试获取现有集合
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )
            except Exception as e:
                # 检查是否是集合不存在的错误
                error_str = str(e).lower()
                if "doesn't exist" in error_str or "not found" in error_str:
                    print(f"📦 集合 {self.collection_name} 不存在，正在创建...")

                    # 1. 获取向量维度
                    test_text = "test"
                    test_vector = self.embeddings.embed_query(test_text)
                    vector_size = len(test_vector)

                    # 2. 创建集合
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ 集合创建成功，维度: {vector_size}")

                    # 3. 重新创建 vector_store
                    self.vector_store = QdrantVectorStore(
                        client=self.client,
                        collection_name=self.collection_name,
                        embedding=self.embeddings,
                    )
                else:
                    # 其他错误直接抛出
                    raise e

        return self.vector_store

    def _make_qdrant_filter(self, metadata_filter: Optional[Dict[str, str]] = None):
        """把通用 metadata dict 转为 Qdrant Filter 对象，支持 metadata aware 检索。"""
        if not metadata_filter:
            return None

        must_conditions = []
        for key, value in metadata_filter.items():
            if value is None:
                continue
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度，用于 rerank。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rerank_documents(self, question: str, docs: List[Document], metadata_filter: Optional[Dict[str, str]] = None, top_k: int = 5) -> List[Document]:
        """基于向量语义 + metadata 匹配做一个简单 rerank，避免直接拼接 context。"""
        if not docs:
            return []

        # 1. 计算 question 向量，后面用于 semantic score 计算。
        question_vector = self.embeddings.embed_query(question)

        ranked = []
        for doc in docs:
            # 2. 尽量使用检索返回的距离/score信息，若没有则重新计算。
            semantic_score = 0.0
            if 'score' in doc.metadata:
                semantic_score = float(doc.metadata['score'])
            elif 'similarity' in doc.metadata:
                semantic_score = float(doc.metadata['similarity'])
            elif 'distance' in doc.metadata:
                semantic_score = 1.0 - float(doc.metadata['distance'])

            if semantic_score <= 0:
                # fallback: 计算内容向量相似度
                doc_vector = self.embeddings.embed_documents([doc.page_content])[0]
                semantic_score = self._cosine_similarity(question_vector, doc_vector)

            # 3. metadata 身份加权（page_source、section 等能影响优先级）
            metadata_score = 1.0
            if metadata_filter:
                matches = 0
                for mk, mv in metadata_filter.items():
                    if mk in doc.metadata and str(doc.metadata.get(mk)) == str(mv):
                        matches += 1
                metadata_score = matches / max(len(metadata_filter), 1)

            # 4. 简单线性组合打分
            final_score = semantic_score * 0.8 + metadata_score * 0.2

            ranked.append((final_score, doc))

        # 5. 降序排序，返回 top_k 文档
        ranked.sort(key=lambda item: item[0], reverse=True)

        reranked_docs = [item[1] for item in ranked[:top_k]]
        return reranked_docs

    def retrieve(self, question: str, k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[Document]:
        """支持 metadata aware 的基础检索接口。"""
        vector_store = self._get_vector_store()
        print(f"🔍 正在从 Qdrant 检索相关信息: '{question}' metadata_filter={metadata_filter}")

        qdrant_filter = self._make_qdrant_filter(metadata_filter)
        search_kwargs = {'k': k}
        if qdrant_filter:
            search_kwargs['filter'] = qdrant_filter

        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(question)

        # 进行 rerank，不直接拼接 context，避免 prompt injection 和无效拼接
        reranked = self._rerank_documents(question, docs, metadata_filter, top_k=k)
        return reranked

    def multi_query_search(self, question: str, llm, k: int = 3, metadata_filter: Optional[Dict[str, str]] = None) -> List[Document]:
        """多问法检索 + 去重 + metadata aware rerank。"""
        parser = JsonOutputParser(pydantic_object=MultiQueries)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个专业的AI检索优化专家。"
                "你的任务是将用户的原始查询改写成 3 个不同角度的等价查询，"
                "以提高在向量数据库中的检索召回率。\n\n{format_instructions}"
            ),
            ("human", "原始查询：{question}")
        ]).partial(
            format_instructions=parser.get_format_instructions()
        )

        print(f"🌀 启动原生 Multi-Query 引擎，正在裂变问题: '{question}'")

        chain = prompt | llm | parser
        response = chain.invoke({"question": question})
        generated_queries = response.get("queries", [])

        print(f"改写后的问题：\n{generated_queries}")

        vector_store = self._get_vector_store()
        qdrant_filter = self._make_qdrant_filter(metadata_filter)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": k, **({'filter': qdrant_filter} if qdrant_filter else {})})

        unique_docs = {}
        all_queries = [question] + generated_queries

        for q in all_queries:
            docs = base_retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc

        docs_list = list(unique_docs.values())
        print(f"✅ 裂变检索完成，共合并去重得到 {len(docs_list)} 个独立文档块。")

        reranked_docs = self._rerank_documents(question, docs_list, metadata_filter, top_k=k)

        # 打印详细的元数据信息，便于调试
        # for i, doc in enumerate(reranked_docs):
        #     print(f"\n📄 文档 {i+1}:")
        #     print(f"  内容预览: {doc.page_content[:50]}...")
        #     print(f"  元数据:")
        #     for key, value in doc.metadata.items():
        #         print(f"    {key}: {value}")
        #     print(f"  {'='*50}")

        return reranked_docs
    
    def add_documents(self, docs: List[Document]):
        """将文本列表批量向量化并存入 Qdrant"""
        vector_store = self._get_vector_store()
        vector_store.add_documents(docs)
        print(f"✅ 成功将 {len(docs)} 个知识块存入 Qdrant 集合: {self.collection_name}")
