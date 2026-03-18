from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
from core.config import Config


class MemoryStore:
    """长期记忆管理器（用户的笔记本）"""

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.EMBEDDING_URL
        )
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self.collection_name = Config.MEMORY_COLLECTION_NAME
        self.vector_store = None

    def _get_vector_store(self):
        """懒加载 vector store，避免程序启动时因为 Qdrant 短暂不可用直接崩掉"""
        if self.vector_store is None:
            try:
                exists = self.client.collection_exists(self.collection_name)
            except Exception as e:
                raise RuntimeError(
                    f"Qdrant 连接失败，无法检查集合 '{self.collection_name}' 是否存在。"
                    f"请先确认 Qdrant 服务是否正常运行。原始错误: {e}"
                ) from e

            if not exists:
                print(f"⚠️ 记忆集合 '{self.collection_name}' 不存在，正在创建...")

                test_vector = self.embeddings.embed_query("测试维度")
                vector_dim = len(test_vector)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ 记忆集合创建成功，向量维度: {vector_dim}")

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        return self.vector_store

    def add_note(
        self,
        content: str,
        concept: str = "general",
        note_type: str = "note",
        importance: float = 0.8,
        source: str = "user",
    ):
        """
        添加一条学习笔记，并补全结构化 metadata
        """
        from datetime import datetime

        vector_store = self._get_vector_store()
        timestamp = datetime.utcnow().isoformat() + "Z"

        print(f"📝 正在将笔记存入大脑: [{concept}] ({note_type}) {content[:20]}...")

        doc = Document(
            page_content=content,
            metadata={
                "type": note_type,
                "concept": concept,
                "importance": importance,
                "timestamp": timestamp,
                "source": source,
            },
        )

        vector_store.add_documents([doc])
        print("✅ 笔记保存成功！")

    def search(self, query: str, k: int = 3):
        """
        检索历史记忆
        """
        vector_store = self._get_vector_store()

        print(f"🧠 正在翻阅历史记忆: '{query}'")
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        results = [doc.page_content for doc in docs]
        return results if results else []