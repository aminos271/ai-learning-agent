from langchain_openai import ChatOpenAI

from rag.retriever import QdrantRetriever
from memory.memory_store import MemoryStore
from core.config import Config



def build_runtime():
    # LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL_ID,
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        temperature=0.1
    )

    # 检索器
    retriever = QdrantRetriever()
    
    # 学习笔记
    memory_store = MemoryStore()
    
    return {
        "llm": llm,
        "retriever": retriever,
        "memory_store": memory_store
    }