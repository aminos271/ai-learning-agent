import os
import dotenv

dotenv.load_dotenv()

class Config:
    # 配置embedding
    EMBEDDING_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "bge-m3"

    # 文本切分配置
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 100
    
    # 配置LLM
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")

    # Qdrant 配置 
    QDRANT_URL = "http://localhost:6333" # 你的 Docker Qdrant 地址
    COLLECTION_NAME = "pdf_knowledge_base" # 知识库集合名称
    MEMORY_COLLECTION_NAME = "user_memory_note-v1" # 笔记本集合