import os
import traceback
from typing import Dict, Any
from rag.ingest import MarkdownIngestor
from rag.retriever import QdrantRetriever


class DocumentManager:
    """文档处理与知识库构建的全局管家"""

    def __init__(self, retriever: QdrantRetriever):
        # 传入全局的检索器和记忆库实例，确保和 Agent 用的是同一个大脑
        self.retriever = retriever
        
        # 实例化 Markdown 清洗流水线
        self.ingestor = MarkdownIngestor()


    def process_and_store(self, file_path: str) -> Dict[str, Any]:
        """
        一键处理上传的文档：解析 -> 切分 -> 向量化入库 
        """

        if not file_path or not os.path.exists(file_path):
            return {"success": False, "message": "❌ 文件不存在或路径错误！"}
        
        file_name = os.path.basename(file_path)
        print(f"📥 [管家] 接收到新文档: {file_name}")

        try:
            # 1. 启动 Markdown 清洗与智能分块
            chunks = self.ingestor.process_file(file_path)
            
            # 2. 存入 Qdrant 向量库
            print(f"📦 [管家] 正在将 {len(chunks)} 个知识块打入向量库...")
            
            self.retriever.add_documents(chunks)
            
            return {
                "success": True,
                "message": f"✅ 文档已成功入库: {file_name}",
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "message": f"❌ 处理失败: {str(e)}"}

