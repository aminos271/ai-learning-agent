from typing import TypedDict, Optional, Annotated, Dict, Any
from langgraph.graph.message import add_messages


# 定义全局的数据结构

class AgentState(TypedDict):

    messages: Annotated[list, add_messages] #存放对话历史
    question: str          # 用户的原始问题
    rewritten_question: str     # 改写后的独立问题
    context: Optional[str] # 检索到的背景知识
    answer: Optional[str]  # 最终生成的回答
    route: Optional[str]   # 路由决策（走向 rag 还是 recall）
    metadata: Optional[Dict[str, Any]]  # 添加 metadata 字段，用于存放额外的元数据

