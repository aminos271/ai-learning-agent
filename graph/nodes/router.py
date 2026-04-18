from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from graph.prompts import router_prompt
from graph.state import AgentState


class RouteDecision(BaseModel):
    """大模型路由决策的输出格式"""
    route: Literal["rag", "note_recall", "chat", "note_store"] = Field(
        description="""
        判断用户意图。如果是查阅以前的笔记，回顾学习，返回 'note_recall'。
        如果用户要求记录新的笔记、保存当前对话中的知识点或学习心得，返回 'note_store'。
        如果是询问客观知识、文档内容，返回 'rag'。
        单纯聊天或追问刚才的话题走 'chat'。
        """
    )
    reason: str | None = Field(
        default=None,
        description="决策原因"
    )
    confidence: float = Field(
        default=1.0,
        description="决策置信度 (0-1)",
        ge=0,
        le=1
    )


parser = JsonOutputParser(pydantic_object=RouteDecision)


def router_node(state: AgentState, llm) -> AgentState:
    """
    路由节点：判断应该用RAG检索还是记忆召回
    """
    question = state["question"]
    messages = state.get("messages", [])
    rewritten_question = state.get("rewritten_question", question)

    history_text = ""
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"AI: {msg.content}\n"

    router_chain = router_prompt | llm | parser

    response = router_chain.invoke({
        "rewritten_question": rewritten_question,
        "question": question,
        "messages": history_text,
        "format_instructions": parser.get_format_instructions()
    })

    route = response["route"]
    reason = response.get("reason")
    confidence = response.get("confidence", 1.0)

    print(f"🔄 路由决策: {route} (置信度: {confidence})")
    if reason:
        print(f"📝 决策原因: {reason}")

    return {
        **state,
        "route": route,
        "metadata": {
            **state.get("metadata", {}),
            "route": route,
            "route_reason": reason,
            "route_confidence": confidence
        }
    }
