from langchain_core.messages import AIMessage
from graph.prompts import chat_prompt
from graph.state import AgentState


def chat_node(state: AgentState, llm) -> AgentState:
    """闲聊与上下文梳理节点：不检索任何库，纯靠历史 messages 回答"""

    question = state["question"]
    messages = state.get("messages", [])

    print("💬 正在进行上下文对话分析...")
    chat_chain = chat_prompt | llm

    response = chat_chain.invoke({
        "messages": messages,
        "question": question
    })

    print("✅ 对话生成完成")

    return {
        **state,
        "answer": response.content,
        "messages": [AIMessage(content=response.content)]
    }
