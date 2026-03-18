from langchain_core.messages import AIMessage
from graph.prompts import note_recall_prompt
from graph.state import AgentState


def note_recall_node(state: AgentState, llm, memory_store) -> AgentState:
    """
    记忆节点的职责：
    1. 从 memory_store 检索历史笔记
    2. 将笔记交给大模型整理润色
    3. 返回回答
    4. 更新对话历史
    """
    question = state["question"]
    messages = state.get("messages", [])
    query = state.get("rewritten_question", question)

    print("🔍 正在检索记忆库...")
    memories = memory_store.search(query)
    context = "\n".join(memories) if memories else "没有找到相关历史记录。"

    print("🧠 正在帮用户整理回忆...")
    recall_chain = note_recall_prompt | llm

    response = recall_chain.invoke({
        "context": context,
        "messages": messages,
        "question": question
    })

    print("✅ 回忆整理完成")

    return {
        **state,
        "context": context,
        "answer": response.content,
        "messages": [
            AIMessage(content=response.content)
        ],
        "metadata": {
            **state.get("metadata", {}),
            "memory_count": len(memories)
        }
    }
