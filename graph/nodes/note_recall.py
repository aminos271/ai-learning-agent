from langchain_core.messages import AIMessage
from graph.prompts import note_recall_prompt
from graph.state import AgentState


def _extract_recall_filters(state: AgentState) -> dict:
    metadata = state.get("metadata", {}) or {}
    raw_filters = (
        metadata.get("recall_filters")
        or metadata.get("memory_filters")
        or {}
    )

    if not isinstance(raw_filters, dict):
        return {}

    allowed_keys = ("concept", "note_type", "save_mode", "source")
    return {
        key: raw_filters[key]
        for key in allowed_keys
        if raw_filters.get(key) is not None
    }


def note_recall_node(state: AgentState, llm, memory_store) -> AgentState:
    """
    recall 节点职责：
    1. 检索 memory
    2. 构建 context
    3. 生成回答
    4. 写入 metadata["recall"]
    """
    question = state["question"]
    messages = state.get("messages", [])
    query = state.get("rewritten_question") or question
    recall_filters = _extract_recall_filters(state)

    print("🔍 正在检索记忆库...")
    memories = memory_store.search_notes(
        query,
        metadata_filter=recall_filters,
    )

    context = "\n".join(item.content for item in memories) if memories else "没有找到相关历史记录。"

    print("🧠 正在帮用户整理回忆...")
    recall_chain = note_recall_prompt | llm

    response = recall_chain.invoke({
        "context": context,
        "messages": messages,
        "question": question
    })

    print("✅ 回忆整理完成")

    metadata = state.get("metadata", {}) or {}
    recall_meta = {
        "query": query,
        "filters": recall_filters,
        "retrieved_count": len(memories),
        "top_items": [
            {
                "content_preview": item.content[:120],
                "metadata": item.metadata,
                "retrieval_meta": item.retrieval_meta,
            }
            for item in memories[:3]
        ],
    }

    return {
        **state,
        "answer": response.content,
        "messages": [
            AIMessage(content=response.content)
        ],
        "metadata": {
            **metadata,
            "recall": recall_meta
        }
    }
