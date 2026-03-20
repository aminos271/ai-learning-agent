from langchain_core.messages import AIMessage
from graph.prompts import note_recall_prompt
from graph.state import AgentState


# 注意：
# filter工具类已实现，Retriever（检索器）也已完全支持。
# 但当前主流程尚未从用户查询或状态中生成 recall_filters。
# 因此，召回流水线中暂未实际启用 Metadata Filtering

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


def _format_recall_context(memories) -> str:
    if not memories:
        return "没有找到相关历史记录。"

    lines = []
    for idx, item in enumerate(memories, start=1):
        meta = item.metadata or {}
        rmeta = item.retrieval_meta or {}

        lines.append(
            "\n".join(
                [
                    f"[结果{idx}]",
                    f"内容: {item.content}",
                    f"相关性排序分: {rmeta.get('rerank_score', 0.0):.4f}",
                    f"语义分: {rmeta.get('similarity', 0.0):.4f}",
                    f"关键词分: {rmeta.get('keyword_score', 0.0):.4f}",
                    f"重要性分: {rmeta.get('importance_score', 0.0):.4f}",
                    f"时效分: {rmeta.get('recency_score', 0.0):.4f}",
                    f"概念: {meta.get('concept', '')}",
                    f"类型: {meta.get('note_type', '')}",
                    f"来源: {meta.get('source', '')}",
                    f"时间: {meta.get('timestamp', '')}",
                ]
            )
        )

    return "\n\n".join(lines)


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

    context = _format_recall_context(memories)
    
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
