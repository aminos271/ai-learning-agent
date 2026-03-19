from langchain_core.messages import AIMessage
from graph.prompts import rag_prompt
from graph.state import AgentState


def extract_summary_from_answer(answer: str) -> str:
    marker = "核心结论："
    if marker in answer:
        return answer.split(marker, 1)[1].strip()
    text = (answer or "").strip().replace("\n", " ")
    return text[:100]


def _extract_rag_metadata_filter(state: AgentState) -> dict:
    metadata = state.get("metadata", {}) or {}
    raw_filter = (
        metadata.get("rag_filter")
        or metadata.get("rag_filters")
        or metadata.get("metadata_filter")
        or {}
    )

    if not isinstance(raw_filter, dict):
        return {}

    allowed_keys = ("source", "section_path", "h1", "h2")
    return {
        key: raw_filter[key]
        for key in allowed_keys
        if raw_filter.get(key) is not None
    }


def rag_node(state: AgentState, llm, retriever) -> AgentState:
    question = state["question"]
    messages = state.get("messages", [])
    query = state.get("rewritten_question", question)
    metadata_filter = _extract_rag_metadata_filter(state)

    print("🔍 正在检索知识库...")
    docs = retriever.multi_query_search(query, llm, metadata_filter=metadata_filter)
    context = "\n\n".join(doc.content for doc in docs) if docs else "未找到相关文档。"
    sources = [
        {
            "source": doc.metadata.get("source"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "section_path": doc.metadata.get("section_path"),
            "h1": doc.metadata.get("h1"),
            "h2": doc.metadata.get("h2"),
            "retrieval_meta": doc.retrieval_meta,
        }
        for doc in docs
    ]

    print("🧠 正在呼叫大模型思考中...")
    rag_chain = rag_prompt | llm

    response = rag_chain.invoke({
        "context": context,
        "messages": messages,
        "question": query
    })

    answer = response.content
    summary = extract_summary_from_answer(answer)

    print("✅ RAG回答生成完成")

    return {
        **state,
        "answer": answer,
        "messages": [
            AIMessage(content=answer)
        ],
        "metadata": {
            **state.get("metadata", {}),
            "last_answer_summary": summary,
            "save_content": summary,
            "rag": {
                "query": query,
                "metadata_filter": metadata_filter,
                "doc_count": len(docs) if docs else 0,
                "sources": sources
            }
        }
    }
