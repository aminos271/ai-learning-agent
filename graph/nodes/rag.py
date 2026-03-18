from langchain_core.messages import AIMessage
from graph.prompts import rag_prompt
from graph.state import AgentState


def extract_summary_from_answer(answer: str) -> str:
    marker = "核心结论："
    if marker in answer:
        return answer.split(marker, 1)[1].strip()
    text = (answer or "").strip().replace("\n", " ")
    return text[:100]


def rag_node(state: AgentState, llm, retriever) -> AgentState:
    question = state["question"]
    messages = state.get("messages", [])
    query = state.get("rewritten_question", question)

    print("🔍 正在检索知识库...")
    docs = retriever.multi_query_search(query, llm)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else "未找到相关文档。"
    sources = [
        {
            "source": doc.metadata.get("source"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "section_path": doc.metadata.get("section_path")
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
        "context": context,
        "answer": answer,
        "messages":  [
            AIMessage(content=answer)
        ],
        "metadata": {
            **state.get("metadata", {}),
            "doc_count": len(docs) if docs else 0,
            "sources": sources,
            "last_answer_summary": summary,
            "save_content": summary
        }
    }