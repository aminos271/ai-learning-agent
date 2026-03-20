from langchain_core.messages import HumanMessage, AIMessage
from graph.prompts import rewrite_prompt
from graph.state import AgentState


def rewrite_node(state: AgentState, llm) -> AgentState:
    """
    改写节点：改写后作为辅助语义表示，供 router 和后续节点共同使用
    """
    original_question = state["question"]
    messages = state.get("messages", [])

    history_text = ""
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"AI: {msg.content}\n"

    print("✍️ [重写节点] 正在分析上下文并重写问题...")
    chain = rewrite_prompt | llm

    response = chain.invoke({
        "messages": history_text,
        "question": original_question
    })

    rewritten_q = response.content.strip()
    if rewritten_q != original_question:
        print(f"🔄 [重写节点] 发生指代消解: \n   原问题: '{original_question}' \n   新问题: '{rewritten_q}'")
    else:
        print(f"⏩ [重写节点] 意图明确，未做修改。")

    return {**state,
            "rewritten_question": rewritten_q,
            "metadata": {
                **state.get("metadata", {}),
                "rewrite": {
                    "original_question": original_question,
                    "rewritten_question": rewritten_q,
                    "changed": rewritten_q != original_question
                }
            }
    }
