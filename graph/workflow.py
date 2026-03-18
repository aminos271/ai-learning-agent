from langgraph.graph import StateGraph, END
from graph.state import AgentState
from langgraph.checkpoint.memory import MemorySaver 
from graph.nodes import router_node, note_store_node, rag_node, note_recall_node, chat_node, rewrite_node

def create_workflow(runtime: dict):

    llm = runtime["llm"]
    retriever = runtime["retriever"]
    memory_store = runtime["memory_store"]

    # 1. 初始化图并传入状态定义
    workflow = StateGraph(AgentState)

    # 2. 添加节点
    workflow.add_node("rewrite", lambda state: rewrite_node(state, llm))
    workflow.add_node("note_store", lambda state: note_store_node(state, memory_store))
    workflow.add_node("router", lambda state: router_node(state, llm))
    workflow.add_node("rag", lambda state: rag_node(state, llm, retriever))
    workflow.add_node("note_recall", lambda state: note_recall_node(state, llm, memory_store))
    workflow.add_node("chat", lambda state: chat_node(state, llm))

    # 3. 设置入口起点
    workflow.set_entry_point("rewrite")

    workflow.add_edge("rewrite", "router")

    # 4. 添加条件边 (Conditional Edges)，决定下一步去哪
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"], # 根据状态中的 route 字段判断
        {
            "rag": "rag",
            "note_recall": "note_recall",
            "chat": "chat",
            "note_store":"note_store",
        }
    )


    # 5. 结束边
    workflow.add_edge("rag", END)
    workflow.add_edge("note_recall", END)
    workflow.add_edge("chat", END)
    workflow.add_edge("note_store", END)

    memory = MemorySaver()
    # 6. 编译图
    app = workflow.compile(checkpointer=memory)

    return app