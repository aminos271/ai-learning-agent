from graph.state import AgentState


def note_store_node(state: AgentState, memory_store) -> AgentState:
    metadata = state.get("metadata", {}) or {}
    original_text = (state.get("question") or "").strip()
    rewritten_text = (state.get("rewritten_question") or "").strip()

    text_for_judge = rewritten_text or original_text
    concept = metadata.get("concept", "general")

    summary_triggers = ["总结", "总结一下", "帮我总结"]
    correction_triggers = ["纠错", "更正", "修正"]
    reminder_triggers = ["记住", "牢记", "提醒我记住", "记录"]

    note_type = "note"
    if any(t in text_for_judge for t in correction_triggers):
        note_type = "correction"
    elif any(t in text_for_judge for t in summary_triggers):
        note_type = "summary"
    elif any(t in text_for_judge for t in reminder_triggers):
        note_type = "reminder"

    content_to_save = (
        metadata.get("save_content")
        or metadata.get("last_answer_summary")
        or rewritten_text
        or original_text
    )

    if not content_to_save:
        print("🤖 Storage Agent: 没有可存储内容。")
        return {
            **state,
            "answer": "我理解你想记录内容，但当前没有提取到可保存的信息。",
            "metadata": {
                **metadata,
                "note_saved": False
            }
        }

    memory_store.add_note(
        content=content_to_save,
        concept=concept,
        note_type=note_type,
        importance=0.8,
        source="user",
    )

    print(f"🤖 Storage Agent: 已存储笔记({note_type}) -> {content_to_save[:40]}...")

    return {
        **state,
        "answer": f"已帮你记录：{content_to_save[:80]}",
        "metadata": {
            **metadata,
            "note_saved": True,
            "note_type": note_type,
            "note_saved_text": content_to_save[:120]
        }
    }