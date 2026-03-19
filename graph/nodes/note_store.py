from graph.state import AgentState


summary_triggers = ["总结", "总结一下", "帮我总结"]
correction_triggers = ["纠错", "更正", "修正"]
reminder_triggers = ["记住", "牢记", "提醒我记住", "记录"]


def detect_save_mode(text: str) -> str:
    """
    判断这次保存属于哪种“保存模式”
    注意：这里不是笔记内容类型，而是保存意图/模式
    """
    text = (text or "").strip()

    if any(t in text for t in correction_triggers):
        return "correction_note"
    if any(t in text for t in summary_triggers):
        return "summary_note"
    if any(t in text for t in reminder_triggers):
        return "raw_note"
    return "raw_note"


def resolve_content_to_save(state: AgentState) -> tuple[str, str]:
    """
    决定要保存什么内容，并显式返回内容来源
    返回: (content, content_source)
    """
    metadata = state.get("metadata", {}) or {}
    original_text = (state.get("question") or "").strip()

    if metadata.get("save_content"):
        return str(metadata["save_content"]).strip(), "save_content"

    if metadata.get("last_answer_summary"):
        return str(metadata["last_answer_summary"]).strip(), "last_answer_summary"

    return original_text, "question"


def build_note_store_meta(state: AgentState) -> dict:
    """
    构造 note_store 的中间状态
    统一收进 metadata['note_store']
    """
    metadata = state.get("metadata", {}) or {}
    original_text = (state.get("question") or "").strip()
    rewritten_text = (state.get("rewritten_question") or "").strip()

    text_for_judge = rewritten_text or original_text
    content_to_save, content_source = resolve_content_to_save(state)
    save_mode = detect_save_mode(text_for_judge)

    note_type = metadata.get("note_type") or "general"
    concept = metadata.get("concept")
    importance = metadata.get("importance")
    timestamp = metadata.get("timestamp")
    source = metadata.get("source") or "user"

    return {
        "content": content_to_save.strip(),
        "content_source": content_source,
        "save_mode": save_mode,
        "note_type": note_type,
        "concept": concept,
        "importance": importance,
        "timestamp": timestamp,
        "source": source,
        "saved": None,
        "saved_text": "",
        "error": None,
    }


def note_store_prepare_node(state: AgentState) -> AgentState:
    """
    准备 note_store 所需的全部中间信息
    """
    metadata = state.get("metadata", {}) or {}
    note_store_meta = build_note_store_meta(state)
    content_to_save = note_store_meta["content"]

    if not content_to_save:
        print("🤖 Storage Agent: 没有可存储内容。")
        note_store_meta["saved"] = False
        note_store_meta["error"] = "empty_content"

        return {
            **state,
            "answer": "我理解你想记录内容，但当前没有提取到可保存的信息。",
            "metadata": {
                **metadata,
                "note_store": note_store_meta,
            },
        }

    return {
        **state,
        "metadata": {
            **metadata,
            "note_store": note_store_meta,
        },
    }


def note_store_save_node(state: AgentState, memory_store) -> AgentState:
    """
    读取 note_store 中间状态并执行写库
    """
    metadata = state.get("metadata", {}) or {}
    note_store_meta = metadata.get("note_store", {}) or {}

    content_to_save = (note_store_meta.get("content") or "").strip()
    concept = note_store_meta.get("concept")
    note_type = note_store_meta.get("note_type", "general")
    save_mode = note_store_meta.get("save_mode", "raw_note")
    importance = note_store_meta.get("importance")
    timestamp = note_store_meta.get("timestamp")
    source = note_store_meta.get("source") or "user"

    if not content_to_save:
        answer = state.get("answer") or "我理解你想记录内容，但当前没有提取到可保存的信息。"
        return {
            **state,
            "answer": answer,
        }

    try:
        memory_store.add_note(
            content=content_to_save,
            concept=concept,
            note_type=note_type,
            save_mode=save_mode,
            importance=importance,
            timestamp=timestamp,
            source=source,
        )

        print(f"🤖 Storage Agent: 已存储笔记({save_mode}/{note_type}) -> {content_to_save[:40]}...")

        updated_note_store_meta = {
            **note_store_meta,
            "saved": True,
            "saved_text": content_to_save[:120],
            "error": None,
        }

        return {
            **state,
            "answer": f"已帮你记录：{content_to_save[:80]}",
            "metadata": {
                **metadata,
                "note_store": updated_note_store_meta,
            },
        }

    except Exception as e:
        print(f"❌ Storage Agent: 存储笔记失败，原因：{e}")

        updated_note_store_meta = {
            **note_store_meta,
            "saved": False,
            "error": str(e),
        }

        return {
            **state,
            "answer": "存储失败，稍后请重试。",
            "metadata": {
                **metadata,
                "note_store": updated_note_store_meta,
            },
        }


def note_store_node(state: AgentState, memory_store) -> AgentState:
    """
    兼容旧调用方式：串行执行 prepare + save
    """
    state = note_store_prepare_node(state)
    state = note_store_save_node(state, memory_store)
    return state
