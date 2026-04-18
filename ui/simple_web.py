import json
import uuid
from typing import Any

import gradio as gr
from langchain_core.messages import HumanMessage

from core.runtime import build_runtime
from graph.workflow import create_workflow
from rag.document_manager import DocumentManager


APP_CSS = """
body, .gradio-container { background: #f5f7fb; }
.gradio-container { max-width: 1240px !important; }
.app-shell {
    border: 1px solid #e5e7eb;
    background: rgba(255, 255, 255, 0.92);
    border-radius: 22px;
    box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(8px);
}
.panel {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    background: #ffffff;
}
.hero {
    padding: 10px 2px 8px 2px;
}
.hero h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
}
.hero p {
    margin: 10px 0 0 0;
    color: #475569;
    font-size: 15px;
}
"""


def _new_session_state() -> dict[str, Any]:
    return {
        "thread_id": str(uuid.uuid4()),
        "turn_count": 0,
    }


class WebAssistant:
    def __init__(self) -> None:
        runtime = build_runtime()
        self.agent = create_workflow(runtime)
        self.doc_manager = DocumentManager(runtime["retriever"])

    def ask(self, session_state: dict[str, Any], question: str) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(session_state or _new_session_state())
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空。")

        config = {"configurable": {"thread_id": state["thread_id"]}}
        result = self.agent.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )

        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        return state, result

    def ingest_files(self, file_paths: list[str] | None) -> tuple[str, str]:
        if not file_paths:
            return "未选择文件。", "[]"

        summaries: list[dict[str, Any]] = []
        success_count = 0

        for file_path in file_paths:
            result = self.doc_manager.process_and_store(file_path)
            summaries.append(
                {
                    "file": file_path,
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                    "chunk_count": result.get("chunk_count", 0),
                }
            )
            if result.get("success"):
                success_count += 1

        status = f"已处理 {len(file_paths)} 个文件，成功 {success_count} 个。"
        return status, json.dumps(summaries, ensure_ascii=False, indent=2)


def _format_status(state: dict[str, Any]) -> str:
    return f"会话 ID：`{state.get('thread_id', '-')}`  |  当前轮次：`{state.get('turn_count', 0)}`"


def _format_sources(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    rag_meta = metadata.get("rag") or {}
    recall_meta = metadata.get("recall") or {}
    note_store_meta = metadata.get("note_store") or {}

    if rag_meta.get("sources"):
        lines = ["**知识来源**"]
        for index, source in enumerate(rag_meta["sources"], start=1):
            retrieval_meta = source.get("retrieval_meta") or {}
            source_name = source.get("source") or "未知文档"
            section_path = source.get("section_path") or "-"
            similarity = retrieval_meta.get("similarity")
            rerank_score = retrieval_meta.get("rerank_score")
            lines.append(
                f"{index}. `{source_name}` | 章节：`{section_path}` | "
                f"similarity={similarity if similarity is not None else '-'} | "
                f"rerank={rerank_score if rerank_score is not None else '-'}"
            )
        return "\n".join(lines)

    if recall_meta:
        return f"**记忆召回**\n检索数量：`{recall_meta.get('retrieved_count', 0)}`"

    if note_store_meta:
        saved = "成功" if note_store_meta.get("saved") else "未保存"
        saved_text = note_store_meta.get("saved_text") or "-"
        return f"**笔记写入**\n状态：`{saved}`\n内容：{saved_text}"

    return "暂无来源信息。"


def _format_debug_info(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    payload = {
        "route": result.get("route"),
        "route_reason": metadata.get("route_reason"),
        "route_confidence": metadata.get("route_confidence"),
        "metadata": metadata,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_app() -> gr.Blocks:
    assistant = WebAssistant()

    with gr.Blocks(title="RAG Learning Assistant") as demo:
        initial_state = _new_session_state()
        session_state = gr.State(initial_state)

        gr.HTML(
            """
            <div class="hero">
              <h1>RAG Learning Assistant</h1>
              <p>文档上传、知识检索、记忆写入与多轮对话统一在一个简洁界面中完成。</p>
            </div>
            """
        )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=4, min_width=320):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 文档库")
                    upload_files = gr.File(
                        label="上传文档",
                        file_count="multiple",
                        type="filepath",
                    )
                    ingest_button = gr.Button("导入到知识库", variant="primary")
                    ingest_status = gr.Markdown("等待导入。")
                    ingest_result = gr.Code(label="导入结果", language="json", value="[]")

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 会话")
                    session_status = gr.Markdown(_format_status(initial_state))
                    reset_button = gr.Button("新建会话")
                    clear_button = gr.Button("清空界面记录")

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 本轮信息")
                    sources_markdown = gr.Markdown("暂无来源信息。")
                    debug_info = gr.Code(label="调试信息", language="json", value="{}")

            with gr.Column(scale=8, min_width=560):
                with gr.Group(elem_classes=["panel"]):
                    chatbot = gr.Chatbot(
                        label="对话",
                        height=620,
                    )
                    with gr.Row():
                        message_box = gr.Textbox(
                            label="输入问题",
                            placeholder="输入问题，或让助手记录/回忆你的笔记内容",
                            lines=3,
                            scale=8,
                        )
                        send_button = gr.Button("发送", variant="primary", scale=1)

        def handle_chat(
            message: str,
            history: list[dict[str, str]] | None,
            state: dict[str, Any],
        ) -> tuple[str, list[dict[str, str]], dict[str, Any], str, str, str]:
            history = history or []
            state = state or _new_session_state()
            message = (message or "").strip()

            if not message:
                return "", history, state, _format_status(state), "请输入问题。", "{}"

            try:
                state, result = assistant.ask(state, message)
                updated_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": result.get("answer", "")},
                ]
                return (
                    "",
                    updated_history,
                    state,
                    _format_status(state),
                    _format_sources(result),
                    _format_debug_info(result),
                )
            except Exception as exc:
                updated_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"处理失败：{exc}"},
                ]
                return (
                    "",
                    updated_history,
                    state,
                    _format_status(state),
                    "本轮执行失败。",
                    json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
                )

        def handle_ingest(file_paths: list[str] | None) -> tuple[str, str]:
            return assistant.ingest_files(file_paths)

        def reset_session() -> tuple[dict[str, Any], list[dict[str, str]], str, str, str]:
            state = _new_session_state()
            return state, [], _format_status(state), "暂无来源信息。", "{}"

        def clear_chat_only(state: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
            current_state = state or _new_session_state()
            return [], _format_status(current_state), "{}"

        send_button.click(
            handle_chat,
            inputs=[message_box, chatbot, session_state],
            outputs=[message_box, chatbot, session_state, session_status, sources_markdown, debug_info],
        )
        message_box.submit(
            handle_chat,
            inputs=[message_box, chatbot, session_state],
            outputs=[message_box, chatbot, session_state, session_status, sources_markdown, debug_info],
        )
        ingest_button.click(
            handle_ingest,
            inputs=[upload_files],
            outputs=[ingest_status, ingest_result],
        )
        reset_button.click(
            reset_session,
            outputs=[session_state, chatbot, session_status, sources_markdown, debug_info],
        )
        clear_button.click(
            clear_chat_only,
            inputs=[session_state],
            outputs=[chatbot, session_status, debug_info],
        )

    return demo


def launch_web(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=APP_CSS,
    )
