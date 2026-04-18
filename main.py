import argparse
import uuid

from langchain_core.messages import HumanMessage


def run_session(agent, session_id: str, questions: list[str]) -> None:
    config = {"configurable": {"thread_id": session_id}}
    print(f"当前会话 ID: {session_id}\n")
    print("=" * 60)

    for question in questions:
        question = question.strip()
        if not question:
            continue

        print("-" * 60)
        result = agent.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )
        route = result.get("route", "unknown")
        answer = result.get("answer", "无回答")

        print(f"[路由系统] 走 {route} 通道")
        print(f"[AI 助手]\n{answer}")
        print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 学习助手")
    parser.add_argument("-q", "--question", action="append", help="单轮或多轮命令行提问")
    parser.add_argument("-i", "--interactive", action="store_true", help="启动命令行交互模式")
    parser.add_argument("--thread-id", help="指定会话 ID，不传则自动生成")
    parser.add_argument("--ingest", help="导入单个文档到知识库")
    parser.add_argument("--web", action="store_true", help="显式启动 Gradio Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务端口")
    parser.add_argument("--share", action="store_true", help="是否启用 Gradio share 链接")
    return parser


def run_cli(args: argparse.Namespace) -> None:
    from core.runtime import build_runtime
    from graph.workflow import create_workflow
    from rag.document_manager import DocumentManager

    runtime = build_runtime()

    if args.ingest:
        doc_manager = DocumentManager(runtime["retriever"])
        result = doc_manager.process_and_store(args.ingest)
        print(result["message"])
        if result.get("chunk_count"):
            print(f"共生成 {result['chunk_count']} 个知识块")
        return

    agent = create_workflow(runtime)
    session_id = args.thread_id or str(uuid.uuid4())

    if args.question:
        run_session(agent, session_id, args.question)
        return

    if args.interactive:
        print("进入交互模式，输入 'exit' 或 'quit' 结束")
        while True:
            try:
                text = input("问题: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已退出交互模式")
                break

            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            run_session(agent, session_id, [text])
        return

    launch_web(host=args.host, port=args.port, share=args.share)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.web:
        from ui.simple_web import launch_web

        launch_web(host=args.host, port=args.port, share=args.share)
        return

    if not any([args.ingest, args.question, args.interactive]):
        from ui.simple_web import launch_web

        launch_web(host=args.host, port=args.port, share=args.share)
        return

    run_cli(args)


if __name__ == "__main__":
    main()
