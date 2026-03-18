import argparse
import uuid
from graph.workflow import create_workflow
from core.runtime import build_runtime


def run_session(agent, session_id, questions):
    config = {"configurable": {"thread_id": session_id}}
    print(f"🎫 当前会话 ID: {session_id}\n")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        question = question.strip()
        if not question:
            continue

        print("-" * 60)

        result = agent.invoke({"question": question}, config=config)
        route = result.get("route", "未知")
        answer = result.get("answer", "无回答")

        print(f"🚦 [路由系统]: 走 {route} 通道")
        print(f"🤖 [AI 助手]:\n{answer}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="终极 AI 学习助手 CLI")
    parser.add_argument("-q", "--question", action="append", help="一次或多次输入问题")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式输入问题")
    parser.add_argument("--thread-id", help="指定会话ID，默认随机生成")

    args = parser.parse_args()

    runtime = build_runtime()
    agent = create_workflow(runtime)
    session_id = args.thread_id or str(uuid.uuid4())

    if args.question:
        run_session(agent, session_id, args.question)
    elif args.interactive:
        print("进入交互模式，输入 'exit' 或 'quit' 结束")
        interactive_questions = []
        count = 0
        while True:
            try:
                text = input(f"问题 #{count+1}: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已退出交互模式")
                break

            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            interactive_questions.append(text)
            count += 1
            run_session(agent, session_id, [text])

        if interactive_questions:
            print("交互模式已结束")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()