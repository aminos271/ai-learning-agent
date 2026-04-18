import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime import build_runtime
from graph.nodes.rewrite import rewrite_node
from graph.nodes.router import router_node


DATA_PATH = ROOT / "eval" / "router_eval.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DETAIL_PATH = RESULTS_DIR / "router_eval_results.jsonl"
SUMMARY_PATH = RESULTS_DIR / "router_eval_summary.md"


def to_messages(history: list[dict]) -> list:
    messages = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def load_rows() -> list[dict]:
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate() -> tuple[list[dict], dict]:
    runtime = build_runtime()
    llm = runtime["llm"]

    rows = load_rows()
    results = []
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = Counter()
    rewrite_changed_count = 0

    for row in rows:
        state = {
            "question": row["question"],
            "messages": to_messages(row.get("history", [])),
            "metadata": {},
        }

        state = rewrite_node(state, llm)
        state = router_node(state, llm)

        predicted = state.get("route")
        expected = row["expected_route"]
        is_correct = predicted == expected
        rewrite_meta = (state.get("metadata") or {}).get("rewrite", {})
        rewrite_changed = bool(rewrite_meta.get("changed", False))
        rewrite_changed_count += int(rewrite_changed)

        per_class[expected]["total"] += 1
        per_class[expected]["correct"] += int(is_correct)
        if not is_correct:
            confusion[f"{expected} -> {predicted}"] += 1

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "rewritten_question": state.get("rewritten_question"),
                "expected_route": expected,
                "predicted_route": predicted,
                "confidence": (state.get("metadata") or {}).get("route_confidence"),
                "rewrite_changed": rewrite_changed,
                "is_correct": is_correct,
                "reason": row.get("reason"),
            }
        )

    correct = sum(1 for item in results if item["is_correct"])
    total = len(results)
    accuracy = correct / total if total else 0.0

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "rewrite_changed_count": rewrite_changed_count,
        "per_class": {
            route: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for route, stats in sorted(per_class.items())
        },
        "confusion": dict(confusion.most_common()),
    }
    return results, metrics


def write_outputs(results: list[dict], metrics: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with DETAIL_PATH.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Router Eval Summary",
        "",
        f"- Total: {metrics['total']}",
        f"- Correct: {metrics['correct']}",
        f"- Accuracy: {metrics['accuracy']:.2%}",
        f"- Rewrite Changed: {metrics['rewrite_changed_count']}/{metrics['total']}",
        "",
        "## Per Class",
    ]

    for route, stats in metrics["per_class"].items():
        lines.append(
            f"- {route}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})"
        )

    lines.extend(["", "## Confusion Pairs"])
    if metrics["confusion"]:
        for pair, count in metrics["confusion"].items():
            lines.append(f"- {pair}: {count}")
    else:
        lines.append("- None")

    wrong = [item for item in results if not item["is_correct"]]
    lines.extend(["", "## Wrong Cases"])
    if wrong:
        for item in wrong:
            lines.append(
                f"- {item['id']}: expected `{item['expected_route']}`, got `{item['predicted_route']}` | "
                f"question: {item['question']} | rewritten: {item['rewritten_question']}"
            )
    else:
        lines.append("- None")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results, metrics = evaluate()
    write_outputs(results, metrics)

    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Rewrite changed: {metrics['rewrite_changed_count']}/{metrics['total']}")
    print("Per-class accuracy:")
    for route, stats in metrics["per_class"].items():
        print(f"  {route}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")

    if metrics["confusion"]:
        print("Confusion pairs:")
        for pair, count in metrics["confusion"].items():
            print(f"  {pair}: {count}")
    else:
        print("Confusion pairs: none")

    print(f"Detailed results written to: {DETAIL_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
