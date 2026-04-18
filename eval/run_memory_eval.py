import argparse
import json
import sys
from pathlib import Path

from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.base_retriever import BaseQdrantStore
from memory.retriever import MemoryRetriever


FIXTURE_PATH = ROOT / "eval" / "memory_eval_fixtures.jsonl"
QUERY_PATH = ROOT / "eval" / "memory_eval_queries.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_PREFIX = "memory_eval"
EVAL_COLLECTION = "user_memory_note_eval_v1"


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_store(collection_name: str) -> BaseQdrantStore:
    return BaseQdrantStore(collection_name=collection_name, collection_label="评测记忆集合")


def build_retriever(collection_name: str) -> MemoryRetriever:
    retriever = MemoryRetriever()
    retriever.collection_name = collection_name
    retriever.collection_label = "评测记忆集合"
    retriever.vector_store = None
    return retriever


def reset_and_seed_collection(collection_name: str, fixtures: list[dict]) -> None:
    store = build_store(collection_name)
    if store.client.collection_exists(collection_name):
        store.client.delete_collection(collection_name)

    vector_store = store._get_vector_store()
    docs = []
    for row in fixtures:
        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "memory_type": "note",
                    "note_type": row["note_type"],
                    "save_mode": row["save_mode"],
                    "concept": row["concept"],
                    "importance": row["importance"],
                    "timestamp": row["timestamp"],
                    "source": row["source"],
                    "eval_note_id": row["eval_note_id"],
                    "label": row["label"],
                },
            )
        )
    vector_store.add_documents(docs)


def semantic_only_search(retriever: MemoryRetriever, query: str, k: int = 3):
    return retriever._similarity_search_items(query, k=k, metadata_filter={})


def memory_search(retriever: MemoryRetriever, query: str, k: int = 3):
    return retriever.retrieve(query, k=k, metadata_filter={})


def find_rank(items, expected_note_id: str) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item.metadata.get("eval_note_id") == expected_note_id:
            return idx
    return None


def summarize_items(items) -> list[dict]:
    rows = []
    for item in items[:3]:
        rows.append(
            {
                "eval_note_id": item.metadata.get("eval_note_id"),
                "label": item.metadata.get("label"),
                "concept": item.metadata.get("concept"),
                "importance": item.metadata.get("importance"),
                "timestamp": item.metadata.get("timestamp"),
                "similarity": item.retrieval_meta.get("similarity"),
                "keyword_score": item.retrieval_meta.get("keyword_score"),
                "importance_score": item.retrieval_meta.get("importance_score"),
                "recency_score": item.retrieval_meta.get("recency_score"),
                "rerank_score": item.retrieval_meta.get("rerank_score"),
                "content_preview": item.content[:160].replace("\n", " "),
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(collection_name: str, fixture_path: Path, query_path: Path):
    fixtures = load_jsonl(fixture_path)
    queries = load_jsonl(query_path)
    reset_and_seed_collection(collection_name, fixtures)

    retriever = build_retriever(collection_name)

    semantic_top1 = []
    semantic_top3 = []
    semantic_mrr = []
    memory_top1 = []
    memory_top3 = []
    memory_mrr = []
    memory_better = 0
    semantic_better = 0
    tie = 0
    results = []

    for row in queries:
        print("=" * 80)
        print(f"[{row['id']}] {row['query']}")

        semantic_items = semantic_only_search(retriever, row["query"], k=3)
        memory_items = memory_search(retriever, row["query"], k=3)

        semantic_rank = find_rank(semantic_items, row["expected_note_id"])
        memory_rank = find_rank(memory_items, row["expected_note_id"])

        semantic_top1.append(1.0 if semantic_rank == 1 else 0.0)
        semantic_top3.append(1.0 if semantic_rank is not None else 0.0)
        semantic_mrr.append(1 / semantic_rank if semantic_rank else 0.0)
        memory_top1.append(1.0 if memory_rank == 1 else 0.0)
        memory_top3.append(1.0 if memory_rank is not None else 0.0)
        memory_mrr.append(1 / memory_rank if memory_rank else 0.0)

        memory_score = 1 / memory_rank if memory_rank else 0.0
        semantic_score = 1 / semantic_rank if semantic_rank else 0.0
        if memory_score > semantic_score:
            memory_better += 1
        elif memory_score < semantic_score:
            semantic_better += 1
        else:
            tie += 1

        results.append(
            {
                "id": row["id"],
                "query": row["query"],
                "reason": row.get("reason"),
                "expected_note_id": row["expected_note_id"],
                "semantic_top1": semantic_rank == 1,
                "semantic_top3": semantic_rank is not None,
                "semantic_rank": semantic_rank,
                "semantic_mrr": 1 / semantic_rank if semantic_rank else 0.0,
                "memory_top1": memory_rank == 1,
                "memory_top3": memory_rank is not None,
                "memory_rank": memory_rank,
                "memory_mrr": 1 / memory_rank if memory_rank else 0.0,
                "memory_better": memory_score > semantic_score,
                "semantic_top_items": summarize_items(semantic_items),
                "memory_top_items": summarize_items(memory_items),
            }
        )

    metrics = {
        "total": len(queries),
        "semantic_top1": mean(semantic_top1),
        "semantic_top3": mean(semantic_top3),
        "semantic_mrr": mean(semantic_mrr),
        "memory_top1": mean(memory_top1),
        "memory_top3": mean(memory_top3),
        "memory_mrr": mean(memory_mrr),
        "top1_delta": mean(memory_top1) - mean(semantic_top1),
        "top3_delta": mean(memory_top3) - mean(semantic_top3),
        "mrr_delta": mean(memory_mrr) - mean(semantic_mrr),
        "memory_better_cases": memory_better,
        "semantic_better_cases": semantic_better,
        "tie_cases": tie,
        "collection_name": collection_name,
    }
    return results, metrics


def write_outputs(results: list[dict], metrics: dict, prefix: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RESULTS_DIR / f"{prefix}_results.jsonl"
    summary_path = RESULTS_DIR / f"{prefix}_summary.md"

    with detail_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Memory Eval Summary",
        "",
        f"- Collection: {metrics['collection_name']}",
        f"- Total: {metrics['total']}",
        f"- Semantic Top1: {metrics['semantic_top1']:.2%}",
        f"- Memory Top1: {metrics['memory_top1']:.2%}",
        f"- Top1 Delta: {metrics['top1_delta']:+.2%}",
        f"- Semantic Top3: {metrics['semantic_top3']:.2%}",
        f"- Memory Top3: {metrics['memory_top3']:.2%}",
        f"- Top3 Delta: {metrics['top3_delta']:+.2%}",
        f"- Semantic MRR: {metrics['semantic_mrr']:.4f}",
        f"- Memory MRR: {metrics['memory_mrr']:.4f}",
        f"- MRR Delta: {metrics['mrr_delta']:+.4f}",
        f"- Memory Better Cases: {metrics['memory_better_cases']}",
        f"- Semantic Better Cases: {metrics['semantic_better_cases']}",
        f"- Tie Cases: {metrics['tie_cases']}",
        "",
        "## Case Breakdown",
    ]

    for item in results:
        lines.append(
            f"- {item['id']}: semantic rank={item['semantic_rank']}, memory rank={item['memory_rank']}, "
            f"semantic top1={item['semantic_top1']}, memory top1={item['memory_top1']}"
        )

    improved = [item for item in results if item["memory_mrr"] > item["semantic_mrr"]]
    regressed = [item for item in results if item["memory_mrr"] < item["semantic_mrr"]]

    lines.extend(["", "## Improved Cases"])
    if improved:
        for item in improved:
            lines.append(f"- {item['id']}: {item['query']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Regressed Cases"])
    if regressed:
        for item in regressed:
            lines.append(f"- {item['id']}: {item['query']}")
    else:
        lines.append("- None")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return detail_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory recall against semantic-only retrieval.")
    parser.add_argument("--fixtures", default=str(FIXTURE_PATH))
    parser.add_argument("--queries", default=str(QUERY_PATH))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--collection", default=EVAL_COLLECTION)
    args = parser.parse_args()

    fixture_path = Path(args.fixtures).resolve()
    query_path = Path(args.queries).resolve()
    results, metrics = evaluate(args.collection, fixture_path, query_path)
    detail_path, summary_path = write_outputs(results, metrics, args.prefix)

    print(f"Semantic Top1: {metrics['semantic_top1']:.2%}")
    print(f"Memory Top1: {metrics['memory_top1']:.2%}")
    print(f"Top1 Delta: {metrics['top1_delta']:+.2%}")
    print(f"Semantic Top3: {metrics['semantic_top3']:.2%}")
    print(f"Memory Top3: {metrics['memory_top3']:.2%}")
    print(f"Top3 Delta: {metrics['top3_delta']:+.2%}")
    print(f"Semantic MRR: {metrics['semantic_mrr']:.4f}")
    print(f"Memory MRR: {metrics['memory_mrr']:.4f}")
    print(f"MRR Delta: {metrics['mrr_delta']:+.4f}")
    print(f"Memory better: {metrics['memory_better_cases']}")
    print(f"Semantic better: {metrics['semantic_better_cases']}")
    print(f"Tie: {metrics['tie_cases']}")
    print(f"Detailed results written to: {detail_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
