import json
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.runtime import build_runtime
from rag.rerank import rerank_documents


DATA_PATH = ROOT / "eval" / "rag_eval.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_rows(data_path: Path) -> list[dict]:
    lines = data_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def normalize(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def single_query_search(retriever, question: str, k: int = 3):
    active_filter = retriever._normalize_metadata_filter({})
    items = retriever._similarity_search_items(
        question,
        k=k,
        metadata_filter=active_filter,
    )
    return rerank_documents(question, items, active_filter, top_k=k)


def item_to_text(item) -> str:
    parts = [
        item.content,
        item.metadata.get("source", ""),
        item.metadata.get("section_path", ""),
        item.metadata.get("h1", ""),
        item.metadata.get("h2", ""),
        item.metadata.get("h3", ""),
    ]
    return normalize("\n".join(p for p in parts if p))


def find_first_hit(items, sample: dict) -> int | None:
    section_terms = [normalize(x) for x in sample.get("expected_section_terms", [])]
    content_terms = [normalize(x) for x in sample.get("expected_content_terms", [])]
    source = normalize(sample.get("expected_source"))

    for idx, item in enumerate(items, start=1):
        haystack = item_to_text(item)
        source_ok = not source or source in haystack
        section_ok = not section_terms or any(term in haystack for term in section_terms)
        content_ok = not content_terms or any(term in haystack for term in content_terms)

        if source_ok and (section_ok or content_ok):
            return idx
    return None


def summarize_top_items(items) -> list[dict]:
    summary = []
    for item in items[:3]:
        summary.append(
            {
                "source": item.metadata.get("source"),
                "section_path": item.metadata.get("section_path"),
                "chunk_id": item.metadata.get("chunk_id"),
                "similarity": item.retrieval_meta.get("similarity"),
                "keyword_score": item.retrieval_meta.get("keyword_score"),
                "rerank_score": item.retrieval_meta.get("rerank_score"),
                "matched_queries": item.retrieval_meta.get("matched_queries"),
                "content_preview": item.content[:180].replace("\n", " "),
            }
        )
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(data_path: Path):
    runtime = build_runtime()
    retriever = runtime["retriever"]
    llm = runtime["llm"]
    rows = load_rows(data_path)

    results = []
    single_hits = []
    multi_hits = []
    single_mrrs = []
    multi_mrrs = []
    multi_better = 0
    single_better = 0
    tie = 0

    for row in rows:
        print("=" * 80)
        print(f"[{row['id']}] {row['question']}")

        single_items = single_query_search(retriever, row["question"], k=3)
        multi_items = retriever.multi_query_search(row["question"], llm, k=3, metadata_filter={})

        single_rank = find_first_hit(single_items, row)
        multi_rank = find_first_hit(multi_items, row)

        single_hit = single_rank is not None
        multi_hit = multi_rank is not None

        single_mrr = 1 / single_rank if single_rank else 0.0
        multi_mrr = 1 / multi_rank if multi_rank else 0.0

        single_hits.append(1.0 if single_hit else 0.0)
        multi_hits.append(1.0 if multi_hit else 0.0)
        single_mrrs.append(single_mrr)
        multi_mrrs.append(multi_mrr)

        if multi_mrr > single_mrr:
            multi_better += 1
        elif multi_mrr < single_mrr:
            single_better += 1
        else:
            tie += 1

        result = {
            "id": row["id"],
            "question": row["question"],
            "reason": row.get("reason"),
            "expected_source": row.get("expected_source"),
            "expected_section_terms": row.get("expected_section_terms", []),
            "expected_content_terms": row.get("expected_content_terms", []),
            "single_hit@3": single_hit,
            "multi_hit@3": multi_hit,
            "single_rank": single_rank,
            "multi_rank": multi_rank,
            "single_mrr": single_mrr,
            "multi_mrr": multi_mrr,
            "multi_better": multi_mrr > single_mrr,
            "single_top_items": summarize_top_items(single_items),
            "multi_top_items": summarize_top_items(multi_items),
        }
        results.append(result)

    metrics = {
        "total": len(rows),
        "single_hit@3": mean(single_hits),
        "multi_hit@3": mean(multi_hits),
        "single_mrr@3": mean(single_mrrs),
        "multi_mrr@3": mean(multi_mrrs),
        "hit@3_delta": mean(multi_hits) - mean(single_hits),
        "mrr@3_delta": mean(multi_mrrs) - mean(single_mrrs),
        "multi_better_cases": multi_better,
        "single_better_cases": single_better,
        "tie_cases": tie,
    }
    return results, metrics


def write_outputs(results: list[dict], metrics: dict, detail_path: Path, summary_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with detail_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# RAG Eval Summary",
        "",
        f"- Total: {metrics['total']}",
        f"- Single Hit@3: {metrics['single_hit@3']:.2%}",
        f"- Multi Hit@3: {metrics['multi_hit@3']:.2%}",
        f"- Hit@3 Delta: {metrics['hit@3_delta']:+.2%}",
        f"- Single MRR@3: {metrics['single_mrr@3']:.4f}",
        f"- Multi MRR@3: {metrics['multi_mrr@3']:.4f}",
        f"- MRR@3 Delta: {metrics['mrr@3_delta']:+.4f}",
        f"- Multi Better Cases: {metrics['multi_better_cases']}",
        f"- Single Better Cases: {metrics['single_better_cases']}",
        f"- Tie Cases: {metrics['tie_cases']}",
        "",
        "## Case Breakdown",
    ]

    for item in results:
        lines.append(
            f"- {item['id']}: single rank={item['single_rank']}, multi rank={item['multi_rank']}, "
            f"single hit={item['single_hit@3']}, multi hit={item['multi_hit@3']}"
        )

    improved = [item for item in results if item["multi_mrr"] > item["single_mrr"]]
    regressed = [item for item in results if item["multi_mrr"] < item["single_mrr"]]

    lines.extend(["", "## Improved Cases"])
    if improved:
        for item in improved:
            lines.append(f"- {item['id']}: {item['question']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Regressed Cases"])
    if regressed:
        for item in regressed:
            lines.append(f"- {item['id']}: {item['question']}")
    else:
        lines.append("- None")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate single-query vs multi-query retrieval.")
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--prefix",
        default="rag_eval",
        help="Output file prefix under eval/results.",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    detail_path = RESULTS_DIR / f"{args.prefix}_results.jsonl"
    summary_path = RESULTS_DIR / f"{args.prefix}_summary.md"

    results, metrics = evaluate(data_path)
    write_outputs(results, metrics, detail_path, summary_path)

    print(f"Single Hit@3: {metrics['single_hit@3']:.2%}")
    print(f"Multi Hit@3: {metrics['multi_hit@3']:.2%}")
    print(f"Hit@3 Delta: {metrics['hit@3_delta']:+.2%}")
    print(f"Single MRR@3: {metrics['single_mrr@3']:.4f}")
    print(f"Multi MRR@3: {metrics['multi_mrr@3']:.4f}")
    print(f"MRR@3 Delta: {metrics['mrr@3_delta']:+.4f}")
    print(f"Multi better: {metrics['multi_better_cases']}")
    print(f"Single better: {metrics['single_better_cases']}")
    print(f"Tie: {metrics['tie_cases']}")
    print(f"Detailed results written to: {detail_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
