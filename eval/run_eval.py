"""Run RAGAS quality gates against eval/golden_dataset.json."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.chain import answer_question
from src.config import settings
from src.retriever import build_retriever

logging.getLogger("httpx").setLevel(logging.WARNING)


def _answer_for_ragas(answer: str) -> str:
    """Exclude required source/page display metadata from semantic answer metrics."""
    return re.sub(r"\s*\[source:\s*[^\]]+\]", "", answer).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocMind RAGAS quality gates.")
    parser.add_argument("--output", type=Path, help="Optional JSON file for scores and gate status")
    args = parser.parse_args()
    rows = json.loads(Path("eval/golden_dataset.json").read_text(encoding="utf-8"))
    if not rows or rows[0]["question"].startswith("Replace this example"):
        raise SystemExit("Populate eval/golden_dataset.json before running evaluation.")
    results = []
    retriever = build_retriever()
    for row in rows:
        documents = retriever.invoke(row["question"])
        response = answer_question(row["question"])
        results.append({
            "question": row["question"],
            "answer": _answer_for_ragas(response["answer"]),
            "contexts": [document.page_content for document in documents],
            "ground_truth": row["ground_truth"],
        })
    scores = evaluate(
        Dataset.from_list(results),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    ).to_pandas().mean(numeric_only=True).to_dict()
    thresholds = {
        "faithfulness": settings.eval_faithfulness_min,
        "answer_relevancy": 0.70,
        "context_precision": 0.50,
        "context_recall": settings.eval_context_recall_min,
    }
    failures = [
        f"{name}={scores.get(name, 0):.3f} < {minimum:.3f}"
        for name, minimum in thresholds.items()
        if scores.get(name, 0) < minimum
    ]
    payload = {"scores": scores, "thresholds": thresholds, "passed": not failures, "failures": failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))
    if failures:
        print("RAGAS gate failed: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
