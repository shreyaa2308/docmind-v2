"""CLI entrypoint for asking DocMind v2 a question.

Usage:
    python scripts/query.py "What does the document say about ...?"
"""

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution (``python scripts/query.py``) from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.chain import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about indexed DocMind PDFs.")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()
    print(json.dumps(answer_question(args.question), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
