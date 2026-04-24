#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Add rag/ to path so relative imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import HybridRagPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALIDATION_QUERIES = [
    "What caused the payments outage on June 14th?",
    "Show me the runbook steps for a payments circuit breaker being open",
    "Which logs had ERROR level during INC-4021?",
    "Was there a breaking deployment before the incident?",
    "How long did INC-4021 take to resolve?",
]


def _print_result(result: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Q: {result['question']}")
    print(f"{'='*70}")
    print(result["answer"])
    print(f"\nLatency: {result['latency_ms']:.0f}ms")
    print("\nSources:")
    for i, src in enumerate(result["sources"], 1):
        dataset = src.get("source_dataset", "unknown")
        service = src.get("service") or "N/A"
        inc_id = src.get("incident_id") or "N/A"
        ts = src.get("timestamp") or "N/A"
        score = src.get("cross_encoder_score", "N/A")
        print(f"  [{i}] {dataset} | svc={service} | inc={inc_id} | ts={ts} | score={score:.4f}" if isinstance(score, float) else f"  [{i}] {dataset} | svc={service} | inc={inc_id} | ts={ts}")
    print()


def cmd_build_index(args) -> None:
    pipeline = HybridRagPipeline()
    pipeline.build_index()

    print("\n--- Running validation queries ---\n")
    for q in VALIDATION_QUERIES:
        try:
            result = pipeline.query(q)
            _print_result(result)
        except Exception as e:
            logger.error("Validation query failed: %s — %s", q, e)
            print(f"\nQuery failed: {q}\nError: {e}\n")


def cmd_query(args) -> None:
    pipeline = HybridRagPipeline()
    pipeline.auto_load()
    result = pipeline.query(args.query)
    _print_result(result)


def cmd_interactive(args) -> None:
    pipeline = HybridRagPipeline()
    pipeline.auto_load()

    print("\nInteractive mode — type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            result = pipeline.query(question)
            _print_result(result)
        except Exception as e:
            logger.error("Query error: %s", e)
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Incident Investigation Copilot — Hybrid RAG Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --build-index flag (also support as positional-style flag)
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build FAISS + BM25 index from source datasets",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUESTION",
        help='Run a single query (e.g. --query "Why did payments go down?")',
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query loop",
    )

    args = parser.parse_args()

    if args.build_index:
        cmd_build_index(args)
    elif args.query:
        cmd_query(args)
    elif args.interactive:
        cmd_interactive(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
