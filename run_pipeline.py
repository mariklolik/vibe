#!/usr/bin/env python3
"""Main entry point for the autonomous research pipeline.

Usage:
    # Single topic
    python run_pipeline.py "efficient attention mechanisms for long sequences"

    # Multiple topics (sequential)
    python run_pipeline.py --topics topics.txt

    # With options
    python run_pipeline.py "topic" --conference neurips --model sonnet --min-novelty 0.7

    # Batch parallel (not recommended — sequential is more reliable)
    python run_pipeline.py --topics topics.txt --parallel 2
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.agents.orchestrator import Orchestrator
from src.agents.client import get_token_usage


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Research Pipeline"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (single topic mode)",
    )
    parser.add_argument(
        "--topics",
        type=str,
        help="Path to file with topics (one per line)",
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="icml",
        choices=["neurips", "icml", "iclr", "cvpr", "acl", "aaai"],
        help="Target conference (default: icml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        help="Model to use (default: sonnet). Proxy normalizes: sonnet, opus, haiku",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Anthropic API base URL (default: http://localhost:3456)",
    )
    parser.add_argument(
        "--min-novelty",
        type=float,
        default=0.7,
        help="Minimum novelty threshold for idea selection (default: 0.7)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Max revision loops after review fails (default: 2)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of topics to run in parallel (default: 1 = sequential)",
    )
    parser.add_argument(
        "--projects-dir",
        type=str,
        default=None,
        help="Directory for research projects (default: ~/research-projects)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Collect topics
    topics = []
    if args.topic:
        topics = [args.topic]
    elif args.topics:
        topics_path = Path(args.topics)
        if not topics_path.exists():
            print(f"Error: topics file not found: {args.topics}")
            sys.exit(1)
        topics = [
            line.strip()
            for line in topics_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Starting pipeline for {len(topics)} topic(s)")
    print(f"Conference: {args.conference}")
    print(f"Model: {args.model}")
    print(f"Min novelty: {args.min_novelty}")
    print()

    # Create orchestrator
    kwargs = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.projects_dir:
        kwargs["projects_dir"] = args.projects_dir

    orchestrator = Orchestrator(**kwargs)

    # Run
    if len(topics) == 1:
        result = asyncio.run(
            orchestrator.run_full_pipeline(
                topics[0],
                conference=args.conference,
                min_novelty=args.min_novelty,
                max_revisions=args.max_revisions,
            )
        )
        results = [result]
    else:
        results = asyncio.run(
            orchestrator.run_batch(
                topics,
                conference=args.conference,
                parallel=args.parallel,
            )
        )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
            continue
        status = r.get("final_status", "UNKNOWN")
        topic = r.get("topic", "?")
        elapsed = r.get("elapsed_seconds", "?")
        project = r.get("project_dir", "?")
        tokens = r.get("token_usage", {})
        print(f"  [{status}] {topic}")
        print(f"    Project: {project}")
        print(f"    Elapsed: {elapsed}s")
        print(f"    Tokens: {tokens}")

        # Per-phase breakdown
        for phase, data in r.get("phases", {}).items():
            if isinstance(data, dict):
                phase_elapsed = data.get("elapsed_seconds", "?")
                phase_tokens = data.get("tokens_used", {})
                print(f"    {phase}: {phase_elapsed}s, tokens={phase_tokens}")

        pdf_path = r.get("phases", {}).get("writing", {}).get("pdf_path")
        if pdf_path:
            print(f"    PDF: {pdf_path}")
        print()

    # Final token usage
    print(f"Total token usage: {get_token_usage()}")


if __name__ == "__main__":
    main()
