"""CLI entry point with subcommands.

Usage:
    python main.py research [--resume] [--config config.yaml]
    python main.py pipeline [--config config.yaml]
    python main.py results [--config config.yaml]
    python -m autovoiceevals research
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autovoiceevals",
        description="Autoresearch for Voice AI Agent Testing",
    )
    sub = parser.add_subparsers(dest="mode")

    # --- research subcommand ---
    res = sub.add_parser(
        "research",
        help="Iterative autoresearch loop (recommended)",
        description=(
            "Propose one change at a time, keep what improves the score, "
            "revert what doesn't. Runs forever until Ctrl+C."
        ),
    )
    res.add_argument(
        "--resume", action="store_true",
        help="Resume from previous autoresearch.json",
    )
    res.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )

    # --- pipeline subcommand ---
    pip = sub.add_parser(
        "pipeline",
        help="Single-pass attack -> improve -> verify",
        description=(
            "Generate adversarial attacks, analyze failures, rewrite the "
            "prompt, then verify. Good for a one-time audit."
        ),
    )
    pip.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )

    # --- compress subcommand ---
    cmp = sub.add_parser(
        "compress",
        help="Shorten prompt while preserving score",
        description=(
            "Iteratively compress the system prompt to reduce latency. "
            "Keeps compressions that maintain score above a configurable floor."
        ),
    )
    cmp.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    cmp.add_argument(
        "--target-words", type=int, default=1200,
        help="Target word count (default: 1200)",
    )
    cmp.add_argument(
        "--max-regression", type=float, default=0.03,
        help="Max allowed score drop from baseline (default: 0.03 = 3%%)",
    )
    cmp.add_argument(
        "--min-score", type=float, default=0.0,
        help="Absolute minimum score floor (default: 0.0, e.g., 0.80 for 80%%)",
    )
    cmp.add_argument(
        "--eval-suite", type=str, default="",
        help="Path to autoresearch.json to reuse its eval suite for comparable scores",
    )

    # --- results subcommand ---
    rsl = sub.add_parser(
        "results",
        help="View results from a completed run",
        description="Show summary, score progression, and best prompt from a completed run.",
    )
    rsl.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        print("\nRun 'python main.py research' to start autoresearch.")
        sys.exit(0)

    if args.mode == "results":
        from .results import show_results
        try:
            cfg = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        show_results(cfg)
        return

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "research":
        from .researcher import run
        run(cfg, resume=args.resume)
    elif args.mode == "compress":
        from .compress import run
        run(cfg, target_words=args.target_words, max_regression=args.max_regression,
            min_score=args.min_score, eval_suite_path=args.eval_suite)
    elif args.mode == "pipeline":
        from .pipeline import run
        run(cfg)
