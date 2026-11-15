"""Command line entry point.

    python -m stock_analyzer AAPL
    python -m stock_analyzer AAPL MSFT NVDA --save
    python -m stock_analyzer TSLA --focus "Is the valuation justified by growth?"
    python -m stock_analyzer AAPL --data-only      # no Claude call, just the numbers
    python -m stock_analyzer --sample              # offline demo with synthetic data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .analyzer import DEFAULT_MODEL, AnalysisError, analyze
from .data import fetch_snapshot
from .prompt import SYSTEM_PROMPT, build_user_prompt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stock_analyzer", description="Claude-powered stock reports")
    p.add_argument("tickers", nargs="*", help="One or more ticker symbols, e.g. AAPL MSFT")
    p.add_argument("--period", default="2y", help="Price history to pull (default 2y)")
    p.add_argument("--benchmark", default="SPY", help="Benchmark for beta (default SPY)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                   help="How hard Claude thinks (default medium)")
    p.add_argument("--focus", help="An extra question for Claude to address")
    p.add_argument("--save", action="store_true", help="Write each report to reports/")
    p.add_argument("--out-dir", default="reports", help="Folder for --save (default reports/)")
    p.add_argument("--data-only", action="store_true", help="Print the data snapshot JSON and stop")
    p.add_argument("--show-prompt", action="store_true", help="Print the full prompt and stop")
    p.add_argument("--sample", action="store_true", help="Use synthetic offline data (ticker DEMO)")
    args = p.parse_args(argv)

    if not args.tickers and not args.sample:
        p.error("give at least one ticker, or use --sample")

    jobs = ["DEMO"] if args.sample else [t.upper() for t in args.tickers]
    exit_code = 0

    for ticker in jobs:
        try:
            if args.sample:
                from .sample import sample_snapshot
                snap = sample_snapshot()
            else:
                print(f"Fetching data for {ticker}...", file=sys.stderr)
                snap = fetch_snapshot(ticker, period=args.period, benchmark=args.benchmark)
        except Exception as e:  # network issues, bad symbols, etc.
            print(f"[{ticker}] could not fetch data: {e}", file=sys.stderr)
            exit_code = 1
            continue

        if args.data_only:
            print(json.dumps(snap.to_dict(), indent=2, default=str))
            continue
        if args.show_prompt:
            print("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n=== USER ===\n" + build_user_prompt(snap, args.focus))
            continue

        print(f"Analyzing {ticker} with {args.model}...\n", file=sys.stderr)
        try:
            report = analyze(
                snap, model=args.model, effort=args.effort, focus=args.focus,
                on_text=lambda s: print(s, end="", flush=True),
            )
        except AnalysisError as e:
            print(f"\n[{ticker}] {e}", file=sys.stderr)
            exit_code = 1
            continue
        print("\n")

        if args.save:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            path = out / f"{ticker}-{stamp}.md"
            path.write_text(report)
            (out / f"{ticker}-{stamp}.json").write_text(json.dumps(snap.to_dict(), indent=2, default=str))
            print(f"Saved {path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
