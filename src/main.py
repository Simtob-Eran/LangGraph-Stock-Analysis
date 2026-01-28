"""Main entry point for Stock Analysis System."""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from src.orchestrator import Orchestrator
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("main")

# Reports directory
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def get_report_filename(ticker: str) -> str:
    """
    Generate report filename with timestamp.

    Format: YYYY-MM-DD-HH-MM-SS-TICKER.md

    Args:
        ticker: Stock ticker symbol

    Returns:
        Formatted filename string
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{timestamp}-{ticker}.md"


def save_report(ticker: str, markdown_report: str) -> Path:
    """
    Save report to reports folder with timestamp filename.

    Args:
        ticker: Stock ticker symbol
        markdown_report: Report content in markdown format

    Returns:
        Path to saved report file
    """
    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename and save
    filename = get_report_filename(ticker)
    report_path = REPORTS_DIR / filename

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    return report_path


async def run_analysis(query: str, output_file: str = None, json_output: bool = False, auto_save: bool = True):
    """
    Run stock analysis for the given query.

    Args:
        query: Analysis query
        output_file: Optional file to save report
        json_output: Whether to output JSON instead of markdown
        auto_save: Whether to automatically save report to reports folder

    Returns:
        Tuple of (exit_code, ticker, report_path)
    """
    print("=" * 80)
    print("STOCK ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Model: {settings.OPENAI_MODEL}")
    print(f"Query: {query}")
    print("=" * 80)
    print()

    # Initialize orchestrator
    print("[INFO] Initializing orchestrator...")
    orchestrator = Orchestrator()

    # Run analysis
    print(f"[INFO] Starting analysis...")
    print()

    try:
        result = await orchestrator.analyze(query)

        # Check status
        if result["status"] == "error":
            print(f"[ERROR] Analysis failed: {result['error_message']}")
            return 1, query, None

        if result["status"] == "partial":
            print(f"[WARNING] Partial results: {result['error_message']}")
            print()

        # Display results
        print(f"[SUCCESS] Analysis completed in {result['execution_time']:.2f}s")
        print()

        analyses = result.get("analyses", [])
        if not analyses:
            print("[ERROR] No analyses generated")
            return 1, query, None

        saved_report_path = None

        for analysis in analyses:
            synthesis = analysis.get("synthesis", {})
            ticker = analysis.get("ticker", "UNKNOWN")

            if not synthesis:
                print(f"[WARNING] No synthesis report for {ticker}")
                continue

            markdown_report = synthesis.get("markdown_report", "")

            # Output based on format
            if json_output:
                json_summary = synthesis.get("json_summary", {})
                print(json.dumps(json_summary, indent=2, default=str))
            else:
                print()
                print("=" * 80)
                print("ANALYSIS REPORT")
                print("=" * 80)
                print()
                print(markdown_report)

            # Auto-save report to reports folder
            if auto_save and markdown_report:
                saved_report_path = save_report(ticker, markdown_report)
                print()
                print(f"[INFO] Report saved to: {saved_report_path}")

            # Save to custom file if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if json_output:
                    with open(output_path, 'w') as f:
                        json.dump(synthesis.get("json_summary", {}), f, indent=2, default=str)
                else:
                    with open(output_path, 'w') as f:
                        f.write(markdown_report)

                print(f"[INFO] Report also saved to: {output_path}")

            # Show key metrics
            print()
            print("=" * 80)
            print("KEY METRICS")
            print("=" * 80)
            print(f"Overall Score: {synthesis.get('overall_score', 0):.1f}/10")
            print(f"Recommendation: {synthesis.get('recommendation', 'N/A').upper().replace('_', ' ')}")
            print(f"Report ID: {synthesis.get('report_id', 'N/A')}")
            print(f"Saved to Database: {'Yes' if synthesis.get('saved_to_db') else 'No'}")
            print()

        return 0, ticker, saved_report_path

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"[ERROR] Unexpected error: {e}")
        return 1, query, None


async def run_batch_analysis(queries_file: str = "queries.json"):
    """
    Run batch analysis from queries.json file.

    Args:
        queries_file: Path to JSON file with queries

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Load queries from file
    queries_path = Path(queries_file)
    if not queries_path.exists():
        print(f"[ERROR] Queries file not found: {queries_file}")
        return 1

    try:
        with open(queries_path, 'r') as f:
            data = json.load(f)
            queries = data.get("queries", [])
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {queries_file}: {e}")
        return 1

    if not queries:
        print("[ERROR] No queries found in file")
        return 1

    print("=" * 80)
    print("BATCH STOCK ANALYSIS")
    print("=" * 80)
    print(f"Total tickers to analyze: {len(queries)}")
    print(f"Tickers: {', '.join(queries)}")
    print("=" * 80)
    print()

    # Track results
    results = []
    successful = 0
    failed = 0

    for i, query in enumerate(queries, 1):
        print()
        print("#" * 80)
        print(f"# ANALYZING {i}/{len(queries)}: {query}")
        print("#" * 80)
        print()

        exit_code, ticker, report_path = await run_analysis(query, auto_save=True)

        results.append({
            "ticker": ticker,
            "status": "success" if exit_code == 0 else "failed",
            "report_path": str(report_path) if report_path else None
        })

        if exit_code == 0:
            successful += 1
        else:
            failed += 1

    # Print summary
    print()
    print("=" * 80)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total: {len(queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print("Reports saved to:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        report_info = result["report_path"] if result["report_path"] else "N/A"
        print(f"  {status_icon} {result['ticker']}: {report_info}")
    print()

    return 0 if failed == 0 else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Multi-Agent Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single stock
  python -m src.main analyze "AAPL"

  # Analyze with custom query
  python -m src.main analyze "Analyze Apple stock"

  # Compare multiple stocks
  python -m src.main analyze "AAPL,MSFT,GOOGL"

  # Save report to custom file
  python -m src.main analyze "AAPL" -o custom_report.md

  # Get JSON output
  python -m src.main analyze "AAPL" --json

  # Run batch analysis from queries.json
  python -m src.main batch

  # Run batch analysis from custom file
  python -m src.main batch -f my_queries.json

For more information, see README.md
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stock(s)")
    analyze_parser.add_argument(
        "query",
        type=str,
        help="Stock ticker or analysis query (e.g., 'AAPL' or 'Analyze Apple')"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path for the report"
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format instead of markdown"
    )
    analyze_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not auto-save report to reports folder"
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch analysis from queries.json")
    batch_parser.add_argument(
        "-f", "--file",
        type=str,
        default="queries.json",
        help="Path to queries JSON file (default: queries.json)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if command provided
    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "analyze":
        exit_code, _, _ = asyncio.run(run_analysis(
            args.query,
            output_file=args.output,
            json_output=args.json,
            auto_save=not args.no_save
        ))
        return exit_code

    elif args.command == "batch":
        return asyncio.run(run_batch_analysis(args.file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
