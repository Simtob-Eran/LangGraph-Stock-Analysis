"""Main entry point for Stock Analysis System."""

import asyncio
import argparse
import sys
from pathlib import Path
from src.orchestrator import Orchestrator
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("main")


async def run_analysis(query: str, output_file: str = None, json_output: bool = False):
    """
    Run stock analysis for the given query.

    Args:
        query: Analysis query
        output_file: Optional file to save report
        json_output: Whether to output JSON instead of markdown
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
            return 1

        if result["status"] == "partial":
            print(f"[WARNING] Partial results: {result['error_message']}")
            print()

        # Display results
        print(f"[SUCCESS] Analysis completed in {result['execution_time']:.2f}s")
        print()

        analyses = result.get("analyses", [])
        if not analyses:
            print("[ERROR] No analyses generated")
            return 1

        for analysis in analyses:
            synthesis = analysis.get("synthesis", {})
            ticker = analysis.get("ticker", "UNKNOWN")

            if not synthesis:
                print(f"[WARNING] No synthesis report for {ticker}")
                continue

            markdown_report = synthesis.get("markdown_report", "")

            # Output based on format
            if json_output:
                import json
                json_summary = synthesis.get("json_summary", {})
                print(json.dumps(json_summary, indent=2, default=str))
            else:
                print()
                print("=" * 80)
                print("ANALYSIS REPORT")
                print("=" * 80)
                print()
                print(markdown_report)

            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if json_output:
                    import json
                    with open(output_path, 'w') as f:
                        json.dump(synthesis.get("json_summary", {}), f, indent=2, default=str)
                else:
                    with open(output_path, 'w') as f:
                        f.write(markdown_report)

                print()
                print(f"[INFO] Report saved to: {output_path}")

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

        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"[ERROR] Unexpected error: {e}")
        return 1


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

  # Save report to file
  python -m src.main analyze "AAPL" -o reports/aapl_analysis.md

  # Get JSON output
  python -m src.main analyze "AAPL" --json

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

    # Parse arguments
    args = parser.parse_args()

    # Check if command provided
    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "analyze":
        return asyncio.run(run_analysis(
            args.query,
            output_file=args.output,
            json_output=args.json
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
