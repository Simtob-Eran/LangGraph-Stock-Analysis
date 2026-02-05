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
from src.mcp.yfinance_client import (
    create_oauth_provider,
    set_oauth_auth,
    pre_authenticate_oauth,
    OAUTH_AVAILABLE,
)

logger = setup_logger("main")


def generate_report_filename(ticker: str = None) -> str:
    """
    Generate report filename with timestamp format: YYYY-MM-DD-HH-MM-SS.md

    Args:
        ticker: Optional ticker symbol to include in filename

    Returns:
        Formatted filename string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if ticker:
        return f"{timestamp}-{ticker}.md"
    return f"{timestamp}.md"


async def run_analysis(query: str, output_file: str = None, json_output: bool = False, auto_save: bool = True):
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
                json_summary = synthesis.get("json_summary", {})
                print(json.dumps(json_summary, indent=2, default=str))
            else:
                print()
                print("=" * 80)
                print("ANALYSIS REPORT")
                print("=" * 80)
                print()
                print(markdown_report)

            # Auto-save report with timestamp if enabled
            if auto_save and not json_output:
                reports_dir = Path("reports")
                reports_dir.mkdir(parents=True, exist_ok=True)

                report_filename = generate_report_filename(ticker)
                report_path = reports_dir / report_filename

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_report)

                print()
                print(f"[INFO] Report automatically saved to: {report_path}")

            # Save to custom file if requested
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


async def run_batch_analysis(queries_file: str = "queries.json", use_oauth: bool = False):
    """
    Run batch analysis from queries.json file.

    Args:
        queries_file: Path to queries JSON file
        use_oauth: Whether to use OAuth authentication for MCP

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup OAuth if requested
    if use_oauth:
        if not OAUTH_AVAILABLE:
            print("[ERROR] OAuth not available - MCP library not installed")
            return 1

        mcp_url = settings.get_oauth_url()
        if not mcp_url:
            print("[ERROR] MCP URL not configured. Set MCP_OAUTH_GITHUB_URL or MCP_URL in .env file")
            return 1

        redirect_uri = settings.get_oauth_redirect_uri()
        scope = settings.get_oauth_scope()

        print("=" * 80)
        print("OAUTH AUTHENTICATION")
        print("=" * 80)
        print(f"MCP URL: {mcp_url}")
        print(f"Redirect URI: {redirect_uri}")
        print(f"Scope: {scope}")
        print("=" * 80)
        print()

        oauth_provider = create_oauth_provider(
            mcp_url=mcp_url,
            redirect_uri=redirect_uri,
            scope=scope,
            client_name=settings.OAUTH_CLIENT_NAME,
        )

        # Pre-authenticate before starting batch processing
        print("[INFO] Starting OAuth authentication flow...")
        print("[INFO] You will need to authorize in your browser and paste the callback URL.")
        print()

        auth_success = await pre_authenticate_oauth(mcp_url, oauth_provider)
        if not auth_success:
            print("[ERROR] OAuth authentication failed. Cannot proceed.")
            return 1

        set_oauth_auth(oauth_provider, mcp_url=mcp_url)
        print("[INFO] OAuth authentication complete. Starting batch analysis...")

    queries_path = Path(queries_file)

    if not queries_path.exists():
        print(f"[ERROR] Queries file not found: {queries_file}")
        print(f"[INFO] Create a file with format:")
        print(json.dumps({"queries": ["AAPL", "GOOGL", "MSFT"]}, indent=2))
        return 1

    # Load queries
    try:
        with open(queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            queries = data.get("queries", [])
    except Exception as e:
        print(f"[ERROR] Failed to load queries file: {e}")
        return 1

    if not queries:
        print(f"[ERROR] No queries found in {queries_file}")
        return 1

    print("=" * 80)
    print("BATCH STOCK ANALYSIS")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Queries: {', '.join(queries)}")
    print("=" * 80)
    print()

    # Process each query
    total_queries = len(queries)
    successful = 0
    failed = 0

    for i, query in enumerate(queries, 1):
        print()
        print("=" * 80)
        print(f"PROCESSING QUERY {i}/{total_queries}: {query}")
        print("=" * 80)
        print()

        try:
            result = await run_analysis(query, auto_save=True, json_output=False)
            if result == 0:
                successful += 1
                print(f"[SUCCESS] Query {i}/{total_queries} completed: {query}")
            else:
                failed += 1
                print(f"[FAILED] Query {i}/{total_queries} failed: {query}")
        except Exception as e:
            failed += 1
            logger.error(f"Error processing query {query}: {e}", exc_info=True)
            print(f"[ERROR] Query {i}/{total_queries} error: {query} - {e}")

        print()

    # Summary
    print("=" * 80)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total queries: {total_queries}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/total_queries*100):.1f}%")
    print("=" * 80)

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

  # Save report to file
  python -m src.main analyze "AAPL" -o reports/aapl_analysis.md

  # Get JSON output
  python -m src.main analyze "AAPL" --json

  # Run batch analysis from queries.json
  python -m src.main batch

  # Run batch analysis from custom file
  python -m src.main batch -f my_queries.json

  # Run batch analysis with OAuth authentication
  python -m src.main batch --oauth

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

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch analysis from queries.json")
    batch_parser.add_argument(
        "-f", "--file",
        type=str,
        default="queries.json",
        help="Path to queries JSON file (default: queries.json)"
    )
    batch_parser.add_argument(
        "--oauth",
        action="store_true",
        help="Use OAuth authentication for MCP server"
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
            json_output=args.json,
            auto_save=True
        ))

    if args.command == "batch":
        return asyncio.run(run_batch_analysis(
            queries_file=args.file,
            use_oauth=args.oauth
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
