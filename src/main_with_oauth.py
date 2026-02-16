"""Main entry point with OAuth support for MCP servers."""

import asyncio
import argparse
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from typing import Optional

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientMetadata,
    OAuthClientInformationFull,
    OAuthToken,
    AnyUrl,
)
from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from src.utils.logger import setup_logger
from config.settings import settings

# Load environment variables
load_dotenv()

logger = setup_logger("main_oauth")


# ---------------- OAuth Token Storage ----------------

class InMemoryTokenStorage(TokenStorage):
    """In-memory token storage for OAuth."""
    
    def __init__(self):
        self.tokens: Optional[OAuthToken] = None
        self.client_info: Optional[OAuthClientInformationFull] = None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken):
        self.tokens = tokens
        logger.info("OAuth tokens saved")

    async def get_client_info(self):
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull):
        self.client_info = client_info
        logger.info("OAuth client info saved")


# ---------------- OAuth Handlers ----------------

async def handle_redirect(auth_url: str) -> None:
    """Display authorization URL to user."""
    print("\n" + "=" * 80)
    print("OAUTH AUTHORIZATION REQUIRED")
    print("=" * 80)
    print(f"\n🔐 Please visit this URL to authorize:\n")
    print(f"   {auth_url}\n")
    print("=" * 80 + "\n")


async def handle_callback() -> tuple[str, Optional[str]]:
    """Handle OAuth callback from user."""
    print("After authorizing, you'll be redirected to a callback URL.")
    print("Please copy the ENTIRE URL from your browser and paste it here.\n")
    
    callback_url = input("Paste callback URL: ").strip()
    
    try:
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)
        
        if "code" not in params:
            raise ValueError("No authorization code found in URL")
        
        code = params["code"][0]
        state = params.get("state", [None])[0]
        
        logger.info("OAuth callback received successfully")
        return code, state
    
    except Exception as e:
        logger.error(f"Failed to parse callback URL: {e}")
        print(f"\n❌ Error: {e}")
        print("Please make sure you copied the complete callback URL.\n")
        raise


# ---------------- OAuth Setup ----------------

def setup_oauth_provider(server_name: str = "github") -> Optional[OAuthClientProvider]:
    """
    Setup OAuth provider for MCP server.
    
    Args:
        server_name: Name of the MCP server (e.g., 'github', 'gmail')
    
    Returns:
        OAuthClientProvider or None if OAuth not configured
    """
    # Check for OAuth configuration in environment
    oauth_url = os.getenv(f"MCP_OAUTH_{server_name.upper()}_URL")
    redirect_uri = os.getenv(f"MCP_OAUTH_{server_name.upper()}_REDIRECT_URI")
    scope = os.getenv(f"MCP_OAUTH_{server_name.upper()}_SCOPE")
    
    if not oauth_url:
        logger.info(f"No OAuth configuration found for {server_name}")
        return None
    
    # Default values
    if not redirect_uri:
        redirect_uri = "https://cbg-obot.com/"
    if not scope:
        scope = "user repo"  # Default GitHub scopes
    
    logger.info(f"Setting up OAuth for {server_name}")
    logger.info(f"  URL: {oauth_url}")
    logger.info(f"  Redirect URI: {redirect_uri}")
    logger.info(f"  Scope: {scope}")
    
    try:
        oauth_provider = OAuthClientProvider(
            server_url=oauth_url,
            client_metadata=OAuthClientMetadata(
                client_name=f"{server_name.title()} MCP Client",
                redirect_uris=[AnyUrl(redirect_uri)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope=scope,
            ),
            storage=InMemoryTokenStorage(),
            redirect_handler=handle_redirect,
            callback_handler=handle_callback,
        )
        
        logger.info(f"OAuth provider created successfully for {server_name}")
        return oauth_provider
    
    except Exception as e:
        logger.error(f"Failed to setup OAuth provider: {e}")
        return None


# ---------------- Report Generation ----------------

def generate_report_filename(ticker: str = None) -> str:
    """Generate report filename with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if ticker:
        return f"{timestamp}-{ticker}.md"
    return f"{timestamp}.md"


# ---------------- Analysis Functions ----------------

async def run_analysis(
    query: str,
    output_file: str = None,
    json_output: bool = False,
    auto_save: bool = True,
    use_oauth: bool = False
):
    """
    Run stock analysis with optional OAuth support.
    
    Args:
        query: Analysis query
        output_file: Optional file to save report
        json_output: Whether to output JSON instead of markdown
        auto_save: Whether to auto-save reports
        use_oauth: Whether to use OAuth for MCP servers
    """
    print("=" * 80)
    print("STOCK ANALYSIS SYSTEM" + (" (WITH OAUTH)" if use_oauth else ""))
    print("=" * 80)
    print(f"Model: {settings.OPENAI_MODEL}")
    print(f"Query: {query}")
    print(f"OAuth: {'Enabled' if use_oauth else 'Disabled'}")
    print("=" * 80)
    print()
    
    # Setup OAuth if enabled
    if use_oauth:
        print("[INFO] Setting up OAuth authentication...")
        oauth_provider = setup_oauth_provider("github")  # Or other server
        
        if oauth_provider:
            print("[INFO] OAuth provider configured")
            # Note: The actual OAuth flow happens when MCP client connects
            # You would need to integrate this with your MCP client
        else:
            print("[WARNING] OAuth setup failed, continuing without OAuth")
    
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
            
            # Auto-save report
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
    Run batch analysis with optional OAuth.
    
    Args:
        queries_file: Path to queries JSON file
        use_oauth: Whether to use OAuth
    """
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
    print("BATCH STOCK ANALYSIS" + (" (WITH OAUTH)" if use_oauth else ""))
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Queries: {', '.join(queries)}")
    print(f"OAuth: {'Enabled' if use_oauth else 'Disabled'}")
    print("=" * 80)
    print()
    
    # Setup OAuth if enabled
    if use_oauth:
        print("[INFO] Setting up OAuth for batch processing...")
        oauth_provider = setup_oauth_provider("github")
        if not oauth_provider:
            print("[WARNING] OAuth setup failed, continuing without OAuth")
    
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
            result = await run_analysis(query, auto_save=True, json_output=False, use_oauth=use_oauth)
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


# ---------------- Main CLI ----------------

def main():
    """Main CLI entry point with OAuth support."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Stock Analysis System with OAuth Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with OAuth
  python main_with_oauth.py analyze "AAPL" --oauth

  # Batch analysis with OAuth
  python main_with_oauth.py batch --oauth

  # Regular analysis (no OAuth)
  python main_with_oauth.py analyze "AAPL"

OAuth Configuration (.env file):
  MCP_OAUTH_GITHUB_URL=https://cbg-obot.com/mcp-connect/default-github-391ae5a6
  MCP_OAUTH_GITHUB_REDIRECT_URI=https://cbg-obot.com/
  MCP_OAUTH_GITHUB_SCOPE=user repo

For more information, see README.md
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stock(s)")
    analyze_parser.add_argument(
        "query",
        type=str,
        help="Stock ticker or analysis query"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path"
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format"
    )
    analyze_parser.add_argument(
        "--oauth",
        action="store_true",
        help="Use OAuth authentication for MCP servers"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch analysis")
    batch_parser.add_argument(
        "-f", "--file",
        type=str,
        default="queries.json",
        help="Path to queries JSON file"
    )
    batch_parser.add_argument(
        "--oauth",
        action="store_true",
        help="Use OAuth authentication"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    if args.command == "analyze":
        return asyncio.run(run_analysis(
            args.query,
            output_file=args.output,
            json_output=args.json,
            auto_save=True,
            use_oauth=args.oauth
        ))
    
    if args.command == "batch":
        return asyncio.run(run_batch_analysis(
            queries_file=args.file,
            use_oauth=args.oauth
        ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())