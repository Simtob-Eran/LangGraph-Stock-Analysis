"""Main orchestrator coordinating autonomous agents using LangGraph."""

import asyncio
import json
import re
import time
import uuid
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from config.settings import settings
from src.utils.database import Database
from src.utils.logger import setup_logger
from src.utils.validators import validate_tickers
from src.mcp.mcp_client_factory import create_mcp_client
from src.agents.agent_factory import create_all_agents

logger = setup_logger("orchestrator")


class AnalysisState(TypedDict):
    """State object for the analysis workflow."""
    ticker: str
    query: str  # Original user query
    run_id: str
    collected_data: Dict[str, Any]
    fundamental: Dict[str, Any]
    technical: Dict[str, Any]
    sentiment: Dict[str, Any]
    debate: Dict[str, Any]
    risk: Dict[str, Any]
    synthesis: Dict[str, Any]
    feedback: Dict[str, Any]
    error: str
    status: str


class Orchestrator:
    """
    Main orchestrator for stock analysis system.

    Coordinates autonomous agents using LangGraph for workflow management.
    Each specialist agent independently fetches data via MCP tools using
    create_agent's ReAct loop.
    """

    def __init__(self):
        """Initialize orchestrator (lightweight -- call initialize() before analyze)."""
        logger.info("Initializing Stock Analysis Orchestrator")
        logger.info(f"OpenAI Model: {settings.OPENAI_MODEL}")

        # Initialize OpenAI client for ticker extraction
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Initialize database
        self.db = Database()

        # These are set during initialize()
        self.mcp_client = None
        self.agents: Dict[str, Any] = {}
        self.workflow = None
        self._initialized = False

        logger.info("Orchestrator created (call initialize() to load MCP tools)")

    async def initialize(self):
        """Load MCP tools and create all autonomous agents. Must call before analyze()."""
        if self._initialized:
            logger.info("Already initialized, skipping")
            return

        logger.info("Loading MCP tools and creating agents...")

        self.mcp_client = create_mcp_client()

        if self.mcp_client:
            try:
                tools = await self.mcp_client.get_tools()
                tool_names = [t.name for t in tools]
                logger.info(f"Loaded {len(tools)} MCP tools: {tool_names}")
                print(f"Loaded {len(tools)} MCP tools: {tool_names}")
            except Exception as e:
                logger.error(f"Failed to load MCP tools: {e}")
                print(f"Warning: Failed to load MCP tools: {e}")
                tools = []
        else:
            tools = []
            logger.warning("No MCP available -- agents will run without live data tools")
            print("Warning: No MCP available -- agents will run without live data tools")

        self.agents = create_all_agents(tools)

        # Build workflow graph
        self.workflow = self._build_workflow()

        self._initialized = True
        logger.info(f"{len(self.agents)} autonomous agents ready")
        print(f"{len(self.agents)} autonomous agents ready")

    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for stock analysis.

        Returns:
            Compiled StateGraph
        """
        # Create state graph
        workflow = StateGraph(AnalysisState)

        # Add nodes for each agent
        workflow.add_node("collect_data", self._collect_data_node)
        workflow.add_node("analyze_fundamental", self._analyze_fundamental_node)
        workflow.add_node("analyze_technical", self._analyze_technical_node)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment_node)
        workflow.add_node("create_debate", self._create_debate_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("feedback_loop", self._feedback_loop_node)

        # Define workflow edges
        workflow.set_entry_point("collect_data")

        # After data collection, run three analyses in parallel
        workflow.add_edge("collect_data", "analyze_fundamental")
        workflow.add_edge("collect_data", "analyze_technical")
        workflow.add_edge("collect_data", "analyze_sentiment")

        # After parallel analyses, create debate
        workflow.add_edge("analyze_fundamental", "create_debate")
        workflow.add_edge("analyze_technical", "create_debate")
        workflow.add_edge("analyze_sentiment", "create_debate")

        # After debate, assess risk
        workflow.add_edge("create_debate", "assess_risk")

        # After risk assessment, synthesize
        workflow.add_edge("assess_risk", "synthesize")

        # After synthesis, run feedback loop
        workflow.add_edge("synthesize", "feedback_loop")

        # Feedback loop is the end
        workflow.add_edge("feedback_loop", END)

        # Compile the graph
        return workflow.compile()

    # ------------------------------------------------------------------
    # Agent invocation helpers
    # ------------------------------------------------------------------

    async def _invoke_agent(self, agent_name: str, user_message: str) -> dict:
        """Invoke a create_agent agent and parse its JSON output."""
        agent = self.agents[agent_name]

        try:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            # Final agent message is in result["messages"][-1]
            final_content = result["messages"][-1].content

            # Try to parse JSON from the response
            return self._extract_json(final_content)

        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return {
                "error": str(e),
                "score": 5.0,
                "confidence": 0.0,
                "recommendation": "hold",
                "reasoning": f"Agent failed: {e}"
            }

    def _extract_json(self, content: str) -> dict:
        """Extract JSON dict from agent response text."""
        # Try direct parse
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try ```json ... ``` block
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try any { ... } block
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback -- return raw text as reasoning
        return {
            "reasoning": content,
            "score": 5.0,
            "confidence": 0.3,
            "recommendation": "hold"
        }

    # ------------------------------------------------------------------
    # Workflow nodes -- each invokes an autonomous agent
    # ------------------------------------------------------------------

    async def _collect_data_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for data collection / ticker validation -- autonomous agent."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running autonomous data validation for {ticker}")

        result = await self._invoke_agent(
            "data_collector",
            f"""Validate the stock ticker: {ticker}

Use your available MCP tools to verify this is a real, tradeable ticker symbol.
Check that price history, financial data, and news are accessible.
Return your validation as JSON in the required format."""
        )

        return {"collected_data": result}

    async def _analyze_fundamental_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for fundamental analysis -- autonomous agent fetches its own data."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running autonomous fundamental analysis for {ticker}")

        analysis = await self._invoke_agent(
            "fundamental_analyst",
            f"""Perform a complete fundamental analysis for stock ticker: {ticker}

Use your available MCP tools to fetch all financial data you need.
Return your analysis as JSON in the required format."""
        )

        return {"fundamental": analysis}

    async def _analyze_technical_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for technical analysis -- autonomous agent fetches its own data."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running autonomous technical analysis for {ticker}")

        analysis = await self._invoke_agent(
            "technical_analyst",
            f"""Perform a complete technical analysis for stock ticker: {ticker}

Use your available MCP tools to fetch 2 years of daily price history.
Calculate all indicators and return your analysis as JSON in the required format."""
        )

        return {"technical": analysis}

    async def _analyze_sentiment_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for sentiment analysis -- autonomous agent fetches its own data."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running autonomous sentiment analysis for {ticker}")

        analysis = await self._invoke_agent(
            "sentiment_analyst",
            f"""Perform a complete sentiment analysis for stock ticker: {ticker}

Use your available MCP tools to fetch recent news (at least 10 articles).
Return your analysis as JSON in the required format."""
        )

        return {"sentiment": analysis}

    async def _create_debate_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for debate creation -- agent receives prior analyses."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Creating investment debate for {ticker}")

        fundamental = state.get("fundamental", {})
        technical = state.get("technical", {})
        sentiment = state.get("sentiment", {})

        debate = await self._invoke_agent(
            "debate_agent",
            f"""Create a bull vs bear debate for: {ticker}

FUNDAMENTAL ANALYSIS:
{json.dumps(fundamental, indent=2, default=str)}

TECHNICAL ANALYSIS:
{json.dumps(technical, indent=2, default=str)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment, indent=2, default=str)}

Build the strongest bull and bear cases, resolve conflicts, provide final verdict.
Return your analysis as JSON in the required format."""
        )

        return {"debate": debate}

    async def _assess_risk_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for risk assessment -- autonomous agent fetches its own data."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running autonomous risk assessment for {ticker}")

        analysis = await self._invoke_agent(
            "risk_manager",
            f"""Perform a comprehensive risk assessment for stock ticker: {ticker}

Use your available MCP tools to fetch financial data for risk analysis.
Return your assessment as JSON in the required format."""
        )

        return {"risk": analysis}

    async def _synthesize_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for synthesis -- agent receives all prior analyses."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Synthesizing final report for {ticker}")

        fundamental = state.get("fundamental", {})
        technical = state.get("technical", {})
        sentiment = state.get("sentiment", {})
        debate = state.get("debate", {})
        risk = state.get("risk", {})

        synthesis_result = await self._invoke_agent(
            "synthesis_agent",
            f"""Create the final comprehensive analysis report for: {ticker}

FUNDAMENTAL ANALYSIS:
{json.dumps(fundamental, indent=2, default=str)}

TECHNICAL ANALYSIS:
{json.dumps(technical, indent=2, default=str)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment, indent=2, default=str)}

DEBATE ANALYSIS:
{json.dumps(debate, indent=2, default=str)}

RISK ASSESSMENT:
{json.dumps(risk, indent=2, default=str)}

Produce the full markdown report in the required format.
Also provide a JSON summary with overall_score, recommendation, and key metrics."""
        )

        # The synthesis agent returns markdown -- wrap it for compatibility
        # with the existing main.py report extraction logic
        markdown_report = synthesis_result.get("reasoning", "")
        if not markdown_report or markdown_report == synthesis_result.get("recommendation", ""):
            # Agent likely returned the report as the full response text
            markdown_report = synthesis_result.get("markdown_report", str(synthesis_result))

        # Calculate overall score from sub-analyses for backward compatibility
        overall_score = self._calculate_overall_score(fundamental, technical, sentiment, risk)
        final_recommendation = debate.get("final_recommendation",
                                          synthesis_result.get("recommendation", "hold"))

        # Save to database
        report_id = str(uuid.uuid4())
        try:
            self.db.save_report(
                report_id=report_id,
                run_id=state.get("run_id", report_id),
                ticker=ticker,
                overall_score=overall_score,
                recommendation=final_recommendation,
                markdown_report=markdown_report,
                json_data=synthesis_result
            )
            saved_to_db = True
        except Exception as e:
            logger.error(f"Failed to save report to database: {e}")
            saved_to_db = False

        return {
            "synthesis": {
                "report_id": report_id,
                "ticker": ticker,
                "overall_score": overall_score,
                "recommendation": final_recommendation,
                "markdown_report": markdown_report,
                "json_summary": synthesis_result,
                "saved_to_db": saved_to_db,
                "confidence": synthesis_result.get("confidence", 0.8),
                "reasoning": "Comprehensive report generated by autonomous agents"
            }
        }

    async def _feedback_loop_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for feedback loop -- QA agent reviews all outputs."""
        ticker = state["ticker"]
        logger.info(f"[Orchestrator] Running feedback loop for {ticker}")

        fundamental = state.get("fundamental", {})
        technical = state.get("technical", {})
        sentiment = state.get("sentiment", {})
        debate = state.get("debate", {})
        risk = state.get("risk", {})
        synthesis = state.get("synthesis", {})

        feedback = await self._invoke_agent(
            "feedback_loop",
            f"""Review the complete analysis for: {ticker}

FUNDAMENTAL ANALYSIS:
{json.dumps(fundamental, indent=2, default=str)}

TECHNICAL ANALYSIS:
{json.dumps(technical, indent=2, default=str)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment, indent=2, default=str)}

DEBATE:
{json.dumps(debate, indent=2, default=str)}

RISK ASSESSMENT:
{json.dumps(risk, indent=2, default=str)}

SYNTHESIS SUMMARY:
Overall Score: {synthesis.get('overall_score', 'N/A')}
Recommendation: {synthesis.get('recommendation', 'N/A')}

Review all analyses for quality, completeness, and consistency.
Return your QA assessment as JSON in the required format."""
        )

        return {
            "feedback": feedback,
            "status": "completed"
        }

    # ------------------------------------------------------------------
    # Scoring helper (backward compatibility with existing report display)
    # ------------------------------------------------------------------

    def _calculate_overall_score(
        self,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> float:
        """Calculate weighted overall score from sub-analyses."""
        fund_score = float(fundamental.get("score", 5.0))

        # Technical: use score directly if available, else infer from trend
        tech_score = float(technical.get("score", 5.0))

        # Sentiment: normalize from [-1, +1] to [0, 10]
        sent_raw = sentiment.get("sentiment_score", 0.0)
        try:
            sentiment_score = (float(sent_raw) + 1) * 5
        except (TypeError, ValueError):
            sentiment_score = 5.0

        # Risk: invert (lower risk = higher contribution)
        risk_raw = float(risk.get("risk_score", 5.0))
        risk_score = 10 - risk_raw

        overall = (
            fund_score * 0.35 +
            tech_score * 0.25 +
            sentiment_score * 0.20 +
            risk_score * 0.20
        )

        return max(0.0, min(10.0, round(overall, 1)))

    # ------------------------------------------------------------------
    # Ticker extraction and analysis entry points (unchanged)
    # ------------------------------------------------------------------

    async def _extract_tickers_with_llm(self, query: str) -> tuple[list[str], str]:
        """
        Extract ticker symbols from query using LLM.

        Args:
            query: User query in any language

        Returns:
            Tuple of (tickers, analysis_type)
        """
        extraction_prompt = f"""Extract stock ticker symbols from the following query.
Return ONLY a JSON object with this exact format:
{{"tickers": ["AAPL", "MSFT"], "analysis_type": "single" or "multiple"}}

Rules:
- Extract valid US stock ticker symbols (1-5 uppercase letters)
- If only one ticker, analysis_type is "single"
- If multiple tickers, analysis_type is "multiple"
- If no tickers found, return {{"tickers": [], "analysis_type": "none"}}
- The query may be in any language, extract the ticker regardless

Query: {query}

JSON response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0,
                max_tokens=100
            )

            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)
            tickers = [t.upper() for t in result.get("tickers", [])]
            analysis_type = result.get("analysis_type", "single")

            logger.info(f"LLM extracted tickers: {tickers}, type: {analysis_type}")
            return tickers, analysis_type

        except Exception as e:
            logger.error(f"LLM ticker extraction failed: {e}")
            # Fallback: treat entire query as potential ticker if it looks like one
            query_clean = query.strip().upper()
            if len(query_clean) <= 5 and query_clean.isalpha():
                return [query_clean], "single"
            return [], "none"

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for analysis.

        Args:
            query: User query in any language (e.g., "Analyze AAPL")

        Returns:
            Analysis result dictionary
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        logger.info(f"Analyzing query: {query}")
        start_time = time.time()

        # Extract tickers using LLM (supports any language)
        tickers, analysis_type = await self._extract_tickers_with_llm(query)
        if not tickers:
            return {
                "status": "error",
                "error_message": "No valid tickers found in query",
                "analyses": [],
                "execution_time": time.time() - start_time
            }

        # Validate tickers
        valid, error_msg, validated_tickers = validate_tickers(tickers)
        if not valid:
            return {
                "status": "error",
                "error_message": error_msg,
                "analyses": [],
                "execution_time": time.time() - start_time
            }

        logger.info(f"Analysis type: {analysis_type}, Tickers: {validated_tickers}")

        # Execute analysis based on type
        if analysis_type == "single" or len(validated_tickers) == 1:
            result = await self._analyze_single(validated_tickers[0], query)
        else:
            result = await self._analyze_multiple(validated_tickers, query)

        result["execution_time"] = time.time() - start_time
        logger.info(f"Analysis completed in {result['execution_time']:.2f}s")

        return result

    async def _analyze_single(self, ticker: str, query: str) -> Dict[str, Any]:
        """
        Perform deep sequential analysis for a single stock.

        Args:
            ticker: Stock ticker
            query: Original query

        Returns:
            Analysis result
        """
        run_id = str(uuid.uuid4())

        # Create database record
        self.db.create_analysis_run(
            run_id=run_id,
            ticker=ticker,
            analysis_type="single",
            user_query=query
        )

        try:
            # Initialize state
            initial_state: AnalysisState = {
                "ticker": ticker,
                "query": query,
                "run_id": run_id,
                "collected_data": {},
                "fundamental": {},
                "technical": {},
                "sentiment": {},
                "debate": {},
                "risk": {},
                "synthesis": {},
                "feedback": {},
                "error": "",
                "status": "running"
            }

            # Run workflow
            logger.info(f"Starting workflow for {ticker}")
            final_state = await self.workflow.ainvoke(initial_state)

            # Update database
            self.db.update_analysis_run(
                run_id=run_id,
                execution_time=time.time(),
                status="completed",
                error_message=None
            )

            return {
                "status": "success",
                "analyses": [{
                    "ticker": ticker,
                    "collected_data": final_state.get("collected_data"),
                    "fundamental": final_state.get("fundamental"),
                    "technical": final_state.get("technical"),
                    "sentiment": final_state.get("sentiment"),
                    "debate": final_state.get("debate"),
                    "risk": final_state.get("risk"),
                    "synthesis": final_state.get("synthesis"),
                    "feedback": final_state.get("feedback")
                }],
                "error_message": None
            }

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)

            # Update database with error
            self.db.update_analysis_run(
                run_id=run_id,
                execution_time=time.time(),
                status="error",
                error_message=str(e)
            )

            return {
                "status": "error",
                "analyses": [],
                "error_message": str(e)
            }

    async def _analyze_multiple(self, tickers: List[str], query: str) -> Dict[str, Any]:
        """
        Perform parallel analysis for multiple stocks.

        Args:
            tickers: List of stock tickers
            query: Original query

        Returns:
            Analysis result with multiple stocks
        """
        logger.info(f"Running parallel analysis for {len(tickers)} stocks")

        # Limit concurrent tasks
        max_concurrent = min(settings.MAX_PARALLEL_TASKS, len(tickers))

        # Create tasks for each ticker
        tasks = []
        for ticker in tickers[:max_concurrent]:
            tasks.append(self._analyze_single(ticker, query))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful analyses
        analyses = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{tickers[i]}: {str(result)}")
            elif result["status"] == "success":
                analyses.extend(result["analyses"])
            else:
                errors.append(f"{tickers[i]}: {result.get('error_message', 'Unknown error')}")

        # Determine overall status
        if not analyses:
            status = "error"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "analyses": analyses,
            "error_message": "; ".join(errors) if errors else None
        }
