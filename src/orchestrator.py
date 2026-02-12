"""Main orchestrator coordinating all agents using LangGraph."""

import asyncio
import time
import uuid
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from config.settings import settings
from src.utils.database import Database
from src.utils.logger import setup_logger
from src.utils.validators import validate_tickers
from src.agents.data_collector import DataCollectorAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.technical_analyst import TechnicalAnalystAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.agents.debate_agent import DebateAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.feedback_loop import FeedbackLoopAgent

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

    Coordinates all agents using LangGraph for workflow management.
    Handles both sequential (deep) and parallel (multiple stocks) analysis.
    """

    def __init__(self):
        """Initialize orchestrator with all agents."""
        logger.info(f"Initializing Stock Analysis Orchestrator")
        logger.info(f"OpenAI Model: {settings.OPENAI_MODEL}")

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Initialize database
        self.db = Database()

        # Initialize all agents
        self.data_collector = DataCollectorAgent(self.openai_client, self.db)
        self.fundamental_analyst = FundamentalAnalystAgent(self.openai_client, self.db)
        self.technical_analyst = TechnicalAnalystAgent(self.openai_client, self.db)
        self.sentiment_analyst = SentimentAnalystAgent(self.openai_client, self.db)
        self.debate_agent = DebateAgent(self.openai_client, self.db)
        self.risk_manager = RiskManagerAgent(self.openai_client, self.db)
        self.synthesis_agent = SynthesisAgent(self.openai_client, self.db)
        self.feedback_loop = FeedbackLoopAgent(self.openai_client, self.db)

        # Build workflow graph
        self.workflow = self._build_workflow()

        logger.info("Orchestrator initialized successfully")

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

    async def _collect_data_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for data collection."""
        logger.info(f"[Orchestrator] Collecting data for {state['ticker']}")

        result = await self.data_collector.run(
            {"ticker": state["ticker"], "query": state.get("query", "")},
            run_id=state["run_id"]
        )

        return {"collected_data": result}

    async def _analyze_fundamental_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for fundamental analysis."""
        logger.info(f"[Orchestrator] Running fundamental analysis for {state['ticker']}")

        # Wait for data collection to complete
        while not state.get("collected_data"):
            await asyncio.sleep(0.1)

        result = await self.fundamental_analyst.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "collected_data": state["collected_data"].get("data")
            },
            run_id=state["run_id"]
        )

        return {"fundamental": result}

    async def _analyze_technical_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for technical analysis."""
        logger.info(f"[Orchestrator] Running technical analysis for {state['ticker']}")

        # Wait for data collection to complete
        while not state.get("collected_data"):
            await asyncio.sleep(0.1)

        result = await self.technical_analyst.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "collected_data": state["collected_data"].get("data")
            },
            run_id=state["run_id"]
        )

        return {"technical": result}

    async def _analyze_sentiment_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for sentiment analysis."""
        logger.info(f"[Orchestrator] Running sentiment analysis for {state['ticker']}")

        # Wait for data collection to complete
        while not state.get("collected_data"):
            await asyncio.sleep(0.1)

        result = await self.sentiment_analyst.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "collected_data": state["collected_data"].get("data")
            },
            run_id=state["run_id"]
        )

        return {"sentiment": result}

    async def _create_debate_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for debate creation."""
        logger.info(f"[Orchestrator] Creating investment debate for {state['ticker']}")

        # Wait for all three analyses to complete
        while not all([
            state.get("fundamental"),
            state.get("technical"),
            state.get("sentiment")
        ]):
            await asyncio.sleep(0.1)

        result = await self.debate_agent.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "fundamental": state["fundamental"],
                "technical": state["technical"],
                "sentiment": state["sentiment"]
            },
            run_id=state["run_id"]
        )

        return {"debate": result}

    async def _assess_risk_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for risk assessment."""
        logger.info(f"[Orchestrator] Assessing risk for {state['ticker']}")

        result = await self.risk_manager.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "collected_data": state["collected_data"].get("data"),
                "fundamental": state["fundamental"],
                "technical": state["technical"],
                "sentiment": state["sentiment"]
            },
            run_id=state["run_id"]
        )

        return {"risk": result}

    async def _synthesize_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for synthesis and report generation."""
        logger.info(f"[Orchestrator] Synthesizing final report for {state['ticker']}")

        result = await self.synthesis_agent.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "run_id": state["run_id"],
                "collected_data": state["collected_data"].get("data"),
                "fundamental": state["fundamental"],
                "technical": state["technical"],
                "sentiment": state["sentiment"],
                "debate": state["debate"],
                "risk": state["risk"]
            },
            run_id=state["run_id"]
        )

        return {"synthesis": result}

    async def _feedback_loop_node(self, state: AnalysisState) -> Dict[str, Any]:
        """Node for feedback loop."""
        logger.info(f"[Orchestrator] Running feedback loop for {state['ticker']}")

        result = await self.feedback_loop.run(
            {
                "ticker": state["ticker"],
                "query": state.get("query", ""),
                "collected_data": state["collected_data"].get("data"),
                "fundamental": state["fundamental"],
                "technical": state["technical"],
                "sentiment": state["sentiment"],
                "debate": state["debate"],
                "risk": state["risk"],
                "synthesis": state["synthesis"]
            },
            run_id=state["run_id"]
        )

        return {
            "feedback": result,
            "status": "completed"
        }

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
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0,
                max_tokens=100
            )

            import json
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
            query: User query in any language (e.g., "Analyze AAPL", "תנתח לי את AAPL")

        Returns:
            Analysis result dictionary
        """
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
                "query": query,  # Pass original query to agents
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
