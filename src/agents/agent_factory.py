"""
Agent Factory -- builds autonomous create_agent instances per specialist.

Each agent:
  - Has its own expert system prompt
  - Receives the full MCP tools list
  - Uses ReAct loop to decide which tools to call autonomously
"""
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from config.settings import settings
from src.models.prompts import (
    DATA_COLLECTOR_PROMPT,
    FUNDAMENTAL_ANALYST_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    DEBATE_AGENT_PROMPT,
    RISK_MANAGER_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
    FEEDBACK_LOOP_PROMPT,
)
from src.utils.logger import setup_logger

logger = setup_logger("agent_factory")


def build_llm() -> ChatOpenAI:
    """Build a ChatOpenAI instance from project settings."""
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )


def create_all_agents(tools: List[BaseTool]) -> Dict[str, Any]:
    """
    Build all specialist agents using create_agent (LangGraph v1).

    All agents share the same tools list -- each agent autonomously
    decides which tools to call based on its system prompt and task.

    Args:
        tools: list[BaseTool] from MultiServerMCPClient.get_tools()
               May be empty if MCP is unavailable -- agents will work
               without live data and note the limitation.

    Returns:
        Dict mapping agent_name -> compiled LangGraph agent graph
    """
    llm = build_llm()

    agent_configs = {
        "data_collector":      DATA_COLLECTOR_PROMPT,
        "fundamental_analyst": FUNDAMENTAL_ANALYST_PROMPT,
        "technical_analyst":   TECHNICAL_ANALYST_PROMPT,
        "sentiment_analyst":   SENTIMENT_ANALYST_PROMPT,
        "debate_agent":        DEBATE_AGENT_PROMPT,
        "risk_manager":        RISK_MANAGER_PROMPT,
        "synthesis_agent":     SYNTHESIS_AGENT_PROMPT,
        "feedback_loop":       FEEDBACK_LOOP_PROMPT,
    }

    agents = {}
    for name, prompt in agent_configs.items():
        agents[name] = create_agent(llm, tools, system_prompt=prompt)
        logger.info(f"Created agent: {name} (tools available: {len(tools)})")

    return agents
