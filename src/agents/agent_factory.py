"""
Agent Factory -- builds autonomous ReAct agent instances per specialist.

Each agent:
  - Has its own expert system prompt
  - Receives the full MCP tools list
  - Uses ReAct loop to decide which tools to call autonomously
  - Uses a message trimmer to stay within the model's context window
"""
from typing import List, Dict, Any, Callable
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import trim_messages, SystemMessage, BaseMessage
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

# Leave ~28k token headroom below the 128k gpt-4o-mini context limit.
# Agents that fetch 2 years of price history can easily exceed 128k without trimming.
MAX_CONTEXT_TOKENS = 100_000


def build_llm() -> ChatOpenAI:
    """Build a ChatOpenAI instance from project settings."""
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )


def _make_trimming_state_modifier(
    llm: ChatOpenAI, system_prompt: str
) -> Callable[[Any], List[BaseMessage]]:
    """
    Build a state_modifier for create_react_agent that:
      1. Prepends the agent's system message to every LLM call.
      2. Trims old messages from the beginning when the total token count
         would exceed MAX_CONTEXT_TOKENS.

    This prevents context_length_exceeded errors caused by large MCP tool
    responses (e.g., 2 years of daily OHLCV data) accumulating in the
    ReAct loop's message history.

    The trimmer always keeps:
      - The system message (include_system=True)
      - The most recent messages (strategy="last")
    It only trims at human-message boundaries (start_on="human") so it
    never splits a tool-call / tool-response pair.

    Args:
        llm:           The ChatOpenAI instance used for token counting.
        system_prompt: The agent's persona / instructions string.

    Returns:
        A callable suitable for the state_modifier parameter of
        create_react_agent.
    """
    trimmer = trim_messages(
        max_tokens=MAX_CONTEXT_TOKENS,
        strategy="last",       # Keep the most recent messages
        token_counter=llm,     # Use the model's own tokenizer for accuracy
        include_system=True,   # Never drop the system message
        allow_partial=False,   # Never cut a message in half
        start_on="human",      # Only trim at human-message boundaries
    )

    def modifier(state: Any) -> List[BaseMessage]:
        # Prepend the system message then trim the combined list
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        return trimmer.invoke(messages)

    return modifier


def create_all_agents(tools: List[BaseTool]) -> Dict[str, Any]:
    """
    Build all specialist agents using create_react_agent (langgraph.prebuilt).

    All agents share the same tools list -- each agent autonomously
    decides which tools to call based on its system prompt and task.

    Each agent is given a state_modifier that trims its message history to
    MAX_CONTEXT_TOKENS before every LLM call, preventing context overflow
    when tool responses (e.g., long price histories) accumulate.

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
    for name, system_prompt in agent_configs.items():
        agents[name] = create_react_agent(
            llm,
            tools,
            state_modifier=_make_trimming_state_modifier(llm, system_prompt),
        )
        logger.info(f"Created agent: {name} (tools available: {len(tools)})")

    return agents
