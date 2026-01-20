"""Base agent class for all specialized agents."""

import time
import uuid
import json
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from config.settings import settings
from src.utils.logger import AgentLogger
from src.utils.database import Database


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    All specialized agents inherit from this class and implement
    the execute() method with their specific logic.
    """

    def __init__(
        self,
        name: str,
        openai_client: AsyncOpenAI,
        db_client: Database
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., "fundamental_analyst")
            openai_client: OpenAI async client
            db_client: Database client for logging
        """
        self.name = name
        self.openai_client = openai_client
        self.db = db_client
        self.logger = AgentLogger(name)

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - must be implemented by subclasses.

        Args:
            inputs: Input data for the agent

        Returns:
            Agent output dictionary
        """
        raise NotImplementedError(f"Agent {self.name} must implement execute()")

    async def run(
        self,
        inputs: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the agent with logging and error handling.

        Args:
            inputs: Input data
            run_id: Optional run ID for database logging

        Returns:
            Agent output with metadata
        """
        execution_id = str(uuid.uuid4())
        ticker = inputs.get("ticker", "UNKNOWN")
        start_time = time.time()

        self.logger.log_execution_start(ticker, inputs)

        try:
            # Execute agent logic
            outputs = await self.execute(inputs)

            # Calculate execution time
            execution_time = time.time() - start_time
            execution_time_ms = int(execution_time * 1000)

            # Get confidence and reasoning from outputs
            confidence = outputs.get("confidence", 0.5)
            reasoning = outputs.get("reasoning", "")

            # Log successful execution
            self.logger.log_execution_complete(ticker, execution_time, confidence)

            # Log to database if run_id provided
            if run_id and self.db:
                self.db.log_agent_execution(
                    execution_id=execution_id,
                    run_id=run_id,
                    agent_name=self.name,
                    ticker=ticker,
                    input_data=inputs,
                    output_data=outputs,
                    reasoning=reasoning,
                    confidence=confidence,
                    execution_time_ms=execution_time_ms
                )

            # Add metadata to outputs
            outputs["_metadata"] = {
                "agent": self.name,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "timestamp": time.time()
            }

            return outputs

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_execution_error(ticker, e)

            # Return error output
            return {
                "error": str(e),
                "ticker": ticker,
                "confidence": 0.0,
                "_metadata": {
                    "agent": self.name,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "status": "error"
                }
            }

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Call OpenAI LLM with retry logic.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            LLM response text
        """
        prompt_length = sum(len(m.get("content", "")) for m in messages)

        try:
            kwargs = {
                "model": settings.OPENAI_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self.openai_client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            self.logger.log_llm_call(prompt_length, len(content))

            return content

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", exc_info=True)
            raise

    async def _call_llm_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON dictionary
        """
        response = await self._call_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response was: {response[:500]}")
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise

    def _format_data_for_prompt(self, data: Dict[str, Any], max_length: int = 5000) -> str:
        """
        Format data for inclusion in prompts.

        Args:
            data: Data dictionary
            max_length: Maximum string length

        Returns:
            Formatted string
        """
        formatted = json.dumps(data, indent=2, default=str)
        if len(formatted) > max_length:
            formatted = formatted[:max_length] + "\n... (truncated)"
        return formatted

    async def _retry_on_failure(
        self,
        func,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Retry a function on failure with exponential backoff.

        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result
        """
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)


# Import asyncio at the end to avoid circular imports
import asyncio
