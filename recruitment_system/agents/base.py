"""Shared base class for all recruitment agents."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from utils.helpers import extract_json_block

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class BaseRecruitmentAgent(ABC):
    """
    All agents share:
      - A ChatGroq LLM instance
      - A structured JSON call method with retry
      - A common invoke interface
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model_name = model or os.getenv("LLM_MODEL", _DEFAULT_MODEL)
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Core LLM helpers
    # ------------------------------------------------------------------

    def _chat(self, system: str, human: str) -> str:
        """Send a chat message and return the raw string response."""
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        response = self.llm.invoke(messages)
        return response.content  # type: ignore[return-value]

    def _chat_json(self, system: str, human: str, retries: int = 2) -> dict[str, Any] | list[Any]:
        """
        Like _chat but parses the response as JSON.
        Retries up to `retries` times on parse failure.
        """
        prompt_suffix = (
            "\n\nCRITICAL INSTRUCTION: Return ONLY a valid JSON object. "
            "No markdown, no ```json fences, no explanation text, no preamble. "
            "Start your response with { and end with }."
        )
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                # Always include JSON instruction on every attempt
                raw = self._chat(system, human + prompt_suffix)
                self.logger.debug("Raw LLM response (attempt %d): %.200s", attempt + 1, raw)
                return extract_json_block(raw)
            except Exception as exc:
                last_err = exc
                self.logger.warning("JSON parse attempt %d/%d failed: %s", attempt + 1, retries + 1, exc)
        raise RuntimeError(f"LLM did not return valid JSON after {retries+1} attempts. Last error: {last_err}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The fixed system prompt for this agent."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the agent and return a typed result."""
