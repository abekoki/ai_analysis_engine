from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from langchain_core.messages import BaseMessage

from ..utils.context_recorder import AgentInteractionLogger


class PromptLoggingMixin:
    """Reusable mixin for agents that need to log prompts, responses, and outputs."""

    prompter: AgentInteractionLogger

    def _log_prompt(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        prompt_messages: Optional[Sequence[BaseMessage]] = None,
        response: Any = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "prompter"):
            return

        messages: Iterable[BaseMessage] = prompt_messages or self._extract_messages(response)
        self.prompter.log_prompt(
            node=node,
            dataset_id=dataset_id,
            prompt_messages=list(messages),
            prompt_context=prompt_context,
        )

    def _log_response(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        response: Any,
        tool_calls: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "prompter"):
            return

        self.prompter.log_response(
            node=node,
            dataset_id=dataset_id,
            response=response,
            tool_calls=tool_calls,
            extra=extra,
        )

    def _log_result(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        result: Any,
        description: Optional[str] = None,
    ) -> None:
        if not hasattr(self, "prompter"):
            return

        self.prompter.log_result(
            node=node,
            dataset_id=dataset_id,
            result=result,
            description=description,
        )

    @staticmethod
    def _extract_messages(response: Any) -> Iterable[BaseMessage]:
        try:
            candidates = None

            if hasattr(response, "provider_responses"):
                provider_responses = getattr(response, "provider_responses", None)
                if provider_responses:
                    for provider in provider_responses:
                        raw = provider.get("raw") if isinstance(provider, dict) else None
                        if raw and "messages" in raw:
                            candidates = raw["messages"]
                            break

            if candidates is None and hasattr(response, "messages"):
                candidates = getattr(response, "messages", None)

            if candidates is None and hasattr(response, "_messages"):
                candidates = getattr(response, "_messages", None)

            if candidates is None and hasattr(response, "prompt"):
                candidates = getattr(response, "prompt", None)

            if candidates is None:
                return []

            if isinstance(candidates, BaseMessage):
                return [candidates]

            return list(candidates)
        except Exception:
            return []
