from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import BaseMessage


class ContextRecorder:
    """Thread-aware recorder for LangGraph agent interactions."""

    def __init__(self) -> None:
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = RLock()
        self._thread_stack: List[str] = []

    @contextmanager
    def thread_scope(self, thread_id: str) -> Iterable[None]:
        self.set_current_thread(thread_id)
        try:
            yield
        finally:
            self.clear_current_thread(thread_id)

    def set_current_thread(self, thread_id: str) -> None:
        with self._lock:
            self._thread_stack.append(thread_id)
            self._events.setdefault(thread_id, [])

    def clear_current_thread(self, thread_id: Optional[str] = None) -> None:
        with self._lock:
            if self._thread_stack:
                self._thread_stack.pop()
            if thread_id:
                # Do not drop events here; callers decide when to reset completely
                pass

    def reset(self, thread_id: str) -> None:
        with self._lock:
            self._events.pop(thread_id, None)

    def append(self, entry: Dict[str, Any], thread_id: Optional[str] = None) -> None:
        normalized = dict(entry)
        normalized.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

        tid = thread_id or self._current_thread_id
        if not tid:
            return

        with self._lock:
            self._events.setdefault(tid, []).append(normalized)

    @property
    def _current_thread_id(self) -> Optional[str]:
        with self._lock:
            return self._thread_stack[-1] if self._thread_stack else None

    def events(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events.get(thread_id, [])]

    def snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {tid: [dict(event) for event in events] for tid, events in self._events.items()}


context_recorder = ContextRecorder()


def _serialize_messages(messages: Iterable[BaseMessage]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "role": getattr(message, "type", getattr(message, "role", "unknown")),
                "content": getattr(message, "content", None),
                "additional_kwargs": getattr(message, "additional_kwargs", None) or {},
            }
        )
    return serialized


def _serialize_tool_calls(tool_calls: Optional[Any]) -> Optional[Any]:
    if not tool_calls:
        return None

    serialized_calls: List[Dict[str, Any]] = []
    for call in tool_calls:
        try:
            serialized_calls.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(call, "name", None),
                    "type": getattr(call, "type", None),
                    "arguments": getattr(call, "arguments", None),
                }
            )
        except Exception:
            serialized_calls.append({"raw": str(call)})
    return serialized_calls


def _to_serializable(payload: Any, max_length: int = 4000) -> Any:
    if payload is None:
        return None

    try:
        import json

        dumped = json.dumps(payload, ensure_ascii=False, default=str)
        if len(dumped) > max_length:
            return dumped[: max_length - 3] + "..."
        return json.loads(dumped)
    except Exception:
        text = str(payload)
        return text[: max_length - 3] + "..." if len(text) > max_length else text


class AgentInteractionLogger:
    """Helper for recording agent prompts, responses, and outputs."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def log_prompt(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        prompt_messages: Iterable[BaseMessage],
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        context_recorder.append(
            {
                "event": "agent_prompt",
                "agent": self.agent_name,
                "node": node,
                "dataset_id": dataset_id,
                "prompt": _serialize_messages(prompt_messages),
                "prompt_context": _to_serializable(prompt_context),
            }
        )

    def log_response(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        response: Any,
        tool_calls: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        content = getattr(response, "content", response)
        context_recorder.append(
            {
                "event": "agent_response",
                "agent": self.agent_name,
                "node": node,
                "dataset_id": dataset_id,
                "response": _to_serializable(content),
                "tool_calls": _serialize_tool_calls(tool_calls if tool_calls is not None else getattr(response, "tool_calls", None)),
                "extra": _to_serializable(extra),
            }
        )

    def log_result(
        self,
        *,
        node: str,
        dataset_id: Optional[str],
        result: Any,
        description: Optional[str] = None,
    ) -> None:
        context_recorder.append(
            {
                "event": "agent_result",
                "agent": self.agent_name,
                "node": node,
                "dataset_id": dataset_id,
                "description": description,
                "result": _to_serializable(result),
            }
        )

