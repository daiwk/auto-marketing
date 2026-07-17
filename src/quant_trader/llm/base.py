"""Canonical chat contracts for external LLM providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

type ChatRole = Literal["system", "user", "assistant"]
MessageContent = Annotated[str, StringConstraints(min_length=1, max_length=12_000)]


class ChatMessage(BaseModel):
    """An immutable, bounded OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: ChatRole
    content: MessageContent

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must be nonblank")
        return value


type MessageInput = ChatMessage | Mapping[str, object]


class SanitizedLLMCause(RuntimeError):
    """A compact chained cause that cannot retain provider inputs or HTTP objects."""

    def __init__(
        self, category: str, *, status_code: int | None = None, attempts: int | None = None
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.attempts = attempts
        details = [category]
        if status_code is not None:
            details.append(f"status={status_code}")
        if attempts is not None:
            details.append(f"attempts={attempts}")
        super().__init__("; ".join(details))


def canonical_messages(messages: Sequence[MessageInput]) -> tuple[ChatMessage, ...]:
    """Validate arbitrary mapping input once and return immutable canonical messages."""
    if isinstance(messages, str) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of ChatMessage values or mappings")
    if not messages:
        raise ValueError("messages must contain at least one message")
    canonical: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if isinstance(message, ChatMessage):
            canonical.append(message)
            continue
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a ChatMessage or mapping")
        invalid_message = False
        try:
            canonical.append(ChatMessage.model_validate(dict(message)))
        except ValueError:
            invalid_message = True
        if invalid_message:
            raise ValueError(f"messages[{index}] is invalid") from SanitizedLLMCause(
                "invalid-message"
            )
    return tuple(canonical)


class LLMReviewer(Protocol):
    """Text-only review provider; strategy and trading decisions stay outside this boundary."""

    def complete(self, messages: Sequence[MessageInput]) -> str: ...
