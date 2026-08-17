"""Stable, domain-neutral synchronous transport contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class TransportRequest:
    """One logical request sent to a transport endpoint.

    ``payload`` is intentionally opaque. Serialization and domain validation
    belong to the service-specific adapter or a future transport backend.
    """

    method: str
    payload: Any = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("transport request method must be non-empty")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("transport request_id must be non-empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class TransportErrorInfo:
    """Serializable error information returned by a remote endpoint."""

    code: str
    message: str
    details: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("transport error code must be non-empty")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("transport error message must be non-empty")


@dataclass(frozen=True)
class TransportResponse:
    """Response envelope corresponding to one ``TransportRequest``."""

    request_id: str
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: TransportErrorInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("transport response request_id must be non-empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def ok(self) -> bool:
        """Whether the response contains no remote error."""

        return self.error is None


RequestHandler = Callable[[TransportRequest], Any]


class TransportClient(Protocol):
    """Synchronous client boundary implemented by concrete transports."""

    def request(self, request: TransportRequest) -> TransportResponse:
        ...

    def close(self) -> None:
        ...


class TransportServer(Protocol):
    """Lifecycle boundary for a concrete request-serving transport."""

    def close(self) -> None:
        ...


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("transport metadata must be a mapping")
    return MappingProxyType(dict(value))
