"""Errors shared by transport clients and servers."""

from __future__ import annotations

from .contracts import TransportResponse


class TransportError(RuntimeError):
    """Base class for transport failures."""


class TransportClosedError(TransportError):
    """Raised when an operation is attempted after close."""


class TransportProtocolError(TransportError):
    """Raised when a transport violates the request/response contract."""


class TransportRemoteError(TransportError):
    """Raised when the remote endpoint returns a failed response."""

    def __init__(self, response: TransportResponse) -> None:
        if response.error is None:
            raise ValueError("TransportRemoteError requires a failed response")
        self.response = response
        super().__init__(
            f"remote transport error {response.error.code!r}: "
            f"{response.error.message}"
        )
