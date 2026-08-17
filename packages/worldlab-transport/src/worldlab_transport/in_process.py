"""In-process transport used by local composition and tests."""

from __future__ import annotations

from .contracts import (
    RequestHandler,
    TransportErrorInfo,
    TransportRequest,
    TransportResponse,
)
from .errors import (
    TransportClosedError,
    TransportProtocolError,
    TransportRemoteError,
)


class InProcessTransportServer:
    """Dispatch requests to registered handlers in the current process."""

    def __init__(self) -> None:
        self._handlers: dict[str, RequestHandler] = {}
        self._closed = False

    def register(self, method: str, handler: RequestHandler) -> None:
        self._ensure_open()
        _validate_method(method)
        if not callable(handler):
            raise TypeError("transport handler must be callable")
        if method in self._handlers:
            raise ValueError(f"transport method already registered: {method!r}")
        self._handlers[method] = handler

    def dispatch(self, request: TransportRequest) -> TransportResponse:
        self._ensure_open()
        if not isinstance(request, TransportRequest):
            raise TransportProtocolError("server received an invalid request")

        handler = self._handlers.get(request.method)
        if handler is None:
            return TransportResponse(
                request_id=request.request_id,
                error=TransportErrorInfo(
                    code="method_not_found",
                    message=f"no handler registered for {request.method!r}",
                ),
            )

        try:
            result = handler(request)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return TransportResponse(
                request_id=request.request_id,
                error=TransportErrorInfo(
                    code="handler_error",
                    message=str(error) or error.__class__.__name__,
                    details={"exception_type": error.__class__.__name__},
                ),
            )

        if isinstance(result, TransportResponse):
            if result.request_id != request.request_id:
                raise TransportProtocolError(
                    "transport handler returned a response for another request"
                )
            return result
        return TransportResponse(request_id=request.request_id, payload=result)

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise TransportClosedError("transport server is closed")


class InProcessTransportClient:
    """Client with the same request/error behavior as a remote client."""

    def __init__(self, server: InProcessTransportServer) -> None:
        if not isinstance(server, InProcessTransportServer):
            raise TypeError("server must be an InProcessTransportServer")
        self._server = server
        self._closed = False

    def request(self, request: TransportRequest) -> TransportResponse:
        self._ensure_open()
        if not isinstance(request, TransportRequest):
            raise TransportProtocolError("client requires a TransportRequest")
        response = self._server.dispatch(request)
        if response.request_id != request.request_id:
            raise TransportProtocolError(
                "transport response request_id does not match the request"
            )
        if response.error is not None:
            raise TransportRemoteError(response)
        return response

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise TransportClosedError("transport client is closed")


def _validate_method(method: str) -> None:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("transport method must be non-empty")
