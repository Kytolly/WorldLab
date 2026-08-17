from __future__ import annotations

import pytest

from worldlab_transport import (
    InProcessTransportClient,
    InProcessTransportServer,
    TransportClosedError,
    TransportErrorInfo,
    TransportProtocolError,
    TransportRemoteError,
    TransportRequest,
    TransportResponse,
)


def test_request_and_response_preserve_opaque_payload_and_metadata() -> None:
    payload = object()
    server = InProcessTransportServer()
    server.register("echo", lambda request: request.payload)
    client = InProcessTransportClient(server)

    response = client.request(
        TransportRequest(
            method="echo",
            payload=payload,
            metadata={"trace_id": "abc"},
        )
    )

    assert response.ok
    assert response.payload is payload
    assert response.request_id
    assert response.metadata == {}


def test_handler_can_return_a_complete_response() -> None:
    server = InProcessTransportServer()

    def handle(request: TransportRequest) -> TransportResponse:
        return TransportResponse(
            request_id=request.request_id,
            payload={"accepted": True},
            metadata={"server": "fake"},
        )

    server.register("submit", handle)
    response = InProcessTransportClient(server).request(
        TransportRequest("submit", {"value": 3})
    )

    assert response.payload == {"accepted": True}
    assert response.metadata == {"server": "fake"}


def test_unknown_method_is_exposed_as_remote_error() -> None:
    client = InProcessTransportClient(InProcessTransportServer())

    with pytest.raises(TransportRemoteError) as raised:
        client.request(TransportRequest("missing"))

    assert raised.value.response.error == TransportErrorInfo(
        code="method_not_found",
        message="no handler registered for 'missing'",
    )


def test_handler_failure_is_exposed_as_remote_error() -> None:
    server = InProcessTransportServer()
    server.register("fail", lambda request: (_ for _ in ()).throw(ValueError("bad")))
    client = InProcessTransportClient(server)

    with pytest.raises(TransportRemoteError) as raised:
        client.request(TransportRequest("fail"))

    assert raised.value.response.error is not None
    assert raised.value.response.error.code == "handler_error"
    assert raised.value.response.error.details == {"exception_type": "ValueError"}


def test_mismatched_handler_response_is_a_protocol_error() -> None:
    server = InProcessTransportServer()
    server.register("bad", lambda request: TransportResponse("other"))

    with pytest.raises(TransportProtocolError):
        InProcessTransportClient(server).request(TransportRequest("bad"))


def test_close_is_terminal_for_client_and_server() -> None:
    server = InProcessTransportServer()
    client = InProcessTransportClient(server)
    client.close()

    with pytest.raises(TransportClosedError):
        client.request(TransportRequest("anything"))

    server.close()
    with pytest.raises(TransportClosedError):
        server.dispatch(TransportRequest("anything"))


def test_contract_validation_rejects_empty_names() -> None:
    with pytest.raises(ValueError):
        TransportRequest("")
    with pytest.raises(ValueError):
        TransportResponse("")

