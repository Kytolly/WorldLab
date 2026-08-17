# worldlab-transport

Domain-neutral synchronous transport contracts for WorldLab services.

The package deliberately does not depend on `worldlab` and does not define
domain payloads. World Model servers, Panel UI readers, OpenPI services, and
optional reward integrations can use the same request/response boundary while
choosing their own payload types and codecs.

The initial implementation provides an in-process transport for composition
and tests. Network transports and serialization codecs remain optional
backends for a later version.

```python
from worldlab_transport import (
    InProcessTransportClient,
    InProcessTransportServer,
    TransportRequest,
)

server = InProcessTransportServer()
server.register("echo", lambda request: request.payload)

client = InProcessTransportClient(server)
response = client.request(
    TransportRequest(method="echo", payload={"value": 1})
)
assert response.payload == {"value": 1}
```

Transport handlers receive a `TransportRequest` and may return either a
payload or a complete `TransportResponse`. Handler failures and unknown
methods are exposed to clients as `TransportRemoteError`.
