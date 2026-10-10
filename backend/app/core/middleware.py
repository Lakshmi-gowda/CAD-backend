from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_request_size_bytes: int):
        self.app = app
        self.max_request_size_bytes = max_request_size_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_request_size_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        body_size = 0
        request_messages: list[Message] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue

            body_part = message.get("body", b"")
            body_size += len(body_part)
            if body_size > self.max_request_size_bytes:
                await self._send_too_large(scope, receive, send)
                return

            request_messages.append(message)
            if not message.get("more_body", False):
                break

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index
            if message_index < len(request_messages):
                message = request_messages[message_index]
                message_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _send_too_large(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": (
                        "Request body exceeds the maximum size of "
                        f"{self.max_request_size_bytes} bytes"
                    ),
                },
            },
        )
        await response(scope, receive, send)
