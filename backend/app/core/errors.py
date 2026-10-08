import logging
from typing import Dict, List, Optional, Any
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("cad_backend")

class AppException(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        fields: Optional[Dict[str, List[str]]] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.fields = fields
        super().__init__(message)

class ValidationError(AppException):
    def __init__(self, message: str, fields: Optional[Dict[str, List[str]]] = None):
        super().__init__(
            status_code=422,
            code="VALIDATION_ERROR",
            message=message,
            fields=fields or {},
        )

class InvalidRequestError(AppException):
    def __init__(self, message: str = "Invalid request format"):
        super().__init__(
            status_code=400,
            code="INVALID_REQUEST",
            message=message,
        )

class ModelUnavailableError(AppException):
    def __init__(self, message: str = "Model artifacts are not loaded or available"):
        super().__init__(
            status_code=503,
            code="MODEL_UNAVAILABLE",
            message=message,
        )

class InternalError(AppException):
    def __init__(self, message: str = "An internal server error occurred"):
        super().__init__(
            status_code=500,
            code="INTERNAL_ERROR",
            message=message,
        )

def format_error_response(
    status_code: int,
    code: str,
    message: str,
    fields: Optional[Dict[str, List[str]]] = None,
) -> JSONResponse:
    error_payload: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if fields:
        error_payload["fields"] = fields

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error_payload,
        },
    )

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return format_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        fields=exc.fields,
    )

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    fields: Dict[str, List[str]] = {}
    for err in exc.errors():
        loc = err.get("loc", [])
        field_name = str(loc[-1]) if loc else "body"
        msg = err.get("msg", "Invalid value")
        if field_name not in fields:
            fields[field_name] = []
        fields[field_name].append(msg)

    return format_error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        fields=fields,
    )

async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if exc.status_code == 400:
        return format_error_response(
            status_code=exc.status_code,
            code="INVALID_REQUEST",
            message=str(exc.detail),
        )
    if exc.status_code == 404:
        return format_error_response(
            status_code=exc.status_code,
            code="NOT_FOUND",
            message=str(exc.detail),
        )
    if exc.status_code == 503:
        return format_error_response(
            status_code=exc.status_code,
            code="MODEL_UNAVAILABLE",
            message=str(exc.detail),
        )
    return format_error_response(
        status_code=exc.status_code,
        code="HTTP_ERROR",
        message=str(exc.detail),
    )

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full exception server-side, never expose internal details/stack to client
    logger.exception("Unhandled server exception: %s", exc)
    return format_error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="An unexpected internal server error occurred",
    )
