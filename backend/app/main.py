import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import get_settings
from backend.app.core.middleware import RequestSizeLimitMiddleware
from backend.app.core.errors import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from backend.app.services.artifact_loader import get_artifact_registry
from backend.app.api.health import router as health_router
from backend.app.api.model_info import router as model_info_router
from backend.app.api.predict import router as predict_router

logger = logging.getLogger("cad_backend")

def load_artifacts_safe():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        try:
            registry.load()
            logger.info("CAD artifacts loaded successfully.")
        except Exception as exc:
            logger.critical(
                f"Model artifacts failed to load: {exc}. "
                "App will run with model_loaded=False (HTTP 503 on health/predict)."
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Initializing CAD Prediction Backend lifespan...")
    load_artifacts_safe()
    yield
    logger.info("Shutting down CAD Prediction Backend.")

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CAD Prediction Backend",
        description="FastAPI service serving XGBoost model for Coronary Artery Disease prediction",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_request_size_bytes=settings.MAX_REQUEST_SIZE_BYTES,
    )

    # PRD Section 11: CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    # Register custom exception handlers for standardized envelopes
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Include API Routers
    app.include_router(health_router)
    app.include_router(model_info_router)
    app.include_router(predict_router)

    return app

# Initialize artifacts on module load (vital for serverless & direct imports)
load_artifacts_safe()
app = create_app()
