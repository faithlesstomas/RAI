"""FastAPI application for the local RAI runtime."""
import os
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send
from dotenv import load_dotenv

from .routers import agents, capabilities, events, execution, history, mcp
from . import __version__
from . import config_manager
from .container import ApplicationContainer
from .dependencies import close_dependencies
from .tools.security.auth import is_authorized


class AuthenticationMiddleware:
    """Pure ASGI authentication middleware that preserves response streaming."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/"):
            if not is_authorized(Headers(scope=scope)):
                response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

async def health_check() -> JSONResponse:
    """
    Simple health check endpoint to confirm the server is running.
    """
    return JSONResponse(content={"status": "ok"})


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    """Create one fully composed RAI ASGI application."""
    load_dotenv()
    application_container = container or ApplicationContainer(config_manager.load_config())

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await application_container.start()
        yield
        await close_dependencies(application_container)

    application = FastAPI(
        title="RAI - Rich AI Runtime",
        description="Local-first agent runtime and secure Linux capability gateway.",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.container = application_container
    application.state.mcp_runtime = mcp.create_mcp_runtime(
        application_container.capability_service
    )
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("RAI_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=bool(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(AuthenticationMiddleware)
    application.include_router(agents.router)
    application.include_router(history.router)
    application.include_router(execution.router)
    application.include_router(capabilities.router)
    application.include_router(events.router)
    application.include_router(mcp.router)
    application.add_api_route("/health", health_check, methods=["GET"], tags=["Server"])
    return application


# Compatibility ASGI target; services are owned by its explicit container.
app = create_app()
