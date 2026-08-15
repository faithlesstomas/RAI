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

from .routers import agents, execution, history, mcp
from . import __version__
from .dependencies import close_dependencies
from .tools.security.auth import is_authorized

load_dotenv()


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

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup logic
    yield
    # Shutdown logic
    await close_dependencies()

app = FastAPI(
    title="RAI - Rich AI Runtime",
    description="Local-first agent runtime and secure Linux capability gateway.",
    version=__version__,
    lifespan=lifespan,
)



cors_origins = [
    origin.strip()
    for origin in os.environ.get("RAI_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthenticationMiddleware)

# --- Include Routers ---
app.include_router(agents.router)
app.include_router(history.router)
app.include_router(execution.router)
app.include_router(mcp.router)

# --- Global Endpoints ---

@app.get("/health", tags=["Server"])
async def health_check() -> JSONResponse:
    """
    Simple health check endpoint to confirm the server is running.
    """
    return JSONResponse(content={"status": "ok"})
