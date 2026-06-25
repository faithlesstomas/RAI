"""
Main file for the FastAPI server.
"""
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routers import agents, execution, history, mcp
from .dependencies import close_dependencies

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup logic
    yield
    # Shutdown logic
    await close_dependencies()

app = FastAPI(
    title="RAI - Rich AI Assistant",
    description="Backend server for the RAI CLI and other clients.",
    version="0.1.0",
    lifespan=lifespan,
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, be specific.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
