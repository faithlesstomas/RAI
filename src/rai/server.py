"""
Main file for the FastAPI server.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .routers import agents, execution, history

app = FastAPI(
    title="RAI - Rich AI Assistant",
    description="Backend server for the RAI CLI and other clients.",
    version="0.1.0",
)

# --- Include Routers ---
app.include_router(agents.router)
app.include_router(history.router)
app.include_router(execution.router)

# --- Global Endpoints ---

@app.get("/health", tags=["Server"])
async def health_check() -> JSONResponse:
    """
    Simple health check endpoint to confirm the server is running.
    """
    return JSONResponse(content={"status": "ok"})
