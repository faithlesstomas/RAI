"""Server lifecycle commands."""

import os
from typing import Optional

import typer

from rai.configuration.storage import load_settings, selected_path

server = typer.Typer(help="Run the local REST/MCP server.", no_args_is_help=True)


@server.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
    uds: Optional[str] = typer.Option(None),
    workers: int = typer.Option(1, min=1),
    reload: bool = typer.Option(False),
) -> None:
    """Validate settings before starting workers, with the same selected file."""
    import uvicorn  # noqa: PLC0415

    load_settings()
    path = selected_path()
    if path.exists():
        os.environ["RAI_CONFIG_FILE"] = str(path)
    os.environ["RAI_SERVE"] = "1"
    uvicorn.run(
        "rai.server:app",
        host=host,
        port=port,
        uds=uds,
        workers=workers,
        reload=reload,
        log_config=None,
    )
