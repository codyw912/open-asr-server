"""CLI entry point for the ASR server."""

from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="openai-asr-server",
    help="OpenAI-compatible ASR server for local transcription.",
)


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host to bind to")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
    preload: Annotated[
        Optional[list[str]],
        typer.Option("--preload", "-m", help="Models to preload at startup"),
    ] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development")
    ] = False,
):
    """Start the transcription server."""
    import uvicorn

    from .config import ServerConfig

    config = ServerConfig(
        host=host,
        port=port,
        preload_models=preload or [],
    )

    # Store config in module for app factory to access
    import openai_asr_server

    openai_asr_server._server_config = config

    uvicorn.run(
        "openai_asr_server:app",
        host=host,
        port=port,
        reload=reload,
    )


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
