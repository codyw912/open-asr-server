"""CLI entry point for the ASR server."""

import os
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="open-asr-server",
    help="OpenAI-compatible ASR server for local transcription.",
)


@app.callback()
def cli():
    """OpenAI-compatible ASR server for local transcription."""


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

    if preload is not None:
        os.environ["OPEN_ASR_SERVER_PRELOAD"] = ",".join(preload)

    uvicorn.run(
        "open_asr_server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
