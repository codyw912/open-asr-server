# Contributing

Thanks for considering a contribution to Open ASR Server.

## Development

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Run tests:

```bash
uv run --extra dev pytest
```

## Testing

- Run coverage locally:

```bash
uv run --extra dev pytest --cov=open_asr_server --cov-report=term
```

- Keep tests hermetic by avoiding reliance on local env vars, caches, or model downloads.

## Pull requests

- Keep changes focused and include a clear description.
- Add or update tests when behavior changes.
- Ensure `uv run --extra dev pytest` passes before submitting.
