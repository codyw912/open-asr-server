# Contributing

Thanks for considering a contribution to Open ASR Server.

## Development

1. Install dependencies:

```bash
uv sync --frozen --extra dev
```

2. Run tests:

```bash
uv run --frozen --extra dev pytest
```

## Testing

- Run coverage locally:

```bash
uv run --frozen --extra dev pytest --cov=open_asr_server --cov-report=term
```

CI checks on PRs and `main` include:

- `uv lock --check`
- Python test matrix (3.11-3.14)
- package build (`uv build --no-sources`)
- coverage artifact generation

- Keep tests hermetic by avoiding reliance on local env vars, caches, or model downloads.

## Pull requests

- Keep changes focused and include a clear description.
- Add or update tests when behavior changes.
- Ensure `uv run --frozen --extra dev pytest` passes before submitting.

## Releases

Use a PR-first release flow. Do not push release commits directly to `main`.

1. Create a release branch from `main` (example: `release-0.2.2`).
2. Update release files on that branch:
   - `pyproject.toml` version
   - `CHANGELOG.md`
   - `uv.lock` (`uv lock`)
3. Run validation before opening the PR:

```bash
uv run --frozen --extra dev pytest
```

4. Open a PR to `main` and wait for required checks/review.
5. Merge the PR through GitHub (no direct pushes to `main`).
6. After merge, sync local `main` and tag the merged commit:

```bash
git checkout main
git pull --ff-only
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

7. Create a GitHub release for that tag (this also documents release notes).

Note: publishing is triggered by tag pushes via `.github/workflows/publish.yml`.
