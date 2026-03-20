# /publish

Bump the package version, build the wheel, and publish to PyPI.

## Usage

```
/publish <patch|minor|major> [--dry-run]
```

## What this command does

1. Runs the full test suite (`pytest`). Aborts on failure.
2. Runs `ruff check src tests` and `mypy src`. Aborts on failure.
3. Bumps the version in `pyproject.toml` according to the argument.
4. Creates a git tag `v<new_version>`.
5. Builds the wheel with `hatch build`.
6. Publishes with `twine upload dist/*` (requires `~/.pypirc` or `TWINE_*` env vars).
7. Pushes the tag to origin.

## Dry-run mode

Pass `--dry-run` to execute all steps except the actual `twine upload` and `git push`.

## Required env vars / config

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
```

Or configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...
```
