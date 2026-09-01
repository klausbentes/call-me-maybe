"""Module entry point for ``uv run python -m src``."""

from .cli import run


if __name__ == "__main__":
    raise SystemExit(run())
