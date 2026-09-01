"""Safe JSON loading and Pydantic validation for input documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import InputError
from .models import FunctionDefinitions, PromptDefinitions


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_json(path: Path, label: str) -> Any:
    """Read and decode one UTF-8 JSON document with helpful failure messages."""
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError as error:
        raise InputError(f"{label} file not found: {path}") from error
    except IsADirectoryError as error:
        raise InputError(f"{label} path is a directory, not a file: {path}") from error
    except PermissionError as error:
        raise InputError(f"cannot read {label} file: {path}") from error
    except UnicodeDecodeError as error:
        raise InputError(f"{label} file is not valid UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        location = f"line {error.lineno}, column {error.colno}"
        raise InputError(f"invalid JSON in {label} file ({location}): {path}") from error
    except OSError as error:
        raise InputError(f"could not read {label} file: {path}") from error


def _validate(model: type[ModelT], data: Any, label: str) -> ModelT:
    """Validate decoded JSON against a Pydantic model."""
    try:
        if model is FunctionDefinitions:
            return FunctionDefinitions.from_json_data(data)  # type: ignore[return-value]
        return PromptDefinitions.from_json_data(data)  # type: ignore[return-value]
    except ValidationError as error:
        details = error.errors(include_url=False)
        raise InputError(f"invalid {label} schema: {details}") from error


def load_function_definitions(path: Path) -> FunctionDefinitions:
    """Load and validate a function-definition JSON file."""
    return _validate(
        FunctionDefinitions,
        _read_json(path, "function definitions"),
        "function definitions",
    )


def load_prompt_definitions(path: Path) -> PromptDefinitions:
    """Load and validate a prompt-input JSON file."""
    return _validate(PromptDefinitions, _read_json(path, "prompt input"), "prompt input")
