"""Command-line interface for validating Call Me Maybe inputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import InputError
from .generation import GenerationError
from .loader import load_function_definitions, load_prompt_definitions
from .workflow import run_calling_flow


DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with the paths required by the subject."""
    parser = argparse.ArgumentParser(
        description="Validate Call Me Maybe input files (LLM generation is not enabled yet)."
    )
    parser.add_argument("--functions_definition", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Parse arguments and validate both input documents without generating calls."""
    parser = build_parser()
    options = parser.parse_args(arguments)
    try:
        functions = load_function_definitions(options.functions_definition)
        prompts = load_prompt_definitions(options.input)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        results = run_calling_flow(functions, prompts, options.output)
    except (GenerationError, InputError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    print(f"Generated {len(results)} function call(s): {options.output}")
    return 0
