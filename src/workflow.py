"""End-to-end processing and output writing for schema-aware function calls."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .constrained import (
    SchemaGenerationLimitError,
    SchemaGenerationMetrics,
    generate_schema_json,
)
from .errors import InputError
from .generation import (
    GenerationError,
    PublicLanguageModel,
    build_generation_prompt,
    create_model,
    decode_tokens,
    load_model_vocabulary,
)
from .models import FunctionCallResult, FunctionDefinitions, PromptDefinitions


class PromptBenchmarkRecord(BaseModel):
    """Record timing and constrained-generation status for one diagnostic prompt."""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt: str
    total_seconds: float
    generated_tokens: int
    function_name: str | None = None
    parameters: dict[str, Any] | None = None
    generated_json: str | None = None
    logits_seconds: float
    constraint_seconds: float
    status: Literal["completed", "failed", "interrupted"]
    error: str | None = None
    prefix: str | None = None
    active_parameter: str | None = None
    active_parameter_type: str | None = None


DiagnosticReporter = Callable[[PromptBenchmarkRecord], None]


def _value_matches_type(value: Any, type_name: str) -> bool:
    """Check the top-level JSON value type declared for a function parameter."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _parse_generated_call(
    generated_text: str, prompt: str, functions: FunctionDefinitions
) -> FunctionCallResult:
    """Parse and defensively verify one schema-aware decoder result."""
    try:
        payload = json.loads(generated_text)
    except json.JSONDecodeError as error:
        raise GenerationError(f"decoder returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict) or set(payload) != {"name", "parameters"}:
        raise GenerationError("decoder result must contain exactly 'name' and 'parameters'")
    name = payload.get("name")
    parameters = payload.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, dict):
        raise GenerationError("decoder result has an invalid name or parameters object")
    selected = next((item for item in functions.root if item.name == name), None)
    if selected is None:
        raise GenerationError("decoder selected a function absent from the definitions")
    if set(parameters) != set(selected.parameters):
        raise GenerationError("decoder parameters do not exactly match the selected function")
    for parameter_name, definition in selected.parameters.items():
        if not _value_matches_type(parameters[parameter_name], definition.type):
            raise GenerationError(
                f"decoder parameter '{parameter_name}' does not match type '{definition.type}'"
            )
    return FunctionCallResult(prompt=prompt, name=name, parameters=parameters)


def process_prompts(
    model: PublicLanguageModel,
    functions: FunctionDefinitions,
    prompts: PromptDefinitions,
    max_new_tokens: int = 128,
) -> list[FunctionCallResult]:
    """Generate and validate one schema-aware function call for every input prompt."""
    results: list[FunctionCallResult] = []
    token_text_cache: dict[int, str] = {}
    vocabulary = load_model_vocabulary(model)
    vocabulary_ids = set(vocabulary.values())
    for index, prompt_definition in enumerate(prompts.root, start=1):
        try:
            model_prompt = build_generation_prompt(prompt_definition.prompt, functions)
            token_ids = generate_schema_json(
                model,
                model_prompt,
                functions,
                max_new_tokens,
                token_text_cache,
                vocabulary,
                vocabulary_ids,
                original_prompt=prompt_definition.prompt,
            )
            generated_text = decode_tokens(model, token_ids)
            if generated_text is None:
                raise GenerationError("the SDK does not expose the required decode method")
            result = _parse_generated_call(generated_text, prompt_definition.prompt, functions)
            results.append(result)
        except GenerationError as error:
            raise GenerationError(f"generation failed for prompt {index}: {error}") from error
    return results


def diagnose_prompts(
    model: PublicLanguageModel,
    functions: FunctionDefinitions,
    prompts: PromptDefinitions,
    max_new_tokens: int = 128,
    reporter: DiagnosticReporter | None = None,
) -> list[PromptBenchmarkRecord]:
    """Run every prompt sequentially and emit a durable record after each attempt.

    This diagnostic path shares the normal vocabulary and decoded-token caches, but
    never writes an output file and never changes production error behavior.
    """
    records: list[PromptBenchmarkRecord] = []
    token_text_cache: dict[int, str] = {}
    vocabulary = load_model_vocabulary(model)
    vocabulary_ids = set(vocabulary.values())
    for index, prompt_definition in enumerate(prompts.root, start=1):
        metrics = SchemaGenerationMetrics()
        started = time.perf_counter()
        try:
            model_prompt = build_generation_prompt(prompt_definition.prompt, functions)
            token_ids = generate_schema_json(
                model,
                model_prompt,
                functions,
                max_new_tokens,
                token_text_cache,
                vocabulary,
                vocabulary_ids,
                metrics=metrics,
                original_prompt=prompt_definition.prompt,
            )
            generated_text = decode_tokens(model, token_ids)
            if generated_text is None:
                raise GenerationError("the SDK does not expose the required decode method")
            result = _parse_generated_call(generated_text, prompt_definition.prompt, functions)
            record = PromptBenchmarkRecord(
                index=index,
                prompt=prompt_definition.prompt,
                total_seconds=time.perf_counter() - started,
                generated_tokens=len(token_ids),
                function_name=result.name,
                parameters=result.parameters,
                generated_json=generated_text,
                logits_seconds=metrics.logits_seconds,
                constraint_seconds=metrics.constraint_seconds,
                status="completed",
            )
        except SchemaGenerationLimitError as error:
            diagnostics = error.diagnostics
            record = PromptBenchmarkRecord(
                index=index,
                prompt=prompt_definition.prompt,
                total_seconds=time.perf_counter() - started,
                generated_tokens=metrics.generated_tokens,
                function_name=diagnostics.selected_function,
                logits_seconds=metrics.logits_seconds,
                constraint_seconds=metrics.constraint_seconds,
                status="failed",
                error=str(error),
                prefix=diagnostics.prefix,
                active_parameter=diagnostics.active_parameter,
                active_parameter_type=diagnostics.active_parameter_type,
            )
        except GenerationError as error:
            record = PromptBenchmarkRecord(
                index=index,
                prompt=prompt_definition.prompt,
                total_seconds=time.perf_counter() - started,
                generated_tokens=metrics.generated_tokens,
                logits_seconds=metrics.logits_seconds,
                constraint_seconds=metrics.constraint_seconds,
                status="failed",
                error=str(error),
            )
        except KeyboardInterrupt:
            record = PromptBenchmarkRecord(
                index=index,
                prompt=prompt_definition.prompt,
                total_seconds=time.perf_counter() - started,
                generated_tokens=metrics.generated_tokens,
                logits_seconds=metrics.logits_seconds,
                constraint_seconds=metrics.constraint_seconds,
                status="interrupted",
            )
            records.append(record)
            if reporter is not None:
                reporter(record)
            break
        records.append(record)
        if reporter is not None:
            reporter(record)
    return records


def write_results(results: Sequence[FunctionCallResult], output_path: Path) -> None:
    """Atomically write the exact required JSON array, creating its directory if needed."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output_path)
    except OSError as error:
        raise InputError(f"could not write output file: {output_path}") from error


def run_calling_flow(
    functions: FunctionDefinitions,
    prompts: PromptDefinitions,
    output_path: Path,
    max_new_tokens: int = 128,
) -> list[FunctionCallResult]:
    """Initialize Qwen once, process all prompts, then write a complete output document."""
    model = create_model()
    results = process_prompts(model, functions, prompts, max_new_tokens)
    write_results(results, output_path)
    return results
