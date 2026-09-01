"""Opt-in performance instrumentation for one real schema-aware Qwen step."""

from __future__ import annotations

import os
import time
import unittest
import json
from pathlib import Path

from src.constrained import SchemaGenerationMetrics, generate_schema_json
from src.generation import (
    GenerationError,
    PublicLanguageModel,
    build_generation_prompt,
    create_model,
    encoded_token_ids,
)
from src.loader import load_function_definitions, load_prompt_definitions
from src.models import FunctionDefinitions


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_PROFILE") == "1",
    "set RUN_QWEN_PROFILE=1 to profile one real Qwen decoding step",
)
class QwenProfileTests(unittest.TestCase):
    """Measure public-SDK inference and constrained-decoder work separately."""

    def test_minimal_schema_generation_step(self) -> None:
        """Print metrics after one real token selection from a minimal valid schema."""
        functions = FunctionDefinitions.from_json_data(
            [
                {
                    "name": "fn_x",
                    "description": "Perform an action.",
                    "parameters": {},
                    "returns": {"type": "null"},
                }
            ]
        )
        model = create_model()
        prompt = build_generation_prompt("Do it.", functions)
        metrics = SchemaGenerationMetrics()
        started = time.perf_counter()
        with self.assertRaisesRegex(GenerationError, "max_new_tokens"):
            generate_schema_json(model, prompt, functions, max_new_tokens=1, metrics=metrics)
        total = time.perf_counter() - started
        print(
            "profile: "
            f"total={total:.3f}s logits={metrics.logits_seconds:.3f}s "
            f"constraint={metrics.constraint_seconds:.3f}s "
            f"tokens={metrics.generated_tokens} rejected={metrics.rejected_candidates}"
        )

    def test_official_first_prompt_generation_step(self) -> None:
        """Profile one constrained token using the official definitions and first prompt."""
        functions = load_function_definitions(Path("data/input/functions_definition.json"))
        prompts = load_prompt_definitions(Path("data/input/function_calling_tests.json"))
        model = create_model()
        prompt = build_generation_prompt(prompts.root[0].prompt, functions)
        metrics = SchemaGenerationMetrics()
        started = time.perf_counter()
        with self.assertRaisesRegex(GenerationError, "max_new_tokens"):
            generate_schema_json(model, prompt, functions, max_new_tokens=1, metrics=metrics)
        total = time.perf_counter() - started
        print(
            "official-profile: "
            f"total={total:.3f}s logits={metrics.logits_seconds:.3f}s "
            f"constraint={metrics.constraint_seconds:.3f}s "
            f"tokens={metrics.generated_tokens} rejected={metrics.rejected_candidates}"
        )

    def test_context_and_response_token_profile(self) -> None:
        """Measure configurable public-API context timings and valid response token counts."""
        functions = load_function_definitions(Path("data/input/functions_definition.json"))
        prompts = load_prompt_definitions(Path("data/input/function_calling_tests.json"))
        prompt_text = prompts.root[0].prompt
        model = create_model()
        mode = os.environ.get("QWEN_PROFILE_MODE", "compact")
        if mode == "legacy":
            definitions = json.dumps(functions.model_dump()["root"], ensure_ascii=False, indent=2)
            context = (
                "You are a function-calling assistant.\n"
                f"Available function definitions:\n{definitions}\n\n"
                f"User request:\n{prompt_text}\n\n"
                "Generate the appropriate function-call JSON."
            )
            self._measure_logits(model, "legacy", context)
            return
        if mode == "compact":
            self._measure_logits(model, "compact", build_generation_prompt(prompt_text, functions))
            return
        if mode == "minimum":
            self._report_minimum_response_tokens(model)
            return
        target = int(mode)
        context = self._context_at_least_tokens(model, target)
        self._measure_logits(model, f"controlled-{target}", context)

    def _measure_logits(self, model: PublicLanguageModel, label: str, context: str) -> None:
        """Encode a context and print the elapsed public next-token logits call."""
        token_ids = encoded_token_ids(model, context)
        started = time.perf_counter()
        logits = model.get_logits_from_input_ids(token_ids)
        elapsed = time.perf_counter() - started
        self.assertGreater(len(logits), 0)
        print(f"context-profile: {label} tokens={len(token_ids)} logits={elapsed:.3f}s", flush=True)

    def _context_at_least_tokens(self, model: PublicLanguageModel, target: int) -> str:
        """Create a neutral repeated-token context with at least the requested token count."""
        context = "x"
        while len(encoded_token_ids(model, context)) < target:
            context += " x"
        return context

    def _report_minimum_response_tokens(self, model: PublicLanguageModel) -> None:
        """Tokenize minimal typed envelopes for every official function."""
        responses = [
            '{"name":"fn_add_numbers","parameters":{"a":0,"b":0}}',
            '{"name":"fn_greet","parameters":{"name":""}}',
            '{"name":"fn_reverse_string","parameters":{"s":""}}',
            '{"name":"fn_get_square_root","parameters":{"a":0}}',
            (
                '{"name":"fn_substitute_string_with_regex","parameters":'
                '{"source_string":"","regex":"","replacement":""}}'
            ),
        ]
        for response in responses:
            count = len(encoded_token_ids(model, response))
            print(f"minimum-response: tokens={count} json={response}", flush=True)
