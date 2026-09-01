"""Opt-in performance instrumentation for one real schema-aware Qwen step."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from src.constrained import SchemaGenerationMetrics, generate_schema_json
from src.generation import GenerationError, build_generation_prompt, create_model
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
