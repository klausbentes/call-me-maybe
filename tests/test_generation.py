"""Unit tests for unconstrained greedy generation helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from src.generation import (
    GenerationError,
    PublicLanguageModel,
    build_generation_prompt,
    generate_text_for_request,
    generate_greedy,
    greedy_token_id,
    load_model_vocabulary,
    load_vocabulary,
)
from src.models import FunctionDefinitions


class FakeEncodedInput:
    """Minimal encoded-input double compatible with the public SDK contract."""

    def __init__(self, token_ids: list[int]) -> None:
        """Store a single batch of input IDs."""
        self._token_ids = token_ids

    def tolist(self) -> list[list[int]]:
        """Return the stored single batch."""
        return [self._token_ids]


class FakeModel:
    """Scripted public-model double that avoids loading Qwen in unit tests."""

    def __init__(self, logits: list[list[float]], vocabulary_path: str = "") -> None:
        """Configure the logits emitted on successive generation steps."""
        self._logits = logits
        self._vocabulary_path = vocabulary_path
        self.inputs: list[list[int]] = []

    def encode(self, text: str) -> FakeEncodedInput:
        """Return stable prompt IDs for the requested text."""
        del text
        return FakeEncodedInput([10, 11])

    def decode(self, ids: list[int]) -> str:
        """Provide a simple decoder compatible with the SDK's public method."""
        return ":".join(str(token_id) for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Record inputs and return the next scripted logits vector."""
        self.inputs.append(list(input_ids))
        return self._logits[len(self.inputs) - 1]

    def get_path_to_vocab_file(self) -> str:
        """Return the configured local vocabulary fixture path."""
        return self._vocabulary_path


class GenerationTests(unittest.TestCase):
    """Cover generation behavior independent of a real model download."""

    def test_greedy_token_id_chooses_largest_logit(self) -> None:
        """Greedy selection returns the index with the greatest score."""
        self.assertEqual(greedy_token_id([-4.0, 1.5, 0.2]), 1)

    def test_greedy_token_id_rejects_empty_logits(self) -> None:
        """An empty vocabulary cannot produce a next token."""
        with self.assertRaisesRegex(GenerationError, "empty logits"):
            greedy_token_id([])

    def test_generation_honors_maximum_token_count(self) -> None:
        """Generation stops at its maximum when no end token is supplied."""
        model = FakeModel([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        generated = generate_greedy(model, "prompt", max_new_tokens=2)
        self.assertEqual(generated, [1, 0])
        self.assertEqual(model.inputs, [[10, 11], [10, 11, 1]])

    def test_generation_stops_at_supplied_end_token(self) -> None:
        """A supplied end-token set stops generation immediately after that token."""
        model = FakeModel([[0.0, 0.0, 4.0], [9.0, 0.0, 0.0]])
        generated = generate_greedy(model, "prompt", max_new_tokens=5, end_token_ids={2})
        self.assertEqual(generated, [2])
        self.assertEqual(model.inputs, [[10, 11]])

    def test_text_generation_builds_prompt_then_decodes_completion(self) -> None:
        """The high-level baseline uses the prompt builder and public decode method."""
        definitions = FunctionDefinitions.from_json_data(
            [{"name": "fn_x", "description": "X.", "parameters": {}, "returns": {"type": "null"}}]
        )
        model = FakeModel([[0.0, 5.0]])
        text = generate_text_for_request(model, "Do x", definitions, max_new_tokens=1)
        self.assertEqual(text, "1")

    def test_build_generation_prompt_includes_request_and_definitions(self) -> None:
        """The prompt serializes arbitrary supplied definitions without hardcoding them."""
        definitions = FunctionDefinitions.from_json_data(
            [
                {
                    "name": "fn_custom",
                    "description": "Custom action.",
                    "parameters": {"value": {"type": "boolean"}},
                    "returns": {"type": "null"},
                }
            ]
        )
        prompt = build_generation_prompt("Turn it on", definitions)
        self.assertIn("Turn it on", prompt)
        self.assertIn("fn_custom(value:boolean)->null:Custom action.", prompt)

    def test_load_vocabulary_from_sdk_path(self) -> None:
        """Vocabulary loading accepts the token-to-ID JSON representation from the SDK."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocab.json"
            path.write_text(json.dumps({"hello": 7, " world": 11}), encoding="utf-8")
            model = FakeModel([], str(path))
            vocabulary = load_model_vocabulary(cast(PublicLanguageModel, model))
        self.assertEqual(vocabulary, {"hello": 7, " world": 11})

    def test_load_vocabulary_rejects_non_mapping_json(self) -> None:
        """Vocabulary must be represented as a JSON object."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocab.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(GenerationError, "JSON object"):
                load_vocabulary(path)
