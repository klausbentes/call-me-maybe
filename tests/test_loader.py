"""Tests for JSON decoding and Pydantic validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.errors import InputError
from src.loader import load_function_definitions, load_prompt_definitions


class LoaderTests(unittest.TestCase):
    """Verify the public input-loading functions."""

    def test_loads_valid_documents(self) -> None:
        """Valid JSON documents are converted into typed Pydantic models."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            functions_path = directory / "functions.json"
            prompts_path = directory / "prompts.json"
            functions_path.write_text(
                (
                    '[{"name":"fn_flag","description":"Set a flag.",'
                    '"parameters":{"value":{"type":"boolean"}},'
                    '"returns":{"type":"null"}}]'
                ),
                encoding="utf-8",
            )
            prompts_path.write_text('[{"prompt":"Enable it"}]', encoding="utf-8")
            functions = load_function_definitions(functions_path)
            prompts = load_prompt_definitions(prompts_path)
        self.assertEqual(functions.root[0].parameters["value"].type, "boolean")
        self.assertEqual(prompts.root[0].prompt, "Enable it")

    def test_rejects_invalid_json(self) -> None:
        """Malformed JSON is translated into an InputError with location data."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.json"
            path.write_text("[", encoding="utf-8")
            with self.assertRaisesRegex(InputError, "invalid JSON.*line 1"):
                load_prompt_definitions(path)

    def test_rejects_unknown_schema_type(self) -> None:
        """Unsupported parameter types are rejected before generation begins."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "functions.json"
            path.write_text(
                (
                    '[{"name":"fn_bad","description":"Bad.",'
                    '"parameters":{},"returns":{"type":"date"}}]'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InputError, "unsupported type 'date'"):
                load_function_definitions(path)

    def test_rejects_extra_prompt_keys(self) -> None:
        """Prompt records must have exactly the documented prompt field."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_text('[{"prompt":"Hi","extra":true}]', encoding="utf-8")
            with self.assertRaisesRegex(InputError, "invalid prompt input schema"):
                load_prompt_definitions(path)
