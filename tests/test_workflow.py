"""End-to-end tests for processing prompts and writing function-call results."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.generation import GenerationError
from src.models import FunctionDefinitions, PromptDefinitions
from src.workflow import (
    PromptBenchmarkRecord,
    _parse_generated_call,
    diagnose_prompts,
    process_prompts,
    write_results,
)


class WorkflowEncodedInput:
    """Provide a one-batch encoded prompt without loading the real tokenizer."""

    def tolist(self) -> list[list[int]]:
        """Return the expected single non-empty batch."""
        return [[99]]


class WorkflowFakeModel:
    """Generate scripted whole-token JSON pieces through the public model shape."""

    def __init__(
        self, pieces: dict[int, str], logits: list[list[float]], vocabulary_path: str
    ) -> None:
        """Store fake decoded pieces, logits, and vocabulary fixture path."""
        self._pieces = pieces
        self._logits = logits
        self._vocabulary_path = vocabulary_path
        self._position = 0

    def encode(self, text: str) -> WorkflowEncodedInput:
        """Return a stable prompt encoding."""
        del text
        return WorkflowEncodedInput()

    def decode(self, ids: list[int]) -> str:
        """Decode fake token IDs to their full textual content."""
        return "".join(self._pieces[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits for each requested generation step."""
        del input_ids
        logits = self._logits[self._position]
        self._position += 1
        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return the temporary vocabulary path."""
        return self._vocabulary_path


class WorkflowTests(unittest.TestCase):
    """Exercise schema-aware generation, conversion, and output serialization."""

    def _definitions(self) -> FunctionDefinitions:
        """Return a non-hardcoded-style definition fixture for end-to-end tests."""
        return FunctionDefinitions.from_json_data(
            [
                {
                    "name": "fn_reply",
                    "description": "Reply with text.",
                    "parameters": {"message": {"type": "string"}},
                    "returns": {"type": "string"},
                }
            ]
        )

    def _prompts(self) -> PromptDefinitions:
        """Return ambiguous prompts to prove the workflow preserves input unchanged."""
        return PromptDefinitions.from_json_data(
            [
                {"prompt": "Reply or do something else: choose the right tool."},
                {"prompt": "Could this mean either operation?"},
            ]
        )

    def test_processes_all_prompts_and_writes_required_array(self) -> None:
        """Every prompt becomes a three-key output record and output directories are created."""
        pieces = ['{"name":"fn_reply",', '"parameters":{"message":"ok"}}']
        vocabulary = {piece: index for index, piece in enumerate(pieces)}
        logits = [[2.0, -1.0], [-1.0, 2.0]] * 2
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            vocabulary_path = directory / "vocabulary.json"
            vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
            model = WorkflowFakeModel(
                {index: piece for index, piece in enumerate(pieces)}, logits, str(vocabulary_path)
            )
            results = process_prompts(model, self._definitions(), self._prompts())
            output_path = directory / "new" / "result.json"
            write_results(results, output_path)
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(output), 2)
        self.assertEqual(set(output[0]), {"prompt", "name", "parameters"})
        self.assertEqual(output[0]["prompt"], "Reply or do something else: choose the right tool.")
        self.assertEqual(output[0]["parameters"], {"message": "ok"})

    def test_generation_failure_identifies_the_prompt_number(self) -> None:
        """A failed generation is reported clearly without producing partial results."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocabulary.json"
            path.write_text('{"prose": 0}', encoding="utf-8")
            model = WorkflowFakeModel({0: "prose"}, [[1.0]], str(path))
            with self.assertRaisesRegex(GenerationError, "prompt 1"):
                process_prompts(model, self._definitions(), self._prompts())

    def test_diagnostic_reports_each_prompt_without_writing_output(self) -> None:
        """Diagnostic mode emits one completed record per sequential prompt."""
        pieces = ['{"name":"fn_reply",', '"parameters":{"message":"ok"}}']
        vocabulary = {piece: index for index, piece in enumerate(pieces)}
        logits = [[2.0, -1.0], [-1.0, 2.0]] * 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocabulary.json"
            path.write_text(json.dumps(vocabulary), encoding="utf-8")
            model = WorkflowFakeModel(
                {index: piece for index, piece in enumerate(pieces)}, logits, str(path)
            )
            reported: list[PromptBenchmarkRecord] = []
            records = diagnose_prompts(
                model, self._definitions(), self._prompts(), reporter=reported.append
            )
        self.assertEqual(records, reported)
        self.assertEqual([record.status for record in records], ["completed", "completed"])
        self.assertEqual([record.generated_tokens for record in records], [2, 2])
        self.assertEqual([record.function_name for record in records], ["fn_reply", "fn_reply"])

    def test_final_validation_rejects_schema_mismatch(self) -> None:
        """A final JSON parse cannot bypass the selected function's parameter type checks."""
        generated = '{"name":"fn_reply","parameters":{"message":42}}'
        with self.assertRaisesRegex(GenerationError, "does not match type 'string'"):
            _parse_generated_call(generated, "Ambiguous prompt", self._definitions())
