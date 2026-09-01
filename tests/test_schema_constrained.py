"""Tests for function-schema-aware constrained decoding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.constrained import (
    StructuralDecoderState,
    consume_schema_token,
    generate_schema_json,
    validate_schema_prefix,
)
from src.models import FunctionDefinitions


def make_definitions() -> FunctionDefinitions:
    """Create varied definitions without relying on demonstration input files."""
    return FunctionDefinitions.from_json_data(
        [
            {
                "name": "fn_add",
                "description": "Add values.",
                "parameters": {"left": {"type": "number"}, "right": {"type": "number"}},
                "returns": {"type": "number"},
            },
            {
                "name": "fn_add_label",
                "description": "Label an addition.",
                "parameters": {"label": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_toggle",
                "description": "Toggle a value.",
                "parameters": {"enabled": {"type": "boolean"}},
                "returns": {"type": "boolean"},
            },
            {
                "name": "fn_record",
                "description": "Record mixed values.",
                "parameters": {
                    "note": {"type": "string"},
                    "amount": {"type": "number"},
                    "active": {"type": "boolean"},
                },
                "returns": {"type": "null"},
            },
        ]
    )


class SchemaFakeEncoded:
    """Provide the one-batch result exposed by the real SDK encoder."""

    def tolist(self) -> list[list[int]]:
        """Return a non-empty input-ID batch."""
        return [[0]]


class SchemaFakeModel:
    """Use whole-text fake tokens to exercise schema-aware masking in the loop."""

    def __init__(self, pieces: dict[int, str], logits: list[list[float]], path: str) -> None:
        """Store decoder pieces, logits, and the vocabulary file path."""
        self._pieces = pieces
        self._logits = logits
        self._path = path
        self._position = 0

    def encode(self, text: str) -> SchemaFakeEncoded:
        """Encode any prompt into a stable one-token batch."""
        del text
        return SchemaFakeEncoded()

    def decode(self, ids: list[int]) -> str:
        """Decode IDs using their complete fake-token text."""
        return "".join(self._pieces[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits for the current generation step."""
        del input_ids
        result = self._logits[self._position]
        self._position += 1
        return result

    def get_path_to_vocab_file(self) -> str:
        """Return the temporary vocabulary fixture path."""
        return self._path


class SchemaConstrainedTests(unittest.TestCase):
    """Verify dynamic function selection and declared parameter restrictions."""

    def setUp(self) -> None:
        """Load independent function definitions for each test."""
        self.functions = make_definitions()

    def test_similar_function_name_prefixes_remain_distinct(self) -> None:
        """Only complete declared names are accepted despite shared prefixes."""
        partial = validate_schema_prefix('{"name":"fn_add', self.functions)
        valid = validate_schema_prefix(
            '{"name":"fn_add","parameters":{"left":1,"right":2}}', self.functions
        )
        unknown = validate_schema_prefix('{"name":"fn_added"', self.functions)
        self.assertTrue(partial.is_possible)
        self.assertTrue(valid.is_complete)
        self.assertFalse(unknown.is_possible)

    def test_declared_string_number_and_boolean_types_are_enforced(self) -> None:
        """Each parameter accepts only the top-level type from its definition."""
        string_value = validate_schema_prefix(
            '{"name":"fn_add_label","parameters":{"label":"sum"}}', self.functions
        )
        number_value = validate_schema_prefix(
            '{"name":"fn_add","parameters":{"left":1.5,"right":2}}', self.functions
        )
        boolean_value = validate_schema_prefix(
            '{"name":"fn_toggle","parameters":{"enabled":false}}', self.functions
        )
        wrong_boolean = validate_schema_prefix(
            '{"name":"fn_toggle","parameters":{"enabled":"false"}}', self.functions
        )
        self.assertTrue(string_value.is_complete)
        self.assertTrue(number_value.is_complete)
        self.assertTrue(boolean_value.is_complete)
        self.assertFalse(wrong_boolean.is_possible)

    def test_missing_and_extra_parameters_are_rejected(self) -> None:
        """Schemas require every declared key and forbid any undeclared key."""
        missing = validate_schema_prefix(
            '{"name":"fn_add","parameters":{"left":1}}', self.functions
        )
        extra = validate_schema_prefix(
            '{"name":"fn_add_label","parameters":{"label":"x","other":1', self.functions
        )
        self.assertFalse(missing.is_possible)
        self.assertFalse(extra.is_possible)

    def test_schema_changes_after_the_selected_name(self) -> None:
        """The same parameter token is allowed or denied based on the selected function."""
        toggle_state = consume_schema_token(
            StructuralDecoderState(), '{"name":"fn_toggle","parameters":{"enabled":', self.functions
        )
        add_state = consume_schema_token(
            StructuralDecoderState(), '{"name":"fn_add","parameters":{"left":', self.functions
        )
        self.assertIsNotNone(toggle_state)
        self.assertIsNotNone(add_state)
        toggle_number = (
            consume_schema_token(toggle_state, "1", self.functions) if toggle_state else None
        )
        add_number = (
            consume_schema_token(add_state, "1", self.functions) if add_state else None
        )
        self.assertIsNone(toggle_number)
        self.assertIsNotNone(add_number)

    def test_multi_character_token_is_checked_against_schema(self) -> None:
        """A whole token with a valid function and typed parameter is accepted atomically."""
        token = '{"name":"fn_toggle","parameters":{"enabled":true}}'
        state = consume_schema_token(StructuralDecoderState(), token, self.functions)
        self.assertIsNotNone(state)
        self.assertTrue(state.is_complete if state else False)

    def test_adversarial_values_and_parameter_orders_remain_schema_valid(self) -> None:
        """Empty/special strings, negative floats, large values, and reordered keys work."""
        value = (
            '{"name":"fn_record","parameters":{'
            '"active":true,"amount":-1234567890.125,'
            '"note":"\\nquote: \\" and emoji: 😀"}}'
        )
        validation = validate_schema_prefix(value, self.functions)
        self.assertTrue(validation.is_complete)

    def test_schema_generation_masks_alternative_function_and_type(self) -> None:
        """The loop completes only the function and typed values favored by valid tokens."""
        pieces = ['{"name":"fn_toggle",', '"parameters":{"enabled":', "true}}"]
        vocabulary = {piece: index for index, piece in enumerate(pieces)}
        logits = []
        for token_id in range(len(pieces)):
            step = [-2.0] * len(pieces)
            step[token_id] = 2.0
            logits.append(step)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocabulary.json"
            path.write_text(json.dumps(vocabulary), encoding="utf-8")
            model = SchemaFakeModel(
                {index: piece for index, piece in enumerate(pieces)}, logits, str(path)
            )
            generated = generate_schema_json(model, "prompt", self.functions, 6)
        self.assertEqual(generated, [0, 1, 2])
