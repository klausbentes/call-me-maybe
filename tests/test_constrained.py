"""Tests for structural JSON constrained decoding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.constrained import (
    StructuralDecoderState,
    allowed_token_ids,
    consume_token,
    generate_structural_json,
    mask_invalid_logits,
    validate_structural_prefix,
)
from src.generation import GenerationError


class FakeEncodedInput:
    """Return one stable input-ID batch without importing a tensor library."""

    def tolist(self) -> list[list[int]]:
        """Return one non-empty batch matching the public SDK contract."""
        return [[42]]


class StructuredFakeModel:
    """Script a small tokenizer and logits stream for constrained-decoder tests."""

    def __init__(self, token_texts: dict[int, str], logits: list[list[float]], path: str) -> None:
        """Store tokenizer texts, next-token logits, and vocabulary fixture path."""
        self._token_texts = token_texts
        self._logits = logits
        self._path = path
        self._calls = 0

    def encode(self, text: str) -> FakeEncodedInput:
        """Return a fixed encoded prompt."""
        del text
        return FakeEncodedInput()

    def decode(self, ids: list[int]) -> str:
        """Decode each scripted token as its full textual piece."""
        return "".join(self._token_texts[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the scripted logits for the current generation iteration."""
        del input_ids
        logits = self._logits[self._calls]
        self._calls += 1
        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return the local vocabulary fixture."""
        return self._path


class ConstrainedDecoderTests(unittest.TestCase):
    """Verify prefix states, token filtering, masking, and structural termination."""

    def test_accepts_multi_character_tokens_only_when_full_piece_is_viable(self) -> None:
        """A token containing multiple JSON characters is validated atomically."""
        state = StructuralDecoderState()
        accepted = consume_token(state, '{"name":')
        rejected = consume_token(state, "hello")
        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        self.assertEqual(accepted.prefix if accepted else "", '{"name":')

    def test_prefix_rejects_extra_key_and_trailing_comma(self) -> None:
        """The envelope permits exactly two ordered keys and no trailing comma."""
        extra = validate_structural_prefix('{"other":"x"')
        trailing = validate_structural_prefix('{"name":"x","parameters":{"a":1,}')
        self.assertFalse(extra.is_possible)
        self.assertFalse(trailing.is_possible)

    def test_allows_whitespace_outside_string_values(self) -> None:
        """Optional JSON whitespace remains valid around structural syntax."""
        prefix = ' { "name" : "x" , "parameters" : { } } '
        validation = validate_structural_prefix(prefix)
        self.assertTrue(validation.is_complete)

    def test_allows_whitespace_inside_string_values(self) -> None:
        """Spaces inside a JSON string remain part of the valid argument value."""
        prefix = '{"name":"hello world","parameters":{"message":"keep this space"}}'
        validation = validate_structural_prefix(prefix)
        self.assertTrue(validation.is_complete)
        self.assertEqual(json.loads(prefix)["parameters"]["message"], "keep this space")

    def test_prefix_supports_string_escapes_and_json_basic_values(self) -> None:
        """Strings, escapes, booleans, null, numbers, arrays, and objects remain valid."""
        prefix = '{"name":"a\\u0041","parameters":{"x":true,"y":null,"z":[1,2]}}'
        validation = validate_structural_prefix(prefix)
        self.assertTrue(validation.is_complete)
        self.assertEqual(json.loads(prefix)["name"], "aA")

    def test_allowed_ids_and_masking_remove_invalid_token(self) -> None:
        """Only structurally viable token texts retain finite logits."""
        vocabulary = {"open": 0, "prose": 1, "combined": 2}
        token_texts = {0: "{", 1: "hello", 2: '{"name":'}
        model = StructuredFakeModel(token_texts, [], "")
        cache: dict[int, str] = {}
        allowed = allowed_token_ids(model, vocabulary, StructuralDecoderState(), cache)
        masked = mask_invalid_logits([1.0, 100.0, 2.0], allowed)
        self.assertEqual(allowed, {0, 2})
        self.assertEqual(masked[1], float("-inf"))

    def test_allows_a_structural_whitespace_token_after_a_colon(self) -> None:
        """A whitespace token remains viable before a required JSON string."""
        vocabulary = {"space": 0, "quote": 1}
        token_texts = {0: " ", 1: '"'}
        model = StructuredFakeModel(token_texts, [], "")
        state = consume_token(StructuralDecoderState(), '{"name":')
        self.assertIsNotNone(state)
        allowed = allowed_token_ids(model, vocabulary, state, {}) if state else set()
        self.assertIn(0, allowed)
        self.assertIn(1, allowed)

    def test_generate_stops_when_complete_object_is_generated(self) -> None:
        """The generation loop returns immediately after a complete structural object."""
        pieces = ['{"name":', '"anything",', '"parameters":', "{}", "}"]
        token_texts = {index: piece for index, piece in enumerate(pieces)}
        vocabulary = {piece: index for index, piece in token_texts.items()}
        logits = []
        for token_id in range(len(pieces)):
            step = [-10.0] * len(pieces)
            step[token_id] = 10.0
            logits.append(step)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocab.json"
            path.write_text(json.dumps(vocabulary), encoding="utf-8")
            model = StructuredFakeModel(token_texts, logits, str(path))
            generated = generate_structural_json(model, "prompt", max_new_tokens=10)
        self.assertEqual(generated, [0, 1, 2, 3, 4])

    def test_no_allowed_token_is_reported_cleanly(self) -> None:
        """A logits vector without a finite allowed choice produces a domain error."""
        with self.assertRaisesRegex(GenerationError, "no structurally valid token"):
            mask_invalid_logits([1.0, 2.0], set())
