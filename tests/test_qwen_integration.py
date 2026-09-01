"""Opt-in integration check against the real Qwen model supplied through llm_sdk."""

from __future__ import annotations

import os
import unittest

from src.generation import (
    create_model,
    decode_tokens,
    encoded_token_ids,
    generate_greedy,
    load_model_vocabulary,
)


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_INTEGRATION") == "1",
    "set RUN_QWEN_INTEGRATION=1 to download/load Qwen/Qwen3-0.6B",
)
class QwenIntegrationTests(unittest.TestCase):
    """Exercise the public SDK API against Qwen without asserting fixed token IDs."""

    def test_public_sdk_encode_decode_and_vocabulary(self) -> None:
        """Load Qwen and inspect public tokenizer operations on representative strings."""
        model = create_model()
        vocabulary = load_model_vocabulary(model)
        self.assertGreater(len(vocabulary), 0)
        completion = generate_greedy(model, "Respond with one word:", max_new_tokens=1)
        self.assertEqual(len(completion), 1)
        for text in ('{"name":', '"parameters"', "fn_add_numbers", "hello world"):
            with self.subTest(text=text):
                token_ids = encoded_token_ids(model, text)
                decoded = decode_tokens(model, token_ids)
                self.assertGreater(len(token_ids), 0)
                self.assertIsInstance(decoded, str)
                print(f"tokenization: {text!r} -> {len(token_ids)} token(s) -> {decoded!r}")
