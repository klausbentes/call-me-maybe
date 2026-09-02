"""Basic unconstrained autoregressive generation using the public LLM SDK."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from .models import FunctionDefinition, FunctionDefinitions


Vocabulary: TypeAlias = dict[str, int]


class EncodedInput(Protocol):
    """Represent the public tensor operation used by ``Small_LLM_Model.encode``."""

    def tolist(self) -> list[list[int]]:
        """Return the two-dimensional batch of token IDs."""


class PublicLanguageModel(Protocol):
    """Describe only the public SDK methods needed by this generation layer."""

    def encode(self, text: str) -> EncodedInput:
        """Encode text into a batch containing token IDs."""

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for an input sequence."""

    def get_path_to_vocab_file(self) -> str:
        """Return a local path to the tokenizer vocabulary file."""


class GenerationError(Exception):
    """Represent a recoverable error raised by the generation layer."""


def create_model() -> PublicLanguageModel:
    """Initialize the SDK's default Qwen/Qwen3-0.6B model through its public API."""
    try:
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

        return cast(PublicLanguageModel, Small_LLM_Model())
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise GenerationError(f"could not initialize Qwen/Qwen3-0.6B: {error}") from error


def load_vocabulary(path: Path) -> Vocabulary:
    """Load a token-to-ID mapping returned by the SDK's public vocabulary path."""
    try:
        with path.open("r", encoding="utf-8") as source:
            raw_vocabulary = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"could not load vocabulary file '{path}': {error}") from error

    if not isinstance(raw_vocabulary, dict):
        raise GenerationError("vocabulary must be a JSON object mapping tokens to IDs")
    vocabulary: Vocabulary = {}
    for token, token_id in raw_vocabulary.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise GenerationError("vocabulary entries must map string tokens to integer IDs")
        vocabulary[token] = token_id
    return vocabulary


def load_model_vocabulary(model: PublicLanguageModel) -> Vocabulary:
    """Load the vocabulary using the SDK's public ``get_path_to_vocab_file`` method."""
    return load_vocabulary(Path(model.get_path_to_vocab_file()))


def build_generation_prompt(request: str, functions: FunctionDefinitions) -> str:
    """Build a stable prompt containing the request and available function definitions."""
    definitions = "\n".join(
        _format_function_definition(function) for function in functions.root
    )
    return (
        "Functions:\n"
        f"{definitions}\n"
        f"Request:{request}\n"
        "Task:choose the function and extract its arguments from the request. Copy supplied "
        "literal string values exactly; do not add spaces, punctuation, or formatting.\n"
        "JSON:"
    )


def _format_function_definition(function: FunctionDefinition) -> str:
    """Serialize one validated definition compactly without losing semantic fields."""
    parameters = ",".join(
        f"{name}:{definition.type}" for name, definition in function.parameters.items()
    )
    return (
        f"{function.name}({parameters})->{function.returns.type}:"
        f"{function.description}"
    )


def greedy_token_id(logits: Sequence[float]) -> int:
    """Return the lowest-ID token with the greatest logit value."""
    if not logits:
        raise GenerationError("cannot choose a token from an empty logits sequence")
    return max(range(len(logits)), key=lambda token_id: logits[token_id])


def encoded_token_ids(model: PublicLanguageModel, text: str) -> list[int]:
    """Encode text through the SDK and extract its single batch of token IDs."""
    batches = model.encode(text).tolist()
    if len(batches) != 1 or not batches[0]:
        raise GenerationError("SDK encode() must return one non-empty batch of token IDs")
    return list(batches[0])


def decode_tokens(model: PublicLanguageModel, token_ids: list[int]) -> str | None:
    """Decode IDs when the model exposes the optional public ``decode`` method."""
    decoder = getattr(model, "decode", None)
    if not callable(decoder):
        return None
    typed_decoder = cast(Callable[[list[int]], str], decoder)
    return typed_decoder(token_ids)


def generate_greedy(
    model: PublicLanguageModel,
    prompt: str,
    max_new_tokens: int = 64,
    end_token_ids: set[int] | None = None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` greedily without constraining logits.

    The provided SDK exposes no public EOS-token accessor. Callers may therefore pass
    a known end-token ID set when another public source makes it available; otherwise
    generation stops only at the configured maximum.
    """
    if max_new_tokens < 0:
        raise GenerationError("max_new_tokens must not be negative")
    input_ids = encoded_token_ids(model, prompt)
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        next_token_id = greedy_token_id(model.get_logits_from_input_ids(input_ids))
        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)
        if end_token_ids is not None and next_token_id in end_token_ids:
            break
    return generated_ids


def generate_text_for_request(
    model: PublicLanguageModel,
    request: str,
    functions: FunctionDefinitions,
    max_new_tokens: int = 64,
    end_token_ids: set[int] | None = None,
) -> str | None:
    """Generate unconstrained text for a request using the supplied definitions.

    This is intentionally only a generation baseline. It does not parse the result,
    choose a function itself, or claim that the decoded text is a valid function call.
    """
    prompt = build_generation_prompt(request, functions)
    token_ids = generate_greedy(model, prompt, max_new_tokens, end_token_ids)
    return decode_tokens(model, token_ids)
