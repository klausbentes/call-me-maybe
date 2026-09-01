"""Structural constrained decoding for the required function-call JSON envelope."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .generation import (
    GenerationError,
    PublicLanguageModel,
    Vocabulary,
    build_generation_prompt,
    decode_tokens,
    encoded_token_ids,
    greedy_token_id,
    load_model_vocabulary,
)
from .models import FunctionDefinition, FunctionDefinitions


_INVALID = -2
_INCOMPLETE = -1
_WHITESPACE = " \t\r\n"


class StructuralDecoderState(BaseModel):
    """Store the generated JSON prefix and whether its required object is complete."""

    model_config = ConfigDict(frozen=True)

    prefix: str = ""
    is_complete: bool = False


class PrefixValidation(BaseModel):
    """Describe whether a text prefix can still become the required JSON object."""

    is_possible: bool
    is_complete: bool


def _skip_whitespace(text: str, index: int) -> int:
    """Return the first index after JSON whitespace."""
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _literal(text: str, index: int, expected: str) -> int:
    """Match a literal, allowing a matching unfinished suffix at end of input."""
    remaining = text[index:]
    if text.startswith(expected, index):
        return index + len(expected)
    if expected.startswith(remaining):
        return _INCOMPLETE
    return _INVALID


def _string(text: str, index: int) -> int:
    """Parse one complete or unfinished JSON string beginning at ``index``."""
    if index >= len(text):
        return _INCOMPLETE
    if text[index] != '"':
        return _INVALID
    index += 1
    while index < len(text):
        character = text[index]
        if character == '"':
            return index + 1
        if ord(character) < 0x20:
            return _INVALID
        if character != "\\":
            index += 1
            continue
        index += 1
        if index >= len(text):
            return _INCOMPLETE
        escaped = text[index]
        if escaped in '"\\/bfnrt':
            index += 1
            continue
        if escaped != "u":
            return _INVALID
        index += 1
        for _ in range(4):
            if index >= len(text):
                return _INCOMPLETE
            if text[index] not in "0123456789abcdefABCDEF":
                return _INVALID
            index += 1
    return _INCOMPLETE


def _number(text: str, index: int) -> int:
    """Parse a JSON number, accepting incomplete but potentially valid endings."""
    if index < len(text) and text[index] == "-":
        index += 1
        if index == len(text):
            return _INCOMPLETE
    if index == len(text) or not text[index].isdigit():
        return _INVALID
    if text[index] == "0":
        index += 1
        if index < len(text) and text[index].isdigit():
            return _INVALID
    else:
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] == ".":
        index += 1
        if index == len(text):
            return _INCOMPLETE
        if not text[index].isdigit():
            return _INVALID
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] in "eE":
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        if index == len(text):
            return _INCOMPLETE
        if not text[index].isdigit():
            return _INVALID
        while index < len(text) and text[index].isdigit():
            index += 1
    return index


def _value(text: str, index: int) -> int:
    """Parse one JSON value and return its end or a prefix sentinel."""
    index = _skip_whitespace(text, index)
    if index == len(text):
        return _INCOMPLETE
    character = text[index]
    if character == '"':
        return _string(text, index)
    if character == "{":
        return _object(text, index)
    if character == "[":
        return _array(text, index)
    if character in "-0123456789":
        return _number(text, index)
    if character == "t":
        return _literal(text, index, "true")
    if character == "f":
        return _literal(text, index, "false")
    if character == "n":
        return _literal(text, index, "null")
    return _INVALID


def _object(text: str, index: int) -> int:
    """Parse a general JSON object while rejecting trailing commas."""
    index += 1
    index = _skip_whitespace(text, index)
    if index == len(text):
        return _INCOMPLETE
    if text[index] == "}":
        return index + 1
    while True:
        index = _string(text, index)
        if index < 0:
            return index
        index = _skip_whitespace(text, index)
        index = _literal(text, index, ":")
        if index < 0:
            return index
        index = _value(text, index)
        if index < 0:
            return index
        index = _skip_whitespace(text, index)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "}":
            return index + 1
        if text[index] != ",":
            return _INVALID
        index = _skip_whitespace(text, index + 1)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "}":
            return _INVALID


def _array(text: str, index: int) -> int:
    """Parse a general JSON array while rejecting trailing commas."""
    index += 1
    index = _skip_whitespace(text, index)
    if index == len(text):
        return _INCOMPLETE
    if text[index] == "]":
        return index + 1
    while True:
        index = _value(text, index)
        if index < 0:
            return index
        index = _skip_whitespace(text, index)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "]":
            return index + 1
        if text[index] != ",":
            return _INVALID
        index = _skip_whitespace(text, index + 1)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "]":
            return _INVALID


def validate_structural_prefix(prefix: str) -> PrefixValidation:
    """Validate a prefix against exactly ``{"name": string, "parameters": object}``."""
    index = _skip_whitespace(prefix, 0)
    index = _literal(prefix, index, "{")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, '"name"')
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ":")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _string(prefix, index)
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ",")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, '"parameters"')
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ":")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    if index == len(prefix):
        return PrefixValidation(is_possible=True, is_complete=False)
    if prefix[index] != "{":
        return PrefixValidation(is_possible=False, is_complete=False)
    index = _object(prefix, index)
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, "}")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    return PrefixValidation(is_possible=index == len(prefix), is_complete=index == len(prefix))


def consume_token(state: StructuralDecoderState, token_text: str) -> StructuralDecoderState | None:
    """Apply a whole decoded token only when it preserves a viable JSON prefix."""
    if not token_text or state.is_complete:
        return None
    candidate = state.prefix + token_text
    validation = validate_structural_prefix(candidate)
    if not validation.is_possible:
        return None
    return StructuralDecoderState(prefix=candidate, is_complete=validation.is_complete)


def allowed_token_ids(
    model: PublicLanguageModel,
    vocabulary: Vocabulary,
    state: StructuralDecoderState,
    token_text_cache: dict[int, str],
) -> set[int]:
    """Return vocabulary IDs whose complete decoded token keeps the state viable."""
    allowed: set[int] = set()
    for token_id in set(vocabulary.values()):
        token_text = token_text_cache.get(token_id)
        if token_text is None:
            decoded = decode_tokens(model, [token_id])
            token_text = decoded if decoded is not None else ""
            token_text_cache[token_id] = token_text
        if consume_token(state, token_text) is not None:
            allowed.add(token_id)
    return allowed


def mask_invalid_logits(logits: Sequence[float], allowed_ids: set[int]) -> list[float]:
    """Replace every disallowed or unavailable logit with negative infinity."""
    masked = [
        logit if token_id in allowed_ids else -math.inf
        for token_id, logit in enumerate(logits)
    ]
    if not any(math.isfinite(logit) for logit in masked):
        raise GenerationError("no structurally valid token is available")
    return masked


def generate_structural_json(
    model: PublicLanguageModel,
    prompt: str,
    max_new_tokens: int = 128,
) -> list[int]:
    """Greedily generate one structurally complete function-call JSON object.

    This only constrains JSON structure. It deliberately does not restrict the name,
    parameter keys, values, or types according to function definitions.
    """
    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")
    input_ids = encoded_token_ids(model, prompt)
    vocabulary = load_model_vocabulary(model)
    cache: dict[int, str] = {}
    state = StructuralDecoderState()
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = allowed_token_ids(model, vocabulary, state, cache)
        next_token_id = greedy_token_id(mask_invalid_logits(logits, allowed_ids))
        next_text = cache[next_token_id]
        next_state = consume_token(state, next_text)
        if next_state is None:
            raise GenerationError("selected token does not preserve the structural state")
        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)
        state = next_state
        if state.is_complete:
            return generated_ids
    raise GenerationError("generation reached max_new_tokens before completing JSON")


def generate_structural_text_for_request(
    model: PublicLanguageModel,
    request: str,
    functions: FunctionDefinitions,
    max_new_tokens: int = 128,
) -> str:
    """Generate and decode a structurally valid JSON envelope for one request."""
    prompt = build_generation_prompt(request, functions)
    token_ids = generate_structural_json(model, prompt, max_new_tokens)
    decoded = decode_tokens(model, token_ids)
    if decoded is None:
        raise GenerationError("constrained generation requires the public decode method")
    return decoded


def _one_of_literals(
    text: str, index: int, choices: dict[str, str]
) -> tuple[int, str | None]:
    """Match one JSON literal choice, retaining an unfinished common prefix."""
    for literal, value in choices.items():
        if text.startswith(literal, index):
            return index + len(literal), value
    remaining = text[index:]
    if any(literal.startswith(remaining) for literal in choices):
        return _INCOMPLETE, None
    return _INVALID, None


def _integer(text: str, index: int) -> int:
    """Parse a JSON integer, excluding fraction and exponent forms."""
    result = _number(text, index)
    if result < 0:
        return result
    if result < len(text) and text[result] in ".eE":
        return _INVALID
    return result


def _typed_value(text: str, index: int, type_name: str) -> int:
    """Parse one value constrained only by its declared top-level JSON type."""
    index = _skip_whitespace(text, index)
    if index == len(text):
        return _INCOMPLETE
    if type_name == "string":
        return _string(text, index)
    if type_name == "number":
        return _number(text, index)
    if type_name == "integer":
        return _integer(text, index)
    if type_name == "boolean":
        return _one_of_literals(text, index, {"true": "", "false": ""})[0]
    if type_name == "null":
        return _literal(text, index, "null")
    if type_name == "object":
        return _object(text, index) if text[index] == "{" else _INVALID
    if type_name == "array":
        return _array(text, index) if text[index] == "[" else _INVALID
    return _INVALID


def _schema_parameters(text: str, index: int, function: FunctionDefinition) -> int:
    """Parse a parameter object with exactly the keys declared by one function."""
    if index == len(text):
        return _INCOMPLETE
    if text[index] != "{":
        return _INVALID
    index = _skip_whitespace(text, index + 1)
    seen: set[str] = set()
    if index == len(text):
        return _INCOMPLETE
    if text[index] == "}":
        return index + 1 if not function.parameters else _INVALID
    while True:
        choices = {
            json.dumps(name, ensure_ascii=False): name
            for name in function.parameters
            if name not in seen
        }
        index, parameter_name = _one_of_literals(text, index, choices)
        if index < 0:
            return index
        if parameter_name is None:
            return _INVALID
        seen.add(parameter_name)
        index = _skip_whitespace(text, index)
        index = _literal(text, index, ":")
        if index < 0:
            return index
        index = _typed_value(text, index, function.parameters[parameter_name].type)
        if index < 0:
            return index
        index = _skip_whitespace(text, index)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "}":
            return index + 1 if len(seen) == len(function.parameters) else _INVALID
        if text[index] != ",":
            return _INVALID
        index = _skip_whitespace(text, index + 1)
        if index == len(text):
            return _INCOMPLETE
        if text[index] == "}":
            return _INVALID


def validate_schema_prefix(prefix: str, functions: FunctionDefinitions) -> PrefixValidation:
    """Validate a prefix against one dynamically selected function definition."""
    index = _skip_whitespace(prefix, 0)
    index = _literal(prefix, index, "{")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, '"name"')
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ":")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    names = {json.dumps(item.name, ensure_ascii=False): item.name for item in functions.root}
    index, function_name = _one_of_literals(prefix, index, names)
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    if function_name is None:
        return PrefixValidation(is_possible=False, is_complete=False)
    selected = next(item for item in functions.root if item.name == function_name)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ",")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, '"parameters"')
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, ":")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _schema_parameters(prefix, index, selected)
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    index = _literal(prefix, index, "}")
    if index < 0:
        return PrefixValidation(is_possible=index == _INCOMPLETE, is_complete=False)
    index = _skip_whitespace(prefix, index)
    return PrefixValidation(is_possible=index == len(prefix), is_complete=index == len(prefix))


def consume_schema_token(
    state: StructuralDecoderState, token_text: str, functions: FunctionDefinitions
) -> StructuralDecoderState | None:
    """Apply one whole token if it preserves the selected function's schema prefix."""
    if not token_text or state.is_complete:
        return None
    candidate = state.prefix + token_text
    validation = validate_schema_prefix(candidate, functions)
    if not validation.is_possible:
        return None
    return StructuralDecoderState(prefix=candidate, is_complete=validation.is_complete)


def allowed_schema_token_ids(
    model: PublicLanguageModel,
    vocabulary: Vocabulary,
    state: StructuralDecoderState,
    functions: FunctionDefinitions,
    token_text_cache: dict[int, str],
) -> set[int]:
    """Return IDs whose decoded token keeps a dynamically selected schema viable."""
    allowed: set[int] = set()
    for token_id in set(vocabulary.values()):
        token_text = token_text_cache.get(token_id)
        if token_text is None:
            decoded = decode_tokens(model, [token_id])
            token_text = decoded if decoded is not None else ""
            token_text_cache[token_id] = token_text
        if consume_schema_token(state, token_text, functions) is not None:
            allowed.add(token_id)
    return allowed


def generate_schema_json(
    model: PublicLanguageModel,
    prompt: str,
    functions: FunctionDefinitions,
    max_new_tokens: int = 128,
    token_text_cache: dict[int, str] | None = None,
    vocabulary: Vocabulary | None = None,
    vocabulary_ids: set[int] | None = None,
) -> list[int]:
    """Greedily generate a complete JSON object constrained by the chosen function schema."""
    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")
    input_ids = encoded_token_ids(model, prompt)
    active_vocabulary = vocabulary if vocabulary is not None else load_model_vocabulary(model)
    active_vocabulary_ids = (
        vocabulary_ids if vocabulary_ids is not None else set(active_vocabulary.values())
    )
    cache = token_text_cache if token_text_cache is not None else {}
    state = StructuralDecoderState()
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = select_schema_token_greedy(
            model, active_vocabulary, active_vocabulary_ids, state, functions, cache, logits
        )
        next_state = consume_schema_token(state, cache[next_token_id], functions)
        if next_state is None:
            raise GenerationError("selected token does not preserve the function schema")
        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)
        state = next_state
        if state.is_complete:
            return generated_ids
    raise GenerationError("generation reached max_new_tokens before completing schema JSON")


def select_schema_token_greedy(
    model: PublicLanguageModel,
    vocabulary: Vocabulary,
    vocabulary_ids: set[int],
    state: StructuralDecoderState,
    functions: FunctionDefinitions,
    token_text_cache: dict[int, str],
    logits: Sequence[float],
) -> int:
    """Select greedily with lazy masking while preserving full-mask semantics.

    Candidates are visited in descending logit order (and ascending ID for ties). Every
    rejected candidate is equivalent to assigning it ``-inf``. The first viable token
    is therefore exactly the greedy result of a fully masked vector, without decoding
    every vocabulary entry merely to prove lower-scoring candidates are irrelevant.
    """
    for token_id in sorted(range(len(logits)), key=lambda item: (-logits[item], item)):
        if token_id not in vocabulary_ids:
            continue
        token_text = token_text_cache.get(token_id)
        if token_text is None:
            decoded = decode_tokens(model, [token_id])
            token_text = decoded if decoded is not None else ""
            token_text_cache[token_id] = token_text
        if consume_schema_token(state, token_text, functions) is not None:
            return token_id
    raise GenerationError("no structurally valid token is available")
