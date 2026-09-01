*This project has been created as part of the 42 curriculum by kbentes-.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into schema-valid function calls.
This repository implements the foundation stage, an isolated greedy generation layer,
and structural constrained decoding for Qwen/Qwen3-0.6B. The CLI processes every
input prompt, lets the model select from declared function names, and writes a verified
JSON result array after the decoder enforces the selected parameter schema.

## Instructions

Python 3.10+ and `uv` are required. Install dependencies with `make install`, then run:

```sh
uv run python -m src
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

The default inputs are under `data/input/`, and the default output is
`data/output/function_calling_results.json`. The output directory is created as needed.
Use `make test`, `make lint`, or `make lint-strict` during development.

## Design decisions

Each external document is decoded using UTF-8 JSON and immediately validated with
Pydantic models. Models forbid undocumented fields, reject blank identifiers/prompts,
ensure unique function names, and restrict type declarations to standard JSON types.
Expected input failures are converted into short, actionable CLI messages.
The model is initialized once per run. Its vocabulary and decoded token texts are cached
across prompts. Output is written atomically only after every prompt has produced a
validated result, so a failed prompt cannot create a partial output document.

## Algorithm explanation

Structural constrained decoding validates each complete decoded token against an
incremental JSON-prefix parser. Invalid token logits are masked before greedy selection,
ensuring an output envelope with exactly `name` and `parameters`, valid string escapes,
and no trailing commas. Once a name is completed, its supplied schema dynamically
restricts parameter keys, requires every key, rejects extras, and enforces declared
top-level JSON types. Results are parsed and verified again before an atomic output-file
write. No heuristic function selection is used.

## Performance analysis

Validation is linear in the size of the JSON inputs. Generation makes one model forward
pass per token because the supplied public SDK exposes no KV-cache API. Candidates are
checked lazily in descending-logit order: this is equivalent to greedy selection after
masking invalid candidates with negative infinity, while avoiding decoding every token
whose score cannot affect the result. Vocabulary data and decoded token text are reused
between prompts.

Profiling one constrained token with the official definitions on the local cached Qwen
model measured 13.388 seconds in `get_logits_from_input_ids` and 0.167 seconds in
constraint filtering. Compacting the context while preserving every definition reduced
the logits measurement to 11.392 seconds; filtering remained below 0.2 seconds. The
public SDK recomputes the full sequence on each token, so inference—not constrained
decoding—is the dominant cost.

The supplied official dataset contains 11 prompts and no expected-output field, so it
cannot measure accuracy automatically. A full CPU benchmark was attempted on the local
Qwen cache, but the execution environment interrupted the silent run after model loading
and before completion. No end-to-end time, average time, accuracy, or output result is
claimed here. The mandatory five-minute target therefore remains unverified.

## Challenges faced

The main constraint is correctness without tokenizer internals: each complete public
decoder token is checked against the incremental JSON and function schema before it can
be selected. CPU performance is limited by the SDK's full-sequence, token-by-token
inference interface, so production timing must be measured on the evaluation hardware.

## Testing strategy

The standard-library `unittest` suite covers valid input, malformed JSON, missing files,
similar function names, parameter ordering, strings and escapes, negative/large floating
numbers, booleans, missing and extra parameters, type mismatches, ambiguous prompts,
output writing, and per-prompt failures. An opt-in integration test loads Qwen and its
real vocabulary. Lint targets run flake8 and mypy as required by the subject.

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [JSON RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)

AI was used to help scaffold this first-stage implementation and documentation. The
generated code was reviewed against the project subject and is covered by local tests.
