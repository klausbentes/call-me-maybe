*This project has been created as part of the 42 curriculum by klaus.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into schema-valid function calls.
This repository implements the foundation stage plus an isolated, unconstrained greedy
generation layer for Qwen/Qwen3-0.6B. The CLI still validates inputs only; function
selection, structured output, and constrained decoding are not implemented yet.

## Instructions

Python 3.10+ and `uv` are required. Install dependencies with `make install`, then run:

```sh
uv run python -m src
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

The default inputs are under `data/input/`. This stage validates them and prints a
summary. It intentionally does not create function-call results. Use `make test`,
`make lint`, or `make lint-strict` during development.

## Design decisions

Each external document is decoded using UTF-8 JSON and immediately validated with
Pydantic models. Models forbid undocumented fields, reject blank identifiers/prompts,
ensure unique function names, and restrict type declarations to standard JSON types.
Expected input failures are converted into short, actionable CLI messages.

## Algorithm explanation

The next stage will use schema-aware constrained decoding: at each token, invalid
continuations will be masked, ensuring
the final JSON has exactly `prompt`, `name`, and `parameters` with the declared types.
No heuristic selection or pseudo-generation is used in this foundation stage.

## Performance analysis

Validation is linear in the size of the JSON inputs and does not load an LLM. The
separate greedy generator makes one model forward pass per generated token, which is a
clear baseline for the constrained decoder that follows.

## Challenges faced

The main boundary is keeping the application useful before model integration without
pretending that calls were generated. The CLI therefore validates and reports readiness
instead of writing incomplete output records.

## Testing strategy

The standard-library `unittest` suite covers valid input, malformed JSON, unsupported
types, unexpected fields, and missing files. Lint targets run flake8 and mypy as
required by the subject.

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [JSON RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)

AI was used to help scaffold this first-stage implementation and documentation. The
generated code was reviewed against the project subject and is covered by local tests.
