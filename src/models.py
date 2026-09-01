"""Pydantic models for Call Me Maybe input documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


SUPPORTED_TYPES = frozenset({"number", "string", "boolean", "integer", "null", "object", "array"})


class ValueDefinition(BaseModel):
    """Describe the JSON type of a function parameter or return value."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: str = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject types that cannot be represented by this initial schema."""
        if value not in SUPPORTED_TYPES:
            available = ", ".join(sorted(SUPPORTED_TYPES))
            raise ValueError(f"unsupported type '{value}'; expected one of: {available}")
        return value


class FunctionDefinition(BaseModel):
    """Represent one callable function supplied in the definitions document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ValueDefinition]
    returns: ValueDefinition

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, value: dict[str, ValueDefinition]
    ) -> dict[str, ValueDefinition]:
        """Ensure every parameter has a non-empty name."""
        if any(not name.strip() for name in value):
            raise ValueError("parameter names must not be empty")
        return value


class FunctionDefinitions(BaseModel):
    """Represent and validate the complete list of available functions."""

    root: list[FunctionDefinition] = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def validate_unique_names(
        cls, value: list[FunctionDefinition]
    ) -> list[FunctionDefinition]:
        """Reject duplicate function names, which would make selection ambiguous."""
        names = [function.name for function in value]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return value

    @classmethod
    def from_json_data(cls, data: Any) -> FunctionDefinitions:
        """Build this root model from decoded JSON data."""
        return cls(root=data)


class PromptDefinition(BaseModel):
    """Represent one natural-language prompt from the test input."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject whitespace-only prompts without modifying the original text."""
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace only")
        return value


class PromptDefinitions(BaseModel):
    """Represent and validate the complete list of input prompts."""

    root: list[PromptDefinition]

    @classmethod
    def from_json_data(cls, data: Any) -> PromptDefinitions:
        """Build this root model from decoded JSON data."""
        return cls(root=data)
