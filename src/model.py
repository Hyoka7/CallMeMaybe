from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

ALLOWED_TYPES = ("string", "number", "boolean")


class JsonFunction(BaseModel):
    """Validated function name, schema, return type, and description."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters(self) -> "JsonFunction":
        """Require every parameter to contain one supported type."""
        for value in self.parameters.values():
            if (
                len(value) != 1
                or "type" not in value
                or value["type"] not in ALLOWED_TYPES
            ):
                raise ValueError("Invalid parameter type definition")
        return self

    @model_validator(mode="after")
    def validate_returns(self) -> "JsonFunction":
        """Require the return definition to contain one supported type."""
        if (
            len(self.returns) != 1
            or "type" not in self.returns
            or self.returns["type"] not in ALLOWED_TYPES
        ):
            raise ValueError("Invalid return type definition")
        return self


class JsonInput(BaseModel):
    """Validated collection of available function definitions."""

    model_config = ConfigDict(extra="forbid")
    func: list[JsonFunction]


class Prompt(BaseModel):
    """One validated natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_key(cls, data: Any) -> Any:
        """Give a focused validation error when the prompt key is absent."""
        if isinstance(data, dict) and "prompt" not in data:
            raise ValueError("Required key 'prompt' is missing")
        return data


class JsonResult(BaseModel):
    """One schema-constrained function call written to the output file."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]


class PromptInput(BaseModel):
    """Validated collection of natural-language prompts."""

    model_config = ConfigDict(extra="forbid")
    prompts: list[Prompt]
