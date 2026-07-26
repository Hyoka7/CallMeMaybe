from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

ALLOWED_TYPES = ("string", "number", "boolean")


class JsonFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters(self) -> "JsonFunction":
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
        if (
            len(self.returns) != 1
            or "type" not in self.returns
            or self.returns["type"] not in ALLOWED_TYPES
        ):
            raise ValueError("Invalid return type definition")
        return self


class JsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    func: list[JsonFunction]


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_key(cls, data: Any):
        if isinstance(data, dict) and "prompt" not in data:
            raise ValueError("Required key 'prompt' is missing")
        return data


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompts: list[Prompt]
