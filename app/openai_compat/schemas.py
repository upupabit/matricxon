"""OpenAI's request and response shapes for /v1 (chat completions, embeddings, models).

Only what Matricxon can honour is modelled; unknown fields are ignored, as OpenAI-compatible servers
usually do. A few non-OpenAI fields many clients send through `extra_body` are accepted too:
`top_k`, `min_p`, `repeat_penalty`, `num_ctx` and `think`.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    # Text, or parts: {"type": "text", "text"} and {"type": "image_url", "image_url": {"url"}}.
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    n: int | None = Field(default=None, ge=1)
    logprobs: bool | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # Matricxon's own options (not OpenAI's), for clients that pass extra fields.
    top_k: int | None = None
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repeat_penalty: float | None = None
    num_ctx: int | None = Field(default=None, ge=256)
    think: bool | None = None


class OpenAIEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "matricxon"


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class OpenAIEmbedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str  # a base64 string when encoding_format is "base64"


class OpenAIEmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIEmbedding]
    model: str
    usage: Usage


class OpenAIChatResponse(BaseModel):
    """`chat.completion`; the streamed `chat.completion.chunk`s are plain dicts (translate.py)."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: Usage
