from pydantic import BaseModel

from app.schemas.chat import ChatOptions


class GenerateOptions(ChatOptions):
    """The same sampling options and stop texts as a chat."""


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    options: GenerateOptions = GenerateOptions()
    stream: bool = True
    keep_alive: int | None = None

    def is_unload_call(self) -> bool:
        return not self.prompt and self.keep_alive == 0

    def is_load_call(self) -> bool:
        """Ollama's "empty prompt loads the model": nothing is generated."""
        return not self.prompt and self.keep_alive != 0


class GenerateChunk(BaseModel):
    response: str
    done: bool = False

    def to_ndjson_dict(self) -> dict:
        return self.model_dump()


class GenerateDoneChunk(BaseModel):
    response: str = ""
    done: bool = True
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_duration: int | None = None
    eval_duration: int | None = None
    done_reason: str | None = None  # "stop" or "length", as in a chat

    def to_ndjson_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
