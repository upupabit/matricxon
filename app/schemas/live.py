from pydantic import BaseModel


class LiveRequestEntry(BaseModel):
    """One reply's progress. `phase`: loading, building, queued, prefill, decode or done."""

    request_id: str
    model: str
    phase: str
    elapsed_seconds: float
    load_seconds: float | None = None
    prompt_tokens: int = 0
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    first_token_seconds: float | None = None
    tokens_per_second: float | None = None
    layer: int | None = None
    layer_count: int | None = None
    # Seconds per forward-pass stage; "matmul" overlaps the others (it runs inside them).
    stage_seconds: dict[str, float] = {}
    stage_timing: bool = False


class LiveSnapshot(BaseModel):
    requests: list[LiveRequestEntry]
