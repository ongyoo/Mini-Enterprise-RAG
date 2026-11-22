from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # extra="forbid": callers cannot smuggle role/scopes/source IDs into the body.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=5000)


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: list[float] = Field(default=[0.25, 0.28, 0.35], min_length=1, max_length=5)
