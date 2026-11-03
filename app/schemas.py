from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="Exact file name from the retrieved passage, e.g. 03_error_codes.md, or 'ticket:T-1001'.")
    quote: str = Field(description="Short verbatim excerpt (max ~25 words) that supports the answer.")


class AgentAnswer(BaseModel):
    answer: str = Field(description="The final answer for the user, in plain language.")
    citations: list[Citation] = Field(default_factory=list)
    found_in_documents: bool = Field(description="False if the documents/tickets do not contain the answer.")
    confidence: Literal["high", "medium", "low"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: str | None = Field(default=None, max_length=200)


class Step(BaseModel):
    step: int
    tool: str
    arguments: dict
    result_preview: str
    is_error: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    estimated_cost_usd: float | None = None


class AskResponse(BaseModel):
    answer: AgentAnswer
    steps: list[Step]
    usage: Usage
    latency_ms: int
    trace_id: str | None = None
    completed: bool = True


class FeedbackRequest(BaseModel):
    trace_id: str
    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=500)
