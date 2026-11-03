"""FastAPI service. Run: uvicorn app.main:app --reload"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import config as cfg
from .agent import run_agent
from .retrieval import get_index
from .schemas import AskRequest, AskResponse, FeedbackRequest
from .tracing import LANGFUSE_ENABLED, flush, score


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(get_index)  # load/build the index once, at startup
    yield
    flush()  # make sure buffered traces are sent before the process exits


app = FastAPI(title="CloudSync Document Q&A Agent", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.LLM_MODEL, "prompt_version": cfg.PROMPT_VERSION,
            "retrieval_mode": cfg.RETRIEVAL_MODE, "tracing": LANGFUSE_ENABLED}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        r = await run_agent(req.question, req.session_id)

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {repr(e)}"
        )

    return AskResponse(
        answer=r.answer,
        steps=r.steps,
        usage=r.usage,
        latency_ms=r.latency_ms,
        trace_id=r.trace_id,
        completed=r.completed,
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """User thumbs up/down becomes a score on the trace in Langfuse."""
    score(req.trace_id, "user-feedback", 1.0 if req.thumbs_up else 0.0, req.comment)
    return {"ok": True, "recorded": LANGFUSE_ENABLED}
