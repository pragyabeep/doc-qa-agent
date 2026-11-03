"""Langfuse wiring. If no Langfuse keys are set, everything here becomes a harmless no-op."""
import os
from contextlib import nullcontext

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if LANGFUSE_ENABLED:
    from langfuse import observe  # noqa: F401
    from langfuse.openai import AsyncOpenAI  # noqa: F401  (drop-in wrapper: traces every LLM call)
else:
    from openai import AsyncOpenAI  # noqa: F401

    def observe(func=None, **_kwargs):
        """No-op stand-in that supports both @observe and @observe(name=...)."""
        if func is None:
            return lambda f: f
        return func


def trace_context(session_id: str | None = None):
    """Name the trace and attach the session id to every observation inside it."""
    if not LANGFUSE_ENABLED:
        return nullcontext()
    from langfuse import propagate_attributes

    kwargs = {"trace_name": "doc-qa-agent", "tags": ["capstone"]}
    if session_id:
        kwargs["session_id"] = session_id[:200]
    try:
        return propagate_attributes(**kwargs)
    except TypeError:  # older SDK without trace_name
        kwargs.pop("trace_name")
        return propagate_attributes(**kwargs)


def current_trace_id() -> str | None:
    if not LANGFUSE_ENABLED:
        return None
    from langfuse import get_client

    return get_client().get_current_trace_id()


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach a score (user feedback, code check, judge) to a trace."""
    if not (LANGFUSE_ENABLED and trace_id):
        return
    from langfuse import get_client

    get_client().create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def flush() -> None:
    """Short-lived processes (evals, shutdown) must flush or traces can be lost."""
    if LANGFUSE_ENABLED:
        from langfuse import get_client

        get_client().flush()
