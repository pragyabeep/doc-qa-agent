"""Single place that builds the Grok client. The OpenAI SDK gives us timeouts and
exponential-backoff retries out of the box (max_retries)."""
from . import config as cfg
from .tracing import LANGFUSE_ENABLED, AsyncOpenAI

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not cfg.XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = AsyncOpenAI(
            api_key=cfg.XAI_API_KEY,
            base_url=cfg.XAI_BASE_URL,
            timeout=cfg.LLM_TIMEOUT,
            max_retries=3,
        )
    return _client


def chat_kwargs(name: str) -> dict:
    kwargs: dict = {"model": cfg.LLM_MODEL}
    if cfg.LLM_TEMPERATURE.strip() != "":
        kwargs["temperature"] = float(cfg.LLM_TEMPERATURE)
    if LANGFUSE_ENABLED:  # the Langfuse wrapper accepts a generation name
        kwargs["name"] = name
    return kwargs
