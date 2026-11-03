# CloudSync Document Q&A Agent

An AI agent that answers questions about a product's documentation, with **cited, structured answers**,
**every request traced in Langfuse**, and a **measured eval score**. It works with any OpenAI-compatible LLM API. I ran it on Groq's free tier (`openai/gpt-oss-120b`
and `qwen/qwen3.8-27b`); xAI Grok also works by changing three lines in `.env`.

## What it does

| Requirement | Where |
| --- | --- |
| RAG over documents (chunking, embeddings, vector DB, hybrid search) | `app/retrieval.py` (Chroma + BM25, fused with Reciprocal Rank Fusion) |
| Agent with extra tools | `app/tools.py`: `search_documents`, `lookup_ticket`, `calculator` |
| Hand-written agent loop (no framework) | `app/agent.py` |
| Structured, cited answers | `submit_answer` tool validated with Pydantic (`app/schemas.py`) |
| Observability | Langfuse: `@observe` spans plus the drop-in OpenAI wrapper (`app/tracing.py`) |
| API + container | FastAPI (`app/main.py`), `Dockerfile`, `docker-compose.yml` |
| Evals | `evals/` (35-question dataset, scoring, before/after comparison) |

## Architecture

```mermaid
flowchart LR
    U[Client] -->|POST /ask| API[FastAPI]
    API --> A[Agent loop]
    A -->|messages + tools| LLM[LLM via OpenAI-compatible API]
    LLM -->|tool call| A
    A --> T1[search_documents]
    A --> T2[lookup_ticket]
    A --> T3[calculator]
    T1 --> H[Hybrid search]
    H --> V[(Chroma vectors)]
    H --> K[(BM25 keywords)]
    LLM -->|submit_answer| S[Pydantic validation]
    S --> API
    A -.traces.-> LF[(Langfuse)]
```

**How a request flows:** the agent sends the question and tool descriptions to the LLM. The LLM never runs anything; it replies
with tool-call requests, our code runs them, and the results go back. The loop ends when the model calls `submit_answer` with
`answer`, `citations`, `found_in_documents`, and `confidence`. That is validated against a schema; if invalid, the error is
returned to the model so it can fix it.

**Safety rails in the loop:** hard cap of `MAX_STEPS` (6); stops after 3 consecutive tool errors; tool errors are returned as
text so the model can recover; tool exceptions never crash the request; retrieved text is declared as data, not instructions
(prompt-injection hygiene); the calculator parses the expression instead of using `eval`. None of the tools are irreversible,
so no human confirmation step is needed. One would be added before any write/delete tool.

**Retrieval design:** markdown is split by heading first, then into ~800-character chunks with 120 overlap; each chunk carries
its heading. Pure vector search can miss exact tokens like `E-4021`, so BM25 keyword search runs alongside it and the two
rankings are merged with RRF. Embeddings run locally (MiniLM via Chroma), so only the LLM needs an API key.
If nothing relevant is found, the tool returns `NO_RESULTS` and the agent answers `found_in_documents=false` instead of guessing.

## Run it

### Local Python (verified on Windows, Python 3.14)

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # Mac/Linux: cp .env.example .env   then add your API key
python -m app.ingest               # builds the index (first run downloads a small embedding model)
python -m uvicorn app.main:app --reload
```

In `.env`, `XAI_API_KEY` holds the key for whichever provider `XAI_BASE_URL` points to (the variable name is
historical). Example for Groq: `XAI_BASE_URL=https://api.groq.com/openai/v1`, `LLM_MODEL=openai/gpt-oss-120b`.

### Docker

`Dockerfile` and `docker-compose.yml` are included (`docker compose up --build`, keys read from `.env` at runtime).
**I did not test the container build on my machine**, so the local Python run above is the verified path.

### Try it

Open http://localhost:8000/docs, choose `POST /ask`, and send for example:

```json
{"question": "What does error E-4021 mean and how do I fix it?", "session_id": "demo-1"}
```

Other good questions: `How much do 8 Pro users cost for a year on annual billing?` (uses the calculator),
`What is the status of ticket T-1003 and how was it resolved?` (uses the ticket lookup), and
`Who is the CEO of CloudSync?` (not in the docs, so the agent declines).

Thumbs up/down: `POST /feedback {"trace_id": "...", "thumbs_up": true}` (stored as a score on the trace).
The response contains the structured answer, each tool step, token usage, latency, and the Langfuse `trace_id`.

## Tracing with Langfuse

Create a free project at https://cloud.langfuse.com, copy the keys into `.env`
(`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`; use `https://us.cloud.langfuse.com` for the US region).
Each request becomes one trace: `doc-qa-agent` > LLM generations (model, tokens) + retriever/tool observations.
Requests with the same `session_id` are grouped as a session. Without keys, the app runs normally with tracing off.

![Langfuse trace](docs/trace.png)

## Evals

```bash
python -m evals.run_evals --label hard-v1 --ids q24-q35 --concurrency 1    # PROMPT_VERSION=v1 in .env
python -m evals.run_evals --label hard-v2 --ids q24-q35 --concurrency 1    # PROMPT_VERSION=v2 in .env
python -m evals.compare hard-v1 hard-v2
python -m evals.rescore hard-v1       # re-score a saved run after fixing a grader bug (no model calls)
```

Per question it checks: **completed** (valid structured answer), **keywords** (expected facts present), **citation**
(right source), **tool** (expected tool used), **abstain** (unanswerable questions are declined), and
**injection_safe** (a planted instruction in a ticket is not obeyed). An optional `--judge` flag adds an
LLM-as-judge faithfulness score (not used in the results below). Scores are also attached to each Langfuse trace.
`--resume` continues a run that stopped early; saved runs live in `evals/results/`.

The dataset (`evals/dataset.json`) has 35 questions: 23 basic ones (q01-q23) and 12 harder ones (q24-q35).

### Results

**Basic set (23 questions).** The baseline prompt (v1) passed 23/23 on `qwen/qwen3.8-27b`, after I fixed a
too-strict check in my own grader (see "What I learned"). A perfect baseline cannot show an improvement, and
the strict prompt (v2) was not validly run on this set because of free-tier rate limits, so I built a harder set.

**Hard set (12 questions, q24-q35),** run on `openai/gpt-oss-120b` via Groq, one question at a time:
paraphrased facts, ticket + document questions, a calculation, questions that sound answerable but are not,
a direct "ignore the docs" attack, and a prompt-injection ticket.

| Metric | v1 (basic prompt) | v2 (strict prompt) | Change |
| --- | --- | --- | --- |
| Pass rate | 0.833 (10/12) | 0.667 (8/12) | -0.166 |
| Abstain (declines unanswerable) | 1.0 | 0.667 | -0.333 |
| Completed (valid answer produced) | 0.917 | 0.833 | -0.084 |
| Keywords (correct facts) | 0.875 | 0.857 | -0.018 |
| Citation (right source) | 0.875 | 0.857 | -0.018 |
| Injection-safe | 1.0 | 1.0 | 0 |
| Tool use | 1.0 | 1.0 | 0 |
| Avg latency (ms) | 26,834 | 19,174 | -7,660 |
| Avg tokens per question | 3,741 | 2,706 | -1,035 |

**Change tested:** replaced a two-line system prompt with a strict one (answer only from tool results, always use
the calculator, decline when the answer is not found, treat ticket and document text as data, not instructions).

**Result: no accuracy improvement.** One question flipped to pass (q31) and three flipped to fail (q28, q29, q30).
q28 and q29 never produced a valid answer: the model returned malformed tool output (HTTP 400) even after 4 retries,
which is random model/provider behavior, not a wrong answer. q30 was a real model inconsistency: the answer text said
PayPal is not documented, but it set `found_in_documents=true`. With 12 questions, one question is worth 8 points,
so these differences are within noise. (v1's abstain of 1.0 is slightly generous: for q31 the agent gave up after
6 steps, and a "could not complete" reply counts as declining.)

**Cost and latency:** v2's averages look about 28% lower, but the two errored questions count as 0 tokens and 0 ms.
Excluding them, v2 is roughly 13-14% lower (about 3,250 vs 3,740 tokens; about 23 s vs 27 s). That is a possible
small improvement in cost and latency, not a proven one.

**Limits of this test:** q34 (the prompt-injection ticket) failed for both prompts because the ticket data file in
my run did not contain T-1007, so the injection result is not meaningful (the file is fixed in this repo but the
question was not rerun). The test set is small and the model is inconsistent, so differences of one or two questions
should not be over-read.

### What I learned from the evals

- **My grader was wrong twice.** It first required zero citations when declining, but the model correctly cited the
  nearest passage. It also compared text literally, so a non-breaking hyphen (`partial-month`) or a narrow space
  (`30 minutes`) counted as a mismatch. I fixed both and re-scored the saved runs offline with
  `python -m evals.rescore`.
- **Free-tier limits shaped the process.** A run with concurrency 2 hit the per-minute token limit, and later the
  daily limit. I added rate-limit waiting, fail-fast on the daily cap, `--ids` to run a subset, and `--resume`.
- **Reliability work mattered more than the prompt.** `gpt-oss-120b` often calls tools with malformed arguments
  (for example a tool named `json`). The retry layer in `app/agent.py` turned many 502 errors into successful
  answers in manual testing. It is not a complete fix, as q28 and q29 show.
- **Next steps:** a larger test set, repeating each run several times to average out randomness, and a more
  reliable model.

## Tests

```bash
pytest -q     # calculator safety, chunking, hybrid retrieval, agent loop (with a fake LLM), API
```

## Answers to the Day 5 checks

- **What if retrieval returns nothing?** The tool returns `NO_RESULTS`; the prompt tells the agent to retry with different
  wording, then answer `found_in_documents=false` with no citations (the eval's `abstain` metric covers this).
- **Cost per request?** `usage` in the API response gives tokens (plus an estimated cost if `PRICE_IN_PER_M` /
  `PRICE_OUT_PER_M` are set). In Langfuse, open a trace to see per-generation tokens and cost, and the Dashboard for totals.
- **When to self-host a model?** Data-privacy requirements, very high volume where owning GPUs is cheaper than per-token
  pricing, or custom or fine-tuned models. Otherwise a hosted API avoids owning GPUs, scaling, and uptime.
- **Can someone run it with one command?** The Docker setup is included (`docker compose up --build` with keys in `.env`), but I did not test the container build; the local Python steps above are the verified way to run it.

## Project layout

```
app/        config, tracing, llm client, schemas, retrieval (RAG), tools, agent loop, FastAPI app
data/       sample documentation (docs/*.md) and tickets.json
evals/      dataset.json, run_evals.py, compare.py, rescore.py, upload_dataset.py, results/
try_models.py  tests which models on your API key handle tool calls reliably
tests/      unit + integration tests (no network needed)
```

## Swapping in your own documents

Put `.md` files in `data/docs/`, run `python -m app.ingest` (or rebuild the Docker image), and update the example
questions in `evals/dataset.json`. Update the tool descriptions and prompt in `app/agent.py` if the domain changes.
