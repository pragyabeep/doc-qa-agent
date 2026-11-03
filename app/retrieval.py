"""RAG: chunk -> embed -> store (Chroma) + keyword index (BM25) -> hybrid search (RRF)."""
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi

from . import config as cfg

# keeps error codes like "e-4021" as ONE token so keyword search can match them exactly
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


# ---------------------------------------------------------------- chunking
def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start + max_chars // 2, end), text.rfind(" ", start + max_chars // 2, end))
            if cut > start:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [o for o in out if o]


def chunk_markdown(source: str, text: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split on markdown headings first (keeps topics together), then by size with overlap.
    The heading is prepended to every chunk so it carries its own context."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.startswith("#"):
            sections.append((line.lstrip("# ").strip(), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in _windows(body, max_chars, overlap):
            chunks.append(Chunk(id=f"{source}::{len(chunks)}", source=source, heading=heading,
                                text=f"{heading}\n{piece}" if heading else piece))
    return chunks


# ---------------------------------------------------------------- embeddings
class HashEmbedding(EmbeddingFunction[Documents]):
    """Tiny offline bag-of-words embedder. Only for tests / no-network demos (EMBEDDINGS=hash)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "hash-embedding"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding(**config)


def _embedding_function():
    if cfg.EMBEDDINGS == "hash":
        return HashEmbedding()
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 (ONNX), runs locally, no API key


# ---------------------------------------------------------------- index
class Index:
    def __init__(self) -> None:
        cfg.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
        self._ef = _embedding_function()
        self._chunks_file = cfg.CHROMA_DIR / "chunks.json"
        self.chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _collection(self):
        return self._client.get_or_create_collection(
            "docs", embedding_function=self._ef, metadata={"hnsw:space": "cosine"})

    def build(self) -> int:
        """(Re)build the whole index from DOCS_DIR."""
        chunks: list[Chunk] = []
        for path in sorted(cfg.DOCS_DIR.glob("*.md")):
            chunks += chunk_markdown(path.name, path.read_text(encoding="utf-8"))
        if not chunks:
            raise RuntimeError(f"No .md documents found in {cfg.DOCS_DIR}")
        try:
            self._client.delete_collection("docs")
        except Exception:
            pass
        col = self._collection()
        col.add(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                metadatas=[{"source": c.source, "heading": c.heading} for c in chunks])
        self._chunks_file.write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
        self._load_chunks(chunks)
        return len(chunks)

    def _load_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.id: c for c in chunks}
        self._ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def ensure(self) -> None:
        """Load the saved index, or build it if it doesn't exist yet."""
        if self._chunks_file.exists() and self._collection().count() > 0:
            data = json.loads(self._chunks_file.read_text(encoding="utf-8"))
            self._load_chunks([Chunk(**d) for d in data])
        else:
            self.build()

    # ------------------------------------------------------------ search
    def _vector_hits(self, query: str, n: int) -> list[tuple[str, float]]:
        res = self._collection().query(query_texts=[query], n_results=min(n, len(self._ids)))
        return [(i, d) for i, d in zip(res["ids"][0], res["distances"][0]) if d <= cfg.MAX_DISTANCE]

    def _keyword_hits(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self._ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[dict]:
        k, mode = k or cfg.TOP_K, mode or cfg.RETRIEVAL_MODE
        pool = k * 3
        vec = self._vector_hits(query, pool) if mode in ("vector", "hybrid") else []
        kw = self._keyword_hits(query, pool) if mode in ("keyword", "hybrid") else []

        # Reciprocal Rank Fusion: score = sum(1 / (60 + rank)) across both result lists
        fused: dict[str, float] = {}
        via: dict[str, set[str]] = {}
        for label, hits in (("vector", vec), ("keyword", kw)):
            for rank, (cid, _) in enumerate(hits):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
                via.setdefault(cid, set()).add(label)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**asdict(self.chunks[c]), "score": round(fused[c], 4), "via": sorted(via[c])} for c in top]


_index: Index | None = None


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index()
        _index.ensure()
    return _index
