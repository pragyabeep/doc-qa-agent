"""Build the vector index. Run: python -m app.ingest  (also runs during `docker build`)."""
from .retrieval import Index

if __name__ == "__main__":
    n = Index().build()
    print(f"Indexed {n} chunks.")
