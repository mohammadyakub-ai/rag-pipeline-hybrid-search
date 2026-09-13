from pathlib import Path

from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.embedding import MockEmbeddingService
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.ingestion import IngestionPipeline


SAMPLE_MD = """---
title: Product Guide
---

# Installation

To install the toolbox run `pip install toolbox`. Then configure the API key
by setting the environment variable TOOLBOX_API_KEY before starting.

## Requirements

Python 3.11 or later is required. The library depends on requests and pydantic.

# Usage

Call the analyze function with your input text to get sentiment scores back.

# Troubleshooting

If the connection fails, check that TOOLBOX_API_KEY is set and the server is
reachable on port 8080.
"""


def build_pipeline(base: Path) -> IngestionPipeline:
    config = ChunkingConfig(chunk_size=200, chunk_overlap=40)
    embedding = MockEmbeddingService(dimensions=128)
    vs = VectorStore(persist_dir=str(base / "chroma"))
    bm25 = BM25Index(persist_dir=str(base / "bm25"))
    loader = DocumentLoader(
        raw_dir=str(base / "raw"),
        processed_dir=str(base / "processed"),
    )
    return IngestionPipeline(
        chunking_config=config,
        embedding_service=embedding,
        vector_store=vs,
        bm25_index=bm25,
        document_loader=loader,
    )


def test_embedding_dimensions(mock_embedding):
    vecs = mock_embedding.embed_batch(["hello world", "another one"])
    assert len(vecs) == 2
    assert all(len(v) == 128 for v in vecs)
    # deterministic
    assert mock_embedding.embed("hello world") == vecs[0]


def test_ingest_indexes_both_stores(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    result = pipe.ingest_file(str(p))

    assert result["chunks_total"] > 0
    assert pipe.vector_store.count() > 0
    assert pipe.bm25_index.count() > 0
    assert pipe.vector_store.count() == pipe.bm25_index.count()


def test_metadata_fields(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    # Read back a chunk from the vector store
    col = pipe.vector_store._collection
    data = col.get(limit=1)
    meta = data["metadatas"][0]
    assert "doc_id" in meta
    assert "chunk_index" in meta
    assert "source_file" in meta
    assert "section_heading" in meta
    assert "chunking_strategy" in meta
    assert "char_count" in meta


def test_dense_query(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    query_vec = pipe.embedding_service.embed("how do I install the toolbox")
    results = pipe.vector_store.query(query_vec, top_k=3)
    assert len(results) > 0
    assert all("chunk_id" in r for r in results)
    assert all("content" in r for r in results)
    assert results[0]["score"] >= 0.0


def test_bm25_query_finds_exact_terms(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    results = pipe.bm25_index.query("TOOLBOX_API_KEY port 8080", top_k=5)
    assert len(results) > 0
    top = results[0]["content"].lower()
    assert "toolbox_api_key" in top or "8080" in top


def test_strategy_switch_preserves_sync(tmp_path):
    pipe = build_pipeline(tmp_path)
    pipe.set_strategy("recursive")
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))
    status = pipe.sync_status()
    assert status["in_sync"], f"Failed: {status}"


def test_idempotent_ingest(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    r1 = pipe.ingest_file(str(p))
    r2 = pipe.ingest_file(str(p))
    # Dense store should not grow (idempotent), but bm25 also stays flat
    assert r2["chunks_added_dense"] == 0


def test_indexes_persist_across_restart(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "guide.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))
    expected = pipe.sync_status()

    # Simulate a server restart: fresh pipeline against the same dirs.
    pipe2 = build_pipeline(tmp_path)
    status = pipe2.sync_status()
    assert status == expected, f"restart broke sync: {status} vs {expected}"
    hits = pipe2.bm25_index.query("chunking strategy", top_k=3)
    assert hits, "BM25 empty after restart"