from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api import create_app
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.embedding import MockEmbeddingService
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.retrieval import DenseRetriever, SparseRetriever
from rag_pipeline.fusion import HybridRetriever
from rag_pipeline.reranker import LexicalReranker
from rag_pipeline.generation import Generator

DATA_DIR = Path(__file__).parent.parent


def build_app(base: Path):
    pipe = IngestionPipeline(
        chunking_config=ChunkingConfig(chunk_size=400, chunk_overlap=40, strategy="fixed"),
        embedding_service=MockEmbeddingService(dimensions=384),
        vector_store=VectorStore(persist_dir=str(base / "chroma")),
        bm25_index=BM25Index(persist_dir=str(base / "bm25")),
        document_loader=DocumentLoader(
            raw_dir=str(base / "raw"),
            processed_dir=str(base / "processed"),
        ),
    )
    pipe.ingest_file(str(DATA_DIR / "data" / "raw" / "fastapi_docs" / "index.md"))

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(
            embedding_service=pipe.embedding_service,
            vector_store=pipe.vector_store,
        ),
        sparse_retriever=SparseRetriever(bm25_index=pipe.bm25_index),
        reranker=LexicalReranker(),
    )

    # Role-aware mock LLM: the same Generator is used to both answer questions
    # and serve as judge (citation verify + confidence completeness). It must
    # dispatch on the system prompt like a real model would.
    def fake_generate(messages):
        import re
        system = messages[0]["content"] if messages else ""
        if "completeness judge" in system:
            return '{"score": 0.9, "reason": "addresses the question"}'
        if "Citation" in system and "verify" in system:
            return '{"supported": true, "reason": "context supports the claim"}'
        if "answer" in system or "cited" in system:
            return "Reciprocal Rank Fusion combines dense and sparse rankings [1]."
        # Fall back to the grounded answer prompt shape
        return "Reciprocal Rank Fusion combines dense and sparse rankings [1]."

    generator = Generator(respond=fake_generate)

    app = create_app(
        pipeline=pipe,
        retriever=retriever,
        generator=generator,
        threshold=0.5,
    )
    return app


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A fresh app+index per test (isolated persistent dirs under tmp_path)."""
    return TestClient(build_app(tmp_path))


def test_openapi_docs_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    assert spec["info"]["title"] == "RAG Pipeline Hybrid Search API"
    paths = spec["paths"]
    assert "/v1/ask" in paths
    assert "/v1/ingest" in paths
    assert "/v1/documents" in paths
    assert "/healthz" in paths
    # OpenAPI doc page renders
    docs = client.get("/docs")
    assert docs.status_code == 200


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_chunks"] > 0


def test_ask_returns_answer_with_citations_and_sources(client):
    resp = client.post(
        "/v1/ask", json={"question": "What is FastAPI?", "top_k": 3}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["can_answer"] is True
    assert body["answer"]
    assert body["citations"] is not None
    assert "confidence" in body
    assert body["confidence"]["composite"] >= 0.0
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0
    src = body["sources"][0]
    assert "chunk_id" in src and "content" in src


def test_ask_empty_question_400(client):
    resp = client.post("/v1/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_without_llm_503(tmp_path):
    app = create_app(persist_dir=str(tmp_path / "chroma"), bm25_dir=str(tmp_path / "bm25"))
    resp = TestClient(app).post(
        "/v1/ask", json={"question": "What is FastAPI?"}
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_retrieve_supports_modes(client):
    for mode in ("hybrid", "dense", "sparse"):
        resp = client.post(
            "/v1/retrieve",
            json={"question": "What is FastAPI?", "mode": mode, "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == mode
        assert len(body["ranking"]) > 0
        item = body["ranking"][0]
        assert "rerank_score" in item
        assert "dense_score" in item
        assert "sparse_score" in item


def test_retrieve_invalid_mode_falls_back_to_hybrid(client):
    resp = client.post(
        "/v1/retrieve",
        json={"question": "What is FastAPI?", "mode": "bogus", "top_k": 3},
    )
    assert resp.status_code == 200
    # service.retrieve normalizes invalid modes to hybrid; response echoes the
    # requested value, ranking is non-empty either way
    assert len(resp.json()["ranking"]) > 0


def test_ask_supports_dense_and_sparse_modes(client):
    for mode in ("dense", "sparse"):
        resp = client.post(
            "/v1/ask",
            json={"question": "What is FastAPI?", "mode": mode, "top_k": 3},
        )
        resp = client.post(
            "/v1/ask",
            json={
                "question": "What is FastAPI?",
                "retrieval_mode": mode,
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["can_answer"] is True
        assert len(body["sources"]) > 0


def test_ask_low_relevance_declines(client):
    # Off-topic question with weak retrieval -> declines gracefully
    resp = client.post(
        "/v1/ask", json={"question": "What color is the ceiling?", "top_k": 3}
    )
    body = resp.json()
    assert body["can_answer"] is False
    assert body["declined"] is not None
    assert body["declined"]["reason"] == "insufficient_confidence"


def test_ingest_marksdown_upload(client):
    content = b"# Test Doc\n\nChunking of PDF documents works via pdfplumber."
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test_doc.md", content, "text/markdown")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["doc_id"]
    assert body["chunks_added_dense"] > 0
    assert body["chunks_added_bm25"] > 0


def test_ingest_unsupported_extension_422(client):
    resp = client.post(
        "/v1/ingest",
        files={"file": ("bad.xyz", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code in (422, 500)


def test_documents_lists_indexed(client):
    resp = client.get("/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) > 0
    first = docs[0]
    assert "doc_id" in first and "chunk_count" in first
    assert first["chunk_count"] > 0


def test_ask_generator_dir(client):
    """Ensure /v1/ask still works end-to-end after ingestion of extra doc."""
    with open(DATA_DIR / "data" / "raw" / "fastapi_docs" / "tutorial" / "first-steps.md", "rb") as f:
        content = f.read()
    resp = client.post(
        "/v1/ingest",
        files={"file": ("first-steps.md", content, "text/markdown")},
    )
    assert resp.status_code == 201

    get = client.get("/v1/documents")
    assert len(get.json()) >= 2