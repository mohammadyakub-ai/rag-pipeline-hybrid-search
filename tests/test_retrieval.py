from pathlib import Path

from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.embedding import MockEmbeddingService
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.deduplication import Deduplicator
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.retrieval import DenseRetriever, SparseRetriever, RetrievalResult

SAMPLE_MD = """# Introduction

The Widget API lets developers integrate widget rendering into their apps.
Authentication uses a bearer token issued by the admin dashboard.

# Pricing

The Widget API is billed at $0.05 per thousand widget renders. Volume discounts
apply above one million renders per month.

# Error Handling

Widget API returns a 429 status when the rate limit is exceeded. The error
body includes a retry-after header and a correlation id.

# Configuration

Set the WIDGET_API_KEY environment variable and point the client at the
production endpoint https://api.widgets.example."""


def build_pipeline(base: Path):
    config = ChunkingConfig(chunk_size=200, chunk_overlap=40, strategy="fixed")
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
        deduplicator=Deduplicator(0.95),
    )


def test_dense_retrieval_returns_top_k(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    retriever = DenseRetriever(
        embedding_service=pipe.embedding_service,
        vector_store=pipe.vector_store,
        top_k=10,
    )
    results = retriever.retrieve("How does authentication work in the Widget API?")

    assert 0 < len(results) <= 10
    # Results must be sorted by dense cosine score, descending
    scores = [r.dense_score for r in results]
    assert scores == sorted(scores, reverse=True), "must be ranked desc"
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.content for r in results)


def test_dense_retrieval_top_k_smaller_than_total(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    retriever = DenseRetriever(
        embedding_service=pipe.embedding_service,
        vector_store=pipe.vector_store,
    )
    total_chunks = pipe.vector_store.count()
    # request more than available -> returns all
    results = retriever.retrieve("widget", top_k=100)
    assert len(results) == total_chunks
    # request fewer -> honors k
    results2 = retriever.retrieve("widget", top_k=3)
    assert len(results2) == 3


def test_dense_result_metadata_present(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    retriever = DenseRetriever(
        embedding_service=pipe.embedding_service,
        vector_store=pipe.vector_store,
    )
    results = retriever.retrieve("rate limit 429", top_k=5)
    meta = results[0].metadata
    assert "doc_id" in meta
    assert "chunk_index" in meta
    assert "source_file" in meta
    assert "section_heading" in meta
    assert "chunking_strategy" in meta
    assert results[0].dense_score is not None


def test_dense_relevance_ranking_with_stub_store():
    # Deterministic wiring test: the DenseRetriever must surface results in the
    # exact order/score the vector store returns (descending cosine similarity),
    # independent of embedding quality.
    class StubVectorStore:
        def query(self, query_embedding, top_k=10, where=None):
            items = [
                {"chunk_id": "d::chunk::2", "score": 0.98,
                 "content": "Pricing is billed per render.", "metadata": {"doc_id": "d"}},
                {"chunk_id": "d::chunk::0", "score": 0.91,
                 "content": "Introduction text.", "metadata": {"doc_id": "d"}},
                {"chunk_id": "d::chunk::1", "score": 0.76,
                 "content": "Error handling.", "metadata": {"doc_id": "d"}},
            ]
            return items[:top_k]

    class StubEmb:
        is_configured = True

        def embed(self, text):
            return [0.5, 0.5]

    retriever = DenseRetriever(
        embedding_service=StubEmb(), vector_store=StubVectorStore()
    )
    results = retriever.retrieve("how much does pricing cost?", top_k=2)

    assert [r.chunk_id for r in results] == ["d::chunk::2", "d::chunk::0"]
    assert [r.dense_score for r in results] == [0.98, 0.91]
    assert results[0].content == "Pricing is billed per render."


def test_sparse_retrieval_returns_top_k_by_bm25(tmp_path):
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    retriever = SparseRetriever(bm25_index=pipe.bm25_index, top_k=10)
    results = retriever.retrieve("rate limit 429 retry after correlation id")

    assert 0 < len(results) <= 10
    # Ranked by BM25 score descending
    scores = [r.sparse_score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The error-handling chunk contains the exact keywords
    top = results[0]
    combined = f"{top.content} {top.metadata}".lower()
    assert "429" in combined or "retry" in combined or "correlation" in combined


def test_sparse_finds_exact_keyword_dense_might_miss(tmp_path):
    # Technical doc with a specific config key. Query the exact key: BM25
    # should surface the chunk that literally contains it.
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    retriever = SparseRetriever(bm25_index=pipe.bm25_index)
    results = retriever.retrieve("WIDGET_API_KEY")
    assert len(results) > 0
    top = results[0]
    combined = (top.content + " " + str(top.metadata)).lower()
    # The config chunk literally contains WIDGET_API_KEY; tokenizer splits
    # it into widget/api/key, so those tokens must appear in the top hit.
    assert "widget" in combined and "api" in combined and "key" in combined


def test_sparse_retrieval_operation_without_embedding(tmp_path):
    # Sparse retrieval is pure text: must work even with no embedding key.
    pipe = build_pipeline(tmp_path)
    p = tmp_path / "widget.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    pipe.ingest_file(str(p))

    # Confirm SparseRetriever needs only the BM25 index (no embeddings).
    retriever = SparseRetriever(bm25_index=pipe.bm25_index)
    res = retriever.retrieve("pricing volume discounts", top_k=3)
    assert len(res) > 0
    assert all(r.sparse_score is not None for r in res)