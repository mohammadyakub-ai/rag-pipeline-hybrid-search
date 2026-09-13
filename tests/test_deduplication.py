from pathlib import Path

from rag_pipeline.document_loader import DocumentLoader, Document
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.embedding import MockEmbeddingService
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.deduplication import Deduplicator
from rag_pipeline.ingestion import IngestionPipeline


DOC_A = """# Overview

The pricing model charges per seat per month. Teams can choose annual billing
for a discount. Contact sales for enterprise pricing.

## API

The API requires an auth token in the Authorization header. Rate limits apply
per account tier.

# Conclusion

This document describes the standard pricing and API setup."""
# Doc B has identical content to Doc A (a true duplicate document).
DOC_B = DOC_A
# Doc C repeats the same information but with light rewriting (near-duplicate).
DOC_C = """# Overview

Our pricing is billed per seat each month. Annual billing is available for a
discount, and enterprise deals are handled by the sales team.

## API

Every API request needs an auth token in the Authorization header. Rate limits
depend on the account tier."""
# Doc D is entirely distinct (unique content).
DOC_D = """# Engineering Notes

The build system compiles with CMake. The test harness uses pytest and runs the
suite in parallel. Release artifacts are signed and pushed to the registry."""


def build_pipeline(base: Path, dedup_threshold: float = 0.95, enable_dedup=True):
    config = ChunkingConfig(chunk_size=300, chunk_overlap=50, strategy="fixed")
    embedding = MockEmbeddingService(dimensions=128)
    vs = VectorStore(persist_dir=str(base / "chroma"))
    bm25 = BM25Index(persist_dir=str(base / "bm25"))
    loader = DocumentLoader(
        raw_dir=str(base / "raw"),
        processed_dir=str(base / "processed"),
    )
    dedup = Deduplicator(threshold=dedup_threshold)
    return IngestionPipeline(
        chunking_config=config,
        embedding_service=embedding,
        vector_store=vs,
        bm25_index=bm25,
        document_loader=loader,
        deduplicator=dedup,
        enable_deduplication=enable_dedup,
    )


def write_doc(base: Path, name: str, content: str) -> str:
    p = base / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_exact_duplicate_document_skipped(tmp_path):
    pipe = build_pipeline(tmp_path)
    r1 = pipe.ingest_file(write_doc(tmp_path, "a.md", DOC_A))
    # Ingest an identical document (same content, different file)
    r2 = pipe.ingest_file(write_doc(tmp_path, "b.md", DOC_B))

    assert r2["chunks_total"] == r1["chunks_total"]
    assert r2["chunks_added_dense"] == 0, "identical doc should add nothing"
    assert r2["chunks_duplicated"] == r1["chunks_total"]
    # Indexes stay in sync (dense did not grow)
    assert pipe.sync_status()["in_sync"]
    assert pipe.vector_store.count() == r1["chunks_total"]


def test_near_duplicate_flagged():
    # Unit-test the dedup *logic* with a controllable stub vector store so we
    # can assert near-duplicate (similarity > 0.95) detection is independent
    # of embedding quality.
    class StubVectorStore:
        def __init__(self, sim_by_index):
            self.sim_by_index = sim_by_index  # [{chunk_id: score}]
            self.calls = 0

        def has(self, cid):
            return False

        def query(self, embedding, top_k=1):
            i = self.calls
            self.calls += 1
            sims = self.sim_by_index[i]
            best_id = max(sims, key=sims.get)
            return [
                {
                    "chunk_id": best_id,
                    "score": sims[best_id],
                    "metadata": {"source_file": "existing.md", "doc_id": "existing"},
                }
            ]

    # chunk0 should be a duplicate (0.97 > 0.95), chunk1 unique (0.80)
    stub = StubVectorStore(
        [{("new", 0): 0.97}, {("new", 1): 0.80}]
    )
    dps = Deduplicator(threshold=0.95)
    results = dps.find_duplicates(
        stub,
        ["new::chunk::0", "new::chunk::1"],
        [[0.1] * 8, [0.2] * 8],
    )
    assert "new::chunk::0" in results
    assert "new::chunk::1" not in results
    assert results["new::chunk::0"]["similarity"] == 0.97
    assert results["new::chunk::0"]["duplicate_of"] == ("new", 0) or results[
        "new::chunk::0"
    ]["duplicate_source"] == "existing.md"


def test_filter_duplicates_flagging():
    class StubVS:
        def __init__(self):
            self.d = {}

        def has(self, cid):
            return False

        def query(self, emb, top_k=1):
            return [{"chunk_id": "old::chunk::0", "score": 0.98,
                     "metadata": {"source_file": "old.md"}}]

    dps = Deduplicator(threshold=0.95)
    keep_ids, keep_embs, keep_docs, keep_meta, dup_info = dps.filter_duplicates(
        StubVS(),
        ["new::chunk::0"],
        [[1.0] * 4],
        ["duplicate text"],
        [{"doc_id": "new", "chunk_index": 0}],
    )
    assert keep_ids == []
    assert "new::chunk::0" in dup_info
    flagged_meta = dup_info["new::chunk::0"]["metadata"]
    assert flagged_meta["is_duplicate"] is True
    assert flagged_meta["duplicate_of"] == "old::chunk::0"
    assert flagged_meta["duplicate_similarity"] == 0.98


def test_unique_document_not_flagged(tmp_path):
    pipe = build_pipeline(tmp_path)
    pipe.ingest_file(write_doc(tmp_path, "a.md", DOC_A))
    r2 = pipe.ingest_file(write_doc(tmp_path, "d.md", DOC_D))
    assert r2["chunks_duplicated"] == 0
    assert r2["chunks_added_dense"] == r2["chunks_total"]


def test_dedup_flag_in_metadata(tmp_path):
    pipe = build_pipeline(tmp_path)
    pipe.ingest_file(write_doc(tmp_path, "a.md", DOC_A))
    pipe.ingest_file(write_doc(tmp_path, "b.md", DOC_B))

    col = pipe.vector_store._collection
    # Confirm no chunk carries the duplicate flag inside the store
    data = col.get(include=["metadatas"])
    flags = [m.get("is_duplicate") for m in data["metadatas"]]
    assert not any(flags), "duplicates must not be inserted"


def test_disable_dedup(tmp_path):
    pipe = build_pipeline(tmp_path, enable_dedup=False)
    r1 = pipe.ingest_file(write_doc(tmp_path, "a.md", DOC_A))
    r2 = pipe.ingest_file(write_doc(tmp_path, "b.md", DOC_B))
    assert r2["chunks_duplicated"] == 0
    assert r2["chunks_added_dense"] == r1["chunks_total"]