from rag_pipeline.retrieval import RetrievalResult, DenseRetriever, SparseRetriever
from rag_pipeline.fusion import ReciprocalRankFusion, HybridRetriever
from rag_pipeline.reranker import LexicalReranker


def result(chunk_id, content="", meta=None, dense=None, sparse=None):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata=meta or {"doc_id": "x"},
        dense_score=dense,
        sparse_score=sparse,
    )


def test_rrf_fuses_both_lists_and_boosts_shared_docs():
    fusion = ReciprocalRankFusion(k=60)

    dense = [
        result("A", dense=0.95),
        result("B", dense=0.90),
        result("C", dense=0.80),
    ]
    sparse = [
        result("B", sparse=7.0),
        result("D", sparse=6.0),
        result("E", sparse=5.0),
    ]

    fused = fusion.fuse(dense, sparse, weight_dense=0.7, weight_sparse=0.3)

    # "B" appears in both lists -> should rank #1 (sum of both rank contributions)
    assert fused[0].chunk_id == "B"
    # D and E appear only in sparse; A and C only in dense
    assert {r.chunk_id for r in fused} == {"A", "B", "C", "D", "E"}
    # Every result has an rrf_score
    assert all(r.rrf_score is not None for r in fused)
    # Sorted descending by rrf_score
    scores = [r.rrf_score for r in fused]
    assert scores == sorted(scores, reverse=True)
    # B's rrf = dense(rank2): 0.7/62 + sparse(rank1): 0.3/61
    assert abs(fused[0].rrf_score - (0.7 / 62.0 + 0.3 / 61.0)) < 1e-6


def test_rrf_weights_configurable():
    fusion = ReciprocalRankFusion(k=60)

    dense = [result("A", dense=0.9), result("B", dense=0.8)]
    sparse = [result("B", sparse=9.0), result("A", sparse=8.0)]

    # With pure dense weight (1.0/0.0): A ranks above B (dense order)
    fused_dense = fusion.fuse(dense, sparse, weight_dense=1.0, weight_sparse=0.0)
    assert fused_dense[0].chunk_id == "A"

    # With pure sparse weight (0.0/1.0): B ranks above A (sparse order)
    fused_sparse = fusion.fuse(dense, sparse, weight_dense=0.0, weight_sparse=1.0)
    assert fused_sparse[0].chunk_id == "B"

    # Different weightings produce different scores for the same doc
    score_b_flat = next(r for r in fused_dense if r.chunk_id == "B").rrf_score
    score_b_sparse = next(r for r in fused_sparse if r.chunk_id == "B").rrf_score
    assert score_b_flat != score_b_sparse


def test_rrf_rank_based_robust_to_score_scale():
    # Because RRF uses rank (not raw scores), wildly different absolute score
    # ranges across lists don't distort the fusion.
    fusion = ReciprocalRankFusion(k=60)
    dense = [result("A", dense=0.999), result("B", dense=0.001)]
    sparse = [result("C", sparse=100.0), result("B", sparse=99.0)]
    fused = fusion.fuse(dense, sparse, weight_dense=0.5, weight_sparse=0.5)
    # B (rank 2 in both) should beat A (rank 1 only in dense) and
    # C (rank 1 only in sparse): B = 0.5/61 + 0.5/62, A = 0.5/61, C = 0.5/61
    assert fused[0].chunk_id == "B"


def test_hybrid_retriever_end_to_end():
    class StubDense(DenseRetriever):
        def retrieve(self, query, top_k=None):
            k = top_k or 20
            return [result("D1", content="dense alpha", dense=0.9),
                    result("D2", content="dense beta", dense=0.7)][:k]

    class StubSparse(SparseRetriever):
        def retrieve(self, query, top_k=None):
            k = top_k or 20
            return [result("S1", content="sparse gamma", sparse=9.0),
                    result("D1", content="dense alpha", sparse=8.0)][:k]

    hybrid = HybridRetriever(
        dense_retriever=StubDense(),
        sparse_retriever=StubSparse(),
        reranker=LexicalReranker(),
        dense_k=20,
        sparse_k=20,
        weight_dense=0.6,
        weight_sparse=0.4,
    )
    results = hybrid.retrieve("alpha beta gamma", top_k=3)
    # D1 is in both lists -> #1
    assert results[0].chunk_id == "D1"
    assert results[0].rrf_score is not None
    assert len(results) == 3


def test_hybrid_retriever_separate_methods():
    class StubDense(DenseRetriever):
        def retrieve(self, query, top_k=None):
            return [result("D1", dense=0.9)]

    class StubSparse(SparseRetriever):
        def retrieve(self, query, top_k=None):
            return [result("S1", sparse=9.0)]

    hybrid = HybridRetriever(dense_retriever=StubDense(), sparse_retriever=StubSparse())
    dense_only = hybrid.retrieve_dense("q")
    sparse_only = hybrid.retrieve_sparse("q")
    assert [r.chunk_id for r in dense_only] == ["D1"]
    assert [r.chunk_id for r in sparse_only] == ["S1"]