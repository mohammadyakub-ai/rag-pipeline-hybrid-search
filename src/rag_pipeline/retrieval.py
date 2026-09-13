from typing import List, Dict, Any, Optional

from .embedding import EmbeddingService, create_embedding_service
from .vector_store import VectorStore
from .bm25_index import BM25Index


class RetrievalResult:
    """A single retrieved chunk with normalized ranking info.

    Room is provided for source-specific scores (dense vs sparse), the fused
    RRF score, and a final rerank score, all populated by later pipeline
    stages.
    """

    def __init__(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
        dense_score: Optional[float] = None,
        sparse_score: Optional[float] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata or {}
        self.dense_score = dense_score
        self.sparse_score = sparse_score
        self.rrf_score: Optional[float] = None
        self.rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
        }

    def __repr__(self) -> str:
        return f"<RetrievalResult {self.chunk_id} dense={self.dense_score} sparse={self.sparse_score} rrf={self.rrf_score}>"


class DenseRetriever:
    """Retrieves the top-k chunks most similar (cosine) to a user question.

    Embeds the query with the configured embedding service, then queries the
    vector store. k=10 by default, per the project guide.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ):
        self.embedding_service = embedding_service or create_embedding_service()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.where = where

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Embed `query` and return top-k chunks by cosine similarity."""
        k = top_k or self.top_k
        if not self.embedding_service.is_configured:
            raise RuntimeError(
                "Embedding service not configured (set OPENAI_API_KEY or pass one)."
            )
        query_embedding = self.embedding_service.embed(query)
        raw = self.vector_store.query(
            query_embedding, top_k=k, where=where or self.where
        )

        results = []
        for item in raw:
            results.append(
                RetrievalResult(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    dense_score=item.get("score"),
                )
            )
        return results


class SparseRetriever:
    """Retrieves the top-k chunks by BM25 keyword scores over the corpus.

    Catches exact keyword matches (function names, config keys, error codes)
    that dense semantic search may miss. Skips document text normalization so
    tokens like error codes and env vars rank correctly.
    """

    def __init__(self, bm25_index: Optional[BM25Index] = None, top_k: int = 10):
        self.bm25_index = bm25_index or BM25Index()
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Run `query` through BM25 and return top-k chunks by BM25 score."""
        k = top_k or self.top_k
        raw = self.bm25_index.query(query, top_k=k)

        results = []
        for item in raw:
            results.append(
                RetrievalResult(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    sparse_score=item.get("score"),
                )
            )
        return results