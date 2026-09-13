"""FastAPI service exposing the RAG pipeline over HTTP.

Endpoints:
    POST /v1/ask       - answer a question with citations + confidence
    POST /v1/retrieve  - ranked chunks for a retrieval mode (no LLM needed)
    POST /v1/ingest    - ingest a new document (file upload)
    GET  /v1/documents - list indexed documents

OpenAPI docs are auto-generated at /docs and /openapi.json.

The service is fully runnable offline: if no chat API key (OPENROUTER_API_KEY
or OPENAI_API_KEY) is present it uses MockEmbeddingService and a pluggable
generator (see create_app).
"""

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root silently

import io
import os
import uuid
from dataclasses import asdict
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .ingestion import IngestionPipeline
from .document_loader import DocumentLoader
from .chunker import ChunkingConfig
from .embedding import MockEmbeddingService, create_embedding_service
from .vector_store import VectorStore
from .bm25_index import BM25Index
from .retrieval import DenseRetriever, SparseRetriever, RetrievalResult
from .fusion import HybridRetriever
from .reranker import LexicalReranker, create_reranker
from .generation import Generator
from .citation import CitationVerifier
from .confidence import ConfidenceScorer
from .unknown import UnknownHandler, AnswerOutcome

DEFAULT_PERSIST_DIR = os.environ.get("RAG_PERSIST_DIR", "data/chroma")
DEFAULT_BM25_DIR = os.environ.get("RAG_BM25_DIR", "data/bm25")
DEFAULT_THRESHOLD = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.5"))


def _chroma_endpoint_from_env() -> Optional[tuple]:
    """Return (host, port) for a standalone ChromaDB server, if configured.

    Reads CHROMA_HOST/CHROMA_PORT. When unset the API uses its embedded
    persistent store (data/chroma); when set it talks to the composed
    ChromaDB service over HTTP.
    """
    host = os.environ.get("CHROMA_HOST")
    if not host:
        return None
    try:
        port = int(os.environ.get("CHROMA_PORT", "8000"))
    except ValueError:
        port = 8000
    return host, port


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: str = "hybrid"  # hybrid | dense | sparse
    threshold: Optional[float] = None  # override service default per request


class AskResponse(BaseModel):
    can_answer: bool
    question: str
    answer: Optional[str] = None
    confidence: Optional[dict] = None
    citations: Optional[dict] = None
    sources: Optional[list] = None
    declined: Optional[dict] = None


class RetrieveRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # hybrid | dense | sparse
    top_k: int = 10


class RetrieveResponse(BaseModel):
    question: str
    mode: str
    ranking: list = []


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added_dense: int
    chunks_added_bm25: int
    duplicates: int
    source_file: str


class DocumentInfo(BaseModel):
    doc_id: str
    source_file: Optional[str] = None
    chunk_count: int
    strategies: List[str] = []


class AskService:
    """Wires the full retrieval -> generation -> verification -> gate flow."""

    RETRIEVAL_MODES = ("hybrid", "dense", "sparse")

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        generator: Generator,
        citation_verifier: Optional[CitationVerifier] = None,
        scorer: Optional[ConfidenceScorer] = None,
        handler: Optional[UnknownHandler] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.citation_verifier = citation_verifier
        self.scorer = scorer or ConfidenceScorer()
        self.handler = handler or UnknownHandler(self.scorer)

    def retrieve(self, question: str, top_k: int = 10, retrieval_mode: str = "hybrid") -> List[RetrievalResult]:
        """Retrieve + rerank for a retrieval mode (hybrid/dense/sparse).

        Hybrid goes through the fused + reranked path. Dense/sparse rerank the
        single-list candidates with the same reranker so rankings are
        comparable side by side.
        """
        retriever = self.hybrid_retriever
        mode = retrieval_mode if retrieval_mode in self.RETRIEVAL_MODES else "hybrid"

        if mode == "dense":
            results = retriever.retrieve_dense(question, top_k=max(top_k, retriever.dense_k))
        elif mode == "sparse":
            results = retriever.retrieve_sparse(question, top_k=max(top_k, retriever.sparse_k))
        else:
            return retriever.retrieve(question, top_k=top_k)[:top_k]

        if retriever.reranker and results:
            results = retriever.reranker.score(question, results)
        return results[:top_k]

    def answer(self, question: str, top_k: int = 10, retrieval_mode: str = "hybrid") -> dict:
        results = self.retrieve(question, top_k=top_k, retrieval_mode=retrieval_mode)

        gen_out = self.generator.generate(question, results)
        answer_text = gen_out.get("answer", "")

        citation_report = None
        if self.citation_verifier and answer_text.strip():
            citation_report = self.citation_verifier.verify(answer_text, results)

        outcome = self.handler.decide(
            question, answer_text, results, citation_report=citation_report
        )

        return self._serialize(outcome, question, results)

    def _serialize(self, outcome: AnswerOutcome, question: str, results: List[RetrievalResult]) -> dict:
        sources = [self._source(r) for r in results]

        if outcome.can_answer:
            return {
                "can_answer": True,
                "question": question,
                "answer": outcome.answer,
                "confidence": outcome.confidence.to_dict() if outcome.confidence else None,
                "citations": outcome.citation_report.to_dict()
                if outcome.citation_report
                else None,
                "sources": sources,
                "declined": None,
            }
        return {
            "can_answer": False,
            "question": question,
            "answer": None,
            "confidence": outcome.confidence.to_dict() if outcome.confidence else None,
            "citations": None,
            "sources": sources,
            "declined": outcome.low_confidence.to_dict()
            if outcome.low_confidence
            else None,
        }

    def _source(self, r: RetrievalResult) -> dict:
        meta = r.metadata or {}
        return {
            "chunk_id": r.chunk_id,
            "source_file": meta.get("source_file"),
            "section": meta.get("section_heading"),
            "chunking_strategy": meta.get("chunking_strategy"),
            "rerank_score": r.rerank_score,
            "rrf_score": r.rrf_score,
            "dense_score": r.dense_score,
            "sparse_score": r.sparse_score,
            "content": r.content[:500],
        }


def create_app(
    pipeline: Optional[IngestionPipeline] = None,
    retriever: Optional[HybridRetriever] = None,
    generator: Optional[Generator] = None,
    citation_verifier: Optional[CitationVerifier] = None,
    handler: Optional[UnknownHandler] = None,
    threshold: float = DEFAULT_THRESHOLD,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    bm25_dir: str = DEFAULT_BM25_DIR,
    chroma_host: Optional[str] = None,
    chroma_port: int = 8000,
) -> FastAPI:
    """Construct the FastAPI app.

    All components can be injected for tests. If none are given, a self-
    contained pipeline is built against persistent stores under data/ (or
    against a standalone ChromaDB server when CHROMA_HOST is set).
    """
    if pipeline is None:
        if chroma_host is None:
            chroma_host, chroma_port = _chroma_endpoint_from_env() or (None, 8000)
        vector_store = (
            VectorStore(host=chroma_host, port=chroma_port)
            if chroma_host
            else VectorStore(persist_dir=persist_dir)
        )
        pipeline = IngestionPipeline(
            chunking_config=ChunkingConfig(),
            embedding_service=create_embedding_service(),
            vector_store=vector_store,
            bm25_index=BM25Index(persist_dir=bm25_dir),
            document_loader=DocumentLoader(
                raw_dir=str(Path(persist_dir).parent / "raw"),
                processed_dir=str(Path(persist_dir).parent / "processed"),
            ),
        )

    if retriever is None:
        retriever = HybridRetriever(
            dense_retriever=DenseRetriever(
                embedding_service=pipeline.embedding_service,
                vector_store=pipeline.vector_store,
            ),
            sparse_retriever=SparseRetriever(bm25_index=pipeline.bm25_index),
            reranker=create_reranker(),
        )

    if generator is None:
        generator = Generator()

    if citation_verifier is None:
        citation_verifier = CitationVerifier(judge=generator)

    scorer = ConfidenceScorer(judge=generator)
    if handler is None:
        handler = UnknownHandler(scorer, threshold=threshold)

    service = AskService(
        hybrid_retriever=retriever,
        generator=generator,
        citation_verifier=citation_verifier,
        scorer=scorer,
        handler=handler,
    )

    app = FastAPI(
        title="RAG Pipeline Hybrid Search API",
        description=(
            "Production-grade RAG over internal docs with hybrid retrieval "
            "(dense + sparse + RRF + rerank), grounded generation, citation "
            "verification, confidence scoring, and graceful 'I don't know' "
            "handling."
        ),
        version="1.0.0",
    )

    app.state.pipeline = pipeline
    app.state.retriever = retriever
    app.state.service = service

    @app.get("/healthz", tags=["system"])
    def healthz():
        return {
            "status": "ok",
            "vector_chunks": pipeline.vector_store.count(),
            "bm25_chunks": pipeline.bm25_index.count(),
        }

    @app.get("/v1/documents", response_model=List[DocumentInfo], tags=["ingestion"])
    def list_documents():
        docs = pipeline.vector_store.list_documents()
        return [
            DocumentInfo(
                doc_id=d["doc_id"],
                source_file=d.get("source_file"),
                chunk_count=d["chunk_count"],
                strategies=d.get("strategies") or [],
            )
            for d in docs
        ]

    @app.post("/v1/reset", tags=["ingestion"], status_code=200)
    def reset_index():
        """Drop all indexed chunks (dense + sparse) and cached uploads."""
        pipeline.vector_store.reset()
        pipeline.bm25_index.reset()
        for d in (pipeline.document_loader.raw_dir, pipeline.document_loader.processed_dir):
            if d.is_dir():
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()
        return {
            "status": "reset",
            "vector_chunks": pipeline.vector_store.count(),
            "bm25_chunks": pipeline.bm25_index.count(),
        }

    @app.post(
        "/v1/ingest",
        response_model=IngestResponse,
        tags=["ingestion"],
        status_code=201,
    )
    async def ingest_document(file: UploadFile = File(...)):
        if not file.filename:
            raise HTTPException(status_code=400, detail="filename required")
        try:
            content = await file.read()
            result = pipeline.ingest_upload(file.filename, content)
        except ValueError as exc:  # unsupported extension
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}")
        return IngestResponse(
            doc_id=result.get("doc_id", ""),
            chunks_added_dense=result.get("chunks_added_dense", 0),
            chunks_added_bm25=result.get("chunks_bm25", 0),
            duplicates=result.get("chunks_duplicated", 0),
            source_file=result.get("source_file", file.filename),
        )

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        tags=["query"],
    )
    async def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        try:
            data = service.answer(
                req.question, top_k=req.top_k, retrieval_mode=req.retrieval_mode
            )
        except RuntimeError as exc:  # e.g. Generator not configured
            raise HTTPException(status_code=503, detail=str(exc))
        return AskResponse(**data)

    @app.post(
        "/v1/retrieve",
        response_model=RetrieveResponse,
        tags=["query"],
    )
    async def retrieve(req: RetrieveRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        results = service.retrieve(req.question, top_k=req.top_k, retrieval_mode=req.mode)
        return RetrieveResponse(
            question=req.question,
            mode=req.mode,
            ranking=[service._source(r) for r in results],
        )

    return app


def run():
    """Entry point: uvicorn rag_pipeline.api:app"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)