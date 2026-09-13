import json
from pathlib import Path
from typing import List, Optional

from .document_loader import DocumentLoader, Document, DocumentChunk
from .chunker import ChunkingConfig, ChunkingService
from .embedding import EmbeddingService, create_embedding_service
from .vector_store import VectorStore
from .bm25_index import BM25Index
from .deduplication import Deduplicator


class IngestionPipeline:
    """Ties together loading -> chunking -> embedding -> dual indexing.

    Guarantees the dense (ChromaDB) and sparse (BM25) indexes stay in sync by
    using one shared chunk-id per chunk and indexing both in a single pass.
    """

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        document_loader: Optional[DocumentLoader] = None,
        deduplicator: Optional[Deduplicator] = None,
        enable_deduplication: bool = True,
    ):
        self.config = chunking_config or ChunkingConfig()
        self.embedding_service = embedding_service or create_embedding_service()
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.bm25_index.load()  # restore sparse index from disk, if any
        # Heal after a crash/restart: if the dense store has chunks but the
        # sparse index is empty, rebuild BM25 from the dense store so both
        # indexes stay in sync.
        if not self.bm25_index.count() and self.vector_store.count():
            self._rebuild_bm25_from_dense()
        self.document_loader = document_loader or DocumentLoader()
        self.deduplicator = deduplicator or Deduplicator()
        self.enable_deduplication = enable_deduplication
        self.chunking_service = ChunkingService(
            self.config, embedding_fn=self.embedding_service.embed
        )

    def set_strategy(self, strategy: str) -> None:
        """Switch the active chunking strategy for subsequent ingests."""
        self.config.strategy = strategy
        self.chunking_service = ChunkingService(
            self.config, embedding_fn=self.embedding_service.embed
        )

    def ingest_file(self, file_path: str) -> dict:
        """Load, chunk, embed, and index a single file into both stores."""
        document = self.document_loader.load_file(file_path)
        result = self._index_document(document)
        result["source_file"] = document.metadata.get("source_file", str(file_path))
        return result

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        """Index an uploaded file (filename + raw bytes) into both stores.

        Writes to a temporary file so the existing multi-format loader can
        infer the format from the extension, then loads and indexes it. The
        original upload filename is stored as the chunk's ``source_file`` so
        retrieval does not leak the temporary file path.
        """
        import tempfile

        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="upload_"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            document = self.document_loader.load_file(tmp_path)
            document.metadata["source_file"] = filename
            result = self._index_document(document)
            result["source_file"] = filename
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def ingest_directory(self, directory: str) -> List[dict]:
        documents = self.document_loader.load_directory(directory)
        results = []
        for doc in documents:
            results.append(self._index_document(doc))
        return results

    def index_document(self, document: Document) -> dict:
        """Index an already-loaded Document into both stores."""
        return self._index_document(document)

    def _index_document(self, document: Document) -> dict:
        chunks = self.chunking_service.chunk(document)
        if not chunks:
            return {"doc_id": document.doc_id, "indexed": 0}

        chunk_ids = [
            self.vector_store.chunk_id(document.doc_id, i)
            for i in range(len(chunks))
        ]
        contents = [c.content for c in chunks]
        embeddings = self.embedding_service.embed_batch(contents)

        metadatas = []
        for c in chunks:
            meta = dict(c.metadata)
            meta.update(
                {
                    "doc_id": document.doc_id,
                    "source_file": document.metadata.get("source_file"),
                    "chunking_strategy": meta.get("chunking_strategy")
                    or self.config.strategy,
                }
            )
            metadatas.append(meta)

        duplicates: dict = {}
        added_dense = 0
        added_bm25 = 0

        # Index both stores in lock-step with the same chunk ids. Deduplicate
        # incrementally so that within-batch duplicates are also caught:
        # each kept chunk is inserted immediately and becomes the reference
        # for subsequent candidates.
        for cid, emb, content, meta in zip(
            chunk_ids, embeddings, contents, metadatas
        ):
            if self.vector_store.has(cid):
                # Exact re-ingest of an already-indexed chunk.
                continue

            if self.enable_deduplication:
                matches = self.vector_store.query(emb, top_k=1)
                if matches:
                    top = matches[0]
                    if top.get("score", 0.0) > self.deduplicator.threshold:
                        info = {
                            "similarity": round(top.get("score", 0.0), 4),
                            "duplicate_of": top.get("chunk_id"),
                            "duplicate_source": (top.get("metadata") or {}).get(
                                "source_file"
                            ),
                        }
                        info["metadata"] = dict(meta)
                        info["metadata"]["is_duplicate"] = True
                        info["metadata"]["duplicate_of"] = top.get("chunk_id")
                        info["metadata"]["duplicate_similarity"] = info[
                            "similarity"
                        ]
                        duplicates[cid] = info
                        continue

            added_dense += self.vector_store.add([cid], [emb], [content], [meta])
            self.bm25_index.add([cid], [content], [meta])
            added_bm25 += 1

        if self.bm25_index.persist_dir is not None:
            self.bm25_index.save()

        return {
            "doc_id": document.doc_id,
            "chunks_total": len(chunks),
            "chunks_added_dense": added_dense,
            "chunks_bm25": added_bm25,
            "chunks_duplicated": len(duplicates),
            "duplicates": duplicates,
            "strategy": self.config.strategy,
        }

    def sync_status(self) -> dict:
        """Report whether both indexes contain the same number of chunks."""
        return {
            "dense_count": self.vector_store.count(),
            "bm25_count": self.bm25_index.count(),
            "in_sync": self.vector_store.count() == self.bm25_index.count(),
        }

    def delete_document(self, doc_id: str) -> int:
        n = self.vector_store.delete_document(doc_id)
        # BM25 uses append-only corpus; full rebuild removes deleted doc.
        self._rebuild_bm25_from_dense()
        return n

    def _rebuild_bm25_from_dense(self) -> None:
        all_data = self.vector_store._collection.get(
            include=["documents", "metadatas"]
        )
        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        fresh = BM25Index(persist_dir=self.bm25_index.persist_dir)
        fresh.add(ids, docs, metas)
        fresh.save()
        self.bm25_index.__dict__.update(fresh.__dict__)