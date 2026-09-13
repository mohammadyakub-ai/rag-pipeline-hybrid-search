import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb


class VectorStore:
    """Dense vector store over the chunk corpus backed by ChromaDB.

    Every chunk is stored with the required metadata (source document, chunk
    index, section heading, chunking strategy, character count) plus the raw
    text so it can be retrieved directly for generation.
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "chunks",
        host: Optional[str] = None,
        port: int = 8000,
    ):
        """A chroma client that is either embedded (PersistentClient) or remote.

        Default: a local persistent client under `persist_dir` (self-contained).
        If `host` is given, connect to a standalone ChromaDB server over HTTP
        (docker-compose runs one), keeping `persist_dir` unused.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.host = host
        self.port = port
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            self._client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def chunk_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}::chunk::{chunk_index}"

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only accepts scalar metadata values (str/int/float/bool).

        Coerce `None` to "" and drop any non-scalar values (lists, dicts,
        numpy arrays) that would otherwise crash the add() call.
        """
        out = {}
        for k, v in (metadata or {}).items():
            if isinstance(v, (str, int, float, bool)) and not isinstance(v, bool):
                out[str(k)] = v
            elif isinstance(v, bool):
                out[str(k)] = v
            elif v is None:
                out[str(k)] = ""
            else:
                continue  # drop lists/dicts/ndarrays
        return out

    def add(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Insert chunks. Skips ids that already exist (idempotent)."""
        if not (len(chunk_ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("chunk_ids, embeddings, documents, metadatas must match")

        to_add = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
        }
        existing = set(self._collection.get(ids=chunk_ids)["ids"])
        for cid, emb, doc, meta in zip(chunk_ids, embeddings, documents, metadatas):
            if cid in existing:
                continue
            to_add["ids"].append(cid)
            to_add["embeddings"].append(emb)
            to_add["documents"].append(doc)
            to_add["metadatas"].append(self._sanitize_metadata(meta))

        if to_add["ids"]:
            self._collection.add(
                ids=to_add["ids"],
                embeddings=to_add["embeddings"],
                documents=to_add["documents"],
                metadatas=to_add["metadatas"],
            )
        return len(to_add["ids"])

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks ranked by cosine similarity to the query."""
        kwargs = dict(query_embeddings=[query_embedding], n_results=top_k)
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)

        out = []
        ids = result["ids"][0] if result["ids"] else []
        distances = result["distances"][0] if result["distances"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []

        for i, cid in enumerate(ids):
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            out.append(
                {
                    "chunk_id": cid,
                    "score": round(score, 4),
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregate stored chunks into a per-document summary.

        Returns one entry per unique doc_id with chunk count and source
        metadata so the API can list all indexed documents.
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception:
            result = {}
        metadatas = result.get("metadatas") or []
        by_doc: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = str(meta.get("doc_id") or "unknown")
            entry = by_doc.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source_file": meta.get("source_file"),
                    "chunk_count": 0,
                    "strategies": set(),
                },
            )
            entry["chunk_count"] += 1
            strat = meta.get("chunking_strategy")
            if strat:
                entry["strategies"].add(str(strat))
        docs = []
        for doc_id, entry in by_doc.items():
            entry["strategies"] = sorted(entry["strategies"])
            docs.append(entry)
        docs.sort(key=lambda d: d["doc_id"])
        return docs

    def has(self, chunk_id: str) -> bool:
        return len(self._collection.get(ids=[chunk_id])["ids"]) > 0

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a source document."""
        ids = self._collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )