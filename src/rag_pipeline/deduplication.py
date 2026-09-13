from typing import List, Dict, Any, Optional, Tuple


class Deduplicator:
    """Near-duplicate detection at ingest time.

    Before a chunk is inserted, its embedding is compared against the chunks
    already stored in the vector store. If cosine similarity against the most
    similar existing chunk exceeds `threshold` (default 0.95), the incoming
    chunk is treated as a duplicate: it is flagged with the id of the "twin"
    chunk it duplicates and skipped, so redundant content does not consume
    context-window slots in the retriever.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def find_duplicates(
        self,
        vector_store,
        chunk_ids: List[str],
        embeddings: List[List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Return {chunk_id: info} for chunks that duplicate existing content.

        `vector_store` must expose a `query(query_embedding, top_k=1)` method
        returning a list of dicts with at least a `score` and `chunk_id` field.

        Only chunks whose identical id is NOT already present are candidates;
        re-indexing the same document should not flag its own chunks.
        """
        duplicates: Dict[str, Dict[str, Any]] = {}
        for cid, emb in zip(chunk_ids, embeddings):
            # Idempotency: if this exact chunk is already indexed, it's a
            # clean re-ingest, not a cross-document duplicate.
            if vector_store.has(cid):
                continue

            matches = vector_store.query(emb, top_k=1)
            if not matches:
                continue

            top = matches[0]
            score = top.get("score", 0.0)
            if score > self.threshold:
                duplicates[cid] = {
                    "similarity": round(score, 4),
                    "duplicate_of": top.get("chunk_id"),
                    "duplicate_source": (top.get("metadata") or {}).get(
                        "source_file"
                    ),
                }
        return duplicates

    def filter_duplicates(
        self,
        vector_store,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[List[float]], List[str], List[Dict[str, Any]], Dict]:
        """Split inputs into (keep lists, duplicate_info dict).

        Kept items are those that pass dedup; duplicates are summarized and
        their metadata flagged with `is_duplicate=True` + twin info.
        """
        dup_info = self.find_duplicates(vector_store, chunk_ids, embeddings)

        keep_ids, keep_embs, keep_docs, keep_meta = [], [], [], []
        for cid, emb, doc, meta in zip(chunk_ids, embeddings, contents, metadatas):
            if cid in dup_info:
                flagged = dict(meta)
                flagged["is_duplicate"] = True
                flagged["duplicate_of"] = dup_info[cid]["duplicate_of"]
                flagged["duplicate_similarity"] = dup_info[cid]["similarity"]
                dup_info[cid]["metadata"] = flagged
            else:
                keep_ids.append(cid)
                keep_embs.append(emb)
                keep_docs.append(doc)
                keep_meta.append(meta)

        return keep_ids, keep_embs, keep_docs, keep_meta, dup_info