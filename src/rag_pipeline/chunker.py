import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .document_loader import Document, DocumentChunk


def extract_section_heading(doc_text: str, chunk_start: int) -> str:
    """Find the nearest markdown heading at or before ``chunk_start``.

    Scans the document text *before* the chunk (not the chunk content
    itself) so the returned heading is a real section heading rather than a
    fragment of the chunk body. Falls back to "Introduction" when no heading
    precedes the chunk.
    """
    lines = doc_text[:chunk_start].split("\n")
    for line in reversed(lines):
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return "Introduction"


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    strategy: str = "fixed"  # one of: fixed, recursive, semantic
    embedding_fn: Optional[Callable[[str], List[float]]] = None
    similarity_threshold: float = 0.8
    min_chunk_chars: int = 50


class BaseChunker(ABC):
    strategy_name: str = "base"

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        pass

    def _flag_strategy(self, chunk: DocumentChunk) -> DocumentChunk:
        chunk.metadata["chunking_strategy"] = self.strategy_name
        return chunk


class FixedSizeChunker(BaseChunker):
    """Baseline: split text into fixed-size chunks with configurable overlap."""

    strategy_name = "fixed"

    def chunk(self, document: Document) -> List[DocumentChunk]:
        raw_text = document.clean_text or document.raw_content
        if not raw_text:
            raw_text = "\n".join(c.content for c in document.chunks)

        text = self._normalize(raw_text)
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            if end < len(text):
                end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_chars:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text),
                        "char_offset": start,
                        "section_heading": extract_section_heading(text, start),
                    }
                )
                chunks.append(self._flag_strategy(chunk))
                chunk_index += 1

            if end >= len(text):
                break
            start = end - self.config.chunk_overlap

        return chunks

    def _normalize(self, text: str) -> str:
        # Collapse horizontal whitespace only; keep newlines so markdown
        # headings can still be located by `extract_section_heading`.
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        window = text[start:end]
        for sep in ['\n\n', '\n', '. ', '! ', '? ']:
            pos = window.rfind(sep)
            if pos != -1 and pos > self.config.chunk_size * 0.5:
                return start + pos + len(sep)
        # Prefer a whitespace boundary to avoid splitting a keyword/token in two.
        ws = text.rfind(' ', start, end)
        if ws != -1 and ws - start > self.config.chunk_size * 0.5:
            return ws + 1
        return end


class RecursiveChunker(BaseChunker):
    """Structure-aware: split recursively, respecting section headers."""

    strategy_name = "recursive"

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.separators = ['\n\n', '\n', '. ', ' ', '']

    def chunk(self, document: Document) -> List[DocumentChunk]:
        raw_text = document.clean_text or document.raw_content
        if not raw_text:
            raw_text = "\n".join(c.content for c in document.chunks)

        return self._split_text(raw_text, document.metadata)

    def _split_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        # Identify section boundaries
        sections = self._detect_sections(text)

        chunks = []
        chunk_index = 0
        for heading, content, content_start in sections:
            sub_chunks = self._recursive_split(content, self.config.chunk_size)
            for sc in sub_chunks:
                if len(sc) < self.config.min_chunk_chars:
                    continue
                chunk = DocumentChunk(
                    content=sc,
                    metadata={
                        "chunk_index": chunk_index,
                        "section_heading": (
                            heading or extract_section_heading(text, content_start)
                        ),
                        "char_count": len(sc),
                    }
                )
                chunks.append(self._flag_strategy(chunk))
                chunk_index += 1

        return chunks

    def _detect_sections(self, text: str) -> List[tuple]:
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        sections = []
        current_heading = None
        current_content = []
        current_start = 0

        def flush() -> None:
            content = '\n'.join(current_content)
            if content.strip():
                sections.append((current_heading, content, current_start))

        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line)
            if match:
                flush()
                current_heading = match.group(2).strip()
                current_start = sum(len(l) + 1 for l in lines[: i + 1])
                current_content = [line]
            else:
                current_content.append(line)

        flush()

        if not sections:
            sections = [(None, text, 0)]

        return sections

    def _recursive_split(self, text: str, chunk_size: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        for sep in self.separators:
            if sep == '':
                break
            split_points = [m.start() for m in re.finditer(re.escape(sep), text)]
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end >= len(text):
                    chunks.append(text[start:])
                    break
                # find split point closest to end
                best = None
                for pos in split_points:
                    if start < pos <= end:
                        best = pos
                    elif pos > end:
                        break
                if best is None or best - start < chunk_size * 0.5:
                    best = end
                chunks.append(text[start:best])
                start = best + len(sep) - self.config.chunk_overlap // 2
                start = max(start, best - self.config.chunk_overlap)

            result = []
            for c in chunks:
                result.extend(self._recursive_split(c, chunk_size))
            return [c for c in result if c.strip()]

        return [text[0:chunk_size]] if text[0:chunk_size].strip() else []


class SemanticChunker(BaseChunker):
    """Split on topic boundaries using embedding similarity."""

    strategy_name = "semantic"

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        if config.embedding_fn is None:
            raise ValueError("SemanticChunker requires an embedding function")

    def chunk(self, document: Document) -> List[DocumentChunk]:
        raw_text = document.clean_text or document.raw_content
        if not raw_text:
            raw_text = "\n".join(c.content for c in document.chunks)

        sentences = self._get_sentences(raw_text)  # list of (text, start_offset)
        if not sentences:
            return []

        texts = [s for s, _ in sentences]
        # Generate embeddings for each sentence
        sentence_embeddings = [self.config.embedding_fn(s) for s in texts]

        # Group sentences by topic boundaries
        groups = self._cluster_sentences(texts, sentence_embeddings)

        chunks = []
        for i, group_indices in enumerate(groups):
            content = ' '.join(texts[j] for j in group_indices).strip()
            if len(content) < self.config.min_chunk_chars:
                continue
            start = sentences[group_indices[0]][1]
            chunk = DocumentChunk(
                content=content,
                metadata={
                    "chunk_index": i,
                    "section_heading": extract_section_heading(raw_text, start),
                    "char_count": len(content),
                    "num_sentences": len(group_indices),
                }
            )
            chunks.append(self._flag_strategy(chunk))

        return chunks

    def _get_sentences(self, text: str) -> List[tuple]:
        # Split on sentence boundaries while recording each sentence's start
        # offset in the original raw text, so `extract_section_heading` can
        # look *before* the chunk for the enclosing markdown heading.
        sentences = []
        for m in re.finditer(r'[^.!?]+(?:[.!?]+(?=\s|$))?', text):
            s = m.group(0).strip()
            if s:
                sentences.append((s, m.start()))
        return sentences

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _cluster_sentences(self, sentences: List[str], embeddings: List[List[float]]) -> List[List[int]]:
        if len(sentences) <= 1:
            return [list(range(len(sentences)))]

        # Track group membership by sentence index to avoid fragile string
        # equality matching.
        group_indices = [[0]]  # each inner list holds indices into `sentences`
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(current_embedding, embeddings[i])
            if sim < self.config.similarity_threshold and self._group_has_substance(
                [sentences[j] for j in group_indices[-1]]
            ):
                group_indices.append([i])
                current_embedding = embeddings[i]
            else:
                group_indices[-1].append(i)
                current_embedding = self._group_average_embedding(
                    embeddings, group_indices[-1]
                )

        # Only merge groups that are too small to stand alone as a chunk; keep
        # real topic boundaries intact.
        return self._reconstruct_groups(sentences, group_indices)

    def _reconstruct_groups(self, sentences: List[str], group_indices: List[List[int]]) -> List[List[int]]:
        # Determine which groups survive based on substance, then join dribble
        # groups onto their predecessor.
        surviving = []
        for idx_group in group_indices:
            gs = [sentences[j] for j in idx_group]
            if self._group_has_substance(gs):
                surviving.append(idx_group)
            elif surviving:
                surviving[-1].extend(idx_group)
            else:
                surviving.append(idx_group)
        return surviving

    def _group_has_substance(self, group_sentences: List[str]) -> bool:
        combined = " ".join(group_sentences).strip()
        return len(combined) >= self.config.min_chunk_chars

    def _group_average_embedding(self, embeddings: List[List[float]], indices: List[int]) -> List[float]:
        dim = len(embeddings[0])
        avg = [0.0] * dim
        for idx in indices:
            for d in range(dim):
                avg[d] += embeddings[idx][d]
        n = max(len(indices), 1)
        return [x / n for x in avg]


class ChunkingService:
    STRATEGIES = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }

    def __init__(self, config: ChunkingConfig, embedding_fn: Optional[Callable[[str], List[float]]] = None):
        self.config = config
        self.config.embedding_fn = embedding_fn
        self.chunker = self._get_chunker(config.strategy)
        self.strategy = config.strategy

    def _get_chunker(self, strategy: str):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown chunking strategy: {strategy}. Valid: {list(self.STRATEGIES.keys())}")
        return self.STRATEGIES[strategy](self.config)

    def chunk(self, document: Document) -> List[DocumentChunk]:
        return self.chunker.chunk(document)

    def re_chunk(self, document: Document, strategy: str) -> List[DocumentChunk]:
        """Switch strategy and rechunk a document."""
        self.config.strategy = strategy
        self.chunker = self._get_chunker(strategy)
        return self.chunk(document)

    @classmethod
    def available_strategies(cls):
        return list(cls.STRATEGIES.keys())


if __name__ == "__main__":
    svc = ChunkingService(ChunkingConfig(strategy="fixed"))
    print(f"Available strategies: {svc.available_strategies()}")
    print("Chunking service ready.")