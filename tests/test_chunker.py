from rag_pipeline.document_loader import Document
from rag_pipeline.chunker import (
    ChunkingConfig,
    ChunkingService,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
)


def make_document(content: str) -> Document:
    return Document(
        doc_id="test-doc",
        source_path="test.md",
        raw_content=content,
        chunks=[],
        metadata={"source_file": "test.md"},
    )


def test_fixed_size_chunker():
    content = (
        "This is the first sentence of the document. "
        "Here is some more content to fill out the chunk size threshold. "
        "We need enough words to potentially exceed a small chunk size. "
        "Continuing to add more text so the chunker produces multiple chunks. "
        "This should be long enough to produce at least two chunks when we "
        "use a small chunk_size value to test the overlap behavior properly. "
        "Adding even more text now to make absolutely sure that we cross the "
        "chunk boundary and get more than one chunk from this single input."
    )
    doc = make_document(content)
    config = ChunkingConfig(strategy="fixed", chunk_size=150, chunk_overlap=30)
    chunker = FixedSizeChunker(config)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0, "Fixed chunker should produce at least one chunk"
    for c in chunks:
        assert c.metadata.get("chunking_strategy") == "fixed"
        assert "section_heading" in c.metadata


def test_recursive_chunker():
    content = """# Introduction

This is the intro. It talks about the overview of the system.

## Setup

Here we describe how to set things up. Step one, step two, step three. More detail here to make the section reasonably long for testing purposes.

## Configuration

Configuration details. Setting environment variables and defaults. Ensure values are correct before deployment.

# Advanced Topics

Deep dive into advanced usage patterns and power user workflows."""
    doc = make_document(content)
    config = ChunkingConfig(strategy="recursive", chunk_size=120, chunk_overlap=20)
    chunker = RecursiveChunker(config)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for c in chunks:
        assert c.metadata.get("chunking_strategy") == "recursive"
        assert "section_heading" in c.metadata


def test_semantic_chunker():
    # Mock embedding that groups keywords topically
    topic_keywords = {
        "python": [1.0, 0.0, 0.0, 0.0],
        "code": [1.0, 0.0, 0.0, 0.0],
        "function": [1.0, 0.0, 0.0, 0.0],
        "database": [0.0, 1.0, 0.0, 0.0],
        "sql": [0.0, 1.0, 0.0, 0.0],
        "query": [0.0, 1.0, 0.0, 0.0],
        "marketing": [0.0, 0.0, 1.0, 0.0],
        "campaign": [0.0, 0.0, 1.0, 0.0],
        "budget": [0.0, 0.0, 1.0, 0.0],
    }

    def mock_embed(text: str) -> list:
        vec = [0.0, 0.0, 0.0, 0.0]
        words = text.lower().split()
        for w in words:
            if w in topic_keywords:
                v = topic_keywords[w]
                for i in range(len(vec)):
                    vec[i] += v[i]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec

    content = (
        "Python code using functions. Python is great for code. "
        "We write functions in Python code. "
        "Database sql query. We run sql database queries. "
        "The sql database query is fast. "
        "Marketing campaign budget. Marketing budget for campaign. "
        "We plan the campaign marketing budget."
    )
    doc = make_document(content)
    config = ChunkingConfig(
        strategy="semantic", embedding_fn=mock_embed, similarity_threshold=0.6
    )
    chunker = SemanticChunker(config)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0, "Semantic chunker should produce chunks"
    for c in chunks:
        assert c.metadata.get("chunking_strategy") == "semantic"


def test_chunking_service_switch():
    content = """# Section A

This is content in section A. It describes several things.

# Section B

This is content in section B. Different topic entirely."""
    doc = make_document(content)
    config = ChunkingConfig(strategy="recursive", chunk_size=100)
    svc = ChunkingService(config)

    recursive_chunks = svc.chunk(doc)
    assert all(c.metadata["chunking_strategy"] == "recursive" for c in recursive_chunks)

    fixed_chunks = svc.re_chunk(doc, "fixed")
    assert all(c.metadata["chunking_strategy"] == "fixed" for c in fixed_chunks)


def test_available_strategies():
    strategies = ChunkingService.available_strategies()
    assert set(strategies) == {"fixed", "recursive", "semantic"}