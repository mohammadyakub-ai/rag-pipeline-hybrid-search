from pathlib import Path

from rag_pipeline.document_loader import DocumentLoader


def test_text_loader(tmp_path):
    loader = DocumentLoader()
    f = tmp_path / "doc.txt"
    f.write_text(
        "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some more content."
    )
    doc = loader.load_file(str(f))
    assert doc.doc_id is not None
    assert len(doc.chunks) > 0
    assert doc.metadata["file_extension"] == ".txt"


def test_markdown_loader(tmp_path):
    loader = DocumentLoader()
    md_content = """---
title: Test Document
author: Test Author
---

# Introduction

This is the introduction section.

## Subsection

Content in subsection.

# Conclusion

Final thoughts."""
    f = tmp_path / "doc.md"
    f.write_text(md_content)
    doc = loader.load_file(str(f))
    assert doc.doc_id is not None
    assert len(doc.chunks) > 0
    assert doc.metadata["title"] == "Test Document"
    assert doc.metadata["author"] == "Test Author"


def test_html_loader(tmp_path):
    loader = DocumentLoader()
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Page</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>First paragraph content.</p>
    <h2>Sub Heading</h2>
    <p>Second paragraph content.</p>
    <section>
        <h3>Section Heading</h3>
        <p>Section content.</p>
    </section>
</body>
</html>"""
    f = tmp_path / "page.html"
    f.write_text(html_content)
    doc = loader.load_file(str(f))
    assert doc.doc_id is not None
    assert len(doc.chunks) > 0
    assert doc.metadata["title"] == "Test HTML Page"


def test_save_and_load_processed(tmp_path):
    loader = DocumentLoader(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    f = tmp_path / "doc.txt"
    f.write_text("Test content for save/load.")
    doc = loader.process_and_save(str(f))
    assert Path(loader.raw_dir / f"{doc.doc_id}.txt").exists()
    assert Path(loader.processed_dir / f"{doc.doc_id}.json").exists()

    loaded = loader.load_processed(doc.doc_id)
    assert loaded is not None
    assert loaded.doc_id == doc.doc_id
    assert len(loaded.chunks) == len(doc.chunks)