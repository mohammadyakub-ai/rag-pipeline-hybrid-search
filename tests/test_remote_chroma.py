import socket

import pytest

from rag_pipeline.vector_store import VectorStore


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_remote_mode_attempts_http_connection():
    # A host that is definitely not serving ChromaDB must fail -- proving the
    # HTTP client path was selected rather than the persistent local store.
    port = _closed_port()
    with pytest.raises(Exception):
        VectorStore(host="127.0.0.1", port=port)


def test_default_mode_uses_persistent_client(tmp_path):
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    assert vs.count() == 0
    vs.add(["c1"], [[0.1, 0.2, 0.3]], ["hello world"], [{"source_file": "x.pdf"}])
    assert vs.count() == 1