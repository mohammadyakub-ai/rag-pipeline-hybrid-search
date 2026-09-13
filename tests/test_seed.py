import socket
import threading
import time

import uvicorn
import requests

from rag_pipeline.api import create_app
import seed as seed_module


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return False


class _Server:
    """Runs a uvicorn server on an ephemeral port in a background thread."""

    def __init__(self, app, port: int):
        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self):
        self._thread.start()
        assert _wait_health(f"http://127.0.0.1:{self.port}/healthz"), "server never became healthy"
        return self

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)


def test_discover_filters_supported_extensions(tmp_path):
    p = tmp_path
    (p / "a.md").write_text("x", encoding="utf-8")
    (p / "b.pdf").write_bytes(b"%PDF-1.4")
    (p / "c.txt").write_text("x", encoding="utf-8")
    (p / "d.xyz").write_text("x", encoding="utf-8")
    (p / "sub").mkdir()
    (p / "sub" / "e.markdown").write_text("x", encoding="utf-8")
    found = seed_module.discover(str(tmp_path))
    names = sorted(f.name for f in found)
    assert names == ["a.md", "b.pdf", "c.txt", "e.markdown"], names


def test_discover_explicit_files_override(tmp_path):
    p = tmp_path
    (p / "only.pdf").write_bytes(b"%PDF-1.4")
    (p / "skip.md").write_text("x", encoding="utf-8")
    found = seed_module.discover("/nonexistent-dir", explicit_files=[str(p / "only.pdf")])
    assert [f.name for f in found] == ["only.pdf"]


def test_dry_run_lists_files_without_touching_api(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text("# Doc\n\nsome content here", encoding="utf-8")
    code = seed_module.main(
        ["--api", "http://127.0.0.1:1", "--corpus-dir", str(tmp_path), "--dry-run"]
    )
    assert code == 0, code


def test_seed_e2e_seeds_corpus_into_live_api(tmp_path):
    persist = str(tmp_path / "chroma")
    bm25 = str(tmp_path / "bm25")
    app = create_app(persist_dir=persist, bm25_dir=bm25)
    port = _free_port()

    with _Server(app, port):
        md = tmp_path / "phaq.md"
        md.write_text(
            "# Operations FAQ\n\n"
            "## Backups\n\nBackups run nightly at 02:00 UTC.\n\n"
            "## On-call\n\nAlerts page the on-call engineer after 15 minutes.\n",
            encoding="utf-8",
        )
        code = seed_module.main(
            ["--api", f"http://127.0.0.1:{port}", "--corpus-dir", str(tmp_path)]
        )
        assert code == 0, code

        docs = requests.get(f"http://127.0.0.1:{port}/v1/documents", timeout=5).json()
        assert len(docs) >= 1 and docs[0]["chunk_count"] > 0, docs

        # Re-running is idempotent: everything reported as duplicate.
        code2 = seed_module.main(
            ["--api", f"http://127.0.0.1:{port}", "--corpus-dir", str(tmp_path)]
        )
        assert code2 == 0, code2
        docs2 = requests.get(f"http://127.0.0.1:{port}/v1/documents", timeout=5).json()
        assert len(docs2) == len(docs), (docs2, docs)


def test_seed_fails_cleanly_when_api_unreachable(tmp_path):
    port = _free_port()
    md = tmp_path / "doc.md"
    md.write_text("# Doc\n\ncontent", encoding="utf-8")
    code = seed_module.main(
        ["--api", f"http://127.0.0.1:{port}", "--corpus-dir", str(tmp_path)]
    )
    assert code == 1, code


def test_reset_wipes_index_and_reseeds(tmp_path):
    persist = str(tmp_path / "chroma")
    bm25 = str(tmp_path / "bm25")
    app = create_app(persist_dir=persist, bm25_dir=bm25)
    port = _free_port()

    with _Server(app, port):
        md = tmp_path / "faq.md"
        md.write_text(
            "# FAQ\n\n"
            "## Backups\n\nBackups run nightly at 02:00 UTC.\n\n"
            "## On-call\n\nAlerts page the on-call engineer after 15 minutes.\n",
            encoding="utf-8",
        )
        code = seed_module.main(
            ["--api", f"http://127.0.0.1:{port}", "--corpus-dir", str(tmp_path)]
        )
        assert code == 0, code
        assert len(requests.get(f"http://127.0.0.1:{port}/v1/documents", timeout=5).json()) == 1

        code = seed_module.main(
            ["--api", f"http://127.0.0.1:{port}", "--corpus-dir", str(tmp_path), "--reset"]
        )
        assert code == 0, code
        docs = requests.get(f"http://127.0.0.1:{port}/v1/documents", timeout=5).json()
        assert len(docs) == 1 and docs[0]["chunk_count"] > 0, docs


def test_reset_flag_rejects_dry_run(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("x", encoding="utf-8")
    try:
        seed_module.main(
            ["--api", "http://127.0.0.1:1", "--corpus-dir", str(tmp_path), "--reset", "--dry-run"]
        )
        raise AssertionError("expected SystemExit from parser.error")
    except SystemExit:
        pass