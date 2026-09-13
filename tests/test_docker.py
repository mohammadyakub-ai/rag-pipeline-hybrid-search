import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def test_compose_config_valid_and_defines_expected_services():
    if not shutil.which("docker"):
        pytest.skip("docker CLI not available")
    proc = _run(["docker", "compose", "config"], cwd=str(ROOT), timeout=120)
    assert proc.returncode == 0, f"docker compose config failed:\n{proc.stderr}"
    out = proc.stdout
    for svc in ("chroma:", "api:", "seed:", "dashboard:"):
        assert f"\n  {svc}" in f"\n{out}", f"service {svc!r} missing from resolved config"
    for vol in ("chroma_data", "rag_data"):
        assert f"{vol}:" in out, f"volume {vol!r} missing"


def test_dockerfiles_have_expected_build_blocks():
    api = (ROOT / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    dash = (ROOT / "docker" / "Dockerfile.dashboard").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in api
    assert "uvicorn" in api and "rag_pipeline.api:create_app" in api
    assert "COPY data/raw/" in api  # seed corpus baked into the image
    assert "streamlit" in dash and "dashboards/query_dashboard.py" in dash
    assert "HEALTHCHECK" in dash


def test_dockerignore_does_not_exclude_seed_corpus():
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    lines = [l.strip() for l in di.splitlines() if l.strip() and not l.startswith("#")]
    assert "data/raw" not in lines, "data/raw must stay embeddable for the seed corpus"
    assert any("data/chroma" in l for l in lines)


def test_chroma_image_injects_curl_for_healthcheck():
    df = (ROOT / "docker" / "Dockerfile.chroma").read_text(encoding="utf-8")
    assert "FROM chromadb/chroma:1.5.9" in df
    assert "curlimages/curl" in df and "COPY --from=curl" in df
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"curl"' in compose and "api/v1/heartbeat" in compose