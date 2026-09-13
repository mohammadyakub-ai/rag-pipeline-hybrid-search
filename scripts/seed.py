"""Seed the RAG API with a sample documentation corpus.

Uploads every supported file found under --corpus-dir to the API's POST
/v1/ingest endpoint so reviewers can test the full stack immediately.

Idempotent: re-running against an already-seeded API simply reports each file
as already indexed (duplicates > 0).

Usage:
    python scripts/seed.py --api http://localhost:8000 --corpus-dir data/raw/fastapi_docs
    python scripts/seed.py --dry-run --corpus-dir /app/corpus
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import mimetypes
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag_pipeline.embedding import describe_embedding_service

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


def discover(corpus_dir: str, explicit_files=None) -> list:
    """Return supported corpus files. Explicit files override discovery."""
    if explicit_files:
        return [Path(f) for f in explicit_files]
    root = Path(corpus_dir)
    if not root.is_dir():
        return []
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def seed_file(
    api_base: str, path: Path, corpus_dir: str | None = None, timeout: int = 120
) -> requests.Response:
    """Upload one file; the multipart filename is relative to the corpus
    root (falling back to the basename for explicit file paths) so the API
    stores a meaningful ``source_file`` instead of a temp path."""
    try:
        name = str(path.relative_to(Path(corpus_dir))) if corpus_dir else str(path)
    except ValueError:
        name = path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        content = fh.read()
    return requests.post(
        f"{api_base}/v1/ingest",
        files={"file": (name, content, mime)},
        timeout=timeout,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload a sample documentation corpus to the RAG API."
    )
    parser.add_argument(
        "--api",
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="API base URL (default: $API_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/raw/fastapi_docs",
        help="Directory scanned for supported corpus files (default: data/raw/fastapi_docs)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit file paths; overrides corpus-dir discovery",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be uploaded without uploading")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-upload timeout in seconds (default: 120)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print failures and the final summary")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the API index (dense + sparse) before seeding")
    args = parser.parse_args(argv)

    if args.reset and args.dry_run:
        parser.error("--reset cannot be combined with --dry-run")

    if args.reset:
        try:
            resp = requests.post(f"{args.api}/v1/reset", timeout=args.timeout)
        except requests.RequestException as exc:
            print(f"FAIL reset: {exc}", file=sys.stderr)
            return 1
        if resp.status_code != 200:
            print(
                f"FAIL reset: HTTP {resp.status_code} {resp.text[:200]}",
                file=sys.stderr,
            )
            return 1
        if not args.quiet:
            print("Reset complete (dense + sparse indexes wiped).")

    sharp = not args.quiet
    if sharp:
        print(f"Embedding service: {describe_embedding_service()}")

    files = discover(args.corpus_dir, args.files)
    if not files:
        print(f"No supported corpus files found in '{args.corpus_dir}'.", file=sys.stderr)
        return 2

    if args.dry_run:
        for f in files:
            print(f"- {f}")
        print(f"[dry-run] {len(files)} file(s) would be seeded to {args.api}")
        return 0

    if not args.quiet:
        print(f"Found {len(files)} documents to index from {args.corpus_dir}/")
        print(f"Seeding {len(files)} file(s) to {args.api} ...")

    failed = 0
    seeded_doc_ids = []
    for path in files:
        try:
            resp = seed_file(args.api, path, corpus_dir=args.corpus_dir, timeout=args.timeout)
            if resp.status_code == 201:
                data = resp.json()
                seeded_doc_ids.append(data.get("doc_id"))
                print(
                    f"OK   {path.name}: +{data.get('chunks_added_dense', 0)} chunks, "
                    f"{data.get('duplicates', 0)} dup(s)  doc_id={data.get('doc_id')}"
                )
            else:
                failed += 1
                print(f"FAIL {path.name}: HTTP {resp.status_code} {resp.text[:200]}")
        except requests.RequestException as exc:
            failed += 1
            print(f"FAIL {path.name}: {exc}")

    if failed:
        print(f"Done: {len(files) - failed}/{len(files)} seeded successfully.")
        return 1

    print(f"Done: all {len(files)} file(s) indexed. doc_ids: {', '.join(seeded_doc_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())