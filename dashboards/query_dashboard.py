"""Streamlit query dashboard for the RAG hybrid-search service.

Runs against the FastAPI backend (rag_pipeline.api) and shows:
  - the generated answer with clickable citations
  - retrieved chunks ranked by relevance
  - confidence broken down by dimension
  - a side-by-side hybrid vs dense-only retrieval comparison

Usage:
    uvicorn rag_pipeline.api:create_app --factory   # backend, port 8000
    streamlit run dashboards/query_dashboard.py      # dashboard
"""

import os

import streamlit as st
import requests

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODES = {
    "Hybrid (dense + sparse)": "hybrid",
    "Dense only": "dense",
    "Sparse only (BM25)": "sparse",
}


def api_get(base: str, path: str, timeout: int = 5):
    try:
        return requests.get(f"{base}{path}", timeout=timeout)
    except requests.RequestException:
        return None


def api_post(base: str, path: str, payload: dict, timeout: int = 30):
    try:
        resp = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return None, None, f"backend unreachable: {exc.__class__.__name__}"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text[:200]
        return None, resp.status_code, detail
    return resp.json(), resp.status_code, None


def fetch_ask(base: str, question: str, mode: str, top_k: int):
    return api_post(base, "/v1/ask", {
        "question": question,
        "retrieval_mode": mode,
        "top_k": top_k,
    })


def fetch_retrieve(base: str, question: str, mode: str, top_k: int):
    return api_post(base, "/v1/retrieve", {
        "question": question,
        "mode": mode,
        "top_k": top_k,
    })


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_chunk(item: dict, rank: int, highlight: bool = False):
    """Render one retrieved chunk with context. `highlight=True` when its rank
    is the target of a clicked citation."""
    meta_lines = []
    if item.get("source_file"):
        meta_lines.append(f"**{item['source_file'].split('/')[-1]}**")
    if item.get("section"):
        meta_lines.append(f"§ {item['section']}")
    if item.get("chunking_strategy"):
        meta_lines.append(f"chunking={item['chunking_strategy']}")
    header = f"**#{rank}** · {' · '.join(meta_lines)}"
    scores = (
        f"rerank **{fmt(item.get('rerank_score'))}** · "
        f"rrf **{fmt(item.get('rrf_score'))}** · "
        f"dense **{fmt(item.get('dense_score'))}** · "
        f"sparse **{fmt(item.get('sparse_score'))}**"
    )
    if highlight:
        st.markdown(f"### {header}")
        st.markdown(f"`{scores}` - ← cited by the answer")
        st.info(item.get("content", ""))
    else:
        st.markdown(f"### {header}")
        st.markdown(f"`{scores}`")
        st.markdown(item.get("content", ""))


def render_confidence(conf: dict):
    if not conf:
        st.write("_Confidence unavailable for this response._")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite", f"{conf.get('composite', 0):.2%}")
    c2.metric("Retrieval", f"{conf.get('retrieval_confidence', 0):.2%}")
    c3.metric("Citation coverage", f"{conf.get('citation_coverage', 0):.2%}")
    c4.metric("Completeness", f"{conf.get('completeness', 0):.2%}")
    w = conf.get("weights")
    if w:
        st.caption(
            "Weights: "
            f"retrieval {w.get('retrieval')}, "
            f"citation {w.get('citation')}, "
            f"completeness {w.get('completeness')}"
        )


def render_citation_buttons(citations: dict, sources: list):
    """Clickable citation badges -> clicks highlight the cited source rank."""
    if not citations or not citations.get("citations"):
        st.markdown("_No citations in this answer._")
        return
    cols = st.columns(min(len(citations["citations"]), 6))
    for col, cit in zip(cols, citations["citations"]):
        n = cit["citation"]  # e.g. "[1]"
        if col.button(
            n,
            key=f"cite_{n}",
            help=(cit.get("claim") or cit.get("reason") or "")[:120],
        ):
            st.session_state["selected_citation"] = n
    if st.session_state.get("selected_citation"):
        st.caption("🔍 Highlighting the chunk for the citation you clicked.")


def render_answer(body: dict):
    if not body:
        return
    st.subheader("Answer")
    if body.get("can_answer"):
        st.markdown(body["answer"])
        st.markdown("---")
        with st.expander("Citations — click a badge to jump to its source", expanded=True):
            render_citation_buttons(body.get("citations"), body.get("sources") or [])
        st.subheader("Confidence")
        render_confidence(body.get("confidence"))
    else:
        declined = body.get("declined") or {}
        reason = declined.get("reason", "low confidence")
        st.warning(f"I'm not confident enough to answer ({reason}).")
        if declined.get("summary"):
            st.write(declined["summary"])
        st.subheader("Why?")
        render_confidence(body.get("confidence"))
        if declined.get("found"):
            st.markdown("_Partially relevant passages found:_")
            for f in declined["found"]:
                st.markdown(f"- {f}")


def render_ranking(base: str, question: str, top_k: int):
    st.subheader("Retrieved chunks by relevance")
    hybrid, _, _ = fetch_retrieve(base, question, "hybrid", top_k)
    if not hybrid:
        st.warning("Retrieval unavailable — is the backend running?")
        return
    select = st.session_state.get("selected_citation")
    for rank, item in enumerate(hybrid["ranking"], start=1):
        citation_target = select == f"[{rank}]"
        with st.expander(
            f"#{rank} · {item.get('source_file','').split('/')[-1]} · "
            f"rerank {fmt(item.get('rerank_score'))}",
            expanded=citation_target,
        ):
            render_chunk(item, rank, highlight=citation_target)


def render_comparison(base: str, question: str, top_k: int):
    st.subheader("Hybrid vs dense-only retrieval")
    hybrid, hs, hdetail = fetch_retrieve(base, question, "hybrid", top_k)
    dense, ds, ddetail = fetch_retrieve(base, question, "dense", top_k)
    if not hybrid and not dense:
        st.warning("Comparison unavailable — could not reach the backend.")
        return
    if hs == 503 or ds == 503:
        st.info("Backend returned 503 — set an OpenRouter/OpenAI key or pass a generator before asking.")
    if hybrid is None:
        st.error(hdetail)
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hybrid")
        if not hybrid["ranking"]:
            st.write("_no results_")
        dense_ids = {item["chunk_id"] for item in dense["ranking"]} if dense else set()
        for rank, item in enumerate(hybrid["ranking"], start=1):
            tag = "🔗 found in both" if item["chunk_id"] in dense_ids else ""
            st.markdown(
                f"**#{rank}** rerank **{fmt(item.get('rerank_score'))}** {tag}"
            )
            st.caption(item.get("content", "")[:220])
    with c2:
        st.markdown("### Dense only")
        if not dense or not dense["ranking"]:
            st.write("_no results_")
        hybrid_ids = {item["chunk_id"] for item in hybrid["ranking"]} if hybrid else set()
        for rank, item in enumerate(dense["ranking"], start=1):
            tag = "🔗 found in both" if item["chunk_id"] in hybrid_ids else ""
            st.markdown(
                f"**#{rank}** rerank **{fmt(item.get('rerank_score'))}** {tag}"
            )
            st.caption(item.get("content", "")[:220])


def render():
    st.set_page_config(
        page_title="RAG Query Dashboard",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("RAG Query Dashboard")
    st.caption(
        "Hybrid retrieval (dense + sparse + RRF + rerank) with grounded answers, "
        "verified citations, confidence scoring and graceful 'I don't know' behavior."
    )

    with st.sidebar:
        st.header("Backend")
        base = st.text_input("API base URL", value=DEFAULT_BASE_URL)
        try:
            health = api_get(base, "/healthz")
            if health is not None and health.ok:
                hb = health.json()
                st.success(f"healthy — {hb['vector_chunks']} chunks indexed")
            else:
                st.warning("backend unreachable")
        except Exception:
            st.warning("backend unreachable")

        try:
            docs = api_get(base, "/v1/documents")
            if docs is not None and docs.ok and docs.json():
                st.caption(f"{len(docs.json())} document(s) indexed")
        except Exception:
            pass

        st.divider()
        st.header("Ask")
        mode_label = st.radio(
            "Retrieval for the answer",
            list(MODES.keys()),
            index=0,
        )
        retrieval_mode = MODES[mode_label]
        top_k = st.slider("Top-K chunks", min_value=3, max_value=20, value=10)
        show_compare = st.toggle(
            "Compare hybrid vs dense-only retrieval", value=True
        )

    question = st.text_input("Ask your question", placeholder="e.g. What is reciprocal rank fusion?")
    asked = st.button("Ask", type="primary")

    if not question.strip():
        st.info("Type a question above and press **Ask**.")
        return

    if asked or st.session_state.get("last_question") == question:
        st.session_state["last_question"] = question
        st.session_state.pop("selected_citation", None)

        with st.spinner("Running the RAG pipeline…"):
            body, code, detail = fetch_ask(base, question, retrieval_mode, top_k)
        if body:
            render_answer(body)
        else:
            st.error(f"/v1/ask failed ({code}): {detail}")
            if code == 503:
                st.warning(
                    "The backend has no LLM configured. Set OPENROUTER_API_KEY "
                    "or OPENAI_API_KEY (or provide a Generator) to get answers; retrieval "
                    "comparison below still works."
                )

        st.markdown("---")
        render_ranking(base, question, top_k)

        if show_compare:
            st.markdown("---")
            render_comparison(base, question, top_k)


if __name__ == "__main__":
    render()