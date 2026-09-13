import pytest

from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.generation import GroundedPrompt, Generator, DEFAULT_SYSTEM_PROMPT


def chunk(cid, content, heading=None, source="guide.md"):
    return RetrievalResult(
        chunk_id=cid,
        content=content,
        metadata={"doc_id": "d", "source_file": source, "section_heading": heading},
    )


def test_system_prompt_contains_grounding_rules():
    for rule in [
        "must cite every factual claim",
        "chunk number from the context below",
        "[1]",
        "does not contain the answer",
        "[2]",
        "cannot find this information",
    ]:
        assert rule.lower() in DEFAULT_SYSTEM_PROMPT.lower()
    # Explicit "cite" mandate plus chunk-notation and I-don't-know handling
    low = DEFAULT_SYSTEM_PROMPT.lower()
    assert "cite" in low
    assert "never write a factual sentence" in low
    assert "not supported by the context" in low


def test_context_blocks_are_numbered_and_metadata_present():
    prompt = GroundedPrompt()
    results = [
        chunk("d::chunk::0", "Alpha content here.", heading="Intro", source="a.md"),
        chunk("d::chunk::1", "Beta content here.", heading="API", source="a.md"),
        chunk("d::chunk::2", "Gamma content here.", source="b.pdf"),
    ]
    context = prompt.format_context(results)

    assert 'id="1"' in context
    assert 'id="2"' in context
    assert 'id="3"' in context
    assert "a.md" in context
    assert "b.pdf" in context
    assert "Intro" in context and "API" in context
    # Each block wraps content in a <context> tag
    assert context.count("<context ") == 3


def test_build_messages_includes_question_and_context():
    prompt = GroundedPrompt()
    results = [chunk("d::chunk::0", "The API requires a bearer token.", heading="Auth")]
    messages = prompt.build_messages("How is the API authenticated?", results)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "grounded" in messages[0]["content"] or "context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert 'id="1"' in user
    assert "How is the API authenticated?" in user
    assert "bearer token" in user
    # The answer format + question are in the user turn
    assert "Answer format" in user


def test_empty_context_handled():
    prompt = GroundedPrompt()
    context = prompt.format_context([])
    assert "No context provided" in context
    messages = prompt.build_messages("any question?", [])
    assert len(messages) == 2


def test_generator_calls_respond_and_returns_structure():
    captured = {}

    def fake_respond(messages):
        captured["messages"] = messages
        return "The token is a bearer token [1]."

    gen = Generator(prompt=GroundedPrompt(), model="test-model", respond=fake_respond)
    results = [chunk("d::chunk::0", "Use a bearer token.", heading="Auth")]
    out = gen.generate("How auth?", results)

    assert out["answer"] == "The token is a bearer token [1]."
    assert out["source_chunks"] == results
    assert out["model"] == "test-model"
    assert len(captured["messages"]) == 2


def test_generator_requires_config(monkeypatch):
    # No OPENAI_API_KEY -> generator is unconfigured and refuses to answer.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = Generator(api_key=None)
    assert not gen.is_configured
    with pytest.raises(RuntimeError):
        gen.generate("q", [chunk("x", "c")])