from pathlib import Path

from rag_evaluation.golden_dataset import GoldenQuestion
from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.generation import Generator
from rag_pipeline.citation import CitationVerifier
from rag_evaluation.eval_metrics import (
    CorrectnessScorer,
    FaithfulnessScorer,
    RetrievalRelevanceScorer,
    CitationAccuracyScorer,
    EvalRunner,
    EvalReport,
)

DATA = Path(__file__).parent.parent / "data" / "golden_dataset.json"


def chunk(content, meta=None):
    return RetrievalResult(
        chunk_id="x", content=content,
        metadata=meta or {"source_file": "Guide.pdf"},
    )


def sample_q(
    category="lookup",
    answer="An api key is required for access.",
    source_sections=("Phase 1.3",),
):
    return GoldenQuestion(
        id="t1", category=category, question="What auth is required?",
        answer=answer, source_documents=["Guide.pdf"],
        source_sections=list(source_sections),
    )


def test_correctness_lexical_overlap():
    scorer = CorrectnessScorer()
    q = sample_q(answer="The api key grants admin access to the dashboard.")
    mr = scorer.score(q, "An api key grants admin access to the dashboard.")
    assert 0.0 <= mr.score <= 1.0
    assert mr.name == "correctness"
    assert mr.score > 0.5


def test_correctness_no_answer_decline_wins():
    scorer = CorrectnessScorer()
    q = sample_q(category="no_answer", answer="Not stated in the corpus.")
    good = scorer.score(q, "I don't have enough information in the context to answer this.")
    assert good.score == 1.0
    bad = scorer.score(q, "The answer is definitely 50 dollars.")
    assert bad.score < 1.0


def test_correctness_llm_judge():
    def fake(messages):
        return '{"score": 0.9, "reason": "matches golden"}'
    scorer = CorrectnessScorer(judge=Generator(respond=fake))
    q = sample_q()
    mr = scorer.score(q, "Api key is required.")
    assert mr.score == 0.9


def test_faithfulness_lexical_grounded():
    scorer = FaithfulnessScorer()
    chunks = [chunk("Authentication requires an api key issued by the admin dashboard.")]
    mr = scorer.score("Authentication requires an api key [1].", chunks)
    assert mr.score > 0.5


def test_faithfulness_lexical_not_grounded():
    scorer = FaithfulnessScorer()
    chunks = [chunk("The sky is blue today in the office.")]
    mr = scorer.score("Authentication requires an api key [1].", chunks)
    assert mr.score == 0.0


def test_faithfulness_llm_judge():
    def fake(messages):
        return '{"grounded": [true, false]}'
    scorer = FaithfulnessScorer(judge=Generator(respond=fake))
    chunks = [chunk("some context content")]
    mr = scorer.score("Claim one. Claim two.", chunks)
    assert mr.score == 0.5


def test_retrieval_relevance_section_and_content():
    scorer = RetrievalRelevanceScorer()
    q = sample_q(
        answer="text-embedding-3-small is used to embed chunks",
        source_sections=["Phase 1.3"],
    )
    chunks = [chunk("We embed every chunk using text-embedding-3-small", meta={"section_heading": "phase 1.3"})]
    mr = scorer.score(q, chunks)
    assert mr.score >= 0.5


def test_retrieval_relevance_empty_chunks():
    scorer = RetrievalRelevanceScorer()
    mr = scorer.score(sample_q(), [])
    assert mr.score == 0.0


def test_citation_accuracy_from_report():
    scorer = CitationAccuracyScorer()
    q = sample_q()
    # reuse it through the citation verifier to produce a report
    verifier = CitationVerifier(judge=Generator(respond=lambda m: '{"supported": true, "reason": "ok"}'))
    chunks = [chunk("The api key is required for access.")]
    report = verifier.verify("Access needs an api key [1].", chunks)
    mr = scorer.score(q, "Access needs an api key [1].", chunks, report)
    assert mr.score == 1.0


def test_citation_accuracy_no_answer_no_citations():
    scorer = CitationAccuracyScorer()
    q = sample_q(category="no_answer")
    mr = scorer.score(q, "I don't know.", chunk("content"))
    assert mr.score == 1.0


def test_citation_accuracy_no_answer_has_citations_failed():
    scorer = CitationAccuracyScorer()
    q = sample_q(category="no_answer")
    mr = scorer.score(q, "The price is 50 [1].", chunk("content"))
    assert mr.score == 0.0


def test_eval_runner_full_suite():
    from rag_evaluation.golden_dataset import GoldenDataset

    ds = GoldenDataset.load(str(DATA))

    lookups = ds.filter("lookup")
    lookups = lookups[:3]
    ds.questions = lookups + ds.filter("no_answer")[:2] + ds.filter("ambiguous")[:1]

    def retrieve_fn(question):
        return [chunk("Authentication requires an api key issued by the admin dashboard.")]

    def generate_fn(question, chunks):
        if "don't" in question.lower() or "price" in question.lower() or "refund" in question.lower() or "not" in question.lower():
            return {"answer": "I don't have enough information in the context to answer this question."}
        return {"answer": "Authentication requires an api key [1]."}

    verifier = CitationVerifier(judge=Generator(respond=lambda m: '{"supported": true, "reason": "ok"}'))
    runner = EvalRunner(
        dataset=ds, retrieve_fn=retrieve_fn, generate_fn=generate_fn,
        citation_verifier=verifier,
    )
    report = runner.run()

    assert isinstance(report, EvalReport)
    assert report.total_questions == len(ds)
    assert "correctness" in report.overall
    assert "faithfulness" in report.overall
    assert "retrieval_relevance" in report.overall
    assert "citation_accuracy" in report.overall
    assert report.by_category
    d = report.to_dict()
    assert d["overall"]["correctness"]["mean"] >= 0.0


def test_eval_runner_handles_errors():
    from rag_evaluation.golden_dataset import GoldenDataset, GoldenQuestion

    ds = GoldenDataset(questions=[
        GoldenQuestion(id="q1", category="lookup", question="Q?", answer="A", source_documents=["d"], source_sections=["s"]),
    ])

    def bad_retrieve(question):
        raise RuntimeError("boom")
    runner = EvalRunner(dataset=ds, retrieve_fn=bad_retrieve, generate_fn=lambda q, c: {"answer": ""})
    report = runner.run()
    assert report.errors == 1
    assert report.results[0].error is not None