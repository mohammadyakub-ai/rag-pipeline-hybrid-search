from rag_evaluation.golden_dataset import GoldenQuestion, GoldenDataset
from rag_pipeline.retrieval import RetrievalResult
from rag_evaluation.eval_metrics import EvalReport, EvalQuestionResult, MetricResult
from rag_evaluation.chunking_comparison import (
    StrategyRow,
    ComparisonResult,
    ChunkingComparison,
    METRICS,
)


def make_report(strategy, score):
    """Build a minimal EvalReport with controlled per-metric means."""
    q = GoldenQuestion(
        id="q1", category="lookup", question="Q?", answer="A",
        source_documents=["d"], source_sections=["s"],
    )
    r = EvalQuestionResult(
        question_id="q1", category="lookup", question="Q?",
        golden_answer="A", generated_answer="Ans [1].",
        correctness=MetricResult(name="correctness", score=score),
        faithfulness=MetricResult(name="faithfulness", score=score + 0.1),
        retrieval_relevance=MetricResult(name="retrieval_relevance", score=score - 0.1),
        citation_accuracy=MetricResult(name="citation_accuracy", score=score),
    )
    rep = EvalReport(results=[r], chunking_strategy=strategy)
    rep.compute()
    return rep


def test_strategy_row_means_and_medians():
    row = StrategyRow(strategy="fixed", report=make_report("fixed", 0.8))
    assert abs(row.means["correctness"] - 0.8) < 1e-6
    assert abs(row.means["faithfulness"] - 0.9) < 1e-6
    assert "q1" in row.per_question
    d = row.to_dict()
    assert d["strategy"] == "fixed" and d["means"]["correctness"] == 0.8


def test_comparison_decides_winners():
    rows = {
        "fixed": StrategyRow("fixed", make_report("fixed", 0.8)),
        "recursive": StrategyRow("recursive", make_report("recursive", 0.6)),
        "semantic": StrategyRow("semantic", make_report("semantic", 0.1)),
    }
    result = ComparisonResult(rows=rows)
    result.decide_winners(margin=0.01)
    # fixed should win correctness, faithfulness (0.9 vs recursive 0.7), citation
    assert result.metric_winners["correctness"] == "fixed"
    assert result.metric_winners["faithfulness"] == "fixed"


def test_comparison_tie_handling():
    rows = {
        "fixed": StrategyRow("fixed", make_report("fixed", 0.8)),
        "recursive": StrategyRow("recursive", make_report("recursive", 0.805)),
    }
    result = ComparisonResult(rows=rows)
    result.decide_winners(margin=0.01)
    # within margin -> tie reported as joined names
    assert set(result.metric_winners["correctness"].split("+")) == {"fixed", "recursive"}


def test_category_winners():
    q_lookup = GoldenQuestion(id="q1", category="lookup", question="Q?", answer="A",
                              source_documents=["d"], source_sections=["s"])
    r = EvalQuestionResult(
        question_id="q1", category="lookup", question="Q?",
        golden_answer="A", generated_answer="Ans [1].",
        correctness=MetricResult(name="correctness", score=0.9),
        faithfulness=MetricResult(name="faithfulness", score=0.9),
        retrieval_relevance=MetricResult(name="retrieval_relevance", score=0.9),
        citation_accuracy=MetricResult(name="citation_accuracy", score=0.9),
    )
    rep = EvalReport(results=[r], chunking_strategy="fixed")
    rep.compute()
    row = StrategyRow("fixed", rep)
    cr = ComparisonResult(rows={"fixed": row})
    cw = cr.category_winners()
    assert cw["lookup"]["correctness"] == "fixed"


def test_chunking_comparison_factory_wiring():
    """Verify the ChunkingComparison invokes the factory once per strategy
    and produces a ComparisonResult with three rows."""
    calls = []

    def pipeline_factory(strategy):
        calls.append(strategy)

        class FakePipeline:
            def retrieve(self, question, top_k=10):
                return [RetrievalResult(
                    chunk_id=f"{strategy}::0", content=f"{strategy} content about api keys",
                    metadata={"source_file": "guide.pdf", "chunking_strategy": strategy},
                )]

        return FakePipeline()

    ds = GoldenDataset(questions=[
        GoldenQuestion(id="q1", category="lookup", question="What auth is required?",
                       answer="An api key", source_documents=["guide.pdf"], source_sections=["Phase 1.3"]),
        GoldenQuestion(id="q2", category="no_answer", question="refund policy?",
                       answer="Not stated.", source_documents=[], source_sections=[]),
    ])
    cmp = ChunkingComparison(
        dataset=ds,
        strategies=["fixed", "recursive", "semantic"],
        pipeline_factory=pipeline_factory,
        generate_fn=lambda q, c: ({"answer": "I don't have enough information to answer this."}
                                  if "refund" in q else {"answer": "An api key is required [1]."}),
    )
    result = cmp.run()
    assert isinstance(result, ComparisonResult)
    assert sorted(result.rows.keys()) == ["fixed", "recursive", "semantic"]
    assert calls == ["fixed", "recursive", "semantic"]
    for s in result.metric_winners.values():
        assert s


def test_summary_and_to_dict():
    rows = {
        "fixed": StrategyRow("fixed", make_report("fixed", 0.7)),
        "recursive": StrategyRow("recursive", make_report("recursive", 0.5)),
    }
    cr = ComparisonResult(rows=rows)
    cr.decide_winners()
    s = cr.summary()
    assert "Chunking Strategy Comparison" in s
    assert "fixed" in s and "correctness" in s
    d = rows["fixed"].to_dict()
    assert d["strategy"] == "fixed"