from pathlib import Path

from rag_evaluation.golden_dataset import GoldenDataset, GoldenQuestion, CATEGORIES

DATA_FILE = Path(__file__).parent.parent / "data" / "golden_dataset.json"


def make_dataset():
    return GoldenDataset.load(str(DATA_FILE))


def test_loads_over_50_questions():
    ds = make_dataset()
    assert len(ds) >= 50, f"expected >= 50, got {len(ds)}"


def test_all_four_categories_present():
    ds = make_dataset()
    counts = ds.counts()
    assert counts["total"] == len(ds)
    for cat in CATEGORIES:
        assert counts[cat] > 0, f"missing category {cat}"


def test_lookup_multihop_have_sources():
    ds = make_dataset()
    for q in ds.filter("lookup") + ds.filter("multi_hop"):
        assert q.source_documents, f"{q.id} missing source docs"
        assert q.source_sections, f"{q.id} missing source sections"


def test_no_answer_has_no_sources():
    ds = make_dataset()
    for q in ds.filter("no_answer"):
        assert not q.source_documents, f"{q.id} should not have sources"


def test_validate_clean():
    ds = make_dataset()
    problems = ds.validate()
    assert problems == [], f"validation problems: {problems}"


def test_validate_catches_bad_dataset():
    bad = GoldenDataset(questions=[
        GoldenQuestion(id="a", category="bogus", question="", answer=""),
        GoldenQuestion(id="a", category="lookup", question="q", answer="a"),
    ])
    problems = bad.validate()
    assert len(problems) > 0
    assert any("duplicate id" in p for p in problems)
    assert any("invalid category" in p for p in problems)


def test_filter_by_category():
    ds = make_dataset()
    lookup = ds.filter("lookup")
    assert all(q.is_lookup for q in lookup)
    assert ds.filter("no_answer") and all(q.is_no_answer for q in ds.filter("no_answer"))


def test_summary_contains_documents():
    ds = make_dataset()
    s = ds.summary()
    assert "total" in s.lower() or "Total" in s
    assert "multi_hop" in s


def test_sample_questions_well_formed():
    ds = make_dataset()
    sample = ds.questions[:3]
    for q in sample:
        assert isinstance(q.question, str) and isinstance(q.answer, str)
        assert q.question.endswith("?") or "?" in q.question