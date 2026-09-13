import re
from typing import List, Optional
from dataclasses import dataclass, field

from .retrieval import RetrievalResult
from .generation import Generator, GroundedPrompt
from .citation import CitationReport, CitationParser


@dataclass
class ConfidenceReport:
    """Per-dimension scores plus a weighted composite confidence score."""

    retrieval_confidence: float = 0.0
    citation_coverage: float = 0.0
    completeness: float = 0.0
    composite: float = 0.0
    weights: dict = field(default_factory=lambda: {
        "retrieval": 0.4,
        "citation": 0.3,
        "completeness": 0.3,
    })
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "completeness": round(self.completeness, 4),
            "composite": round(self.composite, 4),
            "weights": self.weights,
            "details": self.details,
        }


class ConfidenceScorer:
    """Scores an answer on retrieval confidence, citation coverage, and
    completeness, then combines them into a single composite confidence score.

    This is the layer that lets the system decide when an answer is reliable
    enough to return, or when it should instead follow the "I don't know" path.
    """

    def __init__(
        self,
        judge: Optional[Generator] = None,
        weights: Optional[dict] = None,
    ):
        self.judge = judge
        default_weights = {"retrieval": 0.4, "citation": 0.3, "completeness": 0.3}
        self.weights = weights or default_weights

        # normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score(
        self,
        question: str,
        answer: str,
        source_chunks: List[RetrievalResult],
        citation_report: Optional[CitationReport] = None,
    ) -> ConfidenceReport:
        """Compute the three dimension scores and the composite."""
        retrieval_conf = self._retrieval_confidence(source_chunks)
        citation_cov = self._citation_coverage(answer, citation_report)
        completeness = self._completeness(question, answer, source_chunks)

        composite = (
            self.weights["retrieval"] * retrieval_conf
            + self.weights["citation"] * citation_cov
            + self.weights["completeness"] * completeness
        )

        report = ConfidenceReport(
            retrieval_confidence=retrieval_conf,
            citation_coverage=citation_cov,
            completeness=completeness,
            composite=composite,
            weights=self.weights,
            details={
                "citation_verification_done": citation_report is not None,
            },
        )
        return report

    # ---- Dimension 1: retrieval confidence -------------------------------
    def _retrieval_confidence(self, source_chunks: List[RetrievalResult]) -> float:
        """Average relevance of the top retrieved chunks.

        Prefers the rerank score (post-precision-pass), falling back to the
        fused/dense score when no rerank score is present. Normalized to [0,1].
        """
        if not source_chunks:
            return 0.0

        scores = []
        for r in source_chunks:
            s = (
                r.rerank_score
                if r.rerank_score is not None
                else (r.rrf_score if r.rrf_score is not None else r.dense_score)
            )
            if s is not None:
                scores.append(s)

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        return max(0.0, min(1.0, avg))

    # ---- Dimension 2: citation coverage ----------------------------------
    def _citation_coverage(
        self, answer: str, citation_report: Optional[CitationReport]
    ) -> float:
        """Fraction of claims/citations that are verified as supported.

        Uses the citation report when available. Otherwise, heuristically
        measures how much of the answer carries bracketed citations.
        """
        if citation_report is not None and citation_report.total > 0:
            supported = citation_report.supported_count
            return supported / citation_report.total

        # Fallback (no verification run): fraction of sentences with citations
        sentences = CitationParser()._split_sentences(answer)
        if not sentences:
            return 0.0
        cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
        return cited / len(sentences)

    # ---- Dimension 3: completeness ---------------------------------------
    def _completeness(
        self, question: str, answer: str, source_chunks: List[RetrievalResult]
    ) -> float:
        """Did the answer address the question?

        Uses an LLM-as-judge when configured; otherwise a heuristic fallback
        based on question-token coverage and explicit "I don't know" handling.
        """
        if self.judge is not None and self.judge.is_configured:
            return self._judge_completeness(question, answer)

        return self._lexical_completeness(question, answer)

    def _judge_completeness(self, question: str, answer: str) -> float:
        system = (
            "You are a completeness judge. Given a user question and a model "
            "answer, output ONLY a single JSON object: "
            '{"score": <0.0 to 1.0>, "reason": "<short>"} '
            "where score is 1.0 if the answer fully addresses every part of the "
            "question, 0.5 if it addresses part, and 0.0 if it misses the "
            "question or the model says it cannot answer."
        )
        verses = _as_source(
            "Question: " + question + "\n\nAnswer: " + answer
        )
        resp = self.judge.generate(
            "Rate the completeness of the answer to the question.",
            verses,
            system_prompt=system,
        )
        return self._parse_completeness(resp["answer"])

    def _parse_completeness(self, raw: str) -> float:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return 0.0
        try:
            import json
            data = json.loads(m.group(0))
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except Exception:
            return 0.0

    def _lexical_completeness(self, question: str, answer: str) -> float:
        """Heuristic completeness without an LLM judge.

        Rewards an answer that covers the question's meaningful tokens while
        penalizing an explicit "I don't know" (which signals the question was
        not answerable rather than fully addressed).
        """
        if not answer.strip():
            return 0.0

        low = answer.lower()
        if "i don't have enough information" in low or "i don't know" in low:
            # Graceful failure to answer -> low completeness
            return 0.1

        q_tokens = self._tokens(question)
        if not q_tokens:
            return 1.0

        a_tokens = set(self._tokens(answer))
        covered = sum(1 for t in q_tokens if t in a_tokens)
        return covered / len(q_tokens)

    def _tokens(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        stop = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "is",
                "are", "what", "how", "do", "does", "i", "you", "it", "this"}
        return [t for t in text.split() if t and t not in stop]


def _as_source(text: str) -> List[RetrievalResult]:
    r = RetrievalResult(chunk_id="context", content=text, metadata={})
    return [r]