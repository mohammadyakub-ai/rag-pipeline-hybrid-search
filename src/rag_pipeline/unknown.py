from typing import List, Optional, Dict
from dataclasses import dataclass, field
import re

from .retrieval import RetrievalResult
from .generation import Generator, GroundedPrompt
from .confidence import ConfidenceScorer, ConfidenceReport
from .citation import CitationReport


@dataclass
class LowConfidenceAnswer:
    """Structured "I don't know" response produced when confidence is low.

    Instead of hallucinating, the system explains what it found, what it could
    not find, and which documents are worth checking manually.
    """

    question: str
    confidence: ConfidenceReport
    threshold: float
    found: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    candidate_documents: List[dict] = field(default_factory=list)
    summary: str = ""

    @property
    def should_answer(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {
            "can_answer": False,
            "reason": "insufficient_confidence",
            "threshold": self.threshold,
            "confidence": self.confidence.to_dict(),
            "found": self.found,
            "gaps": self.gaps,
            "candidate_documents": self.candidate_documents,
            "summary": self.summary,
        }


@dataclass
class AnswerOutcome:
    """Result of the confidence-gated answer flow.

    Either carries a full answer with citations + confidence (can_answer=True)
    or a LowConfidenceAnswer (can_answer=False) that declines gracefully.
    """

    can_answer: bool
    answer: Optional[str] = None
    confidence: ConfidenceReport = None
    low_confidence: Optional[LowConfidenceAnswer] = None
    citation_report: Optional[CitationReport] = None

    def to_dict(self) -> dict:
        if self.can_answer:
            return {
                "can_answer": True,
                "answer": self.answer,
                "confidence": self.confidence.to_dict(),
                "citations": (
                    self.citation_report.to_dict() if self.citation_report else None
                ),
            }
        return self.low_confidence.to_dict()


class UnknownHandler:
    """Gates answer generation on confidence, producing a graceful
    "I don't know" response when the system is not confident enough.

    This prevents hallucination: rather than fabricating an answer when
    retrieval is weak, it returns a structured explanation of what was found,
    what was not found, and which documents deserve a manual look.
    """

    def __init__(
        self,
        scorer: ConfidenceScorer,
        threshold: float = 0.5,
        found_highlights: int = 4,
        candidate_docs: int = 3,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.scorer = scorer
        self.threshold = threshold
        self.found_highlights = found_highlights
        self.candidate_docs = candidate_docs

    def decide(
        self,
        question: str,
        answer: str,
        source_chunks: List[RetrievalResult],
        citation_report: Optional[CitationReport] = None,
    ) -> AnswerOutcome:
        """Return an AnswerOutcome after applying the confidence gate."""
        confidence = self.scorer.score(
            question, answer, source_chunks, citation_report=citation_report
        )

        if confidence.composite >= self.threshold:
            return AnswerOutcome(
                can_answer=True,
                answer=answer,
                confidence=confidence,
                citation_report=citation_report,
            )

        return AnswerOutcome(
            can_answer=False,
            confidence=confidence,
            low_confidence=self._build_low_confidence(
                question, answer, source_chunks, confidence
            ),
        )

    def _build_low_confidence(
        self,
        question: str,
        answer: str,
        source_chunks: List[RetrievalResult],
        confidence: ConfidenceReport,
    ) -> LowConfidenceAnswer:
        found = self._what_was_found(question, source_chunks)
        gaps = self._what_was_missing(question, answer, found)
        docs = self._candidate_documents(source_chunks)
        summary = self._summarize(question, found, gaps)

        return LowConfidenceAnswer(
            question=question,
            confidence=confidence,
            threshold=self.threshold,
            found=found,
            gaps=gaps,
            candidate_documents=docs,
            summary=summary,
        )

    # ---- What the system found -------------------------------------------
    def _what_was_found(
        self, question: str, source_chunks: List[RetrievalResult]
    ) -> List[str]:
        """Top topics / snippets from the retrieved chunks, so the user sees
        the system did find *something* relevant.
        """
        items = []
        for r in source_chunks[: self.found_highlights]:
            snippet = self._snippet(r.content, max_len=120)
            doc = (r.metadata or {}).get("source_file") or "unknown"
            item = snippet
            if doc != "unknown":
                item = f"{snippet} (from {doc})"
            items.append(item)
        return items

    def _snippet(self, text: str, max_len: int = 120) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")

    # ---- What could not be found -----------------------------------------
    def _what_was_missing(
        self, question: str, answer: str, found: List[str]
    ) -> List[str]:
        """Question terms that did not surface in the answer or found chunks."""
        q_tokens = self.scorer._tokens(question)
        found_text = " ".join(found).lower()
        answer_bits = tokenize(answer)

        missing = []
        for t in q_tokens:
            if t not in answer_bits and t not in found_text:
                missing.append(t)
        return missing

    # ---- Documents worth checking ----------------------------------------
    def _candidate_documents(self, source_chunks: List[RetrievalResult]) -> List[dict]:
        """Deduplicated source documents, ranked by their top chunk score."""
        seen = {}
        for r in source_chunks:
            meta = r.metadata or {}
            doc_id = meta.get("doc_id") or meta.get("source_file") or "unknown"
            title = meta.get("source_file") or doc_id
            score = (
                r.rerank_score
                if r.rerank_score is not None
                else (r.rrf_score if r.rrf_score is not None else r.dense_score)
            )
            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "title": title, "score": score or 0.0}
            else:
                seen[doc_id]["score"] = max(seen[doc_id]["score"], score or 0.0)

        ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
        return ranked[: self.candidate_docs]

    def _summarize(
        self, question: str, found: List[str], gaps: List[str]
    ) -> str:
        parts = []
        if found:
            parts.append(
                f"I found {len(found)} relevant passage(s) related to your question, "
                "but I am not confident enough to give a definitive answer."
            )
        else:
            parts.append(
                "I was unable to find relevant content for this question in the "
                "indexed documents."
            )
        if gaps:
            parts.append(
                "The following aspects could not be confirmed: "
                + ", ".join(repr(g) for g in gaps)
                + "."
            )
        parts.append(
            "Please check the listed documents manually rather than relying on "
            "an AI-generated answer."
        )
        return " ".join(parts)


def tokenize(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        out.add(w)
    return out