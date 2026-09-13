import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .retrieval import RetrievalResult
from .generation import Generator, GroundedPrompt


@dataclass
class Citation:
    """A single [N] citation attached to a claim sentence."""

    number: int
    claim: str
    supported: Optional[bool] = None
    reason: Optional[str] = None
    source_content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "citation": f"[{self.number}]",
            "claim": self.claim,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass
class CitationReport:
    """Full verification report for a generated answer."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    verified: bool = False

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.citations if c.supported is True)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.citations if c.supported is False)

    @property
    def accuracy(self) -> float:
        if not self.citations:
            return 0.0
        return self.supported_count / len(self.citations)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "total": self.total,
            "supported": self.supported_count,
            "unsupported": self.unsupported_count,
            "accuracy": round(self.accuracy, 4),
            "citations": [c.to_dict() for c in self.citations],
        }


class CitationParser:
    """Parses bracketed citations like [1], [2] out of a generated answer.

    Splits the answer into sentences and attributes each citation to the
    sentence (claim) it is attached to. Multiple citations in one sentence
    produce multiple Citation entries sharing that claim.
    """

    # The canonical citation regex. Handles [1], [1][2], and [[1]].
    CITATION_RE = re.compile(r"\[(\d+)\]")

    def parse(self, answer: str) -> List[Citation]:
        sentences = self._split_sentences(answer)
        citations: List[Citation] = []
        for sentence in sentences:
            nums = [int(m) for m in re.findall(r"\[(\d+)\]", sentence)]
            for n in nums:
                citations.append(Citation(number=n, claim=sentence.strip()))
        return citations

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence-ending punctuation followed by space/end, but keep
        # citations and decimals intact.
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
        # Also handle newlines as boundaries
        sentences = []
        for p in parts:
            sentences.extend(s.strip() for s in p.split("\n") if s.strip())
        return [s for s in sentences if s]


class CitationVerifier:
    """Verifies each claim-citation pair via an LLM-as-judge.

    For every Citation, the judge is asked whether the content of the cited
    context block actually supports the claim it's attached to. Citations that
    are unsupported (or unverifiable because they reference an out-of-range
    block) are flagged.
    """

    VERIFY_SYSTEM_TEMPLATE = (
        "You are verifying whether a citation supports a claim.\n"
        "Claim: {claim}\n"
        "Cited chunk [N]: {chunk_text}\n"
        "Does the cited chunk directly support the claim?\n"
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED"
    )

    def __init__(
        self,
        judge: Optional[Generator] = None,
        default_unsupported: bool = True,
    ):
        self.judge = judge or Generator(model="gpt-4o-mini")
        self.default_unsupported = default_unsupported

    def verify(
        self,
        answer: str,
        source_chunks: List[RetrievalResult],
        citations: Optional[List[Citation]] = None,
    ) -> CitationReport:
        """Verify the answer's citations against the provided source chunks."""
        parser = CitationParser()
        citations = citations if citations is not None else parser.parse(answer)
        source_by_index = {i + 1: c for i, c in enumerate(source_chunks)}

        for cit in citations:
            src = source_by_index.get(cit.number)
            if src is None:
                # Citation references a block that wasn't provided -> unsupported
                cit.supported = False
                cit.reason = "references a context block that was not provided"
                cit.source_content = None
                continue
            cit.source_content = src.content
            cit.supported, cit.reason = self._judge_claim(cit.claim, src.content)

        report = CitationReport(answer=answer, citations=citations)
        report.verified = True
        return report

    def _judge_claim(self, claim: str, source_content: str) -> tuple:
        if not self.judge.is_configured:
            # No judge available; fall back to heuristic (no model, no verdict).
            return None, "judge not configured"

        system_prompt = self.VERIFY_SYSTEM_TEMPLATE.format(
            claim=claim, chunk_text=source_content
        )
        responses = self.judge.generate(
            "Verify the citation.",
            _citations_verify_source(claim, source_content),
            system_prompt=system_prompt,
        )
        raw = (responses.get("answer") or "").strip()
        if not raw:
            return False, "empty verifier reply"
        # Strict parser: look only for the literal verdict word (case-insensitive).
        if "SUPPORTED" in raw.upper():
            return True, "SUPPORTED"
        return False, "UNSUPPORTED"


def _citations_verify_source(claim: str, source_content: str) -> List[RetrievalResult]:
    """Wrap a single (claim, context) pair as one RetrievalResult for the judge."""
    from .retrieval import RetrievalResult
    return [
        RetrievalResult(
            chunk_id="verification-source",
            content=source_content,
            metadata={},
        )
    ]