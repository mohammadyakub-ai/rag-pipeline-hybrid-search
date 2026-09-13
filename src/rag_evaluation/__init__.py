"""rag_evaluation: golden datasets, evaluation metrics, and chunking
strategy comparisons.

This package is intentionally separated from ``rag_pipeline`` (retrieval /
generation) so evaluation tooling can be reused without dragging in the
inference stack and vice-versa.
"""

from .golden_dataset import GoldenQuestion, GoldenDataset, CATEGORIES
from .eval_metrics import (
    MetricResult,
    EvalQuestionResult,
    EvalReport,
    CorrectnessScorer,
    FaithfulnessScorer,
    RetrievalRelevanceScorer,
    CitationAccuracyScorer,
    EvalRunner,
)
from .chunking_comparison import StrategyRow, ComparisonResult, ChunkingComparison, METRICS

__all__ = [
    "GoldenQuestion",
    "GoldenDataset",
    "CATEGORIES",
    "MetricResult",
    "EvalQuestionResult",
    "EvalReport",
    "CorrectnessScorer",
    "FaithfulnessScorer",
    "RetrievalRelevanceScorer",
    "CitationAccuracyScorer",
    "EvalRunner",
    "StrategyRow",
    "ComparisonResult",
    "ChunkingComparison",
    "METRICS",
]