"""Prometheus evaluation suite — DeepEval + Phoenix integration (Sprint 13)."""

from prometheus.evals.golden_dataset import (
    GoldenTask,
    TaskTier,
    load_golden_dataset,
)
from prometheus.evals.judge import PrometheusJudge, JudgeVerdict
from prometheus.evals.metrics import (
    TaskCompletionMetric,
    ToolUsageMetric,
    NoHallucinationMetric,
)
from prometheus.evals.runner import EvalRunner, EvalResult, MetricScore
from prometheus.evals.trends import TrendTracker, TrendRow

__all__ = [
    "GoldenTask",
    "TaskTier",
    "load_golden_dataset",
    "PrometheusJudge",
    "JudgeVerdict",
    "TaskCompletionMetric",
    "ToolUsageMetric",
    "NoHallucinationMetric",
    "EvalRunner",
    "EvalResult",
    "MetricScore",
    "TrendTracker",
    "TrendRow",
]
