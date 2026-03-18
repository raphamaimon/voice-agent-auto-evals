"""Scoring logic — single source of truth.

The composite score formula and metrics aggregation live here.
No other module computes scores.
"""

from __future__ import annotations

from .config import ScoringConfig
from .models import EvalResult, Metrics


def composite_score(
    should_results: list[dict],
    should_not_results: list[dict],
    weights: ScoringConfig,
    avg_latency_ms: float = 0.0,
    voice_quality: dict | None = None,
    severity: str = "medium",
) -> tuple[float, float, float, float]:
    """Compute composite score from per-criterion results.

    When weights.latency_weight == 0.0, latency is ignored entirely.
    When weights.quality_weight == 0.0, voice quality is ignored.

    Severity multiplier amplifies the failure portion of the score:
    a critical failure drags the score down harder than a low-severity one.

    Returns:
        (composite, should_score, should_not_score, quality_score)
    """
    s_score = (
        sum(1 for c in should_results if c.get("passed"))
        / max(len(should_results), 1)
    )
    sn_score = (
        sum(1 for c in should_not_results if c.get("passed"))
        / max(len(should_not_results), 1)
    )

    # Quality score from voice quality dimensions (each 0-10, normalized to 0-1)
    quality_score = 0.0
    if voice_quality and weights.quality_weight > 0:
        dims = [
            voice_quality.get(d, {}).get("score", 5)
            for d in ("brevity", "naturalness", "tone", "consistency")
        ]
        quality_score = (sum(dims) / len(dims)) / 10.0

    composite = (
        weights.should_weight * s_score
        + weights.should_not_weight * sn_score
        + weights.quality_weight * quality_score
    )

    # Only include latency if it has non-zero weight
    if weights.latency_weight > 0 and weights.latency_threshold_ms > 0:
        lat_score = 1.0 if avg_latency_ms < weights.latency_threshold_ms else 0.5
        composite += weights.latency_weight * lat_score

    # Severity multiplier: amplify the failure portion for critical scenarios
    multiplier = weights.severity_multipliers.get(severity, 1.0)
    if multiplier != 1.0 and composite < 1.0:
        failure_portion = 1.0 - composite
        composite = 1.0 - (failure_portion * multiplier)
        composite = max(0.0, composite)

    return composite, s_score, sn_score, quality_score


def aggregate(results: list[EvalResult]) -> Metrics:
    """Aggregate per-scenario eval results into summary metrics.

    Uses per-scenario weights when available (default 1.0).
    """
    if not results:
        return Metrics(0.0, 0.0, 0.0, 0, 0)

    total_weight = sum(r.weight for r in results)
    avg_score = sum(r.score * r.weight for r in results) / total_weight
    avg_csat = sum(r.csat_score * r.weight for r in results) / total_weight
    n_passed = sum(1 for r in results if r.passed)
    failures: set[str] = set()
    for r in results:
        failures.update(r.failure_modes)

    return Metrics(
        avg_score=avg_score,
        avg_csat=avg_csat,
        pass_rate=n_passed / len(results),
        n_passed=n_passed,
        n_total=len(results),
        unique_failures=sorted(failures),
    )
