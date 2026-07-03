from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any


UP_TERMS = {
    "increase", "increased", "increases", "rise", "rose", "rising", "surge",
    "surged", "gain", "gained", "growth", "grew", "outperform",
    "outperformed", "bullish", "rally", "rallied", "rebound", "rebounded",
}
DOWN_TERMS = {
    "decrease", "decreased", "decreases", "decline", "declined", "drop",
    "dropped", "fall", "fell", "falling", "loss", "lost", "underperform",
    "underperformed", "bearish", "slump", "slumped", "plunge", "plunged",
    "downturn",
}
STABLE_TERMS = {"stable", "flat", "unchanged", "steady", "stabilized", "stabilised", "remained", "maintained"}
VOLATILE_TERMS = {"volatile", "volatility", "fluctuated", "fluctuation", "mixed", "choppy", "uncertain", "swings", "turbulence"}
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", normalize_text(text))
    return [t for t in tokens if t not in STOPWORDS]


def detect_relative_intent(query_text: str, rewrite_type: str | None = None) -> str:
    q = normalize_text(query_text)
    if any(p in q for p in ("earlier period", "earlier periods", "early period", "early years", "earlier years", "before", "prior to", "previously", "initial period", "beginning")):
        return "earlier"
    if any(p in q for p in ("recent years", "most recent", "recently", "latest", "current period")):
        return "recent"
    if any(p in q for p in ("later period", "later periods", "later years", "after", "following", "subsequent", "toward the end", "by the end")):
        return "later"
    if any(p in q for p in ("over time", "overall", "generally trend", "general trend", "multi-year", "across years", "throughout the period", "long-term", "historical trend")):
        return "broad"
    if rewrite_type == "A":
        return "broad_unknown"
    return "unknown"


def compute_trend_signals(chunk_texts: list[str]) -> dict[str, Any]:
    counts = Counter()
    for text in chunk_texts:
        counts.update(tokenize(text))
    up_count = sum(counts[t] for t in UP_TERMS)
    down_count = sum(counts[t] for t in DOWN_TERMS)
    stable_count = sum(counts[t] for t in STABLE_TERMS)
    volatile_count = sum(counts[t] for t in VOLATILE_TERMS)
    scores = {
        "up": min(1.0, up_count / 3),
        "down": min(1.0, down_count / 3),
        "stable": min(1.0, stable_count / 3),
        "volatile": min(1.0, volatile_count / 3),
    }
    dominant = max(scores, key=scores.get)
    if scores[dominant] == 0:
        dominant = "unknown"
    trend_terms = []
    for terms in (UP_TERMS, DOWN_TERMS, STABLE_TERMS, VOLATILE_TERMS):
        trend_terms.extend([t for t in terms if counts[t] > 0])
    trend_terms = sorted(trend_terms, key=lambda t: counts[t], reverse=True)[:12]
    return {
        "trend_signal_score": max(scores.values()),
        "up_signal_score": scores["up"],
        "down_signal_score": scores["down"],
        "stable_signal_score": scores["stable"],
        "volatile_signal_score": scores["volatile"],
        "dominant_trend_direction": dominant,
        "top_trend_terms": trend_terms,
    }


def safe_mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def safe_std(values: list[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) > 1 else default


def compute_inside_coherence(values: list[float]) -> float:
    if not values:
        return 0.0
    return 1 / (1 + safe_std(values))


def compute_boundary_contrast(years: list[int], relevance_by_year: dict[int, float], start_year: int, end_year: int, context_size: int) -> dict[str, float]:
    inside_left_years = list(range(start_year, min(end_year, start_year + context_size - 1) + 1))
    inside_right_years = list(range(max(start_year, end_year - context_size + 1), end_year + 1))
    left_years = [y for y in range(start_year - context_size, start_year) if y in years]
    right_years = [y for y in range(end_year + 1, end_year + context_size + 1) if y in years]
    inside_left = safe_mean([relevance_by_year.get(y, 0.0) for y in inside_left_years])
    inside_right = safe_mean([relevance_by_year.get(y, 0.0) for y in inside_right_years])
    left_context = safe_mean([relevance_by_year.get(y, 0.0) for y in left_years]) if left_years else 0.0
    right_context = safe_mean([relevance_by_year.get(y, 0.0) for y in right_years]) if right_years else 0.0
    left_contrast = inside_left - left_context if left_years else 0.0
    right_contrast = inside_right - right_context if right_years else 0.0
    positives = []
    if left_years:
        positives.append(max(0.0, left_contrast))
    if right_years:
        positives.append(max(0.0, right_contrast))
    return {
        "left_context_relevance": left_context,
        "right_context_relevance": right_context,
        "left_boundary_contrast": left_contrast,
        "right_boundary_contrast": right_contrast,
        "boundary_contrast": safe_mean(positives) if positives else 0.0,
    }


def compute_trend_coherence(directions: list[str]) -> dict[str, Any]:
    known = [d for d in directions if d and d != "unknown"]
    if not known:
        return {"trend_coherence": 0.0, "dominant_window_trend_direction": "unknown"}
    counts = Counter(known)
    direction, count = counts.most_common(1)[0]
    return {"trend_coherence": count / len(known), "dominant_window_trend_direction": direction}


def compute_length_fit(window_length: int) -> float:
    if window_length <= 2:
        return 0.20
    if 3 <= window_length <= 4:
        return 0.70
    if 5 <= window_length <= 8:
        return 1.00
    if 9 <= window_length <= 10:
        return 0.75
    return 0.45


def compute_intent_fit(relative_intent: str, window_type: str, normalized_center: float, window_length: int, timeline_length: int) -> float:
    if window_type == "content_supported_window":
        return 0.15
    coverage = window_length / max(1, timeline_length)
    if relative_intent == "earlier":
        if window_type in {"prefix_window", "early_half"}:
            return 0.95
        return 0.85 if normalized_center <= 0.45 else 0.45 if normalized_center <= 0.65 else 0.20
    if relative_intent in {"later", "recent"}:
        if window_type in {"suffix_window", "late_half", "recent_5y"}:
            return 0.95 if window_type != "recent_5y" or relative_intent == "recent" else 0.85
        return 0.85 if normalized_center >= 0.55 else 0.45 if normalized_center >= 0.35 else 0.20
    if relative_intent == "broad":
        if 0.45 <= coverage <= 0.80:
            return 0.90
        if 0.30 <= coverage < 0.45:
            return 0.75
        return 0.60 if coverage > 0.80 else 0.40
    if 0.30 <= normalized_center <= 0.70 and 5 <= window_length <= 8:
        return 0.85
    return 0.65 if 5 <= window_length <= 10 else 0.45
