from __future__ import annotations

import re
from typing import Any

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "over",
    "should", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "while", "with", "within", "would",
}

MARKET_SIGNAL_WORDS = {
    "price", "stock", "trend", "performance", "movement", "increase", "increased",
    "decrease", "decreased", "rise", "rose", "rising", "fall", "fell", "decline",
    "declined", "growth", "volatility", "yield", "market", "return", "gain",
    "loss", "bullish", "bearish", "outperform", "underperform",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_.$-]*|\d+(?:\.\d+)?%?", text or "")
    tokens = []
    for token in raw:
        norm = token.lower().strip(".$-")
        if len(norm) < 2 or norm in STOPWORDS:
            continue
        tokens.append(norm)
    return tokens


def compute_lexical_score(query_text: str, chunk_text: str) -> float:
    query_tokens = sorted(set(tokenize(query_text)))
    chunk_tokens = set(tokenize(chunk_text))
    if not query_tokens:
        return 0.0
    return sum(1 for token in query_tokens if token in chunk_tokens) / len(query_tokens)


def compute_signal_bonus(query_text: str, chunk_text: str, ticker: str | None) -> dict[str, Any]:
    chunk = chunk_text or ""
    chunk_tokens = set(tokenize(chunk_text))
    ticker_match = bool(ticker and re.search(rf"\b{re.escape(ticker)}\b", chunk, flags=re.IGNORECASE))
    market_hits = sorted(MARKET_SIGNAL_WORDS & chunk_tokens)
    market_signal_match = bool(market_hits)
    numeric_signal_match = bool(re.search(r"(\d+(?:\.\d+)?%|\$?\d+(?:\.\d+)?)", chunk))
    bonus = 0.0
    if ticker_match:
        bonus += 0.05
    if market_signal_match:
        bonus += 0.05 if len(market_hits) >= 2 else 0.03
    if numeric_signal_match:
        bonus += 0.03
    bonus = min(0.10, bonus)
    reasons = []
    if ticker_match:
        reasons.append("ticker_match")
    if market_signal_match:
        reasons.append("market_signal=" + ",".join(market_hits[:4]))
    if numeric_signal_match:
        reasons.append("numeric_signal")
    return {
        "signal_bonus": bonus,
        "ticker_match": ticker_match,
        "market_signal_match": market_signal_match,
        "numeric_signal_match": numeric_signal_match,
        "match_reason": "; ".join(reasons) if reasons else "no_signal",
    }


def compute_dense_similarity(query_embedding, chunk_embedding) -> float:
    q = np.asarray(query_embedding, dtype=float)
    c = np.asarray(chunk_embedding, dtype=float)
    denom = float(np.linalg.norm(q) * np.linalg.norm(c))
    if denom == 0:
        return 0.0
    cosine = float(np.dot(q, c) / denom)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def compute_hybrid_relevance(query_text: str, chunk_text: str, ticker: str | None, dense_similarity_score: float | None = None) -> dict[str, Any]:
    lexical = compute_lexical_score(query_text, chunk_text)
    signal = compute_signal_bonus(query_text, chunk_text, ticker)
    signal_norm = min(1.0, signal["signal_bonus"] / 0.10) if signal["signal_bonus"] else 0.0
    if dense_similarity_score is None:
        hybrid = 0.75 * lexical + 0.25 * signal_norm
    else:
        hybrid = 0.35 * lexical + 0.55 * dense_similarity_score + 0.10 * signal_norm
    return {
        "lexical_score": lexical,
        "dense_similarity_score": dense_similarity_score,
        "signal_bonus": signal["signal_bonus"],
        "hybrid_relevance_score": max(0.0, min(1.0, hybrid)),
        "ticker_match": signal["ticker_match"],
        "market_signal_match": signal["market_signal_match"],
        "numeric_signal_match": signal["numeric_signal_match"],
        "match_reason": signal["match_reason"],
    }
