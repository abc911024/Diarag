from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "over",
    "should", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "while", "with", "within", "would",
}

IMPORTANT_PHRASES = (
    "trend", "price trend", "stock price", "performance", "movement",
    "volatility", "increase", "decrease", "rise", "fall", "decline", "growth",
)

MARKET_TERMS = (
    "stock", "market", "shares", "price", "prices", "etf", "fund", "yield",
    "return", "returns", "revenue", "earnings", "portfolio", "bond", "equity",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_.$-]*|\d+(?:\.\d+)?%?", text or "")
    tokens = []
    for token in raw:
        norm = token.lower().strip(".$-")
        if len(norm) < 2:
            continue
        if norm in STOPWORDS:
            continue
        tokens.append(norm)
    return tokens


def compute_lexical_relevance(query_text: str, chunk_text: str, ticker: str | None = None) -> dict[str, Any]:
    query_norm = normalize_text(query_text)
    chunk_norm = normalize_text(chunk_text)
    query_tokens = sorted(set(tokenize(query_text)))
    chunk_tokens = set(tokenize(chunk_text))
    if not query_tokens:
        overlap_score = 0.0
    else:
        overlap_score = sum(1 for token in query_tokens if token in chunk_tokens) / len(query_tokens)

    phrase_hits = [phrase for phrase in IMPORTANT_PHRASES if phrase in query_norm and phrase in chunk_norm]
    phrase_bonus = min(0.15, 0.04 * len(phrase_hits))

    ticker_match = False
    if ticker:
        ticker_match = bool(re.search(rf"\b{re.escape(ticker)}\b", chunk_text or "", flags=re.IGNORECASE))
    ticker_bonus = 0.10 if ticker_match else 0.0

    numeric_signal = bool(re.search(r"(\d+(?:\.\d+)?%|\$?\d+(?:\.\d+)?)", chunk_text or ""))
    market_signal = any(term in chunk_tokens for term in MARKET_TERMS)
    market_bonus = (0.05 if numeric_signal else 0.0) + (0.05 if market_signal else 0.0)

    lexical_score = overlap_score
    relevance_score = max(0.0, min(1.0, overlap_score + phrase_bonus + ticker_bonus + market_bonus))
    reasons = []
    if overlap_score > 0:
        reasons.append(f"token_overlap={overlap_score:.2f}")
    if phrase_hits:
        reasons.append("phrase=" + ",".join(phrase_hits[:3]))
    if ticker_match:
        reasons.append("ticker_match")
    if numeric_signal:
        reasons.append("numeric_signal")
    if market_signal:
        reasons.append("market_terms")
    return {
        "relevance_score": relevance_score,
        "lexical_score": lexical_score,
        "ticker_match": ticker_match,
        "match_reason": "; ".join(reasons) if reasons else "no_match",
    }
