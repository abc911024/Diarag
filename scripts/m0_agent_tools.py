from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from psycopg.rows import dict_row


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def make_mock_embedding(text: str, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    tokens = (text or "").lower().split() or [""]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            vector[i] += (digest[i % len(digest)] / 255.0) - 0.5
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / norm, 6) for v in vector]


def get_m0_input(conn, m0_input_id: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                mi.m0_input_id, mi.bridge_id, mi.m0_input_setting, mi.m0_input_json,
                mi.rewrite_type, mi.temporal_need_type, mi.evidence_policy,
                br.rewritten_query, br.ticker, br.stock_name
            FROM qa.m0_inputs mi
            JOIN qa.bridge_records br ON br.bridge_id = mi.bridge_id
            WHERE mi.m0_input_id = %s
            """,
            (m0_input_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"m0 input not found: {m0_input_id}")
    return dict(row)


def get_entity_coverage(conn, ticker: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ticker, min_year, max_year, available_years
            FROM qa.entity_coverage
            WHERE entity_id = %s OR ticker = %s
            LIMIT 1
            """,
            (ticker, ticker),
        )
        row = cur.fetchone()
    return dict(row) if row else {}


def get_gold_target(conn, bridge_id: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT gold_start_year, gold_end_year, gold_range, acceptable_ranges
            FROM qa.temporal_annotations
            WHERE bridge_id = %s
            """,
            (bridge_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}


def get_content_available_years(conn, ticker: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT document_year, COUNT(*) AS chunk_count, COUNT(DISTINCT doc_id) AS doc_count
            FROM content.evidence_chunks
            WHERE mentioned_tickers @> %s::jsonb
              AND document_year IS NOT NULL
            GROUP BY document_year
            ORDER BY document_year
            """,
            (json.dumps([ticker]),),
        )
        rows = cur.fetchall()
    years = [int(r["document_year"]) for r in rows]
    return {
        "content_available_years": years,
        "matched_chunk_count": sum(int(r["chunk_count"]) for r in rows),
        "matched_doc_count": sum(int(r["doc_count"]) for r in rows),
        "chunk_count_by_year": {int(r["document_year"]): int(r["chunk_count"]) for r in rows},
    }


def diagnose_query(m0_input: dict[str, Any]) -> dict[str, Any]:
    rewrite_type = m0_input.get("rewrite_type")
    query = (m0_input.get("rewritten_query") or "").lower()
    if rewrite_type == "A":
        label = "no_time_info"
    elif rewrite_type == "B":
        label = "relative_expression"
    elif rewrite_type == "C":
        label = "event_anchor"
    else:
        label = "unknown"
    relative_intent = detect_relative_intent(query)
    return {
        "rewrite_type": rewrite_type,
        "temporal_need_type": m0_input.get("temporal_need_type", "historical_trend"),
        "has_explicit_time": False,
        "needs_entity_coverage": True,
        "needs_content_probe": True,
        "time_intent_label": label,
        "relative_intent": relative_intent,
    }


def detect_relative_intent(query: str) -> str:
    q = (query or "").lower()
    if any(phrase in q for phrase in ("earlier periods", "earlier period", "early period", "early years", "before")):
        return "earlier"
    if any(phrase in q for phrase in ("later periods", "later period", "later years", "recent years", "most recent")):
        return "later"
    if any(phrase in q for phrase in ("over time", "multi-year", "across years", "throughout the period", "overall trend")):
        return "broad"
    return "unknown"


def _add_window(windows: list[dict[str, Any]], seen: set[tuple[int, int, str]], window_type: str, start: int, end: int, method: str, rationale: str) -> None:
    if start is None or end is None or start > end:
        return
    key = (int(start), int(end), window_type)
    if key in seen:
        return
    seen.add(key)
    windows.append({
        "window_type": window_type,
        "start_year": int(start),
        "end_year": int(end),
        "candidate_range": [int(start), int(end)],
        "generation_method": method,
        "is_visible_to_model": True,
        "uses_gold_label": False,
        "rationale": rationale,
    })


def _add_window_v2(
    windows: list[dict[str, Any]],
    seen: set[tuple[int, int, str]],
    window_type: str,
    start: int,
    end: int,
    method: str,
    priority_hint: str,
    relative_intent: str,
    rationale: str,
    use_as_final_candidate: bool = True,
) -> None:
    if start is None or end is None or int(start) > int(end):
        return
    key = (int(start), int(end), window_type)
    if key in seen:
        return
    seen.add(key)
    windows.append({
        "candidate_range": [int(start), int(end)],
        "start_year": int(start),
        "end_year": int(end),
        "window_type": window_type,
        "generation_method": method,
        "priority_hint": priority_hint,
        "relative_intent": relative_intent,
        "is_visible_to_model": True,
        "uses_gold_label": False,
        "use_as_final_candidate": use_as_final_candidate,
        "candidate_version": "v2",
        "rationale": rationale,
    })


def generate_candidate_windows(m0_input: dict[str, Any], entity_coverage: dict[str, Any], content_years: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    min_year = entity_coverage.get("min_year")
    max_year = entity_coverage.get("max_year")
    if min_year is None or max_year is None:
        return []
    min_year, max_year = int(min_year), int(max_year)
    mid = (min_year + max_year) // 2
    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    query = (m0_input.get("rewritten_query") or "").lower()
    _add_window(windows, seen, "full_entity_range", min_year, max_year, "heuristic_entity_coverage", "Full entity range from QA entity coverage.")
    _add_window(windows, seen, "recent_5y", max(min_year, max_year - 4), max_year, "heuristic_entity_coverage", "Recent five-year window.")
    _add_window(windows, seen, "early_half", min_year, mid, "heuristic_entity_coverage", "Early half of entity coverage.")
    _add_window(windows, seen, "late_half", mid + 1, max_year, "heuristic_entity_coverage", "Late half of entity coverage.")
    if m0_input.get("rewrite_type") == "B":
        if "earlier periods" in query:
            _add_window(windows, seen, "relative_before", min_year, mid, "relative_expression_rule", "Relative phrase suggests earlier periods.")
        elif "later periods" in query:
            _add_window(windows, seen, "relative_after", mid + 1, max_year, "relative_expression_rule", "Relative phrase suggests later periods.")
        elif "over time" in query or "multi-year" in query:
            _add_window(windows, seen, "relative_multi_year", min_year, max_year, "relative_expression_rule", "Relative phrase suggests broad multi-year coverage.")
    payload = m0_input.get("m0_input_json") or {}
    candidate_range = payload.get("candidate_range") if isinstance(payload, dict) else None
    if isinstance(candidate_range, list) and len(candidate_range) == 2:
        _add_window(windows, seen, "verification_candidate", int(candidate_range[0]), int(candidate_range[1]), "verification_input", "Candidate range supplied in verification input.")
    content_available = (content_years or {}).get("content_available_years") or []
    if content_available:
        _add_window(windows, seen, "content_supported_window", min(content_available), max(content_available), "content_coverage_window", "Window spanning content years for ticker.")
    priority = {"relative_before": 0, "relative_after": 0, "relative_multi_year": 0, "full_entity_range": 1}
    return sorted(windows, key=lambda w: priority.get(w["window_type"], 2))


def generate_candidate_windows_v2(
    m0_input: dict[str, Any],
    entity_coverage: dict[str, Any],
    content_years: dict[str, Any] | None = None,
    max_windows: int = 80,
) -> list[dict[str, Any]]:
    min_year = entity_coverage.get("min_year")
    max_year = entity_coverage.get("max_year")
    available_years = [int(y) for y in _as_list(entity_coverage.get("available_years"))]
    if min_year is None or max_year is None:
        return []
    min_year, max_year = int(min_year), int(max_year)
    if not available_years:
        available_years = list(range(min_year, max_year + 1))
    relative_intent = detect_relative_intent(m0_input.get("rewritten_query") or "")
    mid = (min_year + max_year) // 2
    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    _add_window_v2(windows, seen, "full_entity_range", min_year, max_year, "heuristic_entity_coverage", "broad", relative_intent, "Full entity range.")
    _add_window_v2(windows, seen, "recent_5y", max(min_year, max_year - 4), max_year, "heuristic_entity_coverage", "later", relative_intent, "Recent five-year window.")
    _add_window_v2(windows, seen, "early_half", min_year, mid, "heuristic_entity_coverage", "earlier", relative_intent, "Early half of entity years.")
    _add_window_v2(windows, seen, "late_half", mid + 1, max_year, "heuristic_entity_coverage", "later", relative_intent, "Late half of entity years.")

    for size in (3, 5, 8):
        if len(available_years) < size:
            continue
        for i in range(0, len(available_years) - size + 1):
            start = available_years[i]
            end = available_years[i + size - 1]
            if end - start + 1 != size:
                continue
            hint = "earlier" if start <= min_year + 1 else "later" if end >= max_year - 1 else "middle"
            _add_window_v2(windows, seen, f"rolling_{size}y", start, end, "rolling_window", hint, relative_intent, f"Rolling {size}-year candidate.")

    for end in available_years:
        if end - min_year + 1 >= 3:
            hint = "earlier" if end <= mid + 1 else "broad"
            _add_window_v2(windows, seen, "prefix_window", min_year, end, "prefix_window", hint, relative_intent, "Prefix window from entity start.")

    for start in available_years:
        if max_year - start + 1 >= 3:
            hint = "later" if start >= mid else "broad"
            _add_window_v2(windows, seen, "suffix_window", start, max_year, "suffix_window", hint, relative_intent, "Suffix window to entity end.")

    content_available = (content_years or {}).get("content_available_years") or []
    if content_available:
        _add_window_v2(
            windows, seen, "content_supported_window", min(content_available), max(content_available),
            "content_coverage_window", "evidence_availability", relative_intent,
            "Window spanning available content years for ticker.", use_as_final_candidate=False,
        )

    payload = m0_input.get("m0_input_json") or {}
    candidate_range = payload.get("candidate_range") if isinstance(payload, dict) else None
    if isinstance(candidate_range, list) and len(candidate_range) == 2:
        _add_window_v2(
            windows, seen, "verification_candidate", int(candidate_range[0]), int(candidate_range[1]),
            "verification_input", "verification", relative_intent, "Candidate range from verification input.",
        )

    def priority(window: dict[str, Any]) -> tuple[int, int]:
        wtype = window["window_type"]
        start = window["start_year"]
        end = window["end_year"]
        if relative_intent == "earlier":
            order = {"early_half": 0, "prefix_window": 1, "rolling_5y": 2, "rolling_8y": 2}
            return (order.get(wtype, 5), start)
        if relative_intent == "later":
            order = {"late_half": 0, "suffix_window": 1, "recent_5y": 2, "rolling_5y": 3, "rolling_8y": 3}
            return (order.get(wtype, 5), -end)
        if relative_intent == "broad":
            order = {"full_entity_range": 0, "rolling_8y": 1, "prefix_window": 2, "suffix_window": 2}
            return (order.get(wtype, 5), -(end - start))
        return (0 if wtype in {"rolling_8y", "full_entity_range"} else 2, start)

    return sorted(windows, key=priority)[:max_windows]


def probe_content_metadata(conn, query_text: str, ticker: str, start_year: int, end_year: int) -> dict[str, Any]:
    years = list(range(int(start_year), int(end_year) + 1))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT document_year, COUNT(*) AS chunk_count, COUNT(DISTINCT doc_id) AS doc_count
            FROM content.evidence_chunks
            WHERE document_year BETWEEN %s AND %s
              AND mentioned_tickers @> %s::jsonb
            GROUP BY document_year
            ORDER BY document_year
            """,
            (start_year, end_year, json.dumps([ticker])),
        )
        rows = cur.fetchall()
    density = {int(r["document_year"]): int(r["chunk_count"]) for r in rows}
    years_with = sorted(density)
    missing = [year for year in years if year not in density]
    matched_chunks = sum(density.values())
    matched_docs = sum(int(r["doc_count"]) for r in rows)
    if matched_chunks == 0:
        status = "empty"
        action = "no_content_support"
    elif len(years_with) == len(years):
        status = "sufficient"
        action = "keep_range"
    else:
        status = "partial"
        action = "expand_start_year" if missing and missing[0] == start_year else "expand_end_year"
    return {
        "matched_chunk_count": matched_chunks,
        "matched_doc_count": matched_docs,
        "years_with_evidence": years_with,
        "missing_years": missing,
        "evidence_density": density,
        "coverage_status": status,
        "recommended_action": action,
    }


def vector_probe_content(conn, query_text: str, ticker: str, start_year: int, end_year: int, top_k: int = 10, allow_fallback: bool = False) -> dict[str, Any]:
    query_vec = vector_literal(make_mock_embedding(query_text, 8))
    sql = """
        SELECT chunk_id, doc_id, document_year, mentioned_tickers, embedding <=> %s::vector AS distance
        FROM content.evidence_chunks
        WHERE document_year BETWEEN %s AND %s
          AND mentioned_tickers @> %s::jsonb
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [query_vec, start_year, end_year, json.dumps([ticker]), query_vec, top_k]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        used_fallback = False
        if allow_fallback and not rows:
            cur.execute(
                """
                SELECT chunk_id, doc_id, document_year, mentioned_tickers, embedding <=> %s::vector AS distance
                FROM content.evidence_chunks
                WHERE document_year BETWEEN %s AND %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, start_year, end_year, query_vec, top_k),
            )
            rows = cur.fetchall()
            used_fallback = True
    years = sorted({int(r["document_year"]) for r in rows if r["document_year"] is not None})
    return {
        "retrieved_chunk_ids": [r["chunk_id"] for r in rows],
        "retrieved_doc_ids": sorted({r["doc_id"] for r in rows}),
        "retrieved_years": years,
        "matched_chunk_count": len(rows),
        "coverage_status": "sufficient" if rows else ("no_ticker_match" if not used_fallback else "empty"),
        "used_fallback": used_fallback,
    }


def score_candidate_window(candidate: dict[str, Any], metadata_probe: dict[str, Any], vector_probe: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    window_type = candidate["window_type"]
    rewrite_type = diagnosis.get("rewrite_type")
    intent_fit = 0.65
    if rewrite_type == "A" and window_type in {"full_entity_range", "content_supported_window"}:
        intent_fit = 1.0
    elif rewrite_type == "B" and window_type in {"relative_before", "relative_after", "relative_multi_year", "early_half", "late_half"}:
        intent_fit = 0.95
    elif rewrite_type == "C" and window_type == "verification_candidate":
        intent_fit = 0.85
    years_total = max(1, int(candidate["end_year"]) - int(candidate["start_year"]) + 1)
    metadata_coverage_score = min(1.0, len(metadata_probe.get("years_with_evidence", [])) / years_total)
    vector_support_score = min(1.0, vector_probe.get("matched_chunk_count", 0) / 5.0)
    length = years_total
    window_efficiency_score = max(0.35, 1.0 - max(0, length - 8) * 0.03)
    score = 0.35 * intent_fit + 0.30 * metadata_coverage_score + 0.25 * vector_support_score + 0.10 * window_efficiency_score
    return {
        "score": score,
        "score_components": {
            "intent_fit": intent_fit,
            "metadata_coverage_score": metadata_coverage_score,
            "vector_support_score": vector_support_score,
            "window_efficiency_score": window_efficiency_score,
        },
        "rationale": f"{window_type} scored {score:.3f}",
    }


def _window_length(candidate: dict[str, Any]) -> int:
    return int(candidate["end_year"]) - int(candidate["start_year"]) + 1


def _intent_fit_v2(candidate: dict[str, Any], diagnosis: dict[str, Any]) -> float:
    wtype = candidate["window_type"]
    rewrite_type = diagnosis.get("rewrite_type")
    relative_intent = candidate.get("relative_intent") or diagnosis.get("relative_intent") or "unknown"
    start = int(candidate["start_year"])
    if rewrite_type == "A" or (rewrite_type == "B" and relative_intent == "unknown"):
        return {
            "rolling_8y": 0.85,
            "full_entity_range": 0.80,
            "prefix_window": 0.70,
            "suffix_window": 0.65,
            "rolling_5y": 0.65,
            "early_half": 0.60,
            "late_half": 0.60,
            "recent_5y": 0.55,
            "content_supported_window": 0.50,
        }.get(wtype, 0.55)
    if rewrite_type == "B":
        if relative_intent == "earlier":
            if wtype in {"early_half", "prefix_window"}:
                return 0.90
            if wtype in {"rolling_5y", "rolling_8y"} and candidate.get("priority_hint") == "earlier":
                return 0.85
            if wtype == "full_entity_range":
                return 0.65
            if wtype in {"late_half", "recent_5y"}:
                return 0.25
            if wtype == "suffix_window":
                return 0.30
            return 0.55
        if relative_intent == "later":
            if wtype in {"late_half", "suffix_window"}:
                return 0.90
            if wtype == "recent_5y":
                return 0.80
            if wtype == "full_entity_range":
                return 0.65
            if wtype == "early_half":
                return 0.25
            if wtype == "prefix_window" and candidate.get("priority_hint") == "earlier":
                return 0.30
            return 0.55
        if relative_intent == "broad":
            return {
                "full_entity_range": 0.90,
                "rolling_8y": 0.85,
                "rolling_5y": 0.70,
                "prefix_window": 0.70,
                "suffix_window": 0.70,
                "recent_5y": 0.45,
            }.get(wtype, 0.60)
    if rewrite_type == "C":
        return 0.75 if wtype == "verification_candidate" else 0.55
    return 0.55


def score_candidate_window_v2(candidate: dict[str, Any], metadata_probe: dict[str, Any], vector_probe: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    length = _window_length(candidate)
    intent_fit = _intent_fit_v2(candidate, diagnosis)
    if length <= 2:
        temporal_policy_fit = 0.20
    elif 3 <= length <= 5:
        temporal_policy_fit = 0.60
    elif 6 <= length <= 9:
        temporal_policy_fit = 0.90
    elif 10 <= length <= 12:
        temporal_policy_fit = 0.70
    else:
        temporal_policy_fit = 0.50
    if metadata_probe.get("matched_chunk_count", 0) == 0:
        evidence_coverage = 0.0
    else:
        evidence_coverage = min(1.0, len(metadata_probe.get("years_with_evidence", [])) / max(1, length))
    if 3 <= length <= 9:
        window_efficiency = 1.0
    elif 10 <= length <= 12:
        window_efficiency = 0.75
    elif length > 12:
        window_efficiency = 0.50
    else:
        window_efficiency = 0.30
    if diagnosis.get("rewrite_type") == "A" and length >= 10:
        window_efficiency = max(window_efficiency, 0.82)
    retrieved = vector_probe.get("matched_chunk_count", 0)
    top_k = vector_probe.get("top_k", 10) or 10
    vector_support = 1.0 if retrieved >= top_k else 0.5 if retrieved > 0 else 0.0
    score = (
        0.45 * intent_fit
        + 0.20 * temporal_policy_fit
        + 0.20 * evidence_coverage
        + 0.10 * window_efficiency
        + 0.05 * vector_support
    )
    return {
        "score": score,
        "score_components": {
            "intent_fit": intent_fit,
            "temporal_policy_fit": temporal_policy_fit,
            "evidence_coverage": evidence_coverage,
            "window_efficiency": window_efficiency,
            "vector_support": vector_support,
        },
        "rationale": (
            f"{candidate['window_type']} v2 score={score:.3f}; "
            f"intent={intent_fit:.2f}, evidence={evidence_coverage:.2f}, length={length}"
        ),
    }


def choose_best_window(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_candidates:
        return {}
    def key(item: dict[str, Any]) -> tuple[float, int, int]:
        cand = item["candidate"]
        probe = item.get("metadata_probe", {})
        full_bonus = 1 if cand["window_type"] == "full_entity_range" else 0
        return (item["score"], len(probe.get("years_with_evidence", [])), full_bonus)
    return max(scored_candidates, key=key)


def choose_best_window_v2(scored_candidates: list[dict[str, Any]], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if not scored_candidates:
        return {}
    nonzero = [c for c in scored_candidates if c.get("metadata_probe", {}).get("matched_chunk_count", 0) > 0]
    pool = nonzero or scored_candidates
    best_score = max(c["score"] for c in pool)
    near = [c for c in pool if best_score - c["score"] <= 0.03]
    relative_intent = diagnosis.get("relative_intent", "unknown")
    rewrite_type = diagnosis.get("rewrite_type")

    # Content-supported windows are evidence diagnostics. Only keep them if clearly better.
    non_content_best = max((c["score"] for c in pool if c["candidate"]["window_type"] != "content_supported_window"), default=None)
    if non_content_best is not None:
        near = [
            c for c in near
            if c["candidate"]["window_type"] != "content_supported_window"
            or c["score"] > non_content_best + 0.08
        ] or near

    def tiebreak(item: dict[str, Any]) -> tuple[float, float, float, float]:
        cand = item["candidate"]
        length = _window_length(cand)
        evidence_ratio = item.get("score_components", {}).get("evidence_coverage", 0.0)
        length_fit = 1.0 if 5 <= length <= 9 else 0.8 if 3 <= length <= 12 else 0.4
        directional = 0.0
        if rewrite_type == "B" and relative_intent == "earlier":
            directional = -cand["start_year"] / 10000.0
            if cand["window_type"] in {"early_half", "prefix_window"}:
                directional += 1.0
        elif rewrite_type == "B" and relative_intent == "later":
            directional = cand["end_year"] / 10000.0
            if cand["window_type"] in {"late_half", "suffix_window", "recent_5y"}:
                directional += 1.0
        elif rewrite_type == "A":
            if cand["window_type"] in {"rolling_8y", "prefix_window"}:
                directional = 1.0
            elif cand["window_type"] == "full_entity_range":
                directional = 0.8
        return (length_fit, directional, evidence_ratio, 0.0 if cand.get("uses_gold_label") else 1.0)

    chosen = max(near, key=tiebreak)
    chosen["rationale"] = chosen.get("rationale", "") + " Selected by v2 tie-break policy."
    return chosen


def _valid_range(r: Any) -> bool:
    return isinstance(r, list) and len(r) == 2 and r[0] is not None and r[1] is not None and int(r[0]) <= int(r[1])


def _year_set(r: list[int]) -> set[int]:
    return set(range(int(r[0]), int(r[1]) + 1))


def compute_m0_metrics(pred_range: Any, gold_range: Any, acceptable_ranges: Any) -> dict[str, Any]:
    if not _valid_range(pred_range) or not _valid_range(gold_range):
        return {
            "overlap_score": 0.0, "start_boundary_error": None, "end_boundary_error": None,
            "mean_boundary_error": None, "acceptable_accuracy": False, "exact_match_accuracy": False,
            "prediction_length_error": None, "over_extension": None, "under_extension": None,
        }
    pred = [int(pred_range[0]), int(pred_range[1])]
    gold = [int(gold_range[0]), int(gold_range[1])]
    pred_years, gold_years = _year_set(pred), _year_set(gold)
    start_error = abs(pred[0] - gold[0])
    end_error = abs(pred[1] - gold[1])
    acceptable = any(_valid_range(r) and pred == [int(r[0]), int(r[1])] for r in _as_list(acceptable_ranges))
    return {
        "overlap_score": len(pred_years & gold_years) / len(pred_years | gold_years),
        "start_boundary_error": start_error,
        "end_boundary_error": end_error,
        "mean_boundary_error": (start_error + end_error) / 2.0,
        "acceptable_accuracy": acceptable,
        "exact_match_accuracy": pred == gold,
        "prediction_length_error": abs(len(pred_years) - len(gold_years)),
        "over_extension": len(pred_years - gold_years),
        "under_extension": len(gold_years - pred_years),
    }
