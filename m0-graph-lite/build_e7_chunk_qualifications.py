#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from db_utils import add_db_args, connect_db, ensure_output_dirs, execute_sql_file, fetch_dataframe, write_dataframe_csv


DEFAULT_SOURCE_BUILD_VERSION = "graph_lite_v2_hybrid"
DEFAULT_QUALIFICATION_BUILD_VERSION = "e7b_rule_v1"

TREND_TERMS = {
    "increase", "increased", "increases", "rise", "rose", "rising", "surge", "surged",
    "gain", "gained", "growth", "grew", "decline", "declined", "drop", "dropped",
    "fall", "fell", "loss", "lost", "rebound", "rebounded", "recover", "recovered",
    "stable", "flat", "unchanged", "steady", "stabilized", "remained", "volatile",
    "volatility", "fluctuated", "mixed", "outperform", "outperformed", "underperform", "underperformed",
    "shift", "shifted", "accelerated", "slowdown", "improvement", "deterioration",
}
EVENT_TERMS = {
    "crisis", "recession", "pandemic", "covid", "shock", "acquisition", "merger",
    "launch", "announced", "announcement", "policy", "rate hike", "rate cut",
    "bankruptcy", "lawsuit", "split", "dividend", "earnings", "guidance",
}
COMPARISON_TERMS = {
    "compared with", "compared to", "versus", "vs.", "before", "after",
    "year over year", "yoy", "qoq", "prior year", "previous year",
}
CONTINUITY_TERMS = {"continued", "remained", "persisted", "sustained", "maintained", "still"}
START_TERMS = {"began", "started", "launched", "introduced", "initial", "emerged", "entered"}
TURNING_TERMS = {"reversed", "shifted", "turned", "accelerated", "disrupted", "crisis", "shock", "transition"}
END_TERMS = {"ended", "completed", "stabilized", "stabilised", "recovered", "weakened", "slowed"}

YEAR_RE = re.compile(r"\b(?:FY)?(?:19|20)\d{2}\b", re.IGNORECASE)
METRIC_RE = re.compile(
    r"(?:[$]\s?\d|\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:million|billion|trillion|bps|points|shares)\b)",
    re.IGNORECASE,
)
FROM_TO_RE = re.compile(r"\bfrom\b.{1,80}\bto\b", re.IGNORECASE)


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def ticker_in_text(ticker: str, text: str) -> bool:
    if not ticker:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text or "", re.IGNORECASE) is not None


def extract_year_or_period(text: str) -> str | None:
    match = YEAR_RE.search(text or "")
    if match:
        return match.group(0)
    lowered = normalize(text)
    for phrase in ("during the year", "over the period", "in recent years", "before", "after", "prior year", "previous year"):
        if phrase in lowered:
            return phrase
    return None


def qualify_chunk(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("chunk_text") or ""
    lowered = normalize(text)
    ticker = row.get("ticker") or ""

    entity_match = bool(row.get("ticker_match")) or ticker_in_text(ticker, text)
    text_year_or_period = extract_year_or_period(text)
    text_temporal_anchor = bool(text_year_or_period)

    signal_types: list[str] = []
    if contains_any(lowered, TREND_TERMS):
        signal_types.append("trend")
    if contains_any(lowered, EVENT_TERMS):
        signal_types.append("event")
    if METRIC_RE.search(text):
        signal_types.append("metric")
    if contains_any(lowered, COMPARISON_TERMS) or FROM_TO_RE.search(text):
        signal_types.append("comparison")
    if contains_any(lowered, CONTINUITY_TERMS):
        signal_types.append("continuity")

    temporal_signal = bool(signal_types)
    metadata_temporal_anchor = bool(row.get("year") is not None and entity_match and temporal_signal)
    temporal_anchor = text_temporal_anchor or metadata_temporal_anchor
    if text_temporal_anchor and metadata_temporal_anchor:
        temporal_anchor_strength = "both"
    elif text_temporal_anchor:
        temporal_anchor_strength = "text"
    elif metadata_temporal_anchor:
        temporal_anchor_strength = "metadata"
    else:
        temporal_anchor_strength = "none"

    boundary_value = "none"
    if entity_match and temporal_anchor and temporal_signal:
        if any(term in lowered for term in START_TERMS):
            boundary_value = "possible_start"
        elif any(term in lowered for term in TURNING_TERMS):
            boundary_value = "turning_point"
        elif any(term in lowered for term in END_TERMS):
            boundary_value = "possible_end"
        elif any(term in lowered for term in CONTINUITY_TERMS):
            boundary_value = "continuation"
        elif "comparison" in signal_types and ("trend" in signal_types or "event" in signal_types):
            boundary_value = "turning_point"
        else:
            boundary_value = "inside_but_not_boundary"

    if entity_match and temporal_anchor and temporal_signal and boundary_value in {"possible_start", "continuation", "turning_point", "possible_end"}:
        evidence_label = "boundary_temporal_evidence"
        confidence = 0.84 if len(signal_types) >= 2 else 0.74
    elif entity_match and temporal_anchor and temporal_signal:
        evidence_label = "general_temporal_evidence"
        confidence = 0.74
    elif entity_match:
        evidence_label = "supporting_context"
        boundary_value = "outside_or_background"
        confidence = 0.62
    else:
        evidence_label = "low_value_noise"
        boundary_value = "none"
        confidence = 0.70

    if evidence_label == "boundary_temporal_evidence":
        feature_usage = {
            "use_for_inside_relevance": True,
            "use_for_evidence_coverage": True,
            "use_for_boundary_contrast": True,
            "use_for_trend_coherence": True,
        }
    elif evidence_label == "general_temporal_evidence":
        feature_usage = {
            "use_for_inside_relevance": True,
            "use_for_evidence_coverage": True,
            "use_for_boundary_contrast": False,
            "use_for_trend_coherence": False,
        }
    elif evidence_label == "supporting_context":
        feature_usage = {
            "use_for_inside_relevance": True,
            "use_for_evidence_coverage": False,
            "use_for_boundary_contrast": False,
            "use_for_trend_coherence": False,
        }
    else:
        feature_usage = {
            "use_for_inside_relevance": False,
            "use_for_evidence_coverage": False,
            "use_for_boundary_contrast": False,
            "use_for_trend_coherence": False,
        }

    if evidence_label == "boundary_temporal_evidence":
        reason = "The chunk has entity, temporal, and boundary or trend-chain evidence, so it may influence all E6b features."
    elif evidence_label == "general_temporal_evidence":
        reason = "The chunk has entity-specific temporal evidence, but lacks boundary or trend-chain value."
    elif evidence_label == "supporting_context":
        reason = "The chunk is entity-related but lacks enough temporal evidence for coverage, boundary, or trend features."
    else:
        reason = "The chunk lacks a clear entity match or usable temporal evidence."

    return {
        "entity_match": entity_match,
        "text_temporal_anchor": text_temporal_anchor,
        "metadata_temporal_anchor": metadata_temporal_anchor,
        "temporal_anchor_strength": temporal_anchor_strength,
        "temporal_anchor": temporal_anchor,
        "temporal_signal": temporal_signal,
        "signal_types": signal_types,
        "boundary_value": boundary_value,
        "evidence_label": evidence_label,
        "qualification_label": evidence_label,
        **feature_usage,
        "confidence": confidence,
        "extracted_evidence": {
            "entity": ticker if entity_match else None,
            "year_or_period": text_year_or_period or (str(row.get("year")) if metadata_temporal_anchor else None),
            "trend_or_change": "trend signal present" if "trend" in signal_types else None,
            "event_or_cause": "event signal present" if "event" in signal_types else None,
            "metric_or_value": "metric signal present" if "metric" in signal_types else None,
            "comparison": "comparison signal present" if "comparison" in signal_types else None,
        },
        "short_reason": reason,
    }


def apply_schema(conn) -> None:
    path = Path(__file__).resolve().parent / "sql" / "007_create_e7_qualification_tables.sql"
    execute_sql_file(conn, path)
    print(f"applied {path}")


def clear_existing(conn, source_build_version: str, qualification_build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM m0_graph.chunk_temporal_qualifications
            WHERE source_build_version = %s AND qualification_build_version = %s
            """,
            (source_build_version, qualification_build_version),
        )
    conn.commit()


def fetch_source_chunks(conn, source_build_version: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT *
            FROM m0_graph.hybrid_query_relevant_chunks
            WHERE build_version = %s
            ORDER BY m0_input_id, year, rank_in_year
            LIMIT %s
            """,
            (source_build_version, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def insert_qualification(conn, row: dict[str, Any], result: dict[str, Any], qualification_build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.chunk_temporal_qualifications (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year, chunk_id, doc_id,
                source_build_version, qualification_build_version, entity_match, text_temporal_anchor,
                metadata_temporal_anchor, temporal_anchor_strength, temporal_anchor, temporal_signal,
                signal_types, boundary_value, evidence_label, qualification_label, use_for_inside_relevance,
                use_for_evidence_coverage, use_for_boundary_contrast, use_for_trend_coherence, confidence,
                extracted_evidence, short_reason
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
            ON CONFLICT (m0_input_id, year, chunk_id, source_build_version, qualification_build_version)
            DO UPDATE SET
                entity_match = EXCLUDED.entity_match,
                text_temporal_anchor = EXCLUDED.text_temporal_anchor,
                metadata_temporal_anchor = EXCLUDED.metadata_temporal_anchor,
                temporal_anchor_strength = EXCLUDED.temporal_anchor_strength,
                temporal_anchor = EXCLUDED.temporal_anchor,
                temporal_signal = EXCLUDED.temporal_signal,
                signal_types = EXCLUDED.signal_types,
                boundary_value = EXCLUDED.boundary_value,
                evidence_label = EXCLUDED.evidence_label,
                qualification_label = EXCLUDED.qualification_label,
                use_for_inside_relevance = EXCLUDED.use_for_inside_relevance,
                use_for_evidence_coverage = EXCLUDED.use_for_evidence_coverage,
                use_for_boundary_contrast = EXCLUDED.use_for_boundary_contrast,
                use_for_trend_coherence = EXCLUDED.use_for_trend_coherence,
                confidence = EXCLUDED.confidence,
                extracted_evidence = EXCLUDED.extracted_evidence,
                short_reason = EXCLUDED.short_reason
            """,
            (
                row["m0_input_id"], row["bridge_id"], row["ticker"], row["query_text"], row["rewrite_type"],
                row["year"], row["chunk_id"], row["doc_id"], row["build_version"], qualification_build_version,
                result["entity_match"], result["text_temporal_anchor"], result["metadata_temporal_anchor"],
                result["temporal_anchor_strength"], result["temporal_anchor"], result["temporal_signal"],
                dump(result["signal_types"]), result["boundary_value"], result["evidence_label"],
                result["qualification_label"], result["use_for_inside_relevance"],
                result["use_for_evidence_coverage"], result["use_for_boundary_contrast"],
                result["use_for_trend_coherence"], result["confidence"], dump(result["extracted_evidence"]),
                result["short_reason"],
            ),
        )


def build_qualifications(conn, source_build_version: str, qualification_build_version: str, limit: int) -> None:
    for row in fetch_source_chunks(conn, source_build_version, limit):
        insert_qualification(conn, row, qualify_chunk(row), qualification_build_version)
    conn.commit()


def export_outputs(conn) -> None:
    exports = {
        "e7_chunk_temporal_qualifications.csv": "SELECT * FROM m0_graph.chunk_temporal_qualifications ORDER BY m0_input_id, year, chunk_id",
        "e7_chunk_qualification_summary.csv": "SELECT * FROM m0_graph.v_e7_chunk_qualification_summary ORDER BY rewrite_type, qualification_build_version",
    }
    for name, sql in exports.items():
        write_dataframe_csv(fetch_dataframe(conn, sql), Path("m0-graph-lite/outputs/csv") / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--source-build-version", default=DEFAULT_SOURCE_BUILD_VERSION)
    parser.add_argument("--qualification-build-version", default=DEFAULT_QUALIFICATION_BUILD_VERSION)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    with connect_db(args) as conn:
        apply_schema(conn)
        if args.clear_existing:
            clear_existing(conn, args.source_build_version, args.qualification_build_version)
        build_qualifications(conn, args.source_build_version, args.qualification_build_version, args.limit)
        export_outputs(conn)


if __name__ == "__main__":
    main()
