#!/usr/bin/env python3
"""
Bridge Benchmark Pipeline v3
Leakage-aware Implicit Diachronic Temporal Scope Inference Dataset Construction

Core design:
1. Annotation Layer may use metadata to build gold labels.
2. Rewrite Layer is text-only and must not receive gold labels or temporal metadata.

This version fixes three issues from the earlier v3 draft:
- anchor_year / anchor_direction are extracted from context_details_for_sampling when available.
- Type C event-anchor rewrite is conservative and only generated from explicit event-anchor text.
- Type A/B fallback rewrites are more natural and quality checks are stricter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Constants
# =============================================================================

DIACHRONIC_KEYWORDS = [
    "trend",
    "movement",
    "change",
    "changed",
    "evolution",
    "evolve",
    "evolved",
    "fluctuation",
    "fluctuate",
    "fluctuated",
    "performance",
    "comparison",
    "compare",
    "compared",
    "development",
    "develop",
    "moved",
    "move",
]

FACTOID_PATTERNS = [
    r"\bwhat\s+was\s+the\s+(?:exact\s+)?(?:price|value|closing\s+price|opening\s+price)\b",
    r"\bhow\s+much\s+was\b",
    r"\bwhat\s+is\s+the\s+(?:price|value)\b",
]

EVENT_NOUNS = [
    "merger",
    "announcement",
    "policy announcement",
    "policy change",
    "covid-19 shock",
    "covid shock",
    "pandemic",
    "earnings call",
    "earnings announcement",
    "acquisition",
    "pledge",
    "shock",
    "crash",
    "bankruptcy",
    "ipo",
    "stock split",
    "split",
    "dividend announcement",
    "restructuring",
    "regulation",
    "law",
    "deal",
    "contract",
]

QUALITY_INTENT_KEYWORDS = DIACHRONIC_KEYWORDS + ["respond", "response", "behave", "behavior"]


# =============================================================================
# 1. Load Dataset
# =============================================================================

def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array, JSONL file, or a JSON object containing a list field."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Dataset file is empty: {path}")

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("data", "items", "records", "questions"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            raise ValueError("JSON object does not contain a supported list field: data/items/records/questions")
    except json.JSONDecodeError:
        # Fall back to JSONL.
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {lineno} is not an object")
            records.append(obj)
        return records

    raise ValueError("Unsupported JSON dataset format")


# =============================================================================
# 2. Candidate Filtering
# =============================================================================

def parse_date(date_str: Any) -> Optional[datetime]:
    """Parse YYYY-MM-DD date strings."""
    if not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def looks_factoid_only(question: str) -> bool:
    """Conservative factoid-only detector used before accepting a candidate."""
    q = question.lower()
    has_diachronic = any(kw in q for kw in DIACHRONIC_KEYWORDS)
    if has_diachronic:
        return False
    return any(re.search(pattern, q) for pattern in FACTOID_PATTERNS)


def is_valid_bridge_candidate(item: dict[str, Any], min_days: int = 365) -> tuple[bool, str]:
    """
    Decide whether a source item is a valid Bridge Benchmark candidate.

    This filtering step may use metadata because it belongs to dataset construction.
    It is not used to generate rewritten queries.
    """
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        return False, "missing question"

    question_lower = question.lower()

    query_type_info = item.get("query_type_info") or {}
    query_type = str(query_type_info.get("type", ""))
    if query_type == "Specific Time Period":
        return False, "excluded query type: 'Specific Time Period' is a single-period question"

    if looks_factoid_only(question):
        return False, "excluded factoid-only question"

    has_diachronic_keyword = any(kw in question_lower for kw in DIACHRONIC_KEYWORDS)
    if not has_diachronic_keyword:
        return False, "not a diachronic question: missing trend/change/movement intent"

    bounds = item.get("plot_time_bounds") or {}
    start_date = parse_date(bounds.get("start_date"))
    end_date = parse_date(bounds.get("end_date"))
    if start_date is None or end_date is None:
        return False, f"missing or invalid plot_time_bounds: {bounds}"

    if start_date > end_date:
        return False, f"invalid date range: start_date after end_date ({bounds})"

    duration_days = (end_date - start_date).days
    if duration_days <= min_days:
        return False, f"duration {duration_days} days <= min_days {min_days}"

    return True, f"valid diachronic candidate over {duration_days} days"


# =============================================================================
# 3. Annotation Layer
# =============================================================================

def extract_year(date_str: Any) -> Optional[int]:
    """Extract year from a YYYY-MM-DD string."""
    if not isinstance(date_str, str):
        return None
    match = re.match(r"^(\d{4})-\d{2}-\d{2}$", date_str)
    return int(match.group(1)) if match else None


def normalize_temporal_annotation(item: dict[str, Any]) -> dict[str, Any]:
    """
    Build gold temporal annotation from DQABench metadata.

    This function may use metadata because it produces evaluation labels.
    Do not pass its output into rewrite generation except for allowed text fields
    such as original_query / ticker / stock_name.
    """
    bounds = item.get("plot_time_bounds") or {}
    start_date = bounds.get("start_date")
    end_date = bounds.get("end_date")
    start_year = extract_year(start_date)
    end_year = extract_year(end_date)

    query_type_info = item.get("query_type_info") or {}
    context = item.get("context_details_for_sampling") or {}
    query_type = str(query_type_info.get("type", ""))

    if query_type == "Before":
        temporal_scope_type = "before_anchor"
    elif query_type == "After":
        temporal_scope_type = "after_anchor"
    elif (
        any(token in query_type for token in ("Range", "Multi", "Period", "Comparison"))
        or all(k in context for k in ("start_year", "end_year"))
        or all(k in context for k in ("start_date", "end_date"))
    ):
        temporal_scope_type = "explicit_range"
    else:
        temporal_scope_type = "other_diachronic"

    annotated: dict[str, Any] = {
        "source_dataset": "DQABench",
        "source_id": item.get("id", ""),
        "original_query": item.get("question", ""),
        "ticker": item.get("ticker", ""),
        "stock_name": item.get("stock_name", ""),
        "source_temporal_label": item.get("temporal_context_label", ""),
        "gold_start_date": start_date,
        "gold_end_date": end_date,
        "gold_range": [start_year, end_year],
        "temporal_scope_type": temporal_scope_type,
        "annotation_uses_metadata": True,
    }

    # Corrected: anchor metadata is often stored inside context_details_for_sampling.
    anchor_year = context.get("anchor_year", query_type_info.get("anchor_year", item.get("anchor_year")))
    if anchor_year is not None:
        annotated["anchor_year"] = anchor_year

    anchor_direction = (
        context.get("direction")
        or query_type_info.get("direction")
        or query_type_info.get("anchor_direction")
        or item.get("anchor_direction")
    )
    if anchor_direction:
        annotated["anchor_direction"] = anchor_direction

    return annotated


# =============================================================================
# 4. Acceptable Range Generation
# =============================================================================

def generate_acceptable_ranges(gold_range: list[int | None]) -> list[list[int]]:
    """Generate simple ±1-year acceptable ranges for intrinsic M0 evaluation."""
    if (
        not isinstance(gold_range, list)
        or len(gold_range) != 2
        or gold_range[0] is None
        or gold_range[1] is None
    ):
        return []

    start_year = int(gold_range[0])
    end_year = int(gold_range[1])
    candidates = [
        [start_year, end_year],
        [start_year - 1, end_year],
        [start_year, end_year + 1],
        [start_year - 1, end_year + 1],
    ]

    seen: set[tuple[int, int]] = set()
    ranges: list[list[int]] = []
    for start, end in candidates:
        if start > end:
            continue
        key = (start, end)
        if key not in seen:
            ranges.append([start, end])
            seen.add(key)
    return ranges


# =============================================================================
# 5. Leakage-aware Rewrite Layer: Text Only
# =============================================================================

TEMPORAL_PATTERNS = [
    # Longer range forms first.
    r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
    r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
    r"\bover\s+\d{4}\s*[-–—]\s*\d{4}\b",
    r"\b(?:before|after|since|until|during|in|over|following)\s+\d{4}\b",
    r"\bprior\s+to\s+\d{4}\b",
]


def clean_question_text(text: str) -> str:
    """Clean spaces and punctuation after regex removal."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?!.,;:])", r"\1", text)
    # Remove stranded prepositions near the end if a temporal phrase was removed.
    text = re.sub(r"\b(for|during|in|over|before|after|since|until)\s*\?$", "?", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if text and not text.endswith("?"):
        text = text.rstrip(".,;:") + "?"
    return text


def pseudo_rewrite_text_only(original_query: str) -> str:
    """
    Remove explicit temporal expressions from original_query only.

    This function intentionally accepts no metadata. It should not know the
    gold range, temporal scope type, anchor year, or plot_time_bounds.
    """
    query = original_query or ""
    for pattern in TEMPORAL_PATTERNS:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE)
    return clean_question_text(query)


def extract_temporal_pattern_from_query(original_query: str) -> str:
    """Infer a relative-time rewrite category from the original query text only."""
    q = (original_query or "").lower()
    if re.search(r"\bfrom\s+\d{4}\s+to\s+\d{4}\b", q):
        return "range"
    if re.search(r"\bbetween\s+\d{4}\s+and\s+\d{4}\b", q):
        return "range"
    if re.search(r"\bover\s+\d{4}\s*[-–—]\s*\d{4}\b", q):
        return "range"
    if re.search(r"\b(before|until)\s+\d{4}\b", q) or re.search(r"\bprior\s+to\s+\d{4}\b", q):
        return "before"
    if re.search(r"\b(after|since|following)\s+\d{4}\b", q):
        return "after"
    return "other"


def build_natural_base_query(pseudo_query: str, ticker: Optional[str], stock_name: Optional[str]) -> str:
    """Build a more natural no-time query from the pseudo query."""
    entity = (ticker or stock_name or "the stock").strip()
    pseudo = clean_question_text(pseudo_query).rstrip("?").strip()
    pseudo_lower = pseudo.lower()

    entity_escaped = re.escape(entity)

    # Common DQABench stock-price question patterns.
    if re.search(rf"\bwhat\s+was\s+the\s+(?:predominant\s+)?(?:general\s+)?stock\s+price\s+trend\s+for\s+{entity_escaped}\b", pseudo, flags=re.IGNORECASE):
        return f"How did {entity}'s stock price generally trend?"

    if re.search(rf"\bwhat\s+was\s+{entity_escaped}'?s\s+(?:predominant\s+)?stock\s+price\s+trend\b", pseudo, flags=re.IGNORECASE):
        return f"How did {entity}'s stock price generally trend?"

    if re.search(rf"\bwhat\s+was\s+the\s+(?:general\s+)?stock\s+price\s+movement\s+for\s+{entity_escaped}\b", pseudo, flags=re.IGNORECASE):
        return f"How did {entity}'s stock price generally move?"

    if re.search(rf"\bwhat\s+was\s+{entity_escaped}'?s\s+(?:general\s+)?stock\s+price\s+movement\b", pseudo, flags=re.IGNORECASE):
        return f"How did {entity}'s stock price generally move?"

    if re.search(r"\bstock\s+price\b", pseudo_lower) and "trend" in pseudo_lower and entity.lower() in pseudo_lower:
        return f"How did {entity}'s stock price generally trend?"

    if re.search(r"\bstock\s+price\b", pseudo_lower) and any(w in pseudo_lower for w in ("movement", "move", "moved")) and entity.lower() in pseudo_lower:
        return f"How did {entity}'s stock price generally move?"

    if re.search(r"\bstock\s+price\b", pseudo_lower) and "change" in pseudo_lower and entity.lower() in pseudo_lower:
        return f"How did {entity}'s stock price change?"

    if re.search(r"\bstock\s+price\b", pseudo_lower) and "performance" in pseudo_lower and entity.lower() in pseudo_lower:
        return f"How did {entity}'s stock price perform?"

    # Fallback: preserve pseudo query but clean it.
    return clean_question_text(pseudo)


def add_relative_phrase(base_query: str, relative_phrase: str) -> str:
    """Add a relative temporal phrase to a natural base question."""
    base = (base_query or "").rstrip("?").strip()
    # Avoid awkward duplication if the base already has a broad temporal phrase.
    base = re.sub(r"\b(over\s+time|in\s+earlier\s+periods|in\s+later\s+periods|over\s+a\s+multi-year\s+period)\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    return clean_question_text(f"{base} {relative_phrase}")


def extract_event_anchor_phrase(original_query: str) -> Optional[str]:
    """
    Extract an explicit event-anchor phrase from the original query text only.

    Conservative rule: a valid Type C anchor must contain a temporal preposition
    plus an event noun. Year-only anchors such as "before 2020" are excluded.
    """
    q = original_query or ""

    # Exclude direct year anchors from Type C.
    if re.search(r"\b(?:before|after|following|since|until)\s+\d{4}\b", q, flags=re.IGNORECASE):
        # It may still contain another event phrase elsewhere, so do not return yet.
        pass

    event_union = "|".join(re.escape(noun) for noun in sorted(EVENT_NOUNS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b(?P<prep>after|before|following|since|until|prior\s+to)\s+"
        rf"(?P<anchor>(?:(?:the|a|an)\s+)?(?:[A-Za-z0-9\-]+\s+){{0,5}}(?:{event_union})(?:\s+[A-Za-z0-9\-]+){{0,3}})",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(q):
        anchor = match.group("anchor").strip()
        # Do not treat a bare year as an event anchor.
        if re.fullmatch(r"\d{4}", anchor):
            continue
        # Clean trailing punctuation and extra words.
        anchor = anchor.strip(" ?.,;:")
        # Avoid returning over-generic anchor.
        if anchor.lower() in {"event", "the event", "a event", "an event"}:
            continue
        return anchor

    return None


def generate_rewrite_variants_text_only(
    original_query: str,
    pseudo_query: str,
    ticker: Optional[str] = None,
    stock_name: Optional[str] = None,
    include_type_c: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate Type A/B/C rewrite variants from text-only inputs.

    Forbidden by design: no full item dict, no plot_time_bounds, no gold_range,
    no temporal_scope_type, no query_type_info.
    """
    variants: list[dict[str, Any]] = []

    # Type A: no explicit or relative time expression.
    type_a_query = build_natural_base_query(pseudo_query, ticker, stock_name)
    variants.append(
        {
            "rewrite_type": "A",
            "rewritten_query": type_a_query,
            "rewrite_status": "generated",
            "rewrite_reason": "No-time rewrite generated from original query text only.",
        }
    )

    # Type B: relative-time expression derived from original query text pattern.
    temporal_pattern = extract_temporal_pattern_from_query(original_query)
    if temporal_pattern == "before":
        relative_phrase = "in earlier periods"
    elif temporal_pattern == "after":
        relative_phrase = "in later periods"
    elif temporal_pattern == "range":
        relative_phrase = "over a multi-year period"
    else:
        relative_phrase = "over time"

    type_b_query = add_relative_phrase(type_a_query, relative_phrase)
    variants.append(
        {
            "rewrite_type": "B",
            "rewritten_query": type_b_query,
            "rewrite_status": "generated",
            "rewrite_reason": f"Relative-time rewrite uses '{relative_phrase}' derived from original query text pattern '{temporal_pattern}'.",
        }
    )

    # Type C: event-anchor expression only if original query text contains an event anchor.
    if include_type_c:
        event_anchor = extract_event_anchor_phrase(original_query)
        if event_anchor:
            entity = (ticker or stock_name or "the stock").strip()
            event_phrase = event_anchor
            if not event_phrase.lower().startswith(("the ", "a ", "an ")):
                event_phrase = f"the {event_phrase}"
            type_c_query = f"How did {entity}'s stock price respond around {event_phrase}?"
            variants.append(
                {
                    "rewrite_type": "C",
                    "rewritten_query": type_c_query,
                    "rewrite_status": "generated",
                    "rewrite_reason": f"Event-anchor rewrite generated from explicit event phrase in original query: {event_anchor}",
                }
            )
        else:
            variants.append(
                {
                    "rewrite_type": "C",
                    "rewritten_query": None,
                    "rewrite_status": "not_applicable",
                    "rewrite_reason": "No explicit event anchor appears in the original query.",
                }
            )

    return variants


# =============================================================================
# 6. Rewrite Quality Check
# =============================================================================

def check_rewrite_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Check whether a rewrite record satisfies v3 leakage-aware quality rules."""
    rewritten_query = record.get("rewritten_query")
    rewrite_status = record.get("rewrite_status")
    ticker = str(record.get("ticker") or "")
    stock_name = str(record.get("stock_name") or "")
    source_temporal_label = str(record.get("source_temporal_label") or "")

    flags = {
        "contains_explicit_year": False,
        "contains_original_temporal_label": False,
        "keeps_entity": False,
        "keeps_diachronic_intent": False,
        "is_non_empty_question": False,
    }

    if rewrite_status == "not_applicable":
        return {"rewrite_quality_flags": flags, "rewrite_quality_status": "not_applicable"}

    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        return {"rewrite_quality_flags": flags, "rewrite_quality_status": "fail"}

    q = rewritten_query.strip()
    q_lower = q.lower()
    flags["is_non_empty_question"] = True
    flags["contains_explicit_year"] = bool(re.search(r"\b\d{4}\b", q))

    if source_temporal_label:
        flags["contains_original_temporal_label"] = source_temporal_label.lower() in q_lower

    if ticker and re.search(rf"\b{re.escape(ticker)}\b", q, flags=re.IGNORECASE):
        flags["keeps_entity"] = True
    elif stock_name and stock_name.lower() in q_lower:
        flags["keeps_entity"] = True

    flags["keeps_diachronic_intent"] = any(keyword in q_lower for keyword in QUALITY_INTENT_KEYWORDS)

    if flags["contains_explicit_year"] or flags["contains_original_temporal_label"]:
        status = "fail"
    elif flags["is_non_empty_question"] and flags["keeps_entity"] and flags["keeps_diachronic_intent"]:
        status = "pass"
    else:
        status = "warning"

    return {"rewrite_quality_flags": flags, "rewrite_quality_status": status}


# =============================================================================
# 7. Record Construction
# =============================================================================

def construct_final_record(
    annotated_item: dict[str, Any],
    pseudo_query: str,
    variant: dict[str, Any],
    bridge_index: int,
) -> dict[str, Any]:
    """Merge annotation and rewrite fields into one final benchmark record."""
    rewrite_type = variant.get("rewrite_type", "?")
    record: dict[str, Any] = {
        "bridge_id": f"BRIDGE_{bridge_index:06d}_{rewrite_type}",
        "source_id": annotated_item.get("source_id", ""),
        "source_dataset": annotated_item.get("source_dataset", "DQABench"),
        "original_query": annotated_item.get("original_query", ""),
        "pseudo_query": pseudo_query,
        "rewritten_query": variant.get("rewritten_query"),
        "rewrite_type": rewrite_type,
        "gold_range": annotated_item.get("gold_range", []),
        "gold_start_date": annotated_item.get("gold_start_date"),
        "gold_end_date": annotated_item.get("gold_end_date"),
        "acceptable_ranges": annotated_item.get("acceptable_ranges", []),
        "temporal_scope_type": annotated_item.get("temporal_scope_type"),
        "source_temporal_label": annotated_item.get("source_temporal_label"),
        "anchor_year": annotated_item.get("anchor_year"),
        "anchor_direction": annotated_item.get("anchor_direction"),
        "ticker": annotated_item.get("ticker"),
        "stock_name": annotated_item.get("stock_name"),
        "rewrite_input_policy": "text_only",
        "metadata_used_for_rewrite": False,
        "annotation_uses_metadata": True,
        "rewrite_status": variant.get("rewrite_status", "unknown"),
        "rewrite_reason": variant.get("rewrite_reason", ""),
    }

    quality = check_rewrite_quality(record)
    record.update(quality)
    record["rationale"] = (
        "The gold temporal range is derived from DQABench plot_time_bounds in the annotation layer. "
        "The rewritten query is generated only from original_query, ticker, and stock_name by removing explicit temporal expressions, "
        "without using gold_range, plot_time_bounds, temporal_scope_type, or anchor metadata."
    )
    return record


# =============================================================================
# 8. Writers
# =============================================================================

def json_safe_cell(value: Any) -> Any:
    """Serialize nested values for CSV cells."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json_safe_cell(record.get(key)) for key in fieldnames})


def write_rewrite_quality_report(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "bridge_id",
        "source_id",
        "rewrite_type",
        "rewritten_query",
        "rewrite_status",
        "rewrite_quality_status",
        "contains_explicit_year",
        "contains_original_temporal_label",
        "keeps_entity",
        "keeps_diachronic_intent",
        "is_non_empty_question",
        "rewrite_reason",
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        flags = record.get("rewrite_quality_flags", {}) or {}
        rows.append(
            {
                "bridge_id": record.get("bridge_id"),
                "source_id": record.get("source_id"),
                "rewrite_type": record.get("rewrite_type"),
                "rewritten_query": record.get("rewritten_query"),
                "rewrite_status": record.get("rewrite_status"),
                "rewrite_quality_status": record.get("rewrite_quality_status"),
                "contains_explicit_year": flags.get("contains_explicit_year"),
                "contains_original_temporal_label": flags.get("contains_original_temporal_label"),
                "keeps_entity": flags.get("keeps_entity"),
                "keeps_diachronic_intent": flags.get("keeps_diachronic_intent"),
                "is_non_empty_question": flags.get("is_non_empty_question"),
                "rewrite_reason": record.get("rewrite_reason"),
            }
        )
    write_csv(rows, path, fieldnames)


def write_rewrite_prompt(path: Path) -> None:
    prompt = """# Rewrite Prompt v3: Leakage-aware Query Rewrite

Rewrite the following explicit-time diachronic stock question into an implicit-time user question.

Constraints:
- Use only the original question text, ticker, and stock name.
- Do not use metadata such as plot_time_bounds, gold_range, temporal_scope_type, anchor_year, or query_type_info.
- Remove explicit years and explicit time ranges.
- Preserve the entity, topic, and diachronic intent: trend, change, movement, fluctuation, performance, or response.
- Do not add new years, dates, facts, or event anchors.
- Output only the rewritten question.

Rewrite types:
- Type A: no explicit or relative time expression.
- Type B: vague relative time expression derived only from the original query text.
- Type C: event-anchor rewrite only if the original query already contains a clear event anchor.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def write_annotation_guideline(path: Path) -> None:
    guideline = """# Bridge Benchmark v3 Annotation Guideline

## Core principle
The pipeline separates annotation from rewrite generation to avoid temporal answer leakage.

## Annotation Layer
Metadata is allowed here because this layer creates evaluation labels:
- plot_time_bounds -> gold_start_date, gold_end_date, gold_range
- query_type_info / context_details_for_sampling -> temporal_scope_type, anchor_year, anchor_direction
- gold_range -> acceptable_ranges

These fields are evaluation targets and should not be given to the M0 predictor.

## Rewrite Layer
Rewrite generation is text-only. It may use only:
- original_query
- ticker
- stock_name

It must not use:
- plot_time_bounds
- gold_range
- gold_start_date / gold_end_date
- acceptable_ranges
- temporal_scope_type
- anchor_year / anchor_direction
- query_type_info
- context_details_for_sampling

## Rewrite Types
- Type A: remove all time expressions and produce a natural no-time query.
- Type B: use a vague temporal phrase derived from the original query text only.
- Type C: generate only when the original query contains a clear event anchor; otherwise mark not_applicable.

## Quality Flags
- contains_explicit_year
- contains_original_temporal_label
- keeps_entity
- keeps_diachronic_intent
- is_non_empty_question

A rewrite passes only when it is non-empty, contains no explicit year, does not retain the source temporal label, preserves the entity, and preserves diachronic intent.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(guideline, encoding="utf-8")


# =============================================================================
# 9. Internal Tests
# =============================================================================

def run_internal_tests() -> None:
    print("Running internal tests...")

    specific = {
        "id": "x1",
        "ticker": "AGG",
        "stock_name": "AGG",
        "question": "What was the general stock price movement for AGG during 2021?",
        "query_type_info": {"type": "Specific Time Period"},
        "plot_time_bounds": {"start_date": "2021-01-01", "end_date": "2021-12-31"},
    }
    ok, reason = is_valid_bridge_candidate(specific)
    assert not ok and "Specific Time Period" in reason

    before = {
        "id": "x2",
        "ticker": "AGG",
        "stock_name": "AGG",
        "temporal_context_label": "before 2020",
        "question": "What was the predominant stock price trend for AGG before 2020?",
        "query_type_info": {"type": "Before", "direction": "before"},
        "plot_time_bounds": {"start_date": "2012-01-03", "end_date": "2019-12-31"},
        "context_details_for_sampling": {"anchor_year": 2020, "direction": "before"},
    }
    ok, reason = is_valid_bridge_candidate(before)
    assert ok, reason

    annotated = normalize_temporal_annotation(before)
    assert annotated["gold_range"] == [2012, 2019]
    assert annotated["anchor_year"] == 2020
    assert annotated["anchor_direction"] == "before"

    ranges = generate_acceptable_ranges([2012, 2019])
    assert {tuple(r) for r in ranges} == {(2012, 2019), (2011, 2019), (2012, 2020), (2011, 2020)}

    pseudo = pseudo_rewrite_text_only(before["question"])
    assert "2020" not in pseudo and "before" not in pseudo.lower(), pseudo
    assert pseudo == "What was the predominant stock price trend for AGG?", pseudo

    variants = generate_rewrite_variants_text_only(before["question"], pseudo, "AGG", "AGG", include_type_c=True)
    assert variants[0]["rewrite_type"] == "A"
    assert variants[0]["rewritten_query"] == "How did AGG's stock price generally trend?"
    assert "2020" not in variants[0]["rewritten_query"]
    assert variants[1]["rewrite_type"] == "B"
    assert "earlier periods" in variants[1]["rewritten_query"]
    assert variants[2]["rewrite_type"] == "C" and variants[2]["rewrite_status"] == "not_applicable"

    event_query = "How did AGG's stock price trend after the COVID-19 shock?"
    event_pseudo = pseudo_rewrite_text_only(event_query)
    event_variants = generate_rewrite_variants_text_only(event_query, event_pseudo, "AGG", "AGG", include_type_c=True)
    assert event_variants[2]["rewrite_status"] == "generated"
    assert "COVID-19" in event_variants[2]["rewritten_query"]

    record = construct_final_record(annotated, pseudo, variants[0], 1)
    assert record["metadata_used_for_rewrite"] is False
    assert record["rewrite_quality_status"] == "pass", record

    print("All internal tests passed.")


# =============================================================================
# 10. Main Pipeline
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge Benchmark Pipeline v3: leakage-aware construction")
    parser.add_argument("--input", type=Path, help="Input dqabench_MCQA.json or JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v3.1"), help="Output directory")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of source items to process")
    parser.add_argument("--min-days", type=int, default=365, help="Minimum duration in days for candidate filtering")
    parser.add_argument("--include-type-c", action="store_true", help="Generate Type C event-anchor records")
    parser.add_argument("--run-tests", action="store_true", help="Run internal tests and exit")
    args = parser.parse_args()

    if args.run_tests:
        run_internal_tests()
        return

    if args.input is None:
        parser.error("--input is required unless --run-tests is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bridge Benchmark Pipeline v3 - Leakage-aware")
    print("=" * 80)

    dataset = load_json_dataset(args.input)
    total_source_items = len(dataset)
    if args.max_items is not None:
        dataset = dataset[: args.max_items]

    bridge_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    for item in dataset:
        ok, reason = is_valid_bridge_candidate(item, min_days=args.min_days)
        item_copy = dict(item)
        item_copy["bridge_candidate_reason"] = reason
        if ok:
            bridge_candidates.append(item_copy)
        else:
            rejected_candidates.append(item_copy)

    write_jsonl(bridge_candidates, args.output_dir / "bridge_candidates.jsonl")
    write_jsonl(rejected_candidates, args.output_dir / "rejected_candidates.jsonl")

    annotated_candidates: list[dict[str, Any]] = []
    for item in bridge_candidates:
        annotated = normalize_temporal_annotation(item)
        annotated["acceptable_ranges"] = generate_acceptable_ranges(annotated.get("gold_range", []))
        annotated_candidates.append(annotated)

    write_jsonl(annotated_candidates, args.output_dir / "bridge_candidates_annotated.jsonl")

    final_records: list[dict[str, Any]] = []
    for idx, annotated in enumerate(annotated_candidates, start=1):
        original_query = str(annotated.get("original_query", ""))
        ticker = annotated.get("ticker")
        stock_name = annotated.get("stock_name")

        # Leakage-aware: rewrite generation receives only text fields.
        pseudo_query = pseudo_rewrite_text_only(original_query)
        variants = generate_rewrite_variants_text_only(
            original_query=original_query,
            pseudo_query=pseudo_query,
            ticker=ticker,
            stock_name=stock_name,
            include_type_c=args.include_type_c,
        )
        for variant in variants:
            final_records.append(construct_final_record(annotated, pseudo_query, variant, idx))

    write_jsonl(final_records, args.output_dir / "bridge_rewrite_dataset_v3.jsonl")

    final_csv_fields = [
        "bridge_id",
        "source_id",
        "source_dataset",
        "original_query",
        "pseudo_query",
        "rewritten_query",
        "rewrite_type",
        "gold_range",
        "gold_start_date",
        "gold_end_date",
        "acceptable_ranges",
        "temporal_scope_type",
        "source_temporal_label",
        "anchor_year",
        "anchor_direction",
        "ticker",
        "stock_name",
        "rewrite_input_policy",
        "metadata_used_for_rewrite",
        "annotation_uses_metadata",
        "rewrite_status",
        "rewrite_reason",
        "rewrite_quality_status",
        "rewrite_quality_flags",
        "rationale",
    ]
    write_csv(final_records, args.output_dir / "bridge_rewrite_dataset_v3.csv", final_csv_fields)
    write_rewrite_quality_report(final_records, args.output_dir / "rewrite_quality_report_v3.csv")
    write_rewrite_prompt(args.output_dir / "rewrite_prompt_v3.txt")
    write_annotation_guideline(args.output_dir / "annotation_guideline_v3.md")

    type_counts = {"A": 0, "B": 0, "C": 0}
    quality_counts: dict[str, int] = {}
    for record in final_records:
        rewrite_type = record.get("rewrite_type")
        if rewrite_type in type_counts:
            type_counts[rewrite_type] += 1
        status = str(record.get("rewrite_quality_status"))
        quality_counts[status] = quality_counts.get(status, 0) + 1

    print("\nPipeline summary")
    print("-" * 80)
    print(f"Total source items loaded: {total_source_items}")
    print(f"Items processed: {len(dataset)}")
    print(f"Accepted bridge candidates: {len(bridge_candidates)}")
    print(f"Rejected candidates: {len(rejected_candidates)}")
    print(f"Annotated candidates: {len(annotated_candidates)}")
    print(f"Final rewrite records: {len(final_records)}")
    print(f"Type A: {type_counts['A']}")
    print(f"Type B: {type_counts['B']}")
    print(f"Type C: {type_counts['C']}")
    print("Quality counts:")
    for status, count in sorted(quality_counts.items()):
        print(f"  {status}: {count}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
