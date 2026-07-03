#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


QUESTION_FILES = [
    "questions/global_questions_new.json",
    "questions/global_questions_old.json",
    "questions/local_questions_new.json",
    "questions/local_questions_old.json",
]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: as_jsonable(v) for k, v in row.items()}, ensure_ascii=False) + "\n")


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [dict(x) for x in parsed]
    if isinstance(parsed, dict):
        for key in ("data", "questions", "records"):
            if isinstance(parsed.get(key), list):
                return [dict(x) for x in parsed[key]]
        return [parsed]
    return []


def fallback_import_questions(dataset_name: str, output_dir: Path, lines: list[str]) -> bool:
    wrote_any = False
    all_rows = []
    lines += [
        "## Questions Fallback Import",
        "",
        "The Hugging Face `questions` config can fail because its JSON files do not share one fixed schema. The importer therefore downloads each question JSON file separately and preserves the union of columns.",
        "",
    ]
    for file_path in QUESTION_FILES:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--austinmyc--ECT-QA" / "snapshots"
        matches = sorted(cache_root.glob(f"*/{file_path}"))
        if matches:
            downloaded = matches[-1]
        else:
            try:
                downloaded = Path(hf_hub_download(repo_id=dataset_name, filename=file_path, repo_type="dataset", local_files_only=True))
            except Exception:
                try:
                    downloaded = Path(hf_hub_download(repo_id=dataset_name, filename=file_path, repo_type="dataset"))
                except Exception as exc:
                    lines += [f"- `{file_path}` failed: `{exc}`"]
                    continue
        try:
            records = load_json_records(downloaded)
        except Exception as exc:
            lines += [f"- `{file_path}` failed: `{exc}`"]
            continue
        subset = file_path.split("/")[-1].removesuffix(".json")
        for idx, row in enumerate(records):
            row["_hf_config"] = "questions"
            row["_hf_split"] = "train"
            row["_question_file"] = subset
            row.setdefault("_row_id", f"{subset}_{idx}")
        all_rows.extend(records)
        df = pd.DataFrame(records)
        stem = f"raw_questions_{subset}"
        write_jsonl(records, output_dir / f"{stem}.jsonl")
        df.to_csv(output_dir / f"{stem}.csv", index=False)
        wrote_any = True
        lines += [
            f"### questions / {subset}",
            "",
            f"- Rows: {len(df)}",
            f"- Columns: {list(df.columns)}",
            "",
        ]
        print(f"wrote {len(df)} question rows from {file_path}")
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        write_jsonl(all_rows, output_dir / "raw_questions_train.jsonl")
        df_all.to_csv(output_dir / "raw_questions_train.csv", index=False)
        lines += [
            "### questions / combined train",
            "",
            f"- Rows: {len(df_all)}",
            f"- Columns: {list(df_all.columns)}",
            "",
            "First examples:",
            "",
            "```json",
            json.dumps(all_rows[:2], ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
        print(f"wrote {len(df_all)} combined question rows")
    return wrote_any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="austinmyc/ECT-QA")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional Hugging Face dataset config. If omitted, all available configs are imported.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached Hugging Face files.")
    parser.add_argument("--output-dir", default="exp_ect_qa/processed")
    parser.add_argument("--report-dir", default="exp_ect_qa/reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ECT-QA Import Report",
        "",
        f"- Dataset: `{args.dataset_name}`",
        f"- `datasets` version: `{package_version('datasets')}`",
        f"- `pandas` version: `{package_version('pandas')}`",
        "",
    ]

    try:
        if args.config_name:
            config_names = [args.config_name]
        elif args.local_files_only and args.dataset_name == "austinmyc/ECT-QA":
            config_names = ["questions", "corpus"]
        else:
            config_names = get_dataset_config_names(args.dataset_name)
            if args.dataset_name == "austinmyc/ECT-QA" and (not config_names or config_names == ["default"]):
                config_names = ["questions", "corpus"]
            elif not config_names:
                config_names = [None]
    except Exception:
        if args.dataset_name == "austinmyc/ECT-QA" and not args.config_name:
            config_names = ["questions", "corpus"]
        else:
            raise

    lines += [
        f"- Configs imported: `{config_names}`",
        "",
        "## Splits",
        "",
    ]

    loading_errors = []
    for config_name in config_names:
        if config_name == "questions":
            fallback_import_questions(args.dataset_name, output_dir, lines)
            continue
        if args.local_files_only and config_name == "corpus" and (output_dir / "raw_corpus_train.csv").exists():
            lines += [
                "### corpus / train",
                "",
                f"- Reused existing local file: `{output_dir / 'raw_corpus_train.csv'}`",
                "",
            ]
            print(f"reused existing local corpus file -> {output_dir / 'raw_corpus_train.csv'}")
            continue
        try:
            download_config = DownloadConfig(local_files_only=args.local_files_only)
            dataset = (
                load_dataset(args.dataset_name, config_name, download_config=download_config)
                if config_name
                else load_dataset(args.dataset_name, download_config=download_config)
            )
        except Exception as exc:
            loading_errors.append((config_name, str(exc)))
            continue

        config_label = config_name or "default"
        for split_name, split in dataset.items():
            rows = [dict(row) for row in split]
            for row in rows:
                row["_hf_config"] = config_label
                row["_hf_split"] = split_name
            df = pd.DataFrame(rows)
            file_stem = f"raw_{config_label}_{split_name}"
            jsonl_path = output_dir / f"{file_stem}.jsonl"
            csv_path = output_dir / f"{file_stem}.csv"
            write_jsonl(rows, jsonl_path)
            df.to_csv(csv_path, index=False)

            lines += [
                f"### {config_label} / {split_name}",
                "",
                f"- Config: `{config_label}`",
                f"- Split: `{split_name}`",
                f"- Rows: {len(df)}",
                f"- Columns: {list(df.columns)}",
                f"- JSONL: `{jsonl_path}`",
                f"- CSV: `{csv_path}`",
                "",
                "First examples:",
                "",
                "```json",
                json.dumps(rows[:2], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ]
            print(f"wrote {len(df)} rows for config {config_label} split {split_name}")

    if loading_errors:
        lines += ["## Loading Errors", ""]
        for config_name, error in loading_errors:
            lines += [f"### {config_name}", "", f"```text\n{error}\n```", ""]

    (report_dir / "import_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {report_dir / 'import_report.md'}")


if __name__ == "__main__":
    main()
