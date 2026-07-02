#!/usr/bin/env python3
"""Collect Ours/baseline GSO metrics summaries into paper-table CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge GSO manual/baseline metrics_summary.csv files.")
    parser.add_argument("--ours-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--ours-label", default="Ours")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def table_row(row: dict[str, Any], method: str, audit_status: str) -> dict[str, Any]:
    return {
        "method": method,
        "scene_count": first_present(row, ("scene_count",)),
        "CD-L2": first_present(row, ("cd_l2_mean", "cd_l2")),
        "CD-sq": first_present(row, ("cd_mean", "cd")),
        "PSNR": first_present(row, ("psnr_mean", "psnr")),
        "SSIM": first_present(row, ("ssim_mean", "ssim")),
        "LPIPS": first_present(row, ("lpips_mean", "lpips")),
        "image FID": first_present(row, ("image_fid", "image_fid_scene")),
        "P-FID": first_present(row, ("p_fid", "p_fid_scene")),
        "audit status": audit_status,
    }


def load_distribution_status(summary_path: Path) -> str:
    run_path = Path(summary_path).with_name("evaluation_run.json")
    if not run_path.is_file():
        return "unknown"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    status = payload.get("distribution_status", {})
    image_status = status.get("image_fid", {}).get("status", "unknown")
    point_status = status.get("p_fid", {}).get("status", "unknown")
    return "ok" if image_status == "ok" and point_status == "ok" else f"image_fid={image_status};p_fid={point_status}"


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ours_status = load_distribution_status(args.ours_summary)
    for row in read_csv(args.ours_summary):
        if row.get("variant") == "ALL":
            rows.append(table_row(row, str(args.ours_label), ours_status))
    baseline_status = load_distribution_status(args.baseline_summary)
    for row in read_csv(args.baseline_summary):
        variant = row.get("variant", "")
        if variant and variant != "ALL":
            rows.append(table_row(row, variant, baseline_status))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = collect_rows(args)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)


if __name__ == "__main__":
    main()
