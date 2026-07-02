#!/usr/bin/env python3
"""Batch driver for ablation1 GSO manual-registration metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "sam3d_objects").is_dir() and (parent / "streaming").is_dir():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = find_repo_root()
SCRIPT = REPO_ROOT / "docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py"
CONTROLLER = REPO_ROOT / "docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py"
PYTHON = REPO_ROOT / ".env/bin/python"
INPUT_ROOT = REPO_ROOT / "outputs/ablation1_gso_past_solution"
SCRATCH_ROOT = REPO_ROOT / "scratch/sam3d-gso-manual-registration-ablation1"
DEFAULT_VARIANTS = ("mvsam3d_flowedit", "mvsam3d_kvcache", "mvsam3d_last_chunk")
METRIC_NAME = "render_mvs_25_manual_registration"


@dataclass(frozen=True)
class Job:
    variant: str
    scene: str
    chunk_root: Path

    @property
    def chunk_name(self) -> str:
        return self.chunk_root.name

    @property
    def scratch_dir(self) -> Path:
        return SCRATCH_ROOT / self.variant / self.scene / self.chunk_name

    @property
    def output_dir(self) -> Path:
        return INPUT_ROOT / self.variant / self.scene / "metrics" / METRIC_NAME

    @property
    def state_file(self) -> Path:
        return self.scratch_dir / "alignment_state.json"

    @property
    def summary_file(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def per_view_csv(self) -> Path:
        return self.output_dir / "per_view_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation1 GSO manual-registration metric batch.")
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("ABLATION1_GSO_WORKERS", "1")))
    parser.add_argument("--cuda-devices", default=os.environ.get("ABLATION1_GSO_CUDA_DEVICES", "0"))
    parser.add_argument("--views", default="all")
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--gicp-max-iterations", type=int, default=80)
    parser.add_argument("--initial-count", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def discover_jobs(variants: list[str], scenes_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for variant in variants:
        variant_root = INPUT_ROOT / variant
        if not variant_root.is_dir():
            raise FileNotFoundError(f"Variant root not found: {variant_root}")
        for scene_dir in sorted(path for path in variant_root.iterdir() if path.is_dir() and path.name != "metrics"):
            if scenes_filter is not None and scene_dir.name not in scenes_filter:
                continue
            chunk_dirs = sorted(path for path in scene_dir.glob("chunk_*") if path.is_dir())
            if len(chunk_dirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one chunk under {scene_dir}, found {len(chunk_dirs)}: {[path.name for path in chunk_dirs]}"
                )
            jobs.append(Job(variant=variant, scene=scene_dir.name, chunk_root=chunk_dirs[0]))
    return jobs


def base_env(cuda_device: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_device,
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    return env


def run_command(cmd: list[str], *, env: dict[str, str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_views_finite(summary: dict[str, Any]) -> bool:
    if int(summary["view_count"]) != 25:
        return False
    for row in summary["per_view"]:
        for key in ("psnr", "ssim", "lpips", "alpha_mask_iou"):
            value = row[key]
            if value is None or not math.isfinite(float(value)):
                return False
    return True


def review_decision(summary: dict[str, Any]) -> tuple[bool, dict[str, float | bool]]:
    alpha_median = float(summary["metrics"]["alpha_mask_iou"]["median"])
    cd = float(summary["active_mesh_diagnostics"]["cd"])
    bbox = float(summary["target_bbox_diagonal"])
    cd_fraction = float(cd / max(bbox, 1e-8))
    finite = all_views_finite(summary)
    needs_manual_review = not (alpha_median >= 0.25 and cd_fraction <= 0.15 and finite)
    return needs_manual_review, {
        "alpha_mask_iou_median": alpha_median,
        "cd": cd,
        "cd_bbox_fraction": cd_fraction,
        "all_25_views_finite": finite,
    }


def update_state_review_flag(state_file: Path, needs_manual_review: bool, review_metrics: dict[str, Any]) -> None:
    state = load_json(state_file)
    state["needs_manual_review"] = bool(needs_manual_review)
    state["registration_diagnostics"]["auto_review"] = review_metrics
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_render_outputs(job: Job) -> None:
    summary = load_json(job.scratch_dir / "summary.json")
    per_view = job.scratch_dir / "per_view_metrics.csv"
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    job.per_view_csv.write_text(per_view.read_text(encoding="utf-8"), encoding="utf-8")


def export_review_candidates(job: Job, cuda_device: str, top_k: int) -> None:
    env = base_env(cuda_device)
    run_command(
        [
            str(PYTHON),
            str(CONTROLLER),
            "export-registration-candidates",
            "--state-file",
            str(job.state_file),
            "--round-name",
            "registration_candidates",
            "--top-k",
            str(top_k),
        ],
        env=env,
        cwd=REPO_ROOT,
    )
    run_command(
        [
            str(PYTHON),
            str(CONTROLLER),
            "render-candidates",
            "--round-dir",
            str(job.scratch_dir / "manual_adjustment_rounds" / "registration_candidates"),
            "--views",
            "0,12,24",
        ],
        env=env,
        cwd=REPO_ROOT,
    )


def scene_metric_row(job: Job) -> dict[str, Any]:
    summary = load_json(job.summary_file)
    state = load_json(job.state_file)
    return {
        "variant": job.variant,
        "scene": job.scene,
        "chunk_name": job.chunk_name,
        "psnr": summary["metrics"]["psnr"]["mean"],
        "ssim": summary["metrics"]["ssim"]["mean"],
        "lpips": summary["metrics"]["lpips"]["mean"],
        "alpha_mask_iou": summary["metrics"]["alpha_mask_iou"]["mean"],
        "cd": summary["active_mesh_diagnostics"]["cd"],
        "cd_bbox_fraction": summary["active_mesh_diagnostics"]["cd"] / max(float(summary["target_bbox_diagonal"]), 1e-8),
        "source_geometry_kind": state["source_geometry_kind"],
        "registration_method": state["registration_diagnostics"]["method"],
        "needs_manual_review": bool(state.get("needs_manual_review", False)),
        "state_file": str(job.state_file.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    if not subset:
        raise ValueError(f"No rows for variant {variant}")
    return {
        "variant": variant,
        "scene_count": len(subset),
        "manual_review_count": sum(1 for row in subset if row["needs_manual_review"]),
        "psnr_mean": sum(float(row["psnr"]) for row in subset) / len(subset),
        "ssim_mean": sum(float(row["ssim"]) for row in subset) / len(subset),
        "lpips_mean": sum(float(row["lpips"]) for row in subset) / len(subset),
        "alpha_mask_iou_mean": sum(float(row["alpha_mask_iou"]) for row in subset) / len(subset),
        "cd_mean": sum(float(row["cd"]) for row in subset) / len(subset),
        "cd_bbox_fraction_mean": sum(float(row["cd_bbox_fraction"]) for row in subset) / len(subset),
    }


def run_job(job: Job, cuda_device: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_existing and job.summary_file.is_file() and job.state_file.is_file():
        return {"variant": job.variant, "scene": job.scene, "status": "exists"}
    job.scratch_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(cuda_device)
    run_command(
        [
            str(PYTHON),
            str(SCRIPT),
            "estimate",
            "--scene",
            job.scene,
            "--variant",
            job.variant,
            "--chunk-root",
            str(job.chunk_root),
            "--output-dir",
            str(job.scratch_dir),
            "--source-geometry",
            "auto",
            "--registration-method",
            "initial_gicp_scale",
            "--views",
            str(args.views),
            "--sample-points",
            str(args.sample_points),
            "--gicp-max-iterations",
            str(args.gicp_max_iterations),
            "--initial-count",
            str(args.initial_count),
        ],
        env=env,
        cwd=REPO_ROOT,
    )
    summary = load_json(job.scratch_dir / "summary.json")
    needs_manual_review, review_metrics = review_decision(summary)
    update_state_review_flag(job.state_file, needs_manual_review, review_metrics)
    copy_render_outputs(job)
    if needs_manual_review:
        export_review_candidates(job, cuda_device, args.top_k)
    return {
        "variant": job.variant,
        "scene": job.scene,
        "status": "ok",
        "needs_manual_review": needs_manual_review,
    }


def main() -> None:
    args = parse_args()
    scenes_filter = None if not args.scenes else set(args.scenes)
    jobs = discover_jobs(list(args.variants), scenes_filter)
    cuda_devices = [token.strip() for token in str(args.cuda_devices).split(",") if token.strip()]
    if not cuda_devices:
        raise ValueError("No CUDA devices configured.")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(run_job, job, cuda_devices[index % len(cuda_devices)], args): job for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    rows = [scene_metric_row(job) for job in jobs]
    fieldnames = [
        "variant",
        "scene",
        "chunk_name",
        "psnr",
        "ssim",
        "lpips",
        "alpha_mask_iou",
        "cd",
        "cd_bbox_fraction",
        "source_geometry_kind",
        "registration_method",
        "needs_manual_review",
        "state_file",
    ]
    for variant in args.variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        write_csv(INPUT_ROOT / variant / "metrics" / f"{METRIC_NAME}_scene_metrics.csv", variant_rows, fieldnames)
    variant_summary_rows = [summarize_variant(rows, variant) for variant in args.variants]
    write_csv(
        INPUT_ROOT / "metrics" / f"{METRIC_NAME}_variant_summary.csv",
        variant_summary_rows,
        [
            "variant",
            "scene_count",
            "manual_review_count",
            "psnr_mean",
            "ssim_mean",
            "lpips_mean",
            "alpha_mask_iou_mean",
            "cd_mean",
            "cd_bbox_fraction_mean",
        ],
    )


if __name__ == "__main__":
    main()
