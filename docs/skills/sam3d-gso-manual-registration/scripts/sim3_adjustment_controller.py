#!/usr/bin/env python3
"""Constrained Sim(3) candidate controller for manual GSO alignment.

This helper keeps subagents out of raw 4x4 matrix editing.  It proposes
explainable target-space Sim(3) deltas, renders candidate states, and applies
only the selected candidate back to the canonical alignment state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import manual_register_sam3d_gso as manual  # noqa: E402


DEFAULT_VIEWS = "0,12,24"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    preset: str
    description: str
    delta_rotation_deg: tuple[float, float, float]
    delta_translation: tuple[float, float, float]
    delta_scale: float
    pivot: tuple[float, float, float]
    pivot_mode: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, render, and apply constrained Sim(3) edit candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose", help="Write candidate states for a manual adjustment round.")
    propose.add_argument("--state-file", type=Path, required=True)
    propose.add_argument("--output-dir", type=Path)
    propose.add_argument("--round-name")
    propose.add_argument("--preset", choices=("orientation", "micro", "all"), default="orientation")
    propose.add_argument(
        "--pivot-mode",
        choices=("transformed_source_bbox_center", "target_bbox_center", "origin"),
        default="transformed_source_bbox_center",
    )
    propose.add_argument("--rotation-step-deg", type=float, default=8.0)
    propose.add_argument("--translation-step-fraction", type=float, default=0.025)
    propose.add_argument("--scale-step", type=float, default=0.03)

    export = subparsers.add_parser(
        "export-registration-candidates",
        help="Export registration_diagnostics candidates from a state as renderable candidate states.",
    )
    export.add_argument("--state-file", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--round-name", default="registration_candidates")
    export.add_argument("--top-k", type=int, default=24)

    render = subparsers.add_parser("render-candidates", help="Render candidate states for individual comparison review.")
    render.add_argument("--round-dir", type=Path, required=True)
    render.add_argument("--views", default=DEFAULT_VIEWS)
    render.add_argument("--candidate-id", action="append")
    manual.add_render_args(render)

    apply = subparsers.add_parser("apply-candidate", help="Copy one candidate active_sim3 into the canonical state.")
    apply.add_argument("--state-file", type=Path, required=True)
    apply.add_argument("--round-dir", type=Path)
    apply.add_argument("--candidate-id")
    apply.add_argument("--candidate-state", type=Path)
    apply.add_argument("--reason", default="")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def axis_rotation(axis: str, degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported rotation axis: {axis}")


def euler_xyz_rotation(delta_rotation_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = [float(value) for value in delta_rotation_deg]
    return axis_rotation("z", rz) @ axis_rotation("y", ry) @ axis_rotation("x", rx)


def compose_target_space_delta(
    sim3: dict[str, Any],
    *,
    delta_rotation: np.ndarray,
    delta_translation: Sequence[float],
    delta_scale: float,
    pivot: Sequence[float],
) -> dict[str, Any]:
    scale, rotation, translation = manual.sim3_components(sim3)
    delta_rotation = manual.orthonormalize_rotation(delta_rotation)
    delta_translation = np.asarray(delta_translation, dtype=np.float64).reshape(3)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    new_scale = float(scale) * float(delta_scale)
    new_rotation = delta_rotation @ rotation
    new_translation = float(delta_scale) * ((translation - pivot) @ delta_rotation.T) + pivot + delta_translation
    return manual.make_sim3_dict(
        new_scale,
        new_rotation,
        new_translation,
        source_space=sim3["source_space"],
        target_space=sim3["target_space"],
    )


def bbox_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def target_bbox_diagonal(state: dict[str, Any]) -> float:
    target_mesh = manual.load_mesh(manual.target_mesh_path_from_state(state), manual.MVS25_MESH_BASIS)
    return manual.point_bbox_diagonal(np.asarray(target_mesh.vertices, dtype=np.float64))


def resolve_pivot(state: dict[str, Any], pivot_mode: str) -> np.ndarray:
    if pivot_mode == "origin":
        return np.zeros(3, dtype=np.float64)
    if pivot_mode == "target_bbox_center":
        target_mesh = manual.load_mesh(manual.target_mesh_path_from_state(state), manual.MVS25_MESH_BASIS)
        return bbox_center(np.asarray(target_mesh.vertices, dtype=np.float64))
    if pivot_mode == "transformed_source_bbox_center":
        source_points = manual.source_geometry_full_points_from_state(state)
        source_center = bbox_center(source_points)
        return manual.apply_sim3_to_points(source_center[None, :], state["active_sim3"])[0]
    raise ValueError(f"Unsupported pivot mode: {pivot_mode}")


def orientation_specs(pivot: np.ndarray, pivot_mode: str) -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            "current",
            "orientation",
            "No edit; baseline candidate for visual comparison.",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            1.0,
            tuple(pivot.tolist()),
            pivot_mode,
        )
    ]
    for axis_index, axis in enumerate(("x", "y", "z")):
        for degrees in (90.0, 180.0, 270.0):
            rotation = [0.0, 0.0, 0.0]
            rotation[axis_index] = degrees
            specs.append(
                CandidateSpec(
                    f"rot_{axis}_{int(degrees):03d}",
                    "orientation",
                    f"Rotate {degrees:.0f} degrees around target {axis.upper()} axis at pivot.",
                    tuple(rotation),
                    (0.0, 0.0, 0.0),
                    1.0,
                    tuple(pivot.tolist()),
                    pivot_mode,
                )
            )
    return specs


def micro_specs(
    pivot: np.ndarray,
    pivot_mode: str,
    *,
    rotation_step_deg: float,
    translation_step: float,
    scale_step: float,
) -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            "current",
            "micro",
            "No edit; baseline candidate for visual comparison.",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            1.0,
            tuple(pivot.tolist()),
            pivot_mode,
        )
    ]
    for axis_index, axis in enumerate(("x", "y", "z")):
        for sign, label in ((1.0, "p"), (-1.0, "m")):
            rotation = [0.0, 0.0, 0.0]
            rotation[axis_index] = sign * float(rotation_step_deg)
            specs.append(
                CandidateSpec(
                    f"rot_{axis}_{label}{int(abs(rotation_step_deg)):02d}",
                    "micro",
                    f"Small {sign * float(rotation_step_deg):.3f} degree target {axis.upper()} rotation.",
                    tuple(rotation),
                    (0.0, 0.0, 0.0),
                    1.0,
                    tuple(pivot.tolist()),
                    pivot_mode,
                )
            )
            translation = [0.0, 0.0, 0.0]
            translation[axis_index] = sign * float(translation_step)
            specs.append(
                CandidateSpec(
                    f"trans_{axis}_{label}",
                    "micro",
                    f"Small {sign * float(translation_step):.6f} target {axis.upper()} translation.",
                    (0.0, 0.0, 0.0),
                    tuple(translation),
                    1.0,
                    tuple(pivot.tolist()),
                    pivot_mode,
                )
            )
    specs.extend(
        [
            CandidateSpec(
                "scale_p",
                "micro",
                f"Increase uniform scale by {float(scale_step):.3f}.",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                1.0 + float(scale_step),
                tuple(pivot.tolist()),
                pivot_mode,
            ),
            CandidateSpec(
                "scale_m",
                "micro",
                f"Decrease uniform scale by {float(scale_step):.3f}.",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                1.0 - float(scale_step),
                tuple(pivot.tolist()),
                pivot_mode,
            ),
        ]
    )
    return specs


def candidate_specs(
    state: dict[str, Any],
    *,
    preset: str,
    pivot_mode: str,
    rotation_step_deg: float,
    translation_step_fraction: float,
    scale_step: float,
) -> list[CandidateSpec]:
    pivot = resolve_pivot(state, pivot_mode)
    if preset == "orientation":
        return orientation_specs(pivot, pivot_mode)
    translation_step = max(target_bbox_diagonal(state), 1e-8) * float(translation_step_fraction)
    if preset == "micro":
        return micro_specs(
            pivot,
            pivot_mode,
            rotation_step_deg=rotation_step_deg,
            translation_step=translation_step,
            scale_step=scale_step,
        )
    if preset == "all":
        orientation = orientation_specs(pivot, pivot_mode)
        micro = [spec for spec in micro_specs(
            pivot,
            pivot_mode,
            rotation_step_deg=rotation_step_deg,
            translation_step=translation_step,
            scale_step=scale_step,
        ) if spec.candidate_id != "current"]
        return orientation + micro
    raise ValueError(f"Unsupported preset: {preset}")


def candidate_to_row(spec: CandidateSpec, state_path: Path | None = None) -> dict[str, Any]:
    row = {
        "candidate_id": spec.candidate_id,
        "preset": spec.preset,
        "description": spec.description,
        "delta_rotation_deg": list(spec.delta_rotation_deg),
        "delta_translation": list(spec.delta_translation),
        "delta_scale": float(spec.delta_scale),
        "pivot": list(spec.pivot),
        "pivot_mode": spec.pivot_mode,
    }
    if state_path is not None:
        row["state_path"] = str(Path(state_path).resolve())
    return row


def write_candidate_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "preset",
        "description",
        "rank",
        "registration_candidate_id",
        "initial_cd",
        "cd_after_gicp",
        "cd_after_scale_refine",
        "gicp_fitness",
        "gicp_inlier_rmse",
        "delta_rotation_deg",
        "delta_translation",
        "delta_scale",
        "pivot",
        "pivot_mode",
        "state_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field, "")) if isinstance(row.get(field, ""), list) else row.get(field, "")
                    for field in fieldnames
                }
            )


def default_round_dir(state_file: Path, round_name: str | None) -> Path:
    name = round_name or datetime.now(timezone.utc).strftime("round_%Y%m%dT%H%M%SZ")
    return Path(state_file).expanduser().resolve().parent / "manual_adjustment_rounds" / name


def run_propose(args: argparse.Namespace) -> dict[str, Any]:
    state_file = args.state_file.expanduser().resolve()
    state = manual.read_state(state_file)
    round_dir = (
        default_round_dir(state_file, args.round_name)
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(
        state,
        preset=str(args.preset),
        pivot_mode=str(args.pivot_mode),
        rotation_step_deg=float(args.rotation_step_deg),
        translation_step_fraction=float(args.translation_step_fraction),
        scale_step=float(args.scale_step),
    )
    rows = []
    for spec in specs:
        candidate_dir = round_dir / spec.candidate_id
        candidate_state_path = candidate_dir / "alignment_state.candidate.json"
        candidate_state = json.loads(json.dumps(state))
        candidate_state["active_sim3"] = compose_target_space_delta(
            state["active_sim3"],
            delta_rotation=euler_xyz_rotation(spec.delta_rotation_deg),
            delta_translation=spec.delta_translation,
            delta_scale=spec.delta_scale,
            pivot=spec.pivot,
        )
        candidate_state["candidate_metadata"] = candidate_to_row(spec, candidate_state_path)
        candidate_state["candidate_metadata"]["source_state_file"] = str(state_file)
        manual.write_json(candidate_state_path, candidate_state)
        rows.append(candidate_to_row(spec, candidate_state_path))

    table_json = round_dir / "candidate_table.json"
    table_csv = round_dir / "candidate_table.csv"
    payload = {
        "source_state_file": str(state_file),
        "round_dir": str(round_dir),
        "preset": str(args.preset),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": rows,
    }
    manual.write_json(table_json, payload)
    write_candidate_csv(table_csv, rows)
    return {"round_dir": str(round_dir), "candidate_count": len(rows), "candidate_table": str(table_json)}


def run_export_registration_candidates(args: argparse.Namespace) -> dict[str, Any]:
    state_file = args.state_file.expanduser().resolve()
    state = manual.read_state(state_file)
    candidates = list(state["registration_diagnostics"]["candidates"])[: int(args.top_k)]
    round_dir = (
        state_file.parent / "manual_adjustment_rounds" / str(args.round_name)
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, candidate in enumerate(candidates):
        candidate_id = f"rank_{rank:02d}_{candidate['candidate_id']}"
        candidate_dir = round_dir / candidate_id
        candidate_state_path = candidate_dir / "alignment_state.candidate.json"
        candidate_state = json.loads(json.dumps(state))
        candidate_state["active_sim3"] = candidate["final_sim3"]
        candidate_state["candidate_metadata"] = {
            "candidate_id": candidate_id,
            "registration_candidate_id": candidate["candidate_id"],
            "rank": int(rank),
            "source_state_file": str(state_file),
            "cd_after_scale_refine": float(candidate["cd_after_scale_refine"]),
            "cd_after_gicp": float(candidate["cd_after_gicp"]),
            "initial_cd": float(candidate["initial_cd"]),
            "gicp_fitness": float(candidate["gicp"]["fitness"]),
            "gicp_inlier_rmse": float(candidate["gicp"]["inlier_rmse"]),
            "scale_refine": candidate["scale_refine"],
        }
        manual.write_json(candidate_state_path, candidate_state)
        rows.append(
            {
                "candidate_id": candidate_id,
                "preset": "registration_diagnostics",
                "description": f"registration candidate rank {rank}: {candidate['candidate_id']}",
                "rank": int(rank),
                "registration_candidate_id": candidate["candidate_id"],
                "initial_cd": float(candidate["initial_cd"]),
                "cd_after_gicp": float(candidate["cd_after_gicp"]),
                "cd_after_scale_refine": float(candidate["cd_after_scale_refine"]),
                "gicp_fitness": float(candidate["gicp"]["fitness"]),
                "gicp_inlier_rmse": float(candidate["gicp"]["inlier_rmse"]),
                "delta_rotation_deg": [],
                "delta_translation": [],
                "delta_scale": 1.0,
                "pivot": [],
                "pivot_mode": "n/a",
                "state_path": str(candidate_state_path.resolve()),
            }
        )

    table_json = round_dir / "candidate_table.json"
    table_csv = round_dir / "candidate_table.csv"
    payload = {
        "source_state_file": str(state_file),
        "round_dir": str(round_dir),
        "preset": "registration_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": rows,
    }
    manual.write_json(table_json, payload)
    write_candidate_csv(table_csv, rows)
    return {"round_dir": str(round_dir), "candidate_count": len(rows), "candidate_table": str(table_json)}


def read_candidate_table(round_dir: Path) -> dict[str, Any]:
    return json.loads((Path(round_dir) / "candidate_table.json").read_text(encoding="utf-8"))


def selected_candidates(table: dict[str, Any], candidate_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    candidates = list(table["candidates"])
    if not candidate_ids:
        return candidates
    wanted = set(candidate_ids)
    selected = [row for row in candidates if row["candidate_id"] in wanted]
    missing = sorted(wanted - {row["candidate_id"] for row in selected})
    if missing:
        raise ValueError(f"Candidate ids not found: {missing}")
    return selected


def metric_mean(summary: dict[str, Any], key: str) -> float | None:
    value = summary["metrics"][key]["mean"]
    return None if value is None else float(value)


def write_candidate_report(round_dir: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(round_dir) / "candidate_report.csv"
    fieldnames = [
        "candidate_id",
        "alpha_mask_iou_mean",
        "lpips_mean",
        "psnr_mean",
        "ssim_mean",
        "active_mesh_cd",
        "comparison_000",
        "comparison_012",
        "comparison_024",
        "summary_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def report_row_from_summary(candidate_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = {row["view"]: row["comparison_path"] for row in summary["per_view"]}
    return {
        "candidate_id": candidate_id,
        "alpha_mask_iou_mean": metric_mean(summary, "alpha_mask_iou"),
        "lpips_mean": metric_mean(summary, "lpips"),
        "psnr_mean": metric_mean(summary, "psnr"),
        "ssim_mean": metric_mean(summary, "ssim"),
        "active_mesh_cd": summary["active_mesh_diagnostics"]["cd"],
        "comparison_000": comparisons.get("000", ""),
        "comparison_012": comparisons.get("012", ""),
        "comparison_024": comparisons.get("024", ""),
        "summary_json": str(Path(summary["state_file"]).resolve().parent / "summary.json"),
    }


def run_render_candidates(args: argparse.Namespace) -> dict[str, Any]:
    round_dir = args.round_dir.expanduser().resolve()
    table = read_candidate_table(round_dir)
    rows = []
    for candidate in selected_candidates(table, args.candidate_id):
        candidate_id = candidate["candidate_id"]
        state_path = Path(candidate["state_path"])
        output_dir = round_dir / candidate_id
        summary = manual.render_from_state(
            state_path,
            output_dir=output_dir,
            views=str(args.views),
            render_config=manual.render_config_from_args(args),
        )
        rows.append(report_row_from_summary(candidate_id, summary))
    rows.sort(key=lambda row: (row["alpha_mask_iou_mean"] is None, -(row["alpha_mask_iou_mean"] or -1.0)))
    report_path = write_candidate_report(round_dir, rows)
    return {"round_dir": str(round_dir), "rendered_count": len(rows), "candidate_report": str(report_path)}


def candidate_state_from_args(args: argparse.Namespace) -> Path:
    if args.candidate_state is not None:
        return args.candidate_state.expanduser().resolve()
    if args.round_dir is None or args.candidate_id is None:
        raise ValueError("Use either --candidate-state or both --round-dir and --candidate-id.")
    table = read_candidate_table(args.round_dir.expanduser().resolve())
    selected = selected_candidates(table, [str(args.candidate_id)])
    return Path(selected[0]["state_path"]).expanduser().resolve()


def backup_state_path(state_file: Path, candidate_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return state_file.with_name(f"{state_file.stem}.before_apply_{candidate_id}_{timestamp}{state_file.suffix}")


def apply_candidate(state_file: Path, candidate_state_file: Path, reason: str) -> dict[str, Any]:
    state_file = Path(state_file).expanduser().resolve()
    candidate_state_file = Path(candidate_state_file).expanduser().resolve()
    state = manual.read_state(state_file)
    candidate_state = manual.read_state(candidate_state_file)
    candidate_id = candidate_state["candidate_metadata"]["candidate_id"]
    backup_path = backup_state_path(state_file, str(candidate_id))
    manual.write_json(backup_path, state)
    state["active_sim3"] = candidate_state["active_sim3"]
    if "manual_adjustment_history" not in state:
        state["manual_adjustment_history"] = []
    state["manual_adjustment_history"].append(
        {
            "candidate_id": candidate_id,
            "candidate_state_file": str(candidate_state_file),
            "backup_state_file": str(backup_path),
            "reason": str(reason),
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    manual.write_json(state_file, state)
    return {
        "state_file": str(state_file),
        "candidate_state_file": str(candidate_state_file),
        "backup_state_file": str(backup_path),
        "candidate_id": candidate_id,
    }


def run_apply_candidate(args: argparse.Namespace) -> dict[str, Any]:
    return apply_candidate(args.state_file, candidate_state_from_args(args), str(args.reason))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "propose":
        result = run_propose(args)
    elif args.command == "export-registration-candidates":
        result = run_export_registration_candidates(args)
    elif args.command == "render-candidates":
        result = run_render_candidates(args)
    elif args.command == "apply-candidate":
        result = run_apply_candidate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(manual.jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
