#!/usr/bin/env python3
"""Render and evaluate GSO baseline GLB meshes in render_mvs_25 cameras."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import manual_register_sam3d_gso as manual  # noqa: E402


DEFAULT_INPUT_ROOT = manual.REPO_ROOT / "outputs" / "benchmark-meshes" / "gso"
DEFAULT_BENCHMARK_ROOT = Path(
    "/root/autodl-tmp/xinhai/projects/StreamingSAM3D-benchmark/results/StreamSAM3D-benchmark/last/gso"
)
DEFAULT_OUTPUT_ROOT = manual.REPO_ROOT / "outputs" / "sam3d-gso-baseline-mesh-registration"
DEFAULT_ARCHIVE_ROOT = manual.REPO_ROOT / "outputs" / "gso30_all_variants_archived"
BASELINE_MESH_SOURCE_SPACE = "baseline_raw_glb_mesh_basis"
BASELINE_MESH_BASIS = np.eye(3, dtype=np.float64)
Y_CONVENTIONS = ("opencv_y_down", "opencv_y_up")
BASELINE_METHODS = ("mv_sam3d", "sam3d", "trellis2", "trellis2_md", "trellis_md")
TRELLIS_NATIVE_RENDER_BACKEND = "trellis2_pbr_mesh_renderer"


@dataclass(frozen=True)
class BaselineMeshJob:
    scene: str
    method: str
    mesh_path: Path
    chunk: str = "last"
    source_kind: str = "benchmark_meshes"


@dataclass(frozen=True)
class RenderMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    uvs: np.ndarray | None
    texture: np.ndarray | None
    base_color_factor: np.ndarray


@dataclass(frozen=True)
class TrellisNativeRenderContext:
    mesh: Any
    renderer: Any
    envmap: Any
    render_key: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Sim(3) and render baseline GSO GLB meshes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate one baseline mesh alignment and render it.")
    estimate.add_argument("--mesh", type=Path, required=True)
    estimate.add_argument("--scene")
    estimate.add_argument("--method")
    estimate.add_argument("--gt-root", type=Path)
    estimate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    estimate.add_argument("--output-dir", type=Path)
    estimate.add_argument("--trellis-native-state", type=Path)
    add_registration_args(estimate)
    add_render_args(estimate)
    estimate.add_argument("--skip-render", action="store_true")

    render = subparsers.add_parser("render", help="Rerender an existing baseline alignment state.")
    render.add_argument("--state-file", type=Path, required=True)
    render.add_argument("--output-dir", type=Path)
    add_render_args(render)

    run_all = subparsers.add_parser("run-all", help="Discover and process all pred_raw/last baseline meshes.")
    run_all.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    run_all.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    run_all.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_all.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    run_all.add_argument("--variants", default=",".join(f"{method}_random" for method in BASELINE_METHODS))
    run_all.add_argument("--scenes", default="")
    run_all.add_argument("--include-random", action="store_true")
    run_all.add_argument("--include-random-mv-sam3d", action="store_true")
    run_all.add_argument("--include-benchmark-random", action="store_true", default=True)
    run_all.add_argument("--no-include-benchmark-random", dest="include_benchmark_random", action="store_false")
    run_all.add_argument("--write-archive", action="store_true")
    run_all.add_argument("--skip-existing-archive", action="store_true")
    run_all.add_argument("--num-shards", type=int, default=1)
    run_all.add_argument("--shard-index", type=int, default=0)
    run_all.add_argument("--skip-existing", action="store_true")
    run_all.add_argument("--skip-render", action="store_true")
    add_registration_args(run_all)
    add_render_args(run_all)
    return parser


def add_registration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registration-method",
        choices=[method.value for method in manual.RegistrationMethod],
        default=manual.RegistrationMethod.INITIAL24_GICP_SCALE.value,
    )
    parser.add_argument("--sample-points", type=int, default=manual.SAMPLE_POINTS)
    parser.add_argument("--seed", type=int, default=manual.SEED)
    parser.add_argument("--gicp-max-iterations", type=int, default=80)
    parser.add_argument("--gicp-max-correspondence-distance-fraction", type=float, default=0.25)
    parser.add_argument("--scale-refine-trim-fraction", type=float, default=0.8)
    parser.add_argument("--initial-count", type=int, default=manual.DEFAULT_INITIAL_COUNT)
    parser.add_argument("--teaser-voxel-fraction", type=float, default=0.025)
    parser.add_argument("--icp-max-iterations", type=int, default=80)
    parser.add_argument("--icp-max-correspondence-distance-fraction", type=float, default=2.0)


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--views", default="all")
    parser.add_argument("--near", type=float, default=manual.RENDER_NEAR)
    parser.add_argument("--far", type=float, default=manual.RENDER_FAR)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--lpips-max-side", type=int, default=manual.LPIPS_MAX_SIDE)
    parser.add_argument("--y-convention", choices=Y_CONVENTIONS, default=None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def split_csv(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def discover_baseline_meshes(
    input_root: Path,
    *,
    methods: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
    include_random: bool = False,
    include_random_mv_sam3d: bool = False,
) -> list[BaselineMeshJob]:
    input_root = Path(input_root).expanduser().resolve()
    method_filter = None if not methods else set(methods)
    scene_filter = None if not scenes else set(scenes)
    jobs: list[BaselineMeshJob] = []
    for mesh_path in sorted(input_root.glob("*/pred_raw/last/*.glb")):
        scene = mesh_path.parents[2].name
        method = mesh_path.stem
        if scene_filter is not None and scene not in scene_filter:
            continue
        if method_filter is not None and method not in method_filter:
            continue
        jobs.append(BaselineMeshJob(scene=scene, method=method, mesh_path=mesh_path))
    if include_random or include_random_mv_sam3d:
        for mesh_path in sorted(input_root.glob("*/pred_raw/random/*.glb")):
            scene = mesh_path.parents[2].name
            method = f"{mesh_path.stem}_random"
            if include_random_mv_sam3d and not include_random and method != "mv_sam3d_random":
                continue
            if scene_filter is not None and scene not in scene_filter:
                continue
            if method_filter is not None and method not in method_filter:
                continue
            jobs.append(
                BaselineMeshJob(
                    scene=scene,
                    method=method,
                    mesh_path=mesh_path,
                    chunk="random",
                )
            )
    return sorted(jobs, key=lambda item: (item.method, item.scene))


def discover_benchmark_random_meshes(
    benchmark_root: Path,
    *,
    methods: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
) -> list[BaselineMeshJob]:
    benchmark_root = Path(benchmark_root).expanduser().resolve()
    method_filter = None if not methods else set(methods)
    scene_filter = None if not scenes else set(scenes)
    jobs: list[BaselineMeshJob] = []
    for mesh_path in sorted(benchmark_root.glob("*/*/predictions/model.glb")):
        method = mesh_path.parents[2].name
        scene = mesh_path.parents[1].name
        variant = f"{method}_random"
        if method not in BASELINE_METHODS:
            continue
        if method_filter is not None and variant not in method_filter and method not in method_filter:
            continue
        if scene_filter is not None and scene not in scene_filter:
            continue
        jobs.append(
            BaselineMeshJob(
                scene=scene,
                method=variant,
                mesh_path=mesh_path,
                chunk="random",
                source_kind="streamsam3d_benchmark_last_gso",
            )
        )
    return sorted(jobs, key=lambda item: (item.method, item.scene))


def merge_jobs_prefer_local(local_jobs: Sequence[BaselineMeshJob], benchmark_jobs: Sequence[BaselineMeshJob]) -> list[BaselineMeshJob]:
    merged: dict[tuple[str, str, str], BaselineMeshJob] = {}
    for job in benchmark_jobs:
        merged[(job.method, job.scene, job.chunk)] = job
    for job in local_jobs:
        merged[(job.method, job.scene, job.chunk)] = job
    return sorted(merged.values(), key=lambda item: (item.method, item.scene, item.chunk))


def shard_jobs(jobs: Sequence[BaselineMeshJob], num_shards: int, shard_index: int) -> list[BaselineMeshJob]:
    if int(num_shards) <= 0:
        raise ValueError(f"--num-shards must be positive, got {num_shards}")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError(f"--shard-index must be in [0, {int(num_shards) - 1}], got {shard_index}")
    return [job for index, job in enumerate(jobs) if index % int(num_shards) == int(shard_index)]


def output_dir_for_job(output_root: Path, job: BaselineMeshJob) -> Path:
    return Path(output_root).expanduser().resolve() / job.method / job.scene / job.chunk


def infer_job_from_mesh(mesh_path: Path, scene: str | None, method: str | None) -> BaselineMeshJob:
    mesh_path = Path(mesh_path).expanduser().resolve()
    inferred_scene = scene if scene else mesh_path.parents[2].name
    chunk = mesh_path.parent.name
    inferred_method = method if method else ("mv_sam3d_random" if chunk == "random" and mesh_path.stem == "mv_sam3d" else mesh_path.stem)
    return BaselineMeshJob(scene=str(inferred_scene), method=str(inferred_method), mesh_path=mesh_path, chunk=chunk)


def load_scene_meshes(path: Path) -> list[Any]:
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = loaded.dump()
    else:
        meshes = [loaded]
    return [mesh for mesh in meshes if hasattr(mesh, "vertices") and hasattr(mesh, "faces") and len(mesh.vertices) and len(mesh.faces)]


def load_render_mesh(path: Path, basis: np.ndarray) -> RenderMesh:
    meshes = load_scene_meshes(path)
    if not meshes:
        raise ValueError(f"No triangle mesh geometry found in {path}")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_colors: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    texture: np.ndarray | None = None
    base_color_factor = np.ones(4, dtype=np.float32)
    vertex_offset = 0
    visual_kind: str | None = None

    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + vertex_offset)
        visual = mesh.visual
        kind = str(getattr(visual, "kind", "none"))
        visual_kind = kind if visual_kind is None else visual_kind
        if kind == "vertex":
            colors = np.asarray(visual.vertex_colors, dtype=np.float32)
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            vertex_colors.append(colors[:, :3])
        elif kind == "texture":
            uv = np.asarray(visual.uv, dtype=np.float32)
            uvs.append(uv)
            material = visual.material
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            if image is not None and texture is None:
                texture = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None:
                base_color_factor = np.asarray(factor, dtype=np.float32).reshape(-1)
                if base_color_factor.max(initial=0.0) > 1.0:
                    base_color_factor = base_color_factor / 255.0
        else:
            vertex_colors.append(np.ones((len(mesh_vertices), 3), dtype=np.float32) * 0.8)
        vertex_offset += len(mesh_vertices)

    all_vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    all_faces = np.concatenate(faces, axis=0).astype(np.int32)
    if visual_kind == "texture" and uvs and texture is not None:
        return RenderMesh(all_vertices, all_faces, None, np.concatenate(uvs, axis=0).astype(np.float32), texture, base_color_factor)
    if vertex_colors:
        return RenderMesh(all_vertices, all_faces, np.concatenate(vertex_colors, axis=0).astype(np.float32), None, None, base_color_factor)
    return RenderMesh(all_vertices, all_faces, np.ones((len(all_vertices), 3), dtype=np.float32) * 0.8, None, None, base_color_factor)


def load_registration_mesh(path: Path, basis: np.ndarray) -> Any:
    return manual.load_mesh(path, basis)


def transform_vertices(vertices: np.ndarray, sim3: dict[str, Any]) -> np.ndarray:
    return manual.apply_sim3_to_points(vertices, sim3).astype(np.float32)


def export_aligned_mesh(source_path: Path, destination: Path, sim3: dict[str, Any], basis: np.ndarray) -> bool:
    import trimesh

    meshes = load_scene_meshes(source_path)
    if not meshes:
        return False
    transformed = []
    for mesh in meshes:
        mesh_copy = mesh.copy()
        vertices = np.asarray(mesh_copy.vertices, dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
        mesh_copy.vertices = manual.apply_sim3_to_points(vertices, sim3)
        transformed.append(mesh_copy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(transformed) == 1:
        transformed[0].export(str(destination))
    else:
        scene = trimesh.Scene()
        for index, mesh in enumerate(transformed):
            scene.add_geometry(mesh, node_name=f"mesh_{index:03d}")
        scene.export(str(destination))
    return True


def copy_file(src: Path, dst: Path) -> bool:
    src = Path(src)
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path) -> bool:
    src = Path(src)
    if not src.is_dir():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return True


def archive_variant_dir(archive_root: Path, job: BaselineMeshJob) -> Path:
    return Path(archive_root).expanduser().resolve() / "scenes" / job.scene / job.method


def archive_record_from_outputs(
    *,
    archive_root: Path,
    job: BaselineMeshJob,
    registration_output_dir: Path,
) -> dict[str, Any]:
    archive_root = Path(archive_root).expanduser().resolve()
    variant_root = archive_variant_dir(archive_root, job)
    assets_root = variant_root / "assets"
    renderings_root = variant_root / "renderings"
    metrics_root = variant_root / "metrics"
    provenance_root = variant_root / "provenance"
    state_path = registration_output_dir / manual.STATE_FILENAME
    state = manual.read_state(state_path)
    copied = {
        "raw_mesh": copy_file(job.mesh_path, assets_root / "raw_result.glb"),
        "aligned_mesh": export_aligned_mesh(
            job.mesh_path,
            assets_root / "aligned_mesh.glb",
            state["active_sim3"],
            manual.source_geometry_basis_matrix_from_state(state),
        ),
        "alignment_state": copy_file(state_path, provenance_root / manual.STATE_FILENAME),
        "render_audit.json": copy_file(registration_output_dir / "render_audit.json", provenance_root / "render_audit.json"),
        "renders": copy_tree(registration_output_dir / "renders", renderings_root / "renders"),
        "comparisons": copy_tree(registration_output_dir / "comparisons", renderings_root / "comparisons"),
        "gt_render_images": copy_tree(registration_output_dir / "gt", renderings_root / "gt"),
        "summary.json": copy_file(registration_output_dir / "summary.json", metrics_root / "summary.json"),
        "per_view_metrics.csv": copy_file(registration_output_dir / "per_view_metrics.csv", metrics_root / "per_view_metrics.csv"),
    }
    summary_path = registration_output_dir / "summary.json"
    metrics = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = {
            "variant": job.method,
            "scene": job.scene,
            "chunk": job.chunk,
            "state_file": str(state_path.resolve()),
            "output_dir": str(registration_output_dir.resolve()),
            "view_count": str(summary.get("view_count", "")),
            "render_count": str(len(list((registration_output_dir / "renders").glob("*_pred.png")))),
            "comparison_count": str(len(list((registration_output_dir / "comparisons").glob("*_gt_pred.png")))),
            "complete_25": str(len(list((registration_output_dir / "renders").glob("*_pred.png"))) == 25),
            "cd_l2": str(summary.get("active_mesh_diagnostics", {}).get("cd", "")),
            "psnr": str(summary.get("metrics", {}).get("psnr", {}).get("mean", "")),
            "ssim": str(summary.get("metrics", {}).get("ssim", {}).get("mean", "")),
            "lpips": str(summary.get("metrics", {}).get("lpips", {}).get("mean", "")),
            "alpha_mask_iou": str(summary.get("metrics", {}).get("alpha_mask_iou", {}).get("mean", "")),
            "bbox_iou": str(summary.get("metrics", {}).get("bbox_iou", {}).get("mean", "")),
            "needs_manual_review": str(summary.get("needs_manual_review", "")),
        }
    record = {
        "variant": job.method,
        "label": job.method.replace("_", " "),
        "group": "external_random_baseline",
        "scene": job.scene,
        "chunk": job.chunk,
        "source_chunk": str(job.mesh_path.parent),
        "source_kind": job.source_kind,
        "source_mesh": str(job.mesh_path),
        "registration_output_dir": str(registration_output_dir),
        "state_path": str(state_path),
        "metrics": metrics,
        "copied": copied,
    }
    manual.write_json(variant_root / "asset_manifest.json", record)
    return record


def upsert_archive_manifest(archive_root: Path, records: Sequence[dict[str, Any]]) -> None:
    archive_root = Path(archive_root).expanduser().resolve()
    manifest_path = archive_root / "archive_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"archive_root": str(archive_root), "methods": {}, "gt_records": [], "records": [], "variants": [], "scenes": []}
    by_key = {
        (str(record["variant"]), str(record["scene"]), str(record.get("chunk", ""))): record
        for record in manifest.get("records", [])
    }
    for record in records:
        by_key[(str(record["variant"]), str(record["scene"]), str(record.get("chunk", "")))] = record
        manifest.setdefault("methods", {})[str(record["variant"])] = {
            "label": str(record.get("label", record["variant"])),
            "group": str(record.get("group", "external_random_baseline")),
        }
    merged = sorted(by_key.values(), key=lambda item: (str(item["variant"]), str(item["scene"]), str(item.get("chunk", ""))))
    manifest["records"] = merged
    manifest["variants"] = sorted({str(record["variant"]) for record in merged})
    manifest["scenes"] = sorted({str(record["scene"]) for record in merged})
    manifest["variant_count"] = len(manifest["variants"])
    manifest["scene_count"] = len(manifest["scenes"])
    manifest["record_count"] = len(merged)
    manual.write_json(manifest_path, manifest)


def clip_positions(
    vertices: np.ndarray,
    *,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    near: float,
    far: float,
    y_convention: str,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    camera = (np.asarray(w2c, dtype=np.float32)[:3, :3] @ vertices.T + np.asarray(w2c, dtype=np.float32)[:3, 3:4]).T
    z = np.maximum(camera[:, 2], 1e-6)
    x_norm = camera[:, 0] / z
    y_norm = camera[:, 1] / z
    u = float(intrinsic[0, 0]) * x_norm + float(intrinsic[0, 2])
    v = float(intrinsic[1, 1]) * y_norm + float(intrinsic[1, 2])
    x_ndc = 2.0 * u - 1.0
    if y_convention == "opencv_y_down":
        y_ndc = 1.0 - 2.0 * v
    elif y_convention == "opencv_y_up":
        y_ndc = 2.0 * v - 1.0
    else:
        raise ValueError(f"Unsupported y convention: {y_convention}")
    z_ndc = 2.0 * ((z - float(near)) / max(float(far) - float(near), 1e-6)) - 1.0
    return np.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], axis=1).astype(np.float32)


def render_mesh_nvdiffrast(
    mesh: RenderMesh,
    *,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    near: float,
    far: float,
    y_convention: str,
    device: torch.device,
    ctx: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import nvdiffrast.torch as dr

    pos_np = clip_positions(
        mesh.vertices,
        w2c=w2c,
        intrinsic=intrinsic,
        near=near,
        far=far,
        y_convention=y_convention,
    )
    pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device).unsqueeze(0)
    tri = torch.as_tensor(mesh.faces, dtype=torch.int32, device=device)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[int(image_height), int(image_width)])
    alpha = rast[0, :, :, 3] > 0
    if mesh.texture is not None and mesh.uvs is not None:
        uv = torch.as_tensor(mesh.uvs.copy(), dtype=torch.float32, device=device)
        uv[:, 1] = 1.0 - uv[:, 1]
        uv_interp, _ = dr.interpolate(uv.unsqueeze(0), rast, tri)
        tex = torch.as_tensor(mesh.texture, dtype=torch.float32, device=device).unsqueeze(0)
        rgb = dr.texture(tex, uv_interp, filter_mode="linear", boundary_mode="clamp")[0]
        factor = torch.as_tensor(mesh.base_color_factor[:3], dtype=torch.float32, device=device).view(1, 1, 3)
        rgb = rgb * factor
    else:
        colors = torch.as_tensor(mesh.vertex_colors, dtype=torch.float32, device=device).unsqueeze(0)
        rgb, _ = dr.interpolate(colors, rast, tri)
        rgb = rgb[0]
    rgb = torch.where(alpha[..., None], rgb.clamp(0.0, 1.0), torch.zeros_like(rgb))
    return rgb.detach().cpu().numpy().astype(np.float32), alpha.detach().cpu().numpy()


class BaseColorEnvMap:
    def shade(
        self,
        gb_pos: torch.Tensor,
        gb_normal: torch.Tensor,
        kd: torch.Tensor,
        ks: torch.Tensor,
        view_pos: torch.Tensor,
        specular: bool = True,
    ) -> torch.Tensor:
        return kd

    def sample(self, directions: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(directions)


def sim3_matrix_tensor(sim3: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(sim3["matrix"], dtype=np.float32), dtype=torch.float32, device=device)


def trellis_intrinsic_for_renderer(intrinsic: np.ndarray, y_convention: str) -> np.ndarray:
    adjusted = np.asarray(intrinsic, dtype=np.float32).copy()
    if y_convention == "opencv_y_down":
        return adjusted
    if y_convention == "opencv_y_up":
        adjusted[1, 1] = -adjusted[1, 1]
        adjusted[1, 2] = 1.0 - adjusted[1, 2]
        return adjusted
    raise ValueError(f"Unsupported y convention: {y_convention}")


def load_trellis_native_context(
    state: dict[str, Any],
    *,
    resolution: int,
    near: float,
    far: float,
    device: torch.device,
) -> TrellisNativeRenderContext:
    trellis_root = manual.REPO_ROOT / "TRELLIS.2"
    if str(trellis_root) not in sys.path:
        sys.path.insert(0, str(trellis_root))
    from trellis2.renderers import PbrMeshRenderer
    from trellis2.streaming3d.native_state import load_meshwithvoxel_state

    native_state_path = Path(state["trellis_native_state_path"]).expanduser().resolve()
    mesh = load_meshwithvoxel_state(native_state_path, device=device)
    renderer = PbrMeshRenderer(
        {
            "resolution": int(resolution),
            "near": float(near),
            "far": float(far),
            "ssaa": 1,
            "peel_layers": 8,
        },
        device=str(device),
    )
    return TrellisNativeRenderContext(
        mesh=mesh,
        renderer=renderer,
        envmap=BaseColorEnvMap(),
        render_key="base_color",
    )


def render_trellis_native_frame(
    context: TrellisNativeRenderContext,
    *,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    sim3: dict[str, Any],
    y_convention: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    trellis_intrinsic = trellis_intrinsic_for_renderer(intrinsic, y_convention)
    output = context.renderer.render(
        context.mesh,
        torch.as_tensor(np.asarray(w2c, dtype=np.float32), dtype=torch.float32, device=device),
        torch.as_tensor(trellis_intrinsic, dtype=torch.float32, device=device),
        envmap=context.envmap,
        transformation=sim3_matrix_tensor(sim3, device),
    )
    rgb = output[context.render_key].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    alpha = output["alpha"].detach().clamp(0.0, 1.0).cpu().numpy() > 0.5
    return rgb, alpha


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def bbox_center_distance_fraction(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return float("inf")
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float64)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)
    diag = max(np.linalg.norm([b[2] - b[0], b[3] - b[1]]), 1.0)
    return float(np.linalg.norm(ac - bc) / diag)


def alpha_audit_row(view: str, pred_alpha: np.ndarray, gt_alpha: np.ndarray) -> dict[str, Any]:
    pred_bbox = mask_bbox(pred_alpha)
    gt_bbox = mask_bbox(gt_alpha)
    box_iou = bbox_iou(pred_bbox, gt_bbox)
    center_shift = bbox_center_distance_fraction(pred_bbox, gt_bbox)
    alpha_iou = manual.mask_iou(pred_alpha, gt_alpha)
    suspicious = bool(box_iou < 0.25 or center_shift > 0.5 or not math.isfinite(float(alpha_iou)))
    return {
        "view": view,
        "alpha_mask_iou": alpha_iou,
        "bbox_iou": box_iou,
        "bbox_center_shift_fraction": center_shift,
        "pred_alpha_coverage": float(np.asarray(pred_alpha, dtype=bool).mean()),
        "gt_alpha_coverage": float(np.asarray(gt_alpha, dtype=bool).mean()),
        "suspicious": suspicious,
    }


def select_y_convention(
    target_mesh: RenderMesh,
    *,
    frames: Sequence[manual.FramePair],
    focal: dict[str, Any],
    render_args: argparse.Namespace,
    device: torch.device,
    ctx: Any,
) -> dict[str, Any]:
    intrinsic = manual.normalized_intrinsic(float(focal["focal_norm"]))
    audits: list[dict[str, Any]] = []
    for convention in Y_CONVENTIONS:
        rows = []
        for frame in frames:
            _, gt_alpha = manual.load_rgba(frame.png_path)
            height, width = gt_alpha.shape
            _, pred_alpha = render_mesh_nvdiffrast(
                target_mesh,
                w2c=manual.load_w2c_cv(frame),
                intrinsic=intrinsic,
                image_height=height,
                image_width=width,
                near=float(render_args.near),
                far=float(render_args.far),
                y_convention=convention,
                device=device,
                ctx=ctx,
            )
            rows.append(alpha_audit_row(f"{frame.view_index:03d}", pred_alpha, gt_alpha))
        audits.append(
            {
                "y_convention": convention,
                "mean_bbox_iou": float(np.mean([row["bbox_iou"] for row in rows])),
                "median_bbox_iou": float(np.median([row["bbox_iou"] for row in rows])),
                "mean_alpha_iou": float(np.mean([row["alpha_mask_iou"] for row in rows])),
                "per_view": rows,
            }
        )
    audits.sort(key=lambda row: (float(row["mean_bbox_iou"]), float(row["mean_alpha_iou"])), reverse=True)
    best = audits[0]
    return {
        "selected_y_convention": best["y_convention"],
        "status": "ok" if float(best["mean_bbox_iou"]) >= 0.75 else "suspicious",
        "candidates": audits,
    }


def write_per_view_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "view",
        "psnr",
        "ssim",
        "lpips",
        "alpha_mask_iou",
        "render_alpha_coverage",
        "target_mask_coverage",
        "bbox_iou",
        "bbox_center_shift_fraction",
        "suspicious",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def render_from_state(
    state_file: Path,
    *,
    output_dir: Path | None,
    views: str,
    cuda_device: int,
    near: float,
    far: float,
    lpips_max_side: int,
    y_convention: str | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast baseline mesh rendering.")
    import nvdiffrast.torch as dr

    state_path = Path(state_file).expanduser().resolve()
    state = manual.read_state(state_path)
    gt_root = Path(state["gt_root"])
    output_root = state_path.parent if output_dir is None else Path(output_dir).expanduser().resolve()
    frames = manual.discover_render_mvs25_views(gt_root)
    selected_indices = manual.parse_views(views, len(frames))
    selected_frames = [frames[index] for index in selected_indices]

    torch.cuda.set_device(int(cuda_device))
    device = torch.device(f"cuda:{int(cuda_device)}")
    ctx = dr.RasterizeCudaContext(device=device)

    target_mesh_for_calibration = manual.load_mesh(manual.target_mesh_path_from_state(state), manual.MVS25_MESH_BASIS)
    focal = manual.calibrate_focal_norm(gt_root, target_mesh_for_calibration, frames)
    intrinsic = manual.normalized_intrinsic(float(focal["focal_norm"]))
    use_trellis_native = "trellis_native_state_path" in state
    source_mesh = None
    native_context = None
    if use_trellis_native:
        native_context = load_trellis_native_context(
            state,
            resolution=int(Image.open(selected_frames[0].png_path).height),
            near=float(near),
            far=float(far),
            device=device,
        )
    else:
        source_mesh = load_render_mesh(manual.source_geometry_path_from_state(state), manual.source_geometry_basis_matrix_from_state(state))
        source_mesh = RenderMesh(
            vertices=transform_vertices(source_mesh.vertices, state["active_sim3"]),
            faces=source_mesh.faces,
            vertex_colors=source_mesh.vertex_colors,
            uvs=source_mesh.uvs,
            texture=source_mesh.texture,
            base_color_factor=source_mesh.base_color_factor,
        )
    target_render_mesh = load_render_mesh(manual.target_mesh_path_from_state(state), manual.MVS25_MESH_BASIS)

    if y_convention is None:
        gt_self_audit = select_y_convention(
            target_render_mesh,
            frames=frames,
            focal=focal,
            render_args=argparse.Namespace(near=near, far=far),
            device=device,
            ctx=ctx,
        )
        selected_y = str(gt_self_audit["selected_y_convention"])
    else:
        selected_y = str(y_convention)
        gt_self_audit = {"selected_y_convention": selected_y, "status": "forced", "candidates": []}

    lpips_metric = manual.make_lpips_metric("vgg").to(device)
    render_dir = output_root / "renders"
    gt_dir = output_root / "gt"
    comparison_dir = output_root / "comparisons"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for frame in selected_frames:
            target_rgb, target_mask = manual.load_rgba(frame.png_path)
            height, width = target_mask.shape
            if native_context is not None:
                if height != width:
                    raise ValueError(f"TRELLIS native renderer requires square frames, got {width}x{height}.")
                rendered_rgb, rendered_alpha = render_trellis_native_frame(
                    native_context,
                    w2c=manual.load_w2c_cv(frame),
                    intrinsic=intrinsic,
                    sim3=state["active_sim3"],
                    y_convention=selected_y,
                    device=device,
                )
            else:
                rendered_rgb, rendered_alpha = render_mesh_nvdiffrast(
                    source_mesh,
                    w2c=manual.load_w2c_cv(frame),
                    intrinsic=intrinsic,
                    image_height=height,
                    image_width=width,
                    near=float(near),
                    far=float(far),
                    y_convention=selected_y,
                    device=device,
                    ctx=ctx,
                )
            view_name = f"{frame.view_index:03d}"
            render_path = render_dir / f"{view_name}_pred.png"
            gt_path = gt_dir / f"{view_name}_gt.png"
            comparison_path = comparison_dir / f"{view_name}_gt_pred.png"
            manual.save_rgba(render_path, rendered_rgb, rendered_alpha)
            manual.save_rgba(gt_path, target_rgb, target_mask)
            manual.save_gt_pred_comparison(comparison_path, target_rgb, target_mask, rendered_rgb, rendered_alpha)
            audit = alpha_audit_row(view_name, rendered_alpha, target_mask)
            row = {
                "view": view_name,
                "psnr": manual.compute_masked_psnr(rendered_rgb, target_rgb, target_mask),
                "ssim": manual.compute_masked_ssim(rendered_rgb, target_rgb, target_mask),
                "lpips": manual.compute_lpips_vgg(lpips_metric, rendered_rgb, target_rgb, target_mask, max_side=int(lpips_max_side)),
                "alpha_mask_iou": audit["alpha_mask_iou"],
                "render_alpha_coverage": audit["pred_alpha_coverage"],
                "target_mask_coverage": audit["gt_alpha_coverage"],
                "bbox_iou": audit["bbox_iou"],
                "bbox_center_shift_fraction": audit["bbox_center_shift_fraction"],
                "suspicious": audit["suspicious"],
                "render_path": str(render_path.resolve()),
                "target_path": str(gt_path.resolve()),
                "comparison_path": str(comparison_path.resolve()),
            }
            rows.append(row)
            audit_rows.append(audit)

    source_points = manual.source_geometry_points_from_state(state, sample_points_count=manual.SAMPLE_POINTS, seed=manual.SEED)
    active_mesh_diagnostics = manual.mesh_cd_diagnostics_from_points(
        source_points,
        manual.sample_mesh_points(target_mesh_for_calibration, manual.SAMPLE_POINTS, manual.SEED + 1),
        state["active_sim3"],
    )
    active_mesh_diagnostics["source_geometry_kind"] = manual.source_geometry_kind_from_state(state)
    active_mesh_diagnostics["source_geometry_path"] = str(manual.source_geometry_path_from_state(state).resolve())
    render_audit = {
        "status": "suspicious" if any(row["suspicious"] for row in audit_rows) else "ok",
        "gt_self_render": gt_self_audit,
        "pred_vs_gt_alpha": audit_rows,
    }
    summary = {
        "metric_set": "sam3d_gso_baseline_mesh_render_mvs_25",
        "scene": state["scene"],
        "method": state.get("variant", ""),
        "chunk": state.get("chunk", "last"),
        "gt_root": state["gt_root"],
        "state_file": str(state_path),
        "render_split": "render_mvs_25",
        "views": [row["view"] for row in rows],
        "view_count": len(rows),
        "render_backend": TRELLIS_NATIVE_RENDER_BACKEND if use_trellis_native else "nvdiffrast",
        "trellis_native_state_path": state.get("trellis_native_state_path"),
        "trellis_native_render_key": native_context.render_key if native_context is not None else None,
        "y_convention": selected_y,
        "focal_calibration": focal,
        "active_sim3": state["active_sim3"],
        "active_mesh_diagnostics": active_mesh_diagnostics,
        "source_geometry_kind": manual.source_geometry_kind_from_state(state),
        "source_geometry_path": str(manual.source_geometry_path_from_state(state).resolve()),
        "registration_method": state["registration_diagnostics"]["method"],
        "needs_manual_review": bool(render_audit["status"] != "ok"),
        "target_bbox_diagonal": float(state["target_bbox_diagonal"]),
        "metrics": {
            "psnr": manual.metric_stat([row["psnr"] for row in rows]),
            "ssim": manual.metric_stat([row["ssim"] for row in rows]),
            "lpips": manual.metric_stat([row["lpips"] for row in rows]),
            "alpha_mask_iou": manual.metric_stat([row["alpha_mask_iou"] for row in rows]),
            "bbox_iou": manual.metric_stat([row["bbox_iou"] for row in rows]),
        },
        "per_view": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    per_view_csv = output_root / "per_view_metrics.csv"
    summary_json = output_root / "summary.json"
    audit_json = output_root / "render_audit.json"
    write_per_view_csv(per_view_csv, rows)
    manual.write_json(summary_json, summary)
    manual.write_json(audit_json, render_audit)

    state["render_eval"] = {
        "summary": str(summary_json.resolve()),
        "metrics": summary["metrics"],
        "active_mesh_diagnostics": active_mesh_diagnostics,
        "render_backend": summary["render_backend"],
        "view_count": len(rows),
        "views": summary["views"],
        "render_audit": str(audit_json.resolve()),
        "updated_at_utc": summary["generated_at_utc"],
    }
    state["needs_manual_review"] = bool(render_audit["status"] != "ok")
    state["artifacts"].update(
        {
            "state_file": str(state_path),
            "output_dir": str(output_root.resolve()),
            "renders_dir": str(render_dir.resolve()),
            "gt_dir": str(gt_dir.resolve()),
            "comparisons_dir": str(comparison_dir.resolve()),
            "per_view_metrics_csv": str(per_view_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "render_audit_json": str(audit_json.resolve()),
        }
    )
    manual.write_json(state_path, state)
    return summary


def estimate_alignment(args: argparse.Namespace, job: BaselineMeshJob, output_dir: Path) -> dict[str, Any]:
    gt_root = (args.gt_root if getattr(args, "gt_root", None) is not None else manual.default_gt_root(job.scene)).expanduser().resolve()
    target_mesh_path = manual.gt_paths(gt_root)["mesh"]
    if not target_mesh_path.is_file():
        raise FileNotFoundError(f"Missing GT render_mvs_25 mesh: {target_mesh_path}")
    target_mesh = manual.load_mesh(target_mesh_path, manual.MVS25_MESH_BASIS)
    source_mesh = load_registration_mesh(job.mesh_path, BASELINE_MESH_BASIS)
    source_points = manual.sample_mesh_points(source_mesh, int(args.sample_points), int(args.seed))
    target_points = manual.sample_mesh_points(target_mesh, int(args.sample_points), int(args.seed) + 1)
    method = manual.RegistrationMethod(str(args.registration_method))
    if method in (manual.RegistrationMethod.INITIAL_GICP_SCALE, manual.RegistrationMethod.INITIAL24_GICP_SCALE):
        config = manual.InitialGicpScaleConfig(
            initial_count=int(args.initial_count),
            gicp_max_iterations=int(args.gicp_max_iterations),
            gicp_max_correspondence_distance_fraction=float(args.gicp_max_correspondence_distance_fraction),
            scale_refine_trim_fraction=float(args.scale_refine_trim_fraction),
        )
        sim3, diagnostics = manual.estimate_initial_gicp_scale(source_points, target_points, config)
        diagnostics["method_requested"] = method.value
    elif method == manual.RegistrationMethod.TEASERPP_ICP:
        config = manual.TeaserIcpConfig(
            teaser_voxel_fraction=float(args.teaser_voxel_fraction),
            icp_max_iterations=int(args.icp_max_iterations),
            icp_max_correspondence_distance_fraction=float(args.icp_max_correspondence_distance_fraction),
        )
        sim3, diagnostics = manual.estimate_teaserpp_fpfh_icp(source_points, target_points, config)
    else:
        raise ValueError(f"Unsupported registration method: {method}")

    sim3 = manual.retarget_sim3_source_space(sim3, BASELINE_MESH_SOURCE_SPACE)
    diagnostics = manual.rewrite_diagnostics_source_space(diagnostics, BASELINE_MESH_SOURCE_SPACE)
    diagnostics["source_geometry_kind"] = manual.SourceGeometryKind.BASELINE_MESH.value
    diagnostics["source_geometry_path"] = str(job.mesh_path.resolve())
    diagnostics["source_geometry_basis_matrix"] = BASELINE_MESH_BASIS.tolist()
    state = manual.state_with_initial_alignment(
        scene=job.scene,
        chunk_root=job.mesh_path.parent,
        gt_root=gt_root,
        source_geometry_kind=manual.SourceGeometryKind.BASELINE_MESH.value,
        source_geometry_path=job.mesh_path,
        source_local_alignment=manual.local_alignment_identity(
            source_space=BASELINE_MESH_SOURCE_SPACE,
            target_space=BASELINE_MESH_SOURCE_SPACE,
        ),
        target_mesh=target_mesh_path,
        target_bbox_diagonal=manual.point_bbox_diagonal(np.asarray(target_mesh.vertices, dtype=np.float64)),
        auto_initial_sim3=sim3,
        diagnostics=diagnostics,
    )
    state["variant"] = job.method
    state["chunk"] = job.chunk
    state["source_geometry_basis_matrix"] = BASELINE_MESH_BASIS.tolist()
    state["provenance"]["baseline_mesh_method"] = job.method
    state["provenance"]["baseline_mesh_source_space"] = BASELINE_MESH_SOURCE_SPACE
    if getattr(args, "trellis_native_state", None) is not None:
        native_state_path = args.trellis_native_state.expanduser().resolve()
        if not native_state_path.is_file():
            raise FileNotFoundError(f"Missing TRELLIS native state: {native_state_path}")
        state["trellis_native_state_path"] = str(native_state_path)
        state["provenance"]["trellis_native_state_path"] = str(native_state_path)
    state_path = output_dir / manual.STATE_FILENAME
    manual.write_json(state_path, state)
    result: dict[str, Any] = {"state_file": str(state_path.resolve()), "registration_diagnostics": diagnostics}
    if not bool(getattr(args, "skip_render", False)):
        result["render_eval"] = render_from_state(
            state_path,
            output_dir=output_dir,
            views=str(args.views),
            cuda_device=int(args.cuda_device),
            near=float(args.near),
            far=float(args.far),
            lpips_max_side=int(args.lpips_max_side),
            y_convention=args.y_convention,
        )
    return result


def state_index_row(job: BaselineMeshJob, output_dir: Path) -> dict[str, Any]:
    return {
        "variant": job.method,
        "scene": job.scene,
        "chunk": job.chunk,
        "state_file": str((output_dir / manual.STATE_FILENAME).resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_mesh": str(job.mesh_path.resolve()),
    }


def write_state_index(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def maybe_write_merged_index(output_root: Path, num_shards: int) -> None:
    output_root = Path(output_root).expanduser().resolve()
    shard_files = [output_root / f"final_state_index_shard_{idx:03d}_of_{int(num_shards):03d}.json" for idx in range(int(num_shards))]
    if not all(path.is_file() for path in shard_files):
        return
    rows: list[dict[str, Any]] = []
    for path in shard_files:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (str(row["variant"]), str(row["scene"]), str(row["chunk"])))
    write_state_index(output_root / "final_state_index.json", rows)


def run_estimate(args: argparse.Namespace) -> dict[str, Any]:
    job = infer_job_from_mesh(args.mesh, args.scene, args.method)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else output_dir_for_job(args.output_root, job)
    output_dir.mkdir(parents=True, exist_ok=True)
    return estimate_alignment(args, job, output_dir)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    methods = split_csv(args.variants)
    scenes = split_csv(args.scenes)
    local_jobs = discover_baseline_meshes(
        args.input_root,
        methods=methods,
        scenes=scenes,
        include_random=bool(args.include_random) or bool(args.include_benchmark_random),
        include_random_mv_sam3d=bool(args.include_random_mv_sam3d),
    )
    benchmark_jobs = (
        discover_benchmark_random_meshes(args.benchmark_root, methods=methods, scenes=scenes)
        if bool(args.include_benchmark_random)
        else []
    )
    jobs = merge_jobs_prefer_local(local_jobs, benchmark_jobs)
    shard = shard_jobs(jobs, int(args.num_shards), int(args.shard_index))
    rows: list[dict[str, Any]] = []
    archive_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job in shard:
        output_dir = output_dir_for_job(args.output_root, job)
        state_path = output_dir / manual.STATE_FILENAME
        summary_path = output_dir / "summary.json"
        archive_manifest_path = archive_variant_dir(args.archive_root, job) / "asset_manifest.json"
        if (
            bool(args.skip_existing)
            and state_path.is_file()
            and summary_path.is_file()
            and (not bool(args.write_archive) or not bool(args.skip_existing_archive) or archive_manifest_path.is_file())
        ):
            rows.append(state_index_row(job, output_dir))
            if bool(args.write_archive) and not archive_manifest_path.is_file():
                archive_records.append(
                    archive_record_from_outputs(
                        archive_root=args.archive_root,
                        job=job,
                        registration_output_dir=output_dir,
                    )
                )
            continue
        try:
            estimate_alignment(args, job, output_dir)
            rows.append(state_index_row(job, output_dir))
            if bool(args.write_archive):
                archive_records.append(
                    archive_record_from_outputs(
                        archive_root=args.archive_root,
                        job=job,
                        registration_output_dir=output_dir,
                    )
                )
        except Exception as exc:
            failures.append(
                {
                    "scene": job.scene,
                    "method": job.method,
                    "mesh_path": str(job.mesh_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    output_root = Path(args.output_root).expanduser().resolve()
    if int(args.num_shards) == 1:
        index_path = output_root / "final_state_index.json"
    else:
        index_path = output_root / f"final_state_index_shard_{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.json"
    write_state_index(index_path, rows)
    if int(args.num_shards) > 1:
        maybe_write_merged_index(output_root, int(args.num_shards))
    if bool(args.write_archive) and archive_records:
        upsert_archive_manifest(args.archive_root, archive_records)
    payload = {
        "discovered_count": len(jobs),
        "processed_count": len(rows),
        "archive_record_count": len(archive_records),
        "failure_count": len(failures),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "state_index": str(index_path.resolve()),
        "failures": failures,
    }
    if failures:
        failure_path = output_root / f"failures_shard_{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.json"
        manual.write_json(failure_path, failures)
        payload["failure_file"] = str(failure_path.resolve())
        raise RuntimeError(f"{len(failures)} baseline mesh jobs failed; details written to {failure_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "estimate":
        result = run_estimate(args)
    elif args.command == "render":
        result = render_from_state(
            args.state_file,
            output_dir=args.output_dir,
            views=str(args.views),
            cuda_device=int(args.cuda_device),
            near=float(args.near),
            far=float(args.far),
            lpips_max_side=int(args.lpips_max_side),
            y_convention=args.y_convention,
        )
    elif args.command == "run-all":
        result = run_all(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(manual.jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
