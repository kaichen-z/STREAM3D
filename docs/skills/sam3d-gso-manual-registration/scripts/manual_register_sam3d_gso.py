#!/usr/bin/env python3
"""Manual/agent-in-the-loop Sim(3) registration for SAM3D GSO chunks.

The script is deliberately standalone inside this skill: it does not import
from other skill directories.  It uses only repository modules under
``streaming`` / ``sam3d_objects`` plus normal third-party geometry packages.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "sam3d_objects").is_dir() and (parent / "streaming").is_dir():
            return parent
    cwd = Path.cwd().resolve()
    if (cwd / "sam3d_objects").is_dir() and (cwd / "streaming").is_dir():
        return cwd
    raise RuntimeError("Could not find repository root containing sam3d_objects and streaming.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORMAT_VERSION = 1
STATE_FILENAME = "alignment_state.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "gso" / "GSO30"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scratch" / "sam3d-gso-manual-registration"
DEFAULT_CHUNK_ROOT = REPO_ROOT / "tmp" / "gso30-streaming-ablation" / "k2_4" / "alarm" / "chunk_0016"
REGISTRATION_SPACE = "source_geometry_render_mvs_25"
SOURCE_SPACE = "result_glb_mesh_registration_basis"
PARAMS_SPARSE_SOURCE_SPACE = "params_sparse_canonical_basis"
RESULT_PLY_LOCAL_SOURCE_SPACE = "result_ply_local_gaussian_basis"
TARGET_SPACE = "render_mvs_25_gt_mesh_basis"
SAMPLE_POINTS = 4096
DEFAULT_INITIAL_COUNT = 24
SEED = 20260506
RENDER_NEAR = 0.01
RENDER_FAR = 100.0
RENDER_BACKEND = "gsplat"
LPIPS_MAX_SIDE = 512
PARAMS_GRID_RESOLUTION = 64.0

RESULT_GLB_TO_REGISTRATION_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
MVS25_MESH_BASIS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
MVS25_CAMERA_AXIS = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class FramePair:
    view_index: int
    png_path: Path
    npy_path: Path


class RegistrationMethod(StrEnum):
    TEASERPP_ICP = "teaserpp_icp"
    INITIAL_GICP_SCALE = "initial_gicp_scale"
    INITIAL24_GICP_SCALE = "initial24_gicp_scale"


class SourceGeometryKind(StrEnum):
    RESULT_GLB = "result_glb"
    PARAMS_SPARSE = "params_sparse"
    BASELINE_MESH = "baseline_mesh"


class SourceGeometryMode(StrEnum):
    AUTO = "auto"
    RESULT_GLB = SourceGeometryKind.RESULT_GLB.value
    PARAMS_SPARSE = SourceGeometryKind.PARAMS_SPARSE.value


@dataclass(frozen=True)
class PreparedSourceGeometry:
    kind: SourceGeometryKind
    path: Path
    registration_points: np.ndarray
    full_points: np.ndarray
    source_local_alignment: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TeaserIcpConfig:
    teaser_voxel_fraction: float = 0.025
    icp_max_iterations: int = 80
    icp_max_correspondence_distance_fraction: float = 2.0


@dataclass(frozen=True)
class InitialGicpScaleConfig:
    initial_count: int = DEFAULT_INITIAL_COUNT
    gicp_max_iterations: int = 80
    gicp_max_correspondence_distance_fraction: float = 0.25
    scale_refine_trim_fraction: float = 0.8


Initial24GicpScaleConfig = InitialGicpScaleConfig


@dataclass(frozen=True)
class RenderConfig:
    backend: str = RENDER_BACKEND
    near: float = RENDER_NEAR
    far: float = RENDER_FAR
    ssaa: int = 1
    lpips_max_side: int = LPIPS_MAX_SIDE
    cuda_device: int = 0
    flip_render_x: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate or re-render a manual global Sim(3) alignment for SAM3D GSO render_mvs_25."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate global Sim(3), write state, then render.")
    estimate.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    estimate.add_argument("--scene", required=True)
    estimate.add_argument("--variant")
    estimate.add_argument("--gt-root", type=Path)
    estimate.add_argument("--output-dir", type=Path)
    estimate.add_argument(
        "--source-geometry",
        choices=[mode.value for mode in SourceGeometryMode],
        default=SourceGeometryMode.AUTO.value,
    )
    estimate.add_argument(
        "--registration-method",
        choices=[method.value for method in RegistrationMethod],
        default=RegistrationMethod.TEASERPP_ICP.value,
    )
    estimate.add_argument("--sample-points", type=int, default=SAMPLE_POINTS)
    estimate.add_argument("--seed", type=int, default=SEED)
    estimate.add_argument("--teaser-voxel-fraction", type=float, default=0.025)
    estimate.add_argument("--icp-max-iterations", type=int, default=80)
    estimate.add_argument("--icp-max-correspondence-distance-fraction", type=float, default=2.0)
    estimate.add_argument("--gicp-max-iterations", type=int, default=80)
    estimate.add_argument("--gicp-max-correspondence-distance-fraction", type=float, default=0.25)
    estimate.add_argument("--scale-refine-trim-fraction", type=float, default=0.8)
    estimate.add_argument("--initial-count", type=int, default=DEFAULT_INITIAL_COUNT)
    estimate.add_argument("--views", default="all")
    estimate.add_argument("--skip-render", action="store_true")
    add_render_args(estimate)

    render = subparsers.add_parser("render", help="Render and evaluate the current active_sim3 from an existing state file.")
    render.add_argument("--state-file", type=Path, required=True)
    render.add_argument("--output-dir", type=Path)
    render.add_argument("--views", default="all")
    add_render_args(render)
    return parser


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--render-backend", default=RENDER_BACKEND)
    parser.add_argument("--near", type=float, default=RENDER_NEAR)
    parser.add_argument("--far", type=float, default=RENDER_FAR)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--lpips-max-side", type=int, default=LPIPS_MAX_SIDE)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--flip-render-x", dest="flip_render_x", action="store_true", default=True)
    parser.add_argument("--no-flip-render-x", dest="flip_render_x", action="store_false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def render_config_from_args(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(
        backend=str(args.render_backend),
        near=float(args.near),
        far=float(args.far),
        ssaa=int(args.ssaa),
        lpips_max_side=int(args.lpips_max_side),
        cuda_device=int(args.cuda_device),
        flip_render_x=bool(args.flip_render_x),
    )


def default_gt_root(scene: str) -> Path:
    return DEFAULT_DATA_ROOT / str(scene)


def infer_variant(chunk_root: Path, explicit_variant: str | None) -> str:
    if explicit_variant:
        return explicit_variant
    chunk_root = Path(chunk_root)
    if chunk_root.parent.parent.name:
        return chunk_root.parent.parent.name
    return "unknown_variant"


def default_output_dir(chunk_root: Path, scene: str, variant: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / str(variant) / str(scene) / Path(chunk_root).name


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    variant = infer_variant(args.chunk_root, args.variant)
    return default_output_dir(args.chunk_root, args.scene, variant).resolve()


def gt_paths(gt_root: Path) -> dict[str, Path]:
    split_root = Path(gt_root) / "render_mvs_25"
    return {
        "split_root": split_root,
        "mesh": split_root / "model_norm.glb",
        "model_dir": split_root / "model",
    }


def discover_render_mvs25_views(gt_root: Path) -> list[FramePair]:
    model_dir = gt_paths(gt_root)["model_dir"]
    pngs = {path.stem: path for path in model_dir.glob("*.png")}
    npys = {path.stem: path for path in model_dir.glob("*.npy")}
    stems = sorted(set(pngs) & set(npys), key=lambda item: int(item))
    if len(stems) != 25:
        raise FileNotFoundError(f"Expected 25 render_mvs_25 png/npy pairs under {model_dir}, found {len(stems)}.")
    return [FramePair(int(stem), pngs[stem], npys[stem]) for stem in stems]


def parse_views(spec: str, max_count: int) -> list[int]:
    token = str(spec).strip().lower()
    if token == "all":
        return list(range(max_count))
    if token in {"uniform", "even"}:
        token = "uniform:10"
    if token.startswith("uniform:") or token.startswith("even:"):
        count = int(token.split(":", 1)[1])
        return uniform_view_indices(max_count, count)

    views: list[int] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Bad view slice: {part}")
            start = int(pieces[0]) if pieces[0] else 0
            end = int(pieces[1]) if pieces[1] else max_count
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            views.extend(range(start, end, step))
        else:
            views.append(int(part))
    deduped = list(dict.fromkeys(views))
    bad = [idx for idx in deduped if idx < 0 or idx >= max_count]
    if bad:
        raise ValueError(f"View index out of range 0..{max_count - 1}: {bad}")
    return deduped


def uniform_view_indices(max_count: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError(f"Uniform view count must be positive, got {count}")
    if count >= max_count:
        return list(range(max_count))
    return [int(index) for index in np.linspace(0, max_count - 1, count, dtype=int)]


def load_mesh(path: Path, basis: np.ndarray) -> Any:
    import trimesh

    if not Path(path).is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
    else:
        mesh = loaded
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
    return mesh


def load_point_cloud_vertices(path: Path) -> np.ndarray:
    from plyfile import PlyData

    if not Path(path).is_file():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    ply = PlyData.read(str(path))
    vertex = ply.elements[0]
    return np.stack(
        (
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ),
        axis=1,
    )


def sample_mesh_points(mesh: Any, count: int, seed: int) -> np.ndarray:
    np.random.seed(int(seed))
    return np.asarray(mesh.sample(int(count)), dtype=np.float64)


def sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if count <= 0 or len(points) <= count:
        return points.copy()
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(points), size=int(count), replace=False)
    return points[indices].astype(np.float64)


def point_bbox_diagonal(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def apply_sim3_to_points(points: np.ndarray, sim3: dict[str, Any]) -> np.ndarray:
    scale, rotation, translation = sim3_components(sim3)
    points = np.asarray(points, dtype=np.float64)
    return float(scale) * (points @ rotation.T) + translation


def symmetric_chamfer_distance(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_to_tgt = cKDTree(target).query(source, k=1, workers=-1)[0]
    tgt_to_src = cKDTree(source).query(target, k=1, workers=-1)[0]
    return float(src_to_tgt.mean() + tgt_to_src.mean())


def iter_proper_signed_permutation_rotations() -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float64)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                rotation[row, col] = sign
            if np.linalg.det(rotation) > 0.0:
                name = "perm_{}_sign_{}".format(
                    "".join("xyz"[axis] for axis in perm),
                    "".join("p" if sign > 0.0 else "m" for sign in signs),
                )
                candidates.append((name, rotation))
    candidates.sort(key=lambda item: item[0])
    return candidates


def rotation_from_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    w, x, y, z = (np.asarray([w, x, y, z], dtype=np.float64) / max(norm, 1e-12)).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def iter_hopf_so3_rotations(count: int) -> list[tuple[str, np.ndarray]]:
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    candidates: list[tuple[str, np.ndarray]] = []
    for index in range(int(count)):
        u1 = (index + 0.5) / float(count)
        u2 = ((index + 0.5) / golden_ratio) % 1.0
        u3 = ((index + 0.5) * math.sqrt(2.0)) % 1.0
        x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
        y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
        z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
        w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
        candidates.append((f"so3_hopf_{index:03d}", rotation_from_quaternion_wxyz((w, x, y, z))))
    return candidates


def iter_initial_rotations(initial_count: int) -> list[tuple[str, np.ndarray]]:
    initial_count = int(initial_count)
    if initial_count < 1:
        raise ValueError(f"initial_count must be positive, got {initial_count}")
    base = iter_proper_signed_permutation_rotations()
    if initial_count <= len(base):
        return base[:initial_count]
    return base + iter_hopf_so3_rotations(initial_count - len(base))


def centroid_scale_sim3(source: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rotation = orthonormalize_rotation(rotation)
    source_rotated = source @ rotation.T
    source_mean = source_rotated.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_rms = np.sqrt(np.mean(np.sum((source_rotated - source_mean) ** 2, axis=1)))
    target_rms = np.sqrt(np.mean(np.sum((target - target_mean) ** 2, axis=1)))
    scale = float(target_rms / max(source_rms, 1e-8))
    translation = target_mean - scale * source_mean
    return make_sim3_dict(scale, rotation, translation)


def refine_scale_translation_from_nn(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    *,
    trim_fraction: float,
    pairing_sim3: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rotation = orthonormalize_rotation(rotation)
    rotated = source @ rotation.T
    query_points = apply_sim3_to_points(source, pairing_sim3) if pairing_sim3 is not None else rotated
    distances, indices = cKDTree(target).query(query_points, k=1, workers=-1)
    order = np.argsort(distances)
    keep = max(3, int(round(len(order) * float(trim_fraction))))
    keep_indices = order[: min(len(order), keep)]
    paired_source = rotated[keep_indices]
    paired_target = target[indices[keep_indices]]
    source_mean = paired_source.mean(axis=0)
    target_mean = paired_target.mean(axis=0)
    source_centered = paired_source - source_mean
    target_centered = paired_target - target_mean
    denom = float(np.sum(source_centered * source_centered))
    scale = 1.0 if denom <= 1e-12 else float(np.sum(source_centered * target_centered) / denom)
    translation = target_mean - scale * source_mean
    sim3 = make_sim3_dict(scale, rotation, translation)
    diagnostics = {
        "trim_fraction": float(trim_fraction),
        "paired_count": int(len(keep_indices)),
        "scale": float(scale),
        "mean_nn_distance_before_refit": float(np.mean(distances[keep_indices])),
        "pairing_source": "pairing_sim3" if pairing_sim3 is not None else "rotated_source",
    }
    return sim3, diagnostics


def run_generalized_icp(
    source: np.ndarray,
    target: np.ndarray,
    init_sim3: dict[str, Any],
    config: InitialGicpScaleConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import open3d as o3d

    scale, rotation, translation = sim3_components(init_sim3)
    prealigned = float(scale) * (np.asarray(source, dtype=np.float64) @ rotation.T) + translation
    target = np.asarray(target, dtype=np.float64)
    max_distance = max(point_bbox_diagonal(target), 1e-8) * float(config.gicp_max_correspondence_distance_fraction)
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prealigned))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_distance, max_nn=30))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_distance, max_nn=30))
    result = o3d.pipelines.registration.registration_generalized_icp(
        source_cloud,
        target_cloud,
        max_distance,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(config.gicp_max_iterations)),
    )
    delta = np.asarray(result.transformation, dtype=np.float64)
    delta_rotation = orthonormalize_rotation(delta[:3, :3])
    refined_rotation = delta_rotation @ rotation
    refined_translation = delta_rotation @ translation + delta[:3, 3]
    rigid_sim3 = make_sim3_dict(float(scale), refined_rotation, refined_translation)
    diagnostics = {
        "fixed_scale": float(scale),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "max_correspondence_distance": float(max_distance),
        "max_iterations": int(config.gicp_max_iterations),
        "delta_rotation": delta_rotation.tolist(),
        "delta_translation": delta[:3, 3].tolist(),
    }
    return rigid_sim3, diagnostics


def candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(record["cd_after_scale_refine"]),
        float(record["scale_refine"]["mean_nn_distance_before_refit"]),
        -float(record["gicp"]["fitness"]),
        float(record["gicp"]["inlier_rmse"]),
    )


def estimate_initial_gicp_scale(
    source: np.ndarray,
    target: np.ndarray,
    config: InitialGicpScaleConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cd_before = symmetric_chamfer_distance(source, target)
    records: list[dict[str, Any]] = []
    initial_rotations = iter_initial_rotations(config.initial_count)
    for name, initial_rotation in initial_rotations:
        init_sim3 = centroid_scale_sim3(source, target, initial_rotation)
        initial_cd = symmetric_chamfer_distance(apply_sim3_to_points(source, init_sim3), target)
        gicp_sim3, gicp_diag = run_generalized_icp(source, target, init_sim3, config)
        gicp_cd = symmetric_chamfer_distance(apply_sim3_to_points(source, gicp_sim3), target)
        _, gicp_rotation, _ = sim3_components(gicp_sim3)
        refined_sim3, scale_diag = refine_scale_translation_from_nn(
            source,
            target,
            gicp_rotation,
            trim_fraction=float(config.scale_refine_trim_fraction),
            pairing_sim3=gicp_sim3,
        )
        refined_cd = symmetric_chamfer_distance(apply_sim3_to_points(source, refined_sim3), target)
        records.append(
            {
                "candidate_id": name,
                "initial_sim3": init_sim3,
                "initial_cd": float(initial_cd),
                "gicp_sim3": gicp_sim3,
                "gicp": gicp_diag,
                "cd_after_gicp": float(gicp_cd),
                "scale_refine": scale_diag,
                "final_sim3": refined_sim3,
                "cd_after_scale_refine": float(refined_cd),
            }
        )

    records.sort(key=candidate_sort_key)
    best = records[0]
    diagnostics = {
        "method": f"initial{len(records)}_gicp_scale",
        "cd_before": float(cd_before),
        "cd_after": float(best["cd_after_scale_refine"]),
        "best_candidate_id": best["candidate_id"],
        "candidate_count": len(records),
        "initial_count": int(config.initial_count),
        "signed_permutation_candidate_count": min(24, len(records)),
        "hopf_so3_candidate_count": max(0, len(records) - 24),
        "gicp_max_iterations": int(config.gicp_max_iterations),
        "gicp_max_correspondence_distance_fraction": float(config.gicp_max_correspondence_distance_fraction),
        "scale_refine_trim_fraction": float(config.scale_refine_trim_fraction),
        "source_sample_points": int(source.shape[0]),
        "target_sample_points": int(target.shape[0]),
        "candidates": records,
    }
    return best["final_sim3"], diagnostics


def estimate_initial24_gicp_scale(
    source: np.ndarray,
    target: np.ndarray,
    config: Initial24GicpScaleConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return estimate_initial_gicp_scale(source, target, config)


def estimate_teaserpp_fpfh_icp(
    source: np.ndarray,
    target: np.ndarray,
    config: TeaserIcpConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import teaserpp_python
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Global Sim(3) estimation requires teaserpp_python. Install the TEASER++ Python binding in .env."
        ) from exc

    import open3d as o3d

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cd_before = symmetric_chamfer_distance(source, target)
    target_diagonal = max(point_bbox_diagonal(target), 1e-8)
    voxel_size = max(target_diagonal * float(config.teaser_voxel_fraction), 1e-5)

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source)).voxel_down_sample(voxel_size)
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)).voxel_down_sample(voxel_size)
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    source_fpfh = np.asarray(
        o3d.pipelines.registration.compute_fpfh_feature(
            source_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        ).data
    ).T
    target_fpfh = np.asarray(
        o3d.pipelines.registration.compute_fpfh_feature(
            target_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        ).data
    ).T
    source_points = np.asarray(source_cloud.points, dtype=np.float64)
    target_points = np.asarray(target_cloud.points, dtype=np.float64)
    target_matches = cKDTree(target_fpfh).query(source_fpfh, k=1, workers=-1)[1].astype(np.int64)
    source_matches = cKDTree(source_fpfh).query(target_fpfh, k=1, workers=-1)[1].astype(np.int64)
    source_indices = np.arange(len(source_points), dtype=np.int64)
    mutual = source_matches[target_matches] == source_indices
    source_indices = source_indices[mutual]
    target_indices = target_matches[mutual]
    if len(source_indices) < 3:
        raise RuntimeError(f"TEASER++ FPFH found only {len(source_indices)} mutual correspondences.")

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = voxel_size
    params.estimate_scaling = True
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(source_points[source_indices].T, target_points[target_indices].T)
    solution = solver.getSolution()

    teaser_scale = float(solution.scale)
    teaser_rotation = orthonormalize_rotation(np.asarray(solution.rotation, dtype=np.float64))
    teaser_translation = np.asarray(solution.translation, dtype=np.float64).reshape(3)
    teaser_sim3 = make_sim3_dict(teaser_scale, teaser_rotation, teaser_translation)
    cd_after_teaser = symmetric_chamfer_distance(apply_sim3_to_points(source, teaser_sim3), target)

    refined_sim3, icp_diag = refine_rigid_icp_after_teaser_scale(
        source,
        target,
        teaser_scale=teaser_scale,
        teaser_rotation=teaser_rotation,
        teaser_translation=teaser_translation,
        config=config,
    )
    cd_after = symmetric_chamfer_distance(apply_sim3_to_points(source, refined_sim3), target)
    diagnostics = {
        "method": "teaserpp_fpfh_icp",
        "cd_before": float(cd_before),
        "cd_after_teaser": float(cd_after_teaser),
        "cd_after": float(cd_after),
        "teaser_correspondence_count": int(len(source_indices)),
        "teaser_source_downsample_count": int(len(source_points)),
        "teaser_target_downsample_count": int(len(target_points)),
        "teaser_voxel_size": float(voxel_size),
        "teaser_scale": float(teaser_scale),
        "teaser_rotation": teaser_rotation.tolist(),
        "teaser_translation": teaser_translation.tolist(),
        "icp": icp_diag,
        "source_sample_points": int(source.shape[0]),
        "target_sample_points": int(target.shape[0]),
    }
    return refined_sim3, diagnostics


def refine_rigid_icp_after_teaser_scale(
    source: np.ndarray,
    target: np.ndarray,
    *,
    teaser_scale: float,
    teaser_rotation: np.ndarray,
    teaser_translation: np.ndarray,
    config: TeaserIcpConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import open3d as o3d

    teaser_rotation = orthonormalize_rotation(teaser_rotation)
    teaser_translation = np.asarray(teaser_translation, dtype=np.float64).reshape(3)
    prealigned = float(teaser_scale) * (np.asarray(source, dtype=np.float64) @ teaser_rotation.T) + teaser_translation
    target = np.asarray(target, dtype=np.float64)
    max_distance = max(point_bbox_diagonal(target), 1e-8) * float(config.icp_max_correspondence_distance_fraction)
    result = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prealigned)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)),
        max_distance,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(config.icp_max_iterations)),
    )
    delta = np.asarray(result.transformation, dtype=np.float64)
    delta_rotation = orthonormalize_rotation(delta[:3, :3])
    rotation = delta_rotation @ teaser_rotation
    translation = delta_rotation @ teaser_translation + delta[:3, 3]
    sim3 = make_sim3_dict(float(teaser_scale), rotation, translation)
    diagnostics = {
        "fixed_scale": float(teaser_scale),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "max_correspondence_distance": float(max_distance),
        "max_iterations": int(config.icp_max_iterations),
        "delta_rotation": delta_rotation.tolist(),
        "delta_translation": delta[:3, 3].tolist(),
    }
    return sim3, diagnostics


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    u, _, vt = np.linalg.svd(rotation)
    refined = u @ vt
    if np.linalg.det(refined) < 0.0:
        u[:, -1] *= -1.0
        refined = u @ vt
    return refined


def make_sim3_dict(
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    source_space: str = SOURCE_SPACE,
    target_space: str = TARGET_SPACE,
) -> dict[str, Any]:
    rotation = orthonormalize_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = float(scale) * rotation
    matrix[:3, 3] = translation
    return {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "matrix": matrix.tolist(),
        "source_space": source_space,
        "target_space": target_space,
    }


def sim3_components(sim3: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(sim3["scale"])
    rotation = orthonormalize_rotation(np.asarray(sim3["rotation"], dtype=np.float64))
    translation = np.asarray(sim3["translation"], dtype=np.float64).reshape(3)
    return scale, rotation, translation


def compose_sim3(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_scale, left_rotation, left_translation = sim3_components(left)
    right_scale, right_rotation, right_translation = sim3_components(right)
    scale = left_scale * right_scale
    rotation = left_rotation @ right_rotation
    translation = left_scale * (right_translation @ left_rotation.T) + left_translation
    return make_sim3_dict(
        scale,
        rotation,
        translation,
        source_space=right["source_space"],
        target_space=left["target_space"],
    )


def params_sparse_coords_to_points(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"Expected params coords with shape (N,3) or (N,4), got {coords.shape}")
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    xyz = coords[:, [2, 0, 1]].astype(np.float64)
    return ((xyz + 0.5) / PARAMS_GRID_RESOLUTION) - 0.5


def local_alignment_identity(*, source_space: str, target_space: str) -> dict[str, Any]:
    return {
        "method": "identity",
        "basis_search_count": 0,
        "best_candidate_id": "identity",
        "cd_before": 0.0,
        "cd_after": 0.0,
        "sim3": make_sim3_dict(
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            source_space=source_space,
            target_space=target_space,
        ),
    }


def estimate_params_to_ply_local_sim3(
    params_points: np.ndarray,
    ply_points: np.ndarray,
) -> dict[str, Any]:
    params_points = np.asarray(params_points, dtype=np.float64)
    ply_points = np.asarray(ply_points, dtype=np.float64)
    if len(params_points) < 3 or len(ply_points) < 3:
        raise ValueError("Need at least three points in both params and result.ply clouds.")
    cd_before = symmetric_chamfer_distance(params_points, ply_points)
    records: list[dict[str, Any]] = []
    for candidate_id, rotation in iter_proper_signed_permutation_rotations():
        centroid_sim3 = centroid_scale_sim3(params_points, ply_points, rotation)
        scale, _, translation = sim3_components(centroid_sim3)
        sim3 = make_sim3_dict(
            scale,
            rotation,
            translation,
            source_space=PARAMS_SPARSE_SOURCE_SPACE,
            target_space=RESULT_PLY_LOCAL_SOURCE_SPACE,
        )
        cd_after = symmetric_chamfer_distance(apply_sim3_to_points(params_points, sim3), ply_points)
        records.append(
            {
                "candidate_id": candidate_id,
                "sim3": sim3,
                "cd_after": float(cd_after),
            }
        )
    records.sort(key=lambda item: float(item["cd_after"]))
    best = records[0]
    return {
        "method": "initial24_centroid_scale_signed_permutation",
        "basis_search_count": len(records),
        "best_candidate_id": best["candidate_id"],
        "cd_before": float(cd_before),
        "cd_after": float(best["cd_after"]),
        "candidates": records,
        "sim3": best["sim3"],
    }


def resolve_source_geometry_mode(chunk_root: Path, requested: str) -> tuple[SourceGeometryKind, Path]:
    chunk_root = Path(chunk_root)
    requested_mode = SourceGeometryMode(str(requested))
    result_glb = chunk_root / "result.glb"
    params_npz = chunk_root / "params.npz"
    if requested_mode == SourceGeometryMode.RESULT_GLB:
        if not result_glb.is_file():
            raise FileNotFoundError(f"Requested result_glb source geometry, but file is missing: {result_glb}")
        return SourceGeometryKind.RESULT_GLB, result_glb
    if requested_mode == SourceGeometryMode.PARAMS_SPARSE:
        if not params_npz.is_file():
            raise FileNotFoundError(f"Requested params_sparse source geometry, but file is missing: {params_npz}")
        return SourceGeometryKind.PARAMS_SPARSE, params_npz
    if result_glb.is_file():
        return SourceGeometryKind.RESULT_GLB, result_glb
    if params_npz.is_file():
        return SourceGeometryKind.PARAMS_SPARSE, params_npz
    raise FileNotFoundError(f"No supported source geometry found under {chunk_root}; expected result.glb or params.npz.")


def prepare_source_geometry(
    chunk_root: Path,
    requested_mode: str,
    *,
    sample_points_count: int,
    seed: int,
) -> PreparedSourceGeometry:
    kind, path = resolve_source_geometry_mode(chunk_root, requested_mode)
    chunk_root = Path(chunk_root)
    if kind == SourceGeometryKind.RESULT_GLB:
        mesh = load_mesh(path, RESULT_GLB_TO_REGISTRATION_BASIS)
        full_points = np.asarray(mesh.vertices, dtype=np.float64)
        return PreparedSourceGeometry(
            kind=kind,
            path=path.resolve(),
            registration_points=sample_mesh_points(mesh, sample_points_count, seed),
            full_points=full_points,
            source_local_alignment=local_alignment_identity(
                source_space=SOURCE_SPACE,
                target_space=SOURCE_SPACE,
            ),
            provenance={
                "requested_mode": str(requested_mode),
                "resolved_mode": kind.value,
                "params_path": str((chunk_root / "params.npz").resolve()) if (chunk_root / "params.npz").exists() else None,
                "result_ply_path": str((chunk_root / "result.ply").resolve()),
            },
        )

    params_path = path
    result_ply_path = chunk_root / "result.ply"
    if not result_ply_path.is_file():
        raise FileNotFoundError(f"Missing Gaussian PLY required for params_sparse local alignment: {result_ply_path}")
    with np.load(params_path) as data:
        if "coords" not in data:
            raise KeyError(f"params.npz missing required 'coords' array: {params_path}")
        params_points = params_sparse_coords_to_points(np.asarray(data["coords"]))
        pointmap_scale = None if "pointmap_scale" not in data else np.asarray(data["pointmap_scale"], dtype=np.float64)
        pointmap_shift = None if "pointmap_shift" not in data else np.asarray(data["pointmap_shift"], dtype=np.float64)
    ply_points = load_point_cloud_vertices(result_ply_path)
    ply_sample = sample_points(ply_points, sample_points_count, seed + 17)
    local_alignment = estimate_params_to_ply_local_sim3(params_points, ply_sample)
    full_points = apply_sim3_to_points(params_points, local_alignment["sim3"])
    return PreparedSourceGeometry(
        kind=kind,
        path=params_path.resolve(),
        registration_points=sample_points(full_points, sample_points_count, seed),
        full_points=full_points,
        source_local_alignment=local_alignment,
        provenance={
            "requested_mode": str(requested_mode),
            "resolved_mode": kind.value,
            "result_ply_path": str(result_ply_path.resolve()),
            "result_pose_npz": str((chunk_root / "result_pose.npz").resolve()) if (chunk_root / "result_pose.npz").exists() else None,
            "pointmap_scale": None if pointmap_scale is None else pointmap_scale.tolist(),
            "pointmap_shift": None if pointmap_shift is None else pointmap_shift.tolist(),
        },
    )


def target_mesh_path_from_state(state: dict[str, Any]) -> Path:
    if "target_geometry_path" in state:
        return Path(state["target_geometry_path"])
    return Path(state["target_mesh"])


def source_geometry_path_from_state(state: dict[str, Any]) -> Path:
    if "source_geometry_path" in state:
        return Path(state["source_geometry_path"])
    return Path(state["source_mesh"])


def source_geometry_kind_from_state(state: dict[str, Any]) -> str:
    return str(state.get("source_geometry_kind", SourceGeometryKind.RESULT_GLB.value))


def source_geometry_basis_matrix_from_state(state: dict[str, Any]) -> np.ndarray:
    kind = SourceGeometryKind(source_geometry_kind_from_state(state))
    if kind == SourceGeometryKind.BASELINE_MESH:
        return np.asarray(state["source_geometry_basis_matrix"], dtype=np.float64)
    if kind == SourceGeometryKind.RESULT_GLB:
        return RESULT_GLB_TO_REGISTRATION_BASIS
    raise ValueError(f"Source geometry kind {kind.value} does not use a mesh basis matrix.")


def source_geometry_points_from_state(
    state: dict[str, Any],
    *,
    sample_points_count: int,
    seed: int,
) -> np.ndarray:
    kind = SourceGeometryKind(source_geometry_kind_from_state(state))
    path = source_geometry_path_from_state(state)
    if kind in (SourceGeometryKind.RESULT_GLB, SourceGeometryKind.BASELINE_MESH):
        mesh = load_mesh(path, source_geometry_basis_matrix_from_state(state))
        return sample_mesh_points(mesh, sample_points_count, seed)
    if kind == SourceGeometryKind.PARAMS_SPARSE:
        with np.load(path) as data:
            points = params_sparse_coords_to_points(np.asarray(data["coords"]))
        local_alignment = state.get("source_local_alignment")
        if local_alignment is None or "sim3" not in local_alignment:
            raise KeyError("params_sparse state missing source_local_alignment.sim3")
        aligned = apply_sim3_to_points(points, local_alignment["sim3"])
        return sample_points(aligned, sample_points_count, seed)
    raise ValueError(f"Unsupported source geometry kind in state: {kind}")


def source_geometry_full_points_from_state(state: dict[str, Any]) -> np.ndarray:
    kind = SourceGeometryKind(source_geometry_kind_from_state(state))
    path = source_geometry_path_from_state(state)
    if kind in (SourceGeometryKind.RESULT_GLB, SourceGeometryKind.BASELINE_MESH):
        mesh = load_mesh(path, source_geometry_basis_matrix_from_state(state))
        return np.asarray(mesh.vertices, dtype=np.float64)
    if kind == SourceGeometryKind.PARAMS_SPARSE:
        with np.load(path) as data:
            points = params_sparse_coords_to_points(np.asarray(data["coords"]))
        local_alignment = state.get("source_local_alignment")
        if local_alignment is None or "sim3" not in local_alignment:
            raise KeyError("params_sparse state missing source_local_alignment.sim3")
        return apply_sim3_to_points(points, local_alignment["sim3"])
    raise ValueError(f"Unsupported source geometry kind in state: {kind}")


def retarget_sim3_source_space(sim3: dict[str, Any], source_space: str) -> dict[str, Any]:
    scale, rotation, translation = sim3_components(sim3)
    return make_sim3_dict(scale, rotation, translation, source_space=source_space, target_space=sim3["target_space"])


def rewrite_diagnostics_source_space(diagnostics: dict[str, Any], source_space: str) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(diagnostics))
    if "teaser_scale" in rewritten:
        pass
    for key in ("initial_sim3", "gicp_sim3", "final_sim3"):
        if key in rewritten and isinstance(rewritten[key], dict) and "rotation" in rewritten[key]:
            rewritten[key] = retarget_sim3_source_space(rewritten[key], source_space)
    if "candidates" in rewritten:
        rewritten["candidates"] = [rewrite_diagnostics_source_space(candidate, source_space) for candidate in rewritten["candidates"]]
    return rewritten


def state_with_initial_alignment(
    *,
    scene: str,
    chunk_root: Path,
    gt_root: Path,
    source_geometry_kind: str,
    source_geometry_path: Path,
    source_local_alignment: dict[str, Any],
    target_mesh: Path,
    target_bbox_diagonal: float,
    auto_initial_sim3: dict[str, Any],
    diagnostics: dict[str, Any],
    needs_manual_review: bool = False,
) -> dict[str, Any]:
    pose_path = Path(chunk_root) / "result_pose.npz"
    return {
        "format_version": FORMAT_VERSION,
        "scene": str(scene),
        "chunk_root": str(Path(chunk_root).resolve()),
        "gt_root": str(Path(gt_root).resolve()),
        "registration_space": REGISTRATION_SPACE,
        "source_mesh": str(Path(source_geometry_path).resolve()),
        "source_geometry_kind": str(source_geometry_kind),
        "source_geometry_path": str(Path(source_geometry_path).resolve()),
        "source_local_alignment": source_local_alignment,
        "target_mesh": str(Path(target_mesh).resolve()),
        "target_geometry_path": str(Path(target_mesh).resolve()),
        "target_bbox_diagonal": float(target_bbox_diagonal),
        "needs_manual_review": bool(needs_manual_review),
        "provenance": {
            "result_pose_npz": str(pose_path.resolve()) if pose_path.exists() else None,
            "result_pose_used_for_registration": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "auto_initial_sim3": auto_initial_sim3,
        "active_sim3": json.loads(json.dumps(auto_initial_sim3)),
        "registration_diagnostics": diagnostics,
        "render_eval": None,
        "artifacts": {},
    }


def read_state(path: Path) -> dict[str, Any]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(state["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"Unsupported alignment state format_version={state['format_version']}")
    return state


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return jsonable(float(value))
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
    return value


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[..., :3], array[..., 3] > 0.0


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(np.asarray(rgb) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    alpha_u8 = np.where(np.asarray(alpha, dtype=bool), 255, 0).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb_u8, alpha_u8[..., None]], axis=2)).save(path)


def save_gt_pred_comparison(
    path: Path,
    gt_rgb: np.ndarray,
    gt_alpha: np.ndarray,
    pred_rgb: np.ndarray,
    pred_alpha: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = composite_on_white(gt_rgb, gt_alpha)
    pred = composite_on_white(pred_rgb, pred_alpha)
    separator = np.ones((gt.shape[0], 4, 3), dtype=np.float32)
    image = np.concatenate([gt, separator, pred], axis=1)
    Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)).save(path)


def composite_on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_f = np.asarray(alpha, dtype=np.float32)[..., None]
    return np.asarray(rgb, dtype=np.float32) * alpha_f + 1.0 - alpha_f


def load_w2c_cv(frame: FramePair) -> np.ndarray:
    raw = np.eye(4, dtype=np.float64)
    raw[:3, :4] = np.load(frame.npy_path).astype(np.float64)
    return np.asarray(MVS25_CAMERA_AXIS @ raw, dtype=np.float32)


def normalized_intrinsic(focal_norm: float) -> np.ndarray:
    return np.array(
        [[float(focal_norm), 0.0, 0.5], [0.0, float(focal_norm), 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def calibrate_focal_norm(
    gt_root: Path,
    mesh: Any,
    frames: Sequence[FramePair],
    *,
    max_views: int | None = None,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    eval_frames = list(frames) if max_views is None else list(frames)[: int(max_views)]
    rows: list[dict[str, float | str]] = []
    for frame in eval_frames:
        _, mask = load_rgba(frame.png_path)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError(f"Empty alpha mask in {frame.png_path}")
        height, width = mask.shape
        bbox_width = float(xs.max() - xs.min() + 1)
        bbox_height = float(ys.max() - ys.min() + 1)
        w2c = load_w2c_cv(frame)
        camera_points = (w2c[:3, :3] @ vertices.T + w2c[:3, 3:4]).T
        valid = camera_points[:, 2] > 1e-6
        if not valid.any():
            raise ValueError(f"No positive-depth GT mesh vertices for {frame.png_path}")
        x_norm = camera_points[valid, 0] / camera_points[valid, 2]
        y_norm = camera_points[valid, 1] / camera_points[valid, 2]
        fx = bbox_width / float(x_norm.max() - x_norm.min())
        fy = bbox_height / float(y_norm.max() - y_norm.min())
        rows.append(
            {
                "frame": frame.png_path.stem,
                "fx_norm": float(fx / width),
                "fy_norm": float(fy / height),
            }
        )
    focal_values = np.asarray([row["fx_norm"] for row in rows] + [row["fy_norm"] for row in rows], dtype=np.float64)
    return {
        "method": "median_bbox_fit_to_gt_alpha_masks",
        "split": "render_mvs_25",
        "gt_root": str(Path(gt_root).resolve()),
        "view_count": len(rows),
        "focal_norm": float(np.median(focal_values)),
        "fx_norm_mean": float(np.mean([row["fx_norm"] for row in rows])),
        "fy_norm_mean": float(np.mean([row["fy_norm"] for row in rows])),
        "per_view": rows,
    }


def gaussian_class():
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    return Gaussian


def gaussian_renderer_class():
    from sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render import GaussianRenderer

    return GaussianRenderer


def ply_data_class():
    from plyfile import PlyData

    return PlyData


def infer_ply_sh_degree(plydata: Any) -> int:
    feature_names = [prop.name for prop in plydata.elements[0].properties if prop.name.startswith("f_rest_")]
    if not feature_names:
        return 0
    triplets, remainder = divmod(len(feature_names), 3)
    if remainder != 0:
        raise ValueError(f"Unexpected SH feature count in PLY: {len(feature_names)}")
    degree = int(round(np.sqrt(triplets + 1) - 1.0))
    if (degree + 1) ** 2 != triplets + 1:
        raise ValueError(f"Cannot infer SH degree from {len(feature_names)} f_rest_* properties.")
    return degree


def compute_gaussian_aabb(path: Path) -> tuple[list[float], int]:
    plydata = ply_data_class().read(str(path))
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"], dtype=np.float32),
            np.asarray(plydata.elements[0]["y"], dtype=np.float32),
            np.asarray(plydata.elements[0]["z"], dtype=np.float32),
        ),
        axis=1,
    )
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    aabb = [float(mins[0]), float(mins[1]), float(mins[2]), float(extent[0]), float(extent[1]), float(extent[2])]
    return aabb, infer_ply_sh_degree(plydata)


def _load_gaussian_from_ply(path: Path, *, device: str) -> Any:
    Gaussian = gaussian_class()
    aabb, sh_degree = compute_gaussian_aabb(path)
    gaussian = Gaussian(aabb=aabb, sh_degree=sh_degree, device=device)
    gaussian.max_sh_degree = sh_degree
    gaussian.load_ply(str(path))
    return gaussian


def transform_gaussian_components(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    sim3: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    scale, rotation, translation = sim3_components(sim3)
    device = xyz.device
    rotation_tensor = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    translation_tensor = torch.as_tensor(translation, dtype=torch.float32, device=device)
    xyz_out = float(scale) * (xyz @ rotation_tensor.transpose(0, 1)) + translation_tensor
    scales_out = scales * float(scale)
    rotations_out = matrix_to_quaternion(rotation_tensor[None, :, :] @ quaternion_to_matrix(rotations))
    return xyz_out, scales_out, rotations_out


def transform_gaussian_sim3(source_gaussian: Any, sim3: dict[str, Any]) -> Any:
    xyz_out, scales_out, rotations_out = transform_gaussian_components(
        source_gaussian.get_xyz,
        source_gaussian.get_scaling,
        source_gaussian.get_rotation,
        sim3,
    )
    Gaussian = gaussian_class()
    transformed = Gaussian(device=str(xyz_out.device), **source_gaussian.init_params)
    transformed.max_sh_degree = getattr(source_gaussian, "max_sh_degree", source_gaussian.sh_degree)
    transformed.aabb = torch.tensor(aabb_from_xyz_tensor(xyz_out), dtype=torch.float32, device=xyz_out.device)
    transformed.from_xyz(xyz_out)
    transformed.from_scaling(scales_out)
    transformed.from_rotation(rotations_out)
    transformed.from_opacity(source_gaussian.get_opacity.detach().clone())
    transformed._features_dc = source_gaussian._features_dc.detach().clone()
    transformed._features_rest = (
        None if source_gaussian._features_rest is None else source_gaussian._features_rest.detach().clone()
    )
    transformed.active_sh_degree = source_gaussian.active_sh_degree
    return transformed


def render_gaussian_to_arrays(
    *,
    gaussian: Any,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    backend: str,
    near: float,
    far: float,
    ssaa: int,
) -> tuple[np.ndarray, np.ndarray]:
    renderer = gaussian_renderer_class()(
        rendering_options={
            "resolution": (int(image_height), int(image_width)),
            "near": float(near),
            "far": float(far),
            "ssaa": int(ssaa),
            "bg_color": (0.0, 0.0, 0.0),
            "backend": str(backend),
        }
    )
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True
    device = gaussian.get_xyz.device
    w2c_tensor = torch.as_tensor(w2c, dtype=torch.float32, device=device)
    intrinsic_tensor = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    color = renderer.render(gaussian, w2c_tensor, intrinsic_tensor)["color"]
    image = color.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    image = np.flip(image, axis=1).copy()
    alpha = image.max(axis=2) > 0.0
    return image.astype(np.float32), alpha


def aabb_from_xyz_tensor(xyz: torch.Tensor) -> list[float]:
    mins = xyz.amin(dim=0)
    maxs = xyz.amax(dim=0)
    extent = torch.clamp(maxs - mins, min=1e-6)
    return [
        float(mins[0].item()),
        float(mins[1].item()),
        float(mins[2].item()),
        float(extent[0].item()),
        float(extent[1].item()),
        float(extent[2].item()),
    ]


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(prediction, target).sum() / union)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot compute a mask bbox from an empty mask.")
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def masked_metric_crops(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = mask_bbox(mask)
    mask_crop = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    pred_crop = np.asarray(prediction[y0:y1, x0:x1], dtype=np.float32).copy()
    target_crop = np.asarray(target[y0:y1, x0:x1], dtype=np.float32).copy()
    pred_crop[~mask_crop] = 0.0
    target_crop[~mask_crop] = 0.0
    return pred_crop, target_crop


def resize_metric_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = prediction.shape[:2]
    max_dim = max(height, width)
    if max_dim <= int(max_side):
        return prediction, target
    scale = float(max_side) / float(max_dim)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    pred_img = Image.fromarray((prediction.clip(0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")
    target_img = Image.fromarray((target.clip(0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")
    pred_arr = np.asarray(pred_img.resize(new_size, Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    target_arr = np.asarray(target_img.resize(new_size, Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    return pred_arr, target_arr


def compute_masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot compute masked PSNR with an empty mask.")
    mse = float(np.mean((prediction[mask] - target[mask]) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def compute_masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    y0, y1, x0, x1 = mask_bbox(mask)
    pred_crop = prediction[y0:y1, x0:x1].copy()
    target_crop = target[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]
    pred_crop[~mask_crop] = 0.0
    target_crop[~mask_crop] = 0.0
    min_side = min(pred_crop.shape[:2])
    if min_side < 3:
        return float("nan")
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    return float(structural_similarity(target_crop, pred_crop, channel_axis=2, data_range=1.0, win_size=win_size))


def make_lpips_metric(lpips_net: str) -> Any:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(net_type=str(lpips_net), normalize=True).eval()


def compute_lpips_vgg(
    metric: Any,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int,
) -> float:
    prediction_crop, target_crop = masked_metric_crops(prediction, target, mask)
    prediction_crop, target_crop = resize_metric_pair(prediction_crop, target_crop, max_side=max_side)
    pred_tensor = torch.from_numpy(prediction_crop.transpose(2, 0, 1)).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target_crop.transpose(2, 0, 1)).unsqueeze(0).float()
    device = next(metric.parameters()).device if hasattr(metric, "parameters") else torch.device("cpu")
    with torch.inference_mode():
        value = metric(pred_tensor.to(device), target_tensor.to(device))
    return float(value.detach().cpu().reshape(-1)[0].item())


def metric_stat(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return {
        "finite_count": int(finite.size),
        "mean": None if finite.size == 0 else float(finite.mean()),
        "median": None if finite.size == 0 else float(np.median(finite)),
        "min": None if finite.size == 0 else float(finite.min()),
        "max": None if finite.size == 0 else float(finite.max()),
    }


def mesh_cd_diagnostics_from_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    sim3: dict[str, Any],
) -> dict[str, Any]:
    aligned = apply_sim3_to_points(source_points, sim3)
    return {
        "sample_points": int(source_points.shape[0]),
        "cd": symmetric_chamfer_distance(aligned, target_points),
        "source_space": sim3["source_space"],
        "target_space": sim3["target_space"],
        "mesh_exported": False,
    }


def write_per_view_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty per-view metrics CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["view", "psnr", "ssim", "lpips", "alpha_mask_iou", "render_alpha_coverage", "target_mask_coverage"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_contact_sheet(
    path: Path,
    items: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    thumb_w = 256
    thumb_h = 192
    label_h = 24
    columns = 4
    rows = len(items)
    sheet = Image.new("RGB", (thumb_w * columns, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, (view_name, target_rgb, rendered_rgb, target_mask, rendered_alpha) in enumerate(items):
        y = row_idx * (thumb_h + label_h)
        panels = [
            ("gt " + view_name, target_rgb),
            ("render", rendered_rgb),
            ("gt alpha", np.repeat(np.asarray(target_mask)[..., None].astype(np.float32), 3, axis=2)),
            ("render alpha", np.repeat(np.asarray(rendered_alpha)[..., None].astype(np.float32), 3, axis=2)),
        ]
        for col_idx, (label, image) in enumerate(panels):
            x = col_idx * thumb_w
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
            pil = Image.fromarray(np.clip(np.asarray(image) * 255.0, 0.0, 255.0).round().astype(np.uint8))
            sheet.paste(pil.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR), (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def render_from_state(
    state_file: Path,
    *,
    output_dir: Path | None,
    views: str,
    render_config: RenderConfig,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gaussian rendering and LPIPS evaluation.")

    state_path = Path(state_file).expanduser().resolve()
    state = read_state(state_path)
    chunk_root = Path(state["chunk_root"])
    gt_root = Path(state["gt_root"])
    output_root = state_path.parent if output_dir is None else Path(output_dir).expanduser().resolve()
    frames = discover_render_mvs25_views(gt_root)
    selected_indices = parse_views(views, len(frames))
    selected_frames = [frames[index] for index in selected_indices]

    target_mesh = load_mesh(target_mesh_path_from_state(state), MVS25_MESH_BASIS)
    source_points = source_geometry_points_from_state(state, sample_points_count=SAMPLE_POINTS, seed=SEED)
    active_mesh_diagnostics = mesh_cd_diagnostics_from_points(
        source_points,
        sample_mesh_points(target_mesh, SAMPLE_POINTS, SEED + 1),
        state["active_sim3"],
    )
    active_mesh_diagnostics["source_geometry_kind"] = source_geometry_kind_from_state(state)
    active_mesh_diagnostics["source_geometry_path"] = str(source_geometry_path_from_state(state).resolve())
    focal = calibrate_focal_norm(gt_root, target_mesh, frames)
    intrinsic = normalized_intrinsic(float(focal["focal_norm"]))

    torch.cuda.set_device(render_config.cuda_device)
    lpips_metric = make_lpips_metric("vgg").to("cuda")
    with torch.inference_mode():
        gaussian = _load_gaussian_from_ply(chunk_root / "result.ply", device="cuda")
        transformed = transform_gaussian_sim3(gaussian, state["active_sim3"])

        render_dir = output_root / "renders"
        gt_dir = output_root / "gt"
        comparison_dir = output_root / "comparisons"
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        contact_items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for frame in selected_frames:
            target_rgb, target_mask = load_rgba(frame.png_path)
            height, width = target_mask.shape
            rendered_rgb, rendered_alpha = render_gaussian_to_arrays(
                gaussian=transformed,
                w2c=load_w2c_cv(frame),
                intrinsic=intrinsic,
                image_height=int(height),
                image_width=int(width),
                backend=render_config.backend,
                near=render_config.near,
                far=render_config.far,
                ssaa=render_config.ssaa,
            )
            if render_config.flip_render_x:
                rendered_rgb = np.flip(rendered_rgb, axis=1).copy()
                rendered_alpha = np.flip(rendered_alpha, axis=1).copy()
            view_name = f"{frame.view_index:03d}"
            render_path = render_dir / f"{view_name}_pred.png"
            gt_path = gt_dir / f"{view_name}_gt.png"
            comparison_path = comparison_dir / f"{view_name}_gt_pred.png"
            save_rgba(render_path, rendered_rgb, rendered_alpha)
            save_rgba(gt_path, target_rgb, target_mask)
            save_gt_pred_comparison(comparison_path, target_rgb, target_mask, rendered_rgb, rendered_alpha)

            row = {
                "view": view_name,
                "psnr": compute_masked_psnr(rendered_rgb, target_rgb, target_mask),
                "ssim": compute_masked_ssim(rendered_rgb, target_rgb, target_mask),
                "lpips": compute_lpips_vgg(
                    lpips_metric,
                    rendered_rgb,
                    target_rgb,
                    target_mask,
                    max_side=render_config.lpips_max_side,
                ),
                "alpha_mask_iou": mask_iou(rendered_alpha, target_mask),
                "render_alpha_coverage": float(np.asarray(rendered_alpha, dtype=bool).mean()),
                "target_mask_coverage": float(np.asarray(target_mask, dtype=bool).mean()),
                "render_path": str(render_path.resolve()),
                "target_path": str(frame.png_path.resolve()),
                "comparison_path": str(comparison_path.resolve()),
            }
            rows.append(row)
            contact_items.append((view_name, target_rgb, rendered_rgb, target_mask, rendered_alpha))

    per_view_csv = output_root / "per_view_metrics.csv"
    summary_json = output_root / "summary.json"
    contact_sheet = output_root / "comparison_contact_sheet.png"
    write_per_view_csv(per_view_csv, rows)
    write_contact_sheet(contact_sheet, contact_items)
    summary = {
        "metric_set": "sam3d_gso_manual_registration_render_mvs_25",
        "scene": state["scene"],
        "chunk_root": state["chunk_root"],
        "gt_root": state["gt_root"],
        "state_file": str(state_path),
        "render_split": "render_mvs_25",
        "views": [row["view"] for row in rows],
        "view_count": len(rows),
        "render_backend": render_config.backend,
        "flip_render_x": bool(render_config.flip_render_x),
        "focal_calibration": focal,
        "active_sim3": state["active_sim3"],
        "active_mesh_diagnostics": active_mesh_diagnostics,
        "source_geometry_kind": source_geometry_kind_from_state(state),
        "source_geometry_path": str(source_geometry_path_from_state(state).resolve()),
        "registration_method": state["registration_diagnostics"]["method"],
        "needs_manual_review": bool(state.get("needs_manual_review", False)),
        "target_bbox_diagonal": float(state.get("target_bbox_diagonal", point_bbox_diagonal(np.asarray(target_mesh.vertices, dtype=np.float64)))),
        "metrics": {
            "psnr": metric_stat([row["psnr"] for row in rows]),
            "ssim": metric_stat([row["ssim"] for row in rows]),
            "lpips": metric_stat([row["lpips"] for row in rows]),
            "alpha_mask_iou": metric_stat([row["alpha_mask_iou"] for row in rows]),
        },
        "per_view": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_json, summary)

    state["render_eval"] = {
        "summary": str(summary_json.resolve()),
        "metrics": summary["metrics"],
        "active_mesh_diagnostics": active_mesh_diagnostics,
        "view_count": len(rows),
        "views": summary["views"],
        "updated_at_utc": summary["generated_at_utc"],
    }
    state["artifacts"].update(
        {
            "state_file": str(state_path),
            "output_dir": str(output_root.resolve()),
            "renders_dir": str((output_root / "renders").resolve()),
            "comparisons_dir": str((output_root / "comparisons").resolve()),
            "per_view_metrics_csv": str(per_view_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "comparison_contact_sheet": str(contact_sheet.resolve()),
        }
    )
    write_json(state_path, state)
    return summary


def run_estimate(args: argparse.Namespace) -> dict[str, Any]:
    chunk_root = args.chunk_root.expanduser().resolve()
    gt_root = (args.gt_root if args.gt_root is not None else default_gt_root(args.scene)).expanduser().resolve()
    paths = gt_paths(gt_root)
    target_mesh_path = paths["mesh"]
    if not (chunk_root / "result.ply").is_file():
        raise FileNotFoundError(f"Missing Gaussian PLY: {chunk_root / 'result.ply'}")
    if not target_mesh_path.is_file():
        raise FileNotFoundError(f"Missing GT render_mvs_25 model_norm.glb: {target_mesh_path}")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_mesh = load_mesh(target_mesh_path, MVS25_MESH_BASIS)
    prepared_source = prepare_source_geometry(
        chunk_root,
        str(args.source_geometry),
        sample_points_count=int(args.sample_points),
        seed=int(args.seed),
    )
    source_points = prepared_source.registration_points
    target_points = sample_mesh_points(target_mesh, int(args.sample_points), int(args.seed) + 1)
    method = RegistrationMethod(str(args.registration_method))
    if method == RegistrationMethod.TEASERPP_ICP:
        config = TeaserIcpConfig(
            teaser_voxel_fraction=float(args.teaser_voxel_fraction),
            icp_max_iterations=int(args.icp_max_iterations),
            icp_max_correspondence_distance_fraction=float(args.icp_max_correspondence_distance_fraction),
        )
        sim3, diagnostics = estimate_teaserpp_fpfh_icp(source_points, target_points, config)
    elif method in (RegistrationMethod.INITIAL_GICP_SCALE, RegistrationMethod.INITIAL24_GICP_SCALE):
        config = InitialGicpScaleConfig(
            initial_count=int(args.initial_count),
            gicp_max_iterations=int(args.gicp_max_iterations),
            gicp_max_correspondence_distance_fraction=float(args.gicp_max_correspondence_distance_fraction),
            scale_refine_trim_fraction=float(args.scale_refine_trim_fraction),
        )
        sim3, diagnostics = estimate_initial_gicp_scale(source_points, target_points, config)
        diagnostics["method_requested"] = method.value
    else:
        raise ValueError(f"Unsupported registration method: {method}")
    sim3 = retarget_sim3_source_space(
        sim3,
        prepared_source.source_local_alignment["sim3"]["target_space"],
    )
    diagnostics = rewrite_diagnostics_source_space(
        diagnostics,
        prepared_source.source_local_alignment["sim3"]["target_space"],
    )
    diagnostics["source_geometry_kind"] = prepared_source.kind.value
    diagnostics["source_geometry_path"] = str(prepared_source.path)
    diagnostics["source_local_alignment"] = prepared_source.source_local_alignment
    diagnostics["source_geometry_provenance"] = prepared_source.provenance
    state = state_with_initial_alignment(
        scene=str(args.scene),
        chunk_root=chunk_root,
        gt_root=gt_root,
        source_geometry_kind=prepared_source.kind.value,
        source_geometry_path=prepared_source.path,
        source_local_alignment=prepared_source.source_local_alignment,
        target_mesh=target_mesh_path,
        target_bbox_diagonal=point_bbox_diagonal(np.asarray(target_mesh.vertices, dtype=np.float64)),
        auto_initial_sim3=sim3,
        diagnostics=diagnostics,
    )
    state_path = output_dir / STATE_FILENAME
    write_json(state_path, state)
    summary = {
        "state_file": str(state_path.resolve()),
        "registration_diagnostics": diagnostics,
    }
    if not bool(args.skip_render):
        summary["render_eval"] = render_from_state(
            state_path,
            output_dir=output_dir,
            views=str(args.views),
            render_config=render_config_from_args(args),
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "estimate":
        result = run_estimate(args)
    elif args.command == "render":
        result = render_from_state(
            args.state_file,
            output_dir=args.output_dir,
            views=str(args.views),
            render_config=render_config_from_args(args),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
