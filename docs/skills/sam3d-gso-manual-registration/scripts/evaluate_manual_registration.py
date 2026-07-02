#!/usr/bin/env python3
"""Final evaluation for SAM3D GSO manual Sim(3) registration outputs.

This script is standalone inside the skill.  It may import the sibling
manual-registration script and repository modules, but it does not import from
other ``docs/skills`` directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import manual_register_sam3d_gso as manual  # noqa: E402


DEFAULT_OUTPUT_ROOT = manual.REPO_ROOT / "outputs" / "sam3d-gso-manual-registration"
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_ROOT / "evaluation"
DEFAULT_VARIANTS = ("k2_4", "k2_8", "k2_12", "k2_16")
FID_IMAGE_SIZE = 299
INCEPTION_FEATURE_DIM = 2048
LPIPS_NET = "vgg"
LPIPS_MAX_SIDE = 512


@dataclass(frozen=True)
class SceneRecord:
    variant: str
    scene: str
    chunk: str
    state_file: Path
    output_dir: Path


@dataclass(frozen=True)
class GeometryEvaluation:
    cd: float | None
    cd_l2: float | None
    sample_points: int
    pred_point_path: Path | None
    gt_point_path: Path | None
    status: str
    skipped_reason: str


@dataclass(frozen=True)
class ImageFeatureRecord:
    variant: str
    scene: str
    view: str
    pred_path: Path
    gt_path: Path


class InceptionFeatureExtractor:
    def __init__(self, device: torch.device) -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.device = device
        self.mean = torch.tensor(weights.transforms().mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        self.model = model

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device, non_blocking=True)
        images = F.interpolate(images, size=(FID_IMAGE_SIZE, FID_IMAGE_SIZE), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        features = self.model(images)
        if isinstance(features, tuple):
            features = features[0]
        return features.detach().float().cpu().numpy()


class TimmInceptionFeatureExtractor:
    def __init__(self, device: torch.device) -> None:
        import timm

        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.model = timm.create_model(
            "inception_v3",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        ).eval().to(device)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device, non_blocking=True)
        images = F.interpolate(images, size=(FID_IMAGE_SIZE, FID_IMAGE_SIZE), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        return self.model(images).detach().float().cpu().numpy()


class TorchscriptPointFeatureExtractor:
    def __init__(self, checkpoint: Path, *, device: torch.device, input_layout: str) -> None:
        self.device = device
        self.input_layout = str(input_layout)
        self.model = torch.jit.load(str(checkpoint), map_location=device).eval()

    @torch.inference_mode()
    def __call__(self, point_clouds: torch.Tensor) -> np.ndarray:
        point_clouds = point_clouds.to(self.device, non_blocking=True)
        if self.input_layout == "bcn":
            model_input = point_clouds.transpose(1, 2).contiguous()
        elif self.input_layout == "bnc":
            model_input = point_clouds
        else:
            raise ValueError(f"Unsupported pointnet input layout: {self.input_layout}")
        output = self.model(model_input)
        if isinstance(output, dict):
            for key in ("features", "feat", "embedding", "global_feat"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise KeyError("TorchScript point feature model returned a dict without a known feature key.")
        elif isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1)
        return output.detach().float().cpu().numpy()


class PointNet2SSGFeatureExtractor:
    def __init__(self, repo: Path, checkpoint: Path, *, device: torch.device) -> None:
        repo = Path(repo).expanduser().resolve()
        model_dir = repo / "log" / "classification" / "pointnet2_ssg_wo_normals"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from pointnet2_cls_ssg import get_model

        self.device = device
        self.model = get_model(40, normal_channel=False).to(device).eval()
        payload = torch.load(str(Path(checkpoint).expanduser().resolve()), map_location=device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])

    @torch.inference_mode()
    def __call__(self, point_clouds: torch.Tensor) -> np.ndarray:
        point_clouds = point_clouds.to(self.device, non_blocking=True)
        model_input = point_clouds.transpose(1, 2).contiguous()
        _, global_features = self.model(model_input)
        return global_features.reshape(global_features.shape[0], -1).detach().float().cpu().numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate final SAM3D GSO manual-registration outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state-index", type=Path, default=DEFAULT_OUTPUT_ROOT / "final_state_index.json")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=manual.SEED)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-fid-backend",
        choices=("torchvision_inception", "timm_inception", "skip"),
        default="torchvision_inception",
    )
    parser.add_argument("--image-fid-batch-size", type=int, default=32)
    parser.add_argument("--compute-scene-fid", action="store_true")
    parser.add_argument("--lpips-net", choices=(LPIPS_NET,), default=LPIPS_NET)
    parser.add_argument("--lpips-max-side", type=int, default=LPIPS_MAX_SIDE)
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--point-feature-backend", choices=("torchscript", "pointnet2_ssg", "skip"), default="skip")
    parser.add_argument("--pointnet-checkpoint", type=Path)
    parser.add_argument("--pointnet2-repo", type=Path, default=manual.REPO_ROOT / "scratch" / "external" / "Pointnet_Pointnet2_pytorch")
    parser.add_argument("--pointnet-input-layout", choices=("bnc", "bcn"), default="bnc")
    parser.add_argument("--point-feature-batch-size", type=int, default=16)
    parser.add_argument("--point-feature-normalization", choices=("unit_sphere", "none"), default="unit_sphere")
    parser.add_argument("--write-point-samples", action="store_true")
    parser.add_argument("--allow-missing-distribution-metrics", action="store_true", default=True)
    parser.add_argument("--strict-distribution-metrics", dest="allow_missing_distribution_metrics", action="store_false")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_variants(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def finite_mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def load_scene_records(output_root: Path, state_index: Path, variants: Sequence[str]) -> list[SceneRecord]:
    output_root = Path(output_root).expanduser().resolve()
    index_path = Path(state_index).expanduser().resolve()
    records: list[SceneRecord] = []
    indexed_variants: set[str] = set()
    if index_path.is_file():
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        for row in rows:
            variant = str(row["variant"])
            if variant not in variants:
                continue
            indexed_variants.add(variant)
            state_file = Path(row["state_file"]).expanduser().resolve()
            output_dir = Path(row["output_dir"]).expanduser().resolve()
            records.append(
                SceneRecord(
                    variant=variant,
                    scene=str(row["scene"]),
                    chunk=str(row["chunk"]),
                    state_file=state_file,
                    output_dir=output_dir,
                )
            )
        if indexed_variants == set(variants):
            return sorted(records, key=lambda r: (r.variant, r.scene, r.chunk))

    for variant in variants:
        if variant in indexed_variants:
            continue
        for chunk_dir in sorted((output_root / variant).glob("*/chunk_*")):
            state_file = chunk_dir / manual.STATE_FILENAME
            if state_file.is_file():
                records.append(SceneRecord(variant, chunk_dir.parent.name, chunk_dir.name, state_file, chunk_dir))
    return sorted(records, key=lambda r: (r.variant, r.scene, r.chunk))


def sample_mesh_surface_points(mesh: Any, count: int, seed: int) -> np.ndarray:
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, int(count), seed=int(seed))
    return np.asarray(points, dtype=np.float64)


def chamfer_pytorch3d(source: np.ndarray, target: np.ndarray, device: torch.device) -> float:
    from pytorch3d.loss import chamfer_distance

    x = torch.as_tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    value, _ = chamfer_distance(x, y, batch_reduction="mean", point_reduction="mean", norm=2)
    return float(value.detach().cpu().item())


def evaluate_geometry(
    record: SceneRecord,
    *,
    sample_points: int,
    seed: int,
    device: torch.device,
    eval_dir: Path,
    write_point_samples: bool,
) -> tuple[GeometryEvaluation, np.ndarray | None, np.ndarray | None]:
    state = manual.read_state(record.state_file)
    target_mesh = manual.load_mesh(manual.target_mesh_path_from_state(state), manual.MVS25_MESH_BASIS)
    source_kind = manual.source_geometry_kind_from_state(state)
    if source_kind in (manual.SourceGeometryKind.RESULT_GLB.value, manual.SourceGeometryKind.BASELINE_MESH.value):
        source_mesh_path = manual.source_geometry_path_from_state(state)
        if not source_mesh_path.is_file():
            reason = f"missing pred mesh: {source_mesh_path}"
            return GeometryEvaluation(None, None, sample_points, None, None, "skipped", reason), None, None
        source_mesh = manual.load_mesh(source_mesh_path, manual.source_geometry_basis_matrix_from_state(state))
        pred_source = sample_mesh_surface_points(source_mesh, sample_points, seed)
    else:
        pred_source = manual.source_geometry_points_from_state(
            state,
            sample_points_count=int(sample_points),
            seed=int(seed),
        )
    gt_points = sample_mesh_surface_points(target_mesh, sample_points, seed + 1)
    pred_points = manual.apply_sim3_to_points(pred_source, state["active_sim3"])
    cd = chamfer_pytorch3d(pred_points, gt_points, device)
    cd_l2 = manual.symmetric_chamfer_distance(pred_points, gt_points)

    pred_path: Path | None = None
    gt_path: Path | None = None
    if write_point_samples:
        sample_dir = eval_dir / "point_samples" / record.variant / record.scene / record.chunk
        sample_dir.mkdir(parents=True, exist_ok=True)
        pred_path = sample_dir / "pred_aligned_4096.npy"
        gt_path = sample_dir / "gt_4096.npy"
        np.save(pred_path, pred_points.astype(np.float32))
        np.save(gt_path, gt_points.astype(np.float32))

    return (
        GeometryEvaluation(cd, cd_l2, sample_points, pred_path, gt_path, "ok", ""),
        pred_points.astype(np.float32),
        gt_points.astype(np.float32),
    )


def covariance(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.shape}")
    if features.shape[0] <= 1:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.cov(features, rowvar=False)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, eps: float = 1e-6) -> float:
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Fréchet distance expects two 2D feature arrays.")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {features_a.shape[1]} vs {features_b.shape[1]}")
    mu_a = features_a.mean(axis=0)
    mu_b = features_b.mean(axis=0)
    sigma_a = covariance(features_a)
    sigma_b = covariance(features_b)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a @ sigma_b, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma_a + offset) @ (sigma_b + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1e-3):
            raise ValueError("Fréchet covariance product produced significant imaginary values.")
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean)
    return float(max(value, 0.0))


def frechet_distance_or_none(features_a: np.ndarray, features_b: np.ndarray) -> float | None:
    features_a = np.asarray(features_a)
    features_b = np.asarray(features_b)
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        return None
    return frechet_distance(features_a, features_b)


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[..., :3], arr[..., 3] > 0.0


def composite_rgb_on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)[..., None]
    return (rgb * alpha + 1.0 - alpha).astype(np.float32)


def load_rgba_composited(path: Path) -> np.ndarray:
    rgb, alpha = load_rgba(path)
    return composite_rgb_on_white(rgb, alpha)


def compute_rgb_psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(pred_rgb, dtype=np.float32) - np.asarray(gt_rgb, dtype=np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def compute_full_image_ssim(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred_rgb = np.asarray(pred_rgb, dtype=np.float32)
    gt_rgb = np.asarray(gt_rgb, dtype=np.float32)
    min_side = min(pred_rgb.shape[:2])
    if min_side < 3:
        return float("nan")
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    return float(
        structural_similarity(
            gt_rgb,
            pred_rgb,
            channel_axis=2,
            data_range=1.0,
            win_size=win_size,
        )
    )


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return float("nan")
    intersection = np.logical_and(prediction, target).sum()
    return float(intersection / union)


def resize_metric_pair(prediction: np.ndarray, target: np.ndarray, *, max_side: int) -> tuple[np.ndarray, np.ndarray]:
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


def make_lpips_vgg_metric():
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type=LPIPS_NET, normalize=True, reduction="none")
    return metric.eval()


def image_to_tensor(path: Path) -> torch.Tensor:
    rgb = load_rgba_composited(path)
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))


def image_record_key(item: ImageFeatureRecord) -> tuple[str, str, str, str]:
    return (item.variant, item.scene, item.view, "image")


def evaluate_full_white_photometric(
    records: Sequence[ImageFeatureRecord],
    *,
    device: torch.device,
    lpips_batch_size: int,
    lpips_max_side: int,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    metric = make_lpips_vgg_metric().to(device)
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    batch_pred: list[torch.Tensor] = []
    batch_gt: list[torch.Tensor] = []
    batch_keys: list[tuple[str, str, str, str]] = []
    batch_shape: tuple[int, int, int] | None = None

    def flush() -> None:
        nonlocal batch_shape
        if not batch_keys:
            return
        pred_tensor = torch.stack(batch_pred, dim=0).to(device)
        gt_tensor = torch.stack(batch_gt, dim=0).to(device)
        with torch.inference_mode():
            values = metric(pred_tensor, gt_tensor).detach().float().cpu().reshape(-1).numpy()
        if values.size != len(batch_keys):
            raise RuntimeError(f"LPIPS-VGG returned {values.size} values for batch size {len(batch_keys)}")
        for key, value in zip(batch_keys, values):
            output[key]["lpips"] = float(value)
        batch_pred.clear()
        batch_gt.clear()
        batch_keys.clear()
        batch_shape = None

    for item in records:
        pred_rgb_raw, pred_alpha = load_rgba(item.pred_path)
        gt_rgb_raw, gt_alpha = load_rgba(item.gt_path)
        pred_rgb = composite_rgb_on_white(pred_rgb_raw, pred_alpha)
        gt_rgb = composite_rgb_on_white(gt_rgb_raw, gt_alpha)
        key = image_record_key(item)
        output[key] = {
            "variant": item.variant,
            "scene": item.scene,
            "view": item.view,
            "pred_path": item.pred_path,
            "gt_path": item.gt_path,
            "psnr": compute_rgb_psnr(pred_rgb, gt_rgb),
            "ssim": compute_full_image_ssim(pred_rgb, gt_rgb),
            "alpha_mask_iou": mask_iou(pred_alpha, gt_alpha),
        }

        lpips_pred, lpips_gt = resize_metric_pair(pred_rgb, gt_rgb, max_side=lpips_max_side)
        shape = tuple(lpips_pred.shape)
        if batch_shape is not None and shape != batch_shape:
            flush()
        batch_shape = shape
        batch_pred.append(torch.from_numpy(lpips_pred.transpose(2, 0, 1)).float())
        batch_gt.append(torch.from_numpy(lpips_gt.transpose(2, 0, 1)).float())
        batch_keys.append(key)
        if len(batch_keys) >= int(lpips_batch_size):
            flush()
    flush()
    return output


def extract_image_features(
    records: Sequence[ImageFeatureRecord],
    *,
    device: torch.device,
    batch_size: int,
    backend: str,
) -> dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]]:
    if backend == "torchvision_inception":
        extractor = InceptionFeatureExtractor(device)
    elif backend == "timm_inception":
        extractor = TimmInceptionFeatureExtractor(device)
    else:
        raise ValueError(f"Unsupported image FID backend: {backend}")
    output: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        pred_images = torch.stack([image_to_tensor(item.pred_path) for item in batch], dim=0)
        gt_images = torch.stack([image_to_tensor(item.gt_path) for item in batch], dim=0)
        pred_features = extractor(pred_images)
        gt_features = extractor(gt_images)
        for item, pred_feature, gt_feature in zip(batch, pred_features, gt_features):
            output[(item.variant, item.scene, item.view, "image")] = (pred_feature, gt_feature)
    return output


def normalize_point_cloud(points: np.ndarray, mode: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if mode == "none":
        return points
    if mode != "unit_sphere":
        raise ValueError(f"Unsupported point cloud normalization: {mode}")
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius <= 0.0 or not math.isfinite(float(radius)):
        return centered
    return centered / float(radius)


def fixed_size_point_cloud(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == int(count):
        return points
    rng = np.random.default_rng(int(seed))
    replace = points.shape[0] < int(count)
    indices = rng.choice(points.shape[0], size=int(count), replace=replace)
    return points[indices]


def extract_point_features(
    points_by_scene: Sequence[tuple[SceneRecord, np.ndarray, np.ndarray]],
    *,
    backend: str,
    repo: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    input_layout: str,
    normalization: str,
    sample_points: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    if backend == "torchscript":
        extractor = TorchscriptPointFeatureExtractor(checkpoint, device=device, input_layout=input_layout)
    elif backend == "pointnet2_ssg":
        extractor = PointNet2SSGFeatureExtractor(repo, checkpoint, device=device)
    else:
        raise ValueError(f"Unsupported point feature backend: {backend}")
    output: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(points_by_scene), int(batch_size)):
        batch = points_by_scene[start : start + int(batch_size)]
        pred_clouds = []
        gt_clouds = []
        for offset, item in enumerate(batch):
            pred_clouds.append(fixed_size_point_cloud(item[1], sample_points, seed + start + offset * 2))
            gt_clouds.append(fixed_size_point_cloud(item[2], sample_points, seed + start + offset * 2 + 1))
        pred = torch.from_numpy(np.stack([normalize_point_cloud(points, normalization) for points in pred_clouds], axis=0))
        gt = torch.from_numpy(np.stack([normalize_point_cloud(points, normalization) for points in gt_clouds], axis=0))
        pred_features = extractor(pred)
        gt_features = extractor(gt)
        for (record, _, _), pred_feature, gt_feature in zip(batch, pred_features, gt_features):
            output[(record.variant, record.scene, record.chunk)] = (pred_feature, gt_feature)
    return output


def collect_image_records(record: SceneRecord, summary: dict[str, Any]) -> list[ImageFeatureRecord]:
    state = manual.read_state(record.state_file)
    gt_root = Path(state["gt_root"])
    rows: list[ImageFeatureRecord] = []
    for view in summary["views"]:
        view_name = f"{int(view):03d}" if str(view).isdigit() else str(view)
        pred_path = record.output_dir / "renders" / f"{view_name}_pred.png"
        gt_path = record.output_dir / "gt" / f"{view_name}_gt.png"
        if not gt_path.is_file():
            gt_path = gt_root / "render_mvs_25" / "model" / f"{view_name}.png"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing rendered image for FID: {pred_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT image for FID: {gt_path}")
        rows.append(ImageFeatureRecord(record.variant, record.scene, view_name, pred_path, gt_path))
    return rows


def flatten_scene_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    for key, value in list(flat.items()):
        if isinstance(value, Path):
            flat[key] = str(value)
        elif value is None:
            flat[key] = ""
    return flat


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    variants = parse_variants(args.variants)
    output_root = Path(args.output_root).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records = load_scene_records(output_root, args.state_index, variants)
    if not records:
        raise FileNotFoundError("No scene records found for evaluation.")

    scene_rows: list[dict[str, Any]] = []
    image_records: list[ImageFeatureRecord] = []
    image_records_by_scene: dict[tuple[str, str, str], list[ImageFeatureRecord]] = {}
    point_clouds: list[tuple[SceneRecord, np.ndarray, np.ndarray]] = []
    for record in records:
        summary_path = record.output_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing render summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        comparisons = sorted((record.output_dir / "comparisons").glob("*_gt_pred.png"))
        renders = sorted((record.output_dir / "renders").glob("*_pred.png"))
        geometry, pred_points, gt_points = evaluate_geometry(
            record,
            sample_points=int(args.sample_points),
            seed=int(args.seed),
            device=device,
            eval_dir=eval_dir,
            write_point_samples=bool(args.write_point_samples),
        )
        if pred_points is not None and gt_points is not None:
            point_clouds.append((record, pred_points, gt_points))
        scene_image_records = collect_image_records(record, summary)
        image_records.extend(scene_image_records)
        image_records_by_scene[(record.variant, record.scene, record.chunk)] = scene_image_records
        scene_rows.append(
            {
                "variant": record.variant,
                "scene": record.scene,
                "chunk": record.chunk,
                "state_file": str(record.state_file),
                "output_dir": str(record.output_dir),
                "view_count": int(summary.get("view_count", len(summary.get("views", [])))),
                "comparison_count": len(comparisons),
                "render_count": len(renders),
                "complete_25": bool(len(comparisons) == 25 and len(renders) == 25),
                "cd": geometry.cd,
                "cd_l2": geometry.cd_l2,
                "geometry_status": geometry.status,
                "geometry_skipped_reason": geometry.skipped_reason,
                "geometry_sample_points": geometry.sample_points,
                "pred_point_sample_path": geometry.pred_point_path,
                "gt_point_sample_path": geometry.gt_point_path,
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "alpha_mask_iou": None,
                "image_fid_scene": None,
                "p_fid_scene": None,
            }
        )

    photometric = evaluate_full_white_photometric(
        image_records,
        device=device,
        lpips_batch_size=int(args.lpips_batch_size),
        lpips_max_side=int(args.lpips_max_side),
    )
    for row in scene_rows:
        scene_key = (row["variant"], row["scene"], row["chunk"])
        scene_values = [photometric[image_record_key(item)] for item in image_records_by_scene[scene_key]]
        row["psnr"] = finite_mean(value["psnr"] for value in scene_values)
        row["ssim"] = finite_mean(value["ssim"] for value in scene_values)
        row["lpips"] = finite_mean(value["lpips"] for value in scene_values)
        row["alpha_mask_iou"] = finite_mean(value["alpha_mask_iou"] for value in scene_values)

    distribution_status: dict[str, Any] = {}
    image_features: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    if args.image_fid_backend in ("torchvision_inception", "timm_inception"):
        try:
            image_features = extract_image_features(
                image_records,
                device=device,
                batch_size=int(args.image_fid_batch_size),
                backend=str(args.image_fid_backend),
            )
            distribution_status["image_fid"] = {"status": "ok", "backend": args.image_fid_backend}
        except Exception as exc:
            if not bool(args.allow_missing_distribution_metrics):
                raise
            distribution_status["image_fid"] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        distribution_status["image_fid"] = {"status": "skipped", "reason": "--image-fid-backend skip"}

    point_features: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    if args.point_feature_backend in ("torchscript", "pointnet2_ssg"):
        if args.pointnet_checkpoint is None:
            if not bool(args.allow_missing_distribution_metrics):
                raise ValueError(f"--point-feature-backend {args.point_feature_backend} requires --pointnet-checkpoint")
            distribution_status["p_fid"] = {"status": "skipped", "reason": "missing --pointnet-checkpoint"}
        else:
            try:
                point_features = extract_point_features(
                    point_clouds,
                    backend=str(args.point_feature_backend),
                    repo=Path(args.pointnet2_repo).expanduser().resolve(),
                    checkpoint=Path(args.pointnet_checkpoint).expanduser().resolve(),
                    device=device,
                    batch_size=int(args.point_feature_batch_size),
                    input_layout=str(args.pointnet_input_layout),
                    normalization=str(args.point_feature_normalization),
                    sample_points=int(args.sample_points),
                    seed=int(args.seed),
                )
                distribution_status["p_fid"] = {
                    "status": "ok",
                    "backend": args.point_feature_backend,
                    "repo": str(Path(args.pointnet2_repo).expanduser().resolve())
                    if args.point_feature_backend == "pointnet2_ssg"
                    else "",
                    "checkpoint": str(Path(args.pointnet_checkpoint).expanduser().resolve()),
                    "normalization": str(args.point_feature_normalization),
                }
            except Exception as exc:
                if not bool(args.allow_missing_distribution_metrics):
                    raise
                distribution_status["p_fid"] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        distribution_status["p_fid"] = {"status": "skipped", "reason": "--point-feature-backend skip"}

    if bool(args.compute_scene_fid):
        for row in scene_rows:
            key_prefix = (row["variant"], row["scene"])
            scene_image_pairs = [features for key, features in image_features.items() if key[:2] == key_prefix]
            if scene_image_pairs:
                pred = np.stack([pair[0] for pair in scene_image_pairs], axis=0)
                gt = np.stack([pair[1] for pair in scene_image_pairs], axis=0)
                row["image_fid_scene"] = frechet_distance_or_none(pred, gt)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        subset = [row for row in scene_rows if row["variant"] == variant]
        if not subset:
            continue
        variant_image_pairs = [features for key, features in image_features.items() if key[0] == variant]
        image_fid = None
        if variant_image_pairs:
            image_fid = frechet_distance_or_none(
                np.stack([pair[0] for pair in variant_image_pairs], axis=0),
                np.stack([pair[1] for pair in variant_image_pairs], axis=0),
            )
        variant_point_pairs = [features for key, features in point_features.items() if key[0] == variant]
        p_fid = None
        if variant_point_pairs:
            p_fid = frechet_distance_or_none(
                np.stack([pair[0] for pair in variant_point_pairs], axis=0),
                np.stack([pair[1] for pair in variant_point_pairs], axis=0),
            )
        summary_rows.append(
            {
                "variant": variant,
                "scene_count": len(subset),
                "complete_25_count": sum(1 for row in subset if row["complete_25"]),
                "cd_mean": finite_mean(row["cd"] for row in subset),
                "cd_l2_mean": finite_mean(row["cd_l2"] for row in subset),
                "psnr_mean": finite_mean(row["psnr"] for row in subset),
                "ssim_mean": finite_mean(row["ssim"] for row in subset),
                "lpips_mean": finite_mean(row["lpips"] for row in subset),
                "alpha_mask_iou_mean": finite_mean(row["alpha_mask_iou"] for row in subset),
                "image_fid": image_fid,
                "p_fid": p_fid,
            }
        )

    all_image_pairs = list(image_features.values())
    all_point_pairs = list(point_features.values())
    summary_rows.append(
        {
            "variant": "ALL",
            "scene_count": len(scene_rows),
            "complete_25_count": sum(1 for row in scene_rows if row["complete_25"]),
            "cd_mean": finite_mean(row["cd"] for row in scene_rows),
            "cd_l2_mean": finite_mean(row["cd_l2"] for row in scene_rows),
            "psnr_mean": finite_mean(row["psnr"] for row in scene_rows),
            "ssim_mean": finite_mean(row["ssim"] for row in scene_rows),
            "lpips_mean": finite_mean(row["lpips"] for row in scene_rows),
            "alpha_mask_iou_mean": finite_mean(row["alpha_mask_iou"] for row in scene_rows),
            "image_fid": None
            if not all_image_pairs
            else frechet_distance_or_none(
                np.stack([pair[0] for pair in all_image_pairs], axis=0),
                np.stack([pair[1] for pair in all_image_pairs], axis=0),
            ),
            "p_fid": None
            if not all_point_pairs
            else frechet_distance_or_none(
                np.stack([pair[0] for pair in all_point_pairs], axis=0),
                np.stack([pair[1] for pair in all_point_pairs], axis=0),
            ),
        }
    )

    scene_csv = eval_dir / "scene_metrics.csv"
    scene_json = eval_dir / "scene_metrics.json"
    summary_csv = eval_dir / "metrics_summary.csv"
    summary_json = eval_dir / "metrics_summary.json"
    run_json = eval_dir / "evaluation_run.json"
    write_csv(scene_csv, [flatten_scene_row(row) for row in scene_rows])
    scene_json.write_text(json.dumps(jsonable(scene_rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(summary_csv, [flatten_scene_row(row) for row in summary_rows])
    payload = {
        "metric_set": "sam3d_gso_manual_registration_final_eval_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": output_root,
        "state_index": Path(args.state_index).expanduser().resolve(),
        "eval_dir": eval_dir,
        "variants": variants,
        "sample_points": int(args.sample_points),
        "seed": int(args.seed),
        "cd_definition": "pytorch3d.loss.chamfer_distance, squared L2, mean over both directions; pred/GT surfaces uniformly sampled with trimesh.sample.sample_surface",
        "cd_l2_definition": "mean unsquared nearest-neighbor L2 in both directions, diagnostic only",
        "photometric_definition": "Aligned to docs/skills/sam3d-gso-render: pred and GT RGBA are hard-alpha composited onto white; PSNR/SSIM use full RGB images; LPIPS uses LPIPS-VGG on the same white-background RGB images, resized only when max side exceeds 512 px.",
        "lpips_net": str(args.lpips_net),
        "lpips_max_side": int(args.lpips_max_side),
        "image_fid_definition": "Fréchet distance between ImageNet InceptionV3 features for the same hard-alpha white-background pred/GT render_mvs_25 RGB images used by full-white photometric metrics",
        "p_fid_definition": "Fréchet distance between external PointNet++/TorchScript features extracted from aligned 4096-point pred/GT surface samples",
        "distribution_status": distribution_status,
        "summary": summary_rows,
        "artifacts": {
            "scene_metrics_csv": scene_csv,
            "scene_metrics_json": scene_json,
            "metrics_summary_csv": summary_csv,
            "metrics_summary_json": summary_json,
        },
    }
    summary_json.write_text(json.dumps(jsonable(summary_rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_json.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    result = run_evaluation(parse_args(argv))
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
