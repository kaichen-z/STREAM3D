from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class CameraPoseSource(StrEnum):
    DA3 = "da3"
    DATASET_GT = "dataset_gt"


def depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)


def pointmap_to_sam3d_format(pointmap: np.ndarray) -> np.ndarray:
    return np.asarray(pointmap, dtype=np.float32).transpose(2, 0, 1)


def natural_sort_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    try:
        return (0, int(stem.split("_")[-1]), stem)
    except ValueError:
        return (1, 0, stem)


def to_4x4(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Unexpected camera extrinsic shape: {extrinsic.shape}")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :] = extrinsic
    return matrix


def load_camera_poses(path: Path) -> np.ndarray:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    poses: List[np.ndarray] = []
    for line in lines:
        tokens = [float(token) for token in line.split()]
        if len(tokens) == 16:
            pose = np.asarray(tokens, dtype=np.float32).reshape(4, 4)
        elif len(tokens) == 12:
            pose = np.asarray(tokens, dtype=np.float32).reshape(3, 4)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :] = pose
            pose = mat
        poses.append(pose)
    return np.stack(poses, axis=0)  # Shape: (num_frames, 4, 4)


def load_dataset_gt_camera_parameters(
    dataset_camera_path: Path,
    *,
    expected_num_frames: int,
) -> Dict[str, Any]:
    payload = json.loads(Path(dataset_camera_path).read_text(encoding="utf-8"))
    frames = payload["frames"]
    if len(frames) != int(expected_num_frames):
        raise ValueError(
            f"Dataset GT camera count mismatch: {len(frames)} frames in "
            f"{dataset_camera_path}, expected {expected_num_frames}."
        )

    extrinsics_c2w: List[np.ndarray] = []
    intrinsics_cv: List[np.ndarray] = []
    image_sizes_hw: List[Tuple[int, int]] = []
    for frame in frames:
        extrinsics_c2w.append(to_4x4(np.asarray(frame["transform_matrix"])))
        fl_x = float(frame["fl_x"] if "fl_x" in frame else payload["fl_x"])
        fl_y = float(frame["fl_y"] if "fl_y" in frame else payload["fl_y"])
        cx = float(frame["cx"] if "cx" in frame else payload["cx"])
        cy = float(frame["cy"] if "cy" in frame else payload["cy"])
        width = int(round(frame["w"] if "w" in frame else payload["w"]))
        height = int(round(frame["h"] if "h" in frame else payload["h"]))
        intrinsics_cv.append(
            np.asarray(
                [[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        )
        image_sizes_hw.append((height, width))

    return {
        "dataset_camera_path": str(Path(dataset_camera_path).resolve()),
        "extrinsics_c2w": np.stack(extrinsics_c2w, axis=0),
        "intrinsics_cv": np.stack(intrinsics_cv, axis=0),
        "image_sizes_hw": tuple(image_sizes_hw),
    }


def rescale_intrinsics_to_depth_resolution(
    intrinsic: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    intrinsic[0, 0] *= float(target_width) / float(source_width)
    intrinsic[0, 2] *= float(target_width) / float(source_width)
    intrinsic[1, 1] *= float(target_height) / float(source_height)
    intrinsic[1, 2] *= float(target_height) / float(source_height)
    return intrinsic


def load_da3(
    root: Path,
    *,
    image_files: Sequence[Path],
    camera_pose_source: str | CameraPoseSource = CameraPoseSource.DA3,
    dataset_camera_path: Path | None = None,
) -> Dict[str, Any]:
    """Load DA3 depth/intrinsics and either DA3 or dataset-GT camera poses."""
    results_dir = root / "results_output"
    result_files = sorted(results_dir.glob("frame_*.npz"), key=natural_sort_key)
    if len(result_files) != len(image_files):
        raise ValueError(
            f"DA3 output count mismatch: found {len(result_files)} frame outputs "
            f"in {results_dir}, but image_dir has {len(image_files)} images."
        )

    camera_pose_source = CameraPoseSource(str(camera_pose_source))
    extrinsics = load_camera_poses(root / "camera_poses.txt")
    dataset_gt = None
    if camera_pose_source is CameraPoseSource.DATASET_GT:
        if dataset_camera_path is None:
            raise ValueError("camera_pose_source='dataset_gt' requires dataset_camera_path.")
        dataset_gt = load_dataset_gt_camera_parameters(
            dataset_camera_path,
            expected_num_frames=len(image_files),
        )
        extrinsics = np.asarray(dataset_gt["extrinsics_c2w"], dtype=np.float32)

    pointmaps_sam3d: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    depth_shape_hw: np.ndarray | None = None
    for frame_idx, result_file in enumerate(result_files):
        with np.load(result_file) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            if camera_pose_source is CameraPoseSource.DATASET_GT:
                gt_intrinsic = np.asarray(
                    dataset_gt["intrinsics_cv"][frame_idx],
                    dtype=np.float32,
                )
                gt_height, gt_width = dataset_gt["image_sizes_hw"][frame_idx]
                intrinsic = rescale_intrinsics_to_depth_resolution(
                    gt_intrinsic,
                    source_height=int(gt_height),
                    source_width=int(gt_width),
                    target_height=int(depth.shape[0]),
                    target_width=int(depth.shape[1]),
                )
            else:
                intrinsic = np.asarray(data["intrinsics"], dtype=np.float32)
            pointmap = depth_to_pointmap(depth, intrinsic)
            pointmaps_sam3d.append(pointmap_to_sam3d_format(pointmap))
            intrinsics.append(intrinsic)
            if depth_shape_hw is None:
                depth_shape_hw = np.array(
                    [int(depth.shape[0]), int(depth.shape[1])], dtype=np.int32
                )

    return {
        "output_root": str(root.resolve()),
        "image_stems": [path.stem for path in image_files],
        "image_names": [path.name for path in image_files],
        "pointmaps_sam3d": np.stack(pointmaps_sam3d, axis=0),
        "extrinsics": np.asarray(extrinsics, dtype=np.float32),
        "intrinsics": np.stack(intrinsics, axis=0),
        "depth_shape_hw": depth_shape_hw,
        "num_frames": len(image_files),
        "camera_pose_source": str(camera_pose_source),
        "dataset_camera_path": (
            None if dataset_gt is None else dataset_gt["dataset_camera_path"]
        ),
        "depth_metric_scale": 1.0,
        "depth_metric_scale_source": "none",
        "depth_metric_scale_audit": {
            "scale": 1.0,
            "source": "none",
            "frame_count": 0,
            "sampled_point_count": 0,
            "used_frame_indices": [],
        },
        "pointmap_metric_scale": "raw_da3_depth",
    }


def build_chunk_da3_result(
    scene_da3: Dict[str, Any],
    *,
    global_frame_indices: Sequence[int],
) -> Dict[str, Any]:
    frame_indices = [int(index) for index in global_frame_indices]
    chunk_da3 = dict(scene_da3)
    for key in ("pointmaps_sam3d", "extrinsics", "intrinsics"):
        if key in scene_da3:
            chunk_da3[key] = np.asarray(scene_da3[key])[frame_indices]
    chunk_da3["global_frame_indices"] = frame_indices
    return chunk_da3
