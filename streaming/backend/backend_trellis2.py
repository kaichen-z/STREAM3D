from __future__ import annotations

import gc
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import trimesh
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from streaming.data import StreamingExample


@dataclass(frozen=True)
class Trellis2BackendConfig:
    output_root: Path
    pipeline: dict[str, Any]
    seed: int


def build_trellis2_backend_config(args: DictConfig) -> Trellis2BackendConfig:
    return Trellis2BackendConfig(
        output_root=Path(args.output_root),
        pipeline=dict(OmegaConf.to_container(args.pipeline, resolve=True)),
        seed=int(args.seed),
    )


def _load_masked_rgba(image_path: Path, mask_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path)
    mask = (
        mask_image.getchannel("A")
        if "A" in mask_image.getbands()
        else mask_image.convert("L")
    )
    if image.size != mask.size:
        raise ValueError(
            f"Image/mask size mismatch: {image_path.name} -> {image.size} vs {mask.size}."
        )

    image_np = np.asarray(image, dtype=np.float32)
    alpha_np = np.asarray(mask, dtype=np.float32) / 255.0
    premultiplied = image_np * alpha_np[:, :, None]
    rgba = np.concatenate(
        [
            premultiplied.clip(0, 255).astype(np.uint8),
            np.asarray(mask, dtype=np.uint8)[:, :, None],
        ],
        axis=2,
    )
    return Image.fromarray(rgba, mode="RGBA")


def _save_input_views(
    chunk_dir: Path,
    label: str,
    *,
    indices: list[int],
    images_by_index: dict[int, Image.Image],
) -> list[str]:
    output_dir = chunk_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for index in indices:
        path = output_dir / f"{int(index):03d}.png"
        images_by_index[int(index)].save(path)
        saved_paths.append(str(path))
    return saved_paths


def _selection_scores_to_payload(scores: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for score in scores:
        payload.append(
            {
                "view_index": int(score.view_index),
                "joint_attention_mass": (
                    None
                    if score.joint_attention_mass is None
                    else float(score.joint_attention_mass)
                ),
                "mass_relative": (
                    None if score.mass_relative is None else float(score.mass_relative)
                ),
                "trellis_feature_energy": (
                    None
                    if score.trellis_feature_energy is None
                    else float(score.trellis_feature_energy)
                ),
            }
        )
    return payload


def _to_jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_jsonable(item) for item in payload]
    if isinstance(payload, tuple):
        return [_to_jsonable(item) for item in payload]
    if isinstance(payload, Path):
        return str(payload)
    if torch.is_tensor(payload):
        return _to_jsonable(payload.detach().cpu().tolist())
    if isinstance(payload, np.ndarray):
        return _to_jsonable(payload.tolist())
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


class StreamingTrellis2Backend:
    def __init__(self, config: Trellis2BackendConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        pipeline_cfg = self.config.pipeline
        attn_backend = pipeline_cfg.get("attn_backend")
        if attn_backend:
            os.environ["ATTN_BACKEND"] = str(attn_backend)
        if os.environ.get("SPARSE_CONV_BACKEND") is None:
            has_flex_gemm = importlib.util.find_spec("flex_gemm") is not None
            if not has_flex_gemm and importlib.util.find_spec("spconv") is not None:
                os.environ["SPARSE_CONV_BACKEND"] = "spconv"
                logger.info(
                    "flex_gemm is unavailable; auto-setting SPARSE_CONV_BACKEND=spconv."
                )

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        model_path = str(pipeline_cfg["model_path"])
        device = str(pipeline_cfg["device"])
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("TRELLIS2 requested CUDA but torch.cuda.is_available() is false.")
            pipeline.cuda()
        else:
            pipeline.to(torch.device(device))
        self._pipeline = pipeline
        return pipeline

    def _selection_config(self) -> Any:
        from trellis2.streaming3d.selection import SelectionConfig, SelectionMethod

        selection = self.config.pipeline["selection"]
        return SelectionConfig(
            topk=int(selection["topk"]),
            method=SelectionMethod(str(selection["method"])),
            warmup_steps=int(selection["warmup_steps"]),
            q_chunk_size=int(selection["q_chunk_size"]),
            attention_layer=int(selection["attention_layer"]),
            jam_kappa=float(selection["jam_kappa"]),
            memory_depth=int(selection["memory_depth"]),
            update_margin=float(selection["update_margin"]),
            # Task-43: distance-aware VA selection strength (default 0.1; only used when method=va_div).
            selection_div_lambda=float(selection.get("selection_div_lambda", 0.1)),
            random_seed=int(selection.get("random_seed", 0)),
        )

    def _fusion_config(self) -> dict[str, Any]:
        return dict(self.config.pipeline["fusion"])

    def _export_glb(
        self,
        *,
        pipeline: Any,
        mesh: Any,
        output_path: Path,
        resolution: int,
    ) -> Path:
        export_cfg = self.config.pipeline["export"]
        prefer_ovoxel = bool(export_cfg["prefer_ovoxel"])
        if prefer_ovoxel:
            try:
                from trellis2.streaming3d.export import export_glb

                return export_glb(
                    pipeline,
                    mesh,
                    output_path,
                    resolution=resolution,
                    decimation_target=int(export_cfg["decimation_target"]),
                    texture_size=int(export_cfg["texture_size"]),
                )
            except (ModuleNotFoundError, AttributeError, ImportError, RuntimeError) as exc:
                logger.warning(
                    "TRELLIS2 O-Voxel export unavailable ({}). Falling back to vertex-color GLB.",
                    exc,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertices = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        vertex_colors = np.full((vertices.shape[0], 4), 255, dtype=np.uint8)

        try:
            attrs = mesh.query_vertex_attrs()
            if torch.is_tensor(attrs):
                attrs = attrs.detach().cpu()
            layout = dict(getattr(mesh, "layout", {}))
            base_color_slice = layout.get("base_color", slice(0, 3))
            base_color = attrs[:, base_color_slice].to(torch.float32).clamp(0.0, 1.0)
            rgb = (base_color.numpy() * 255.0).round().astype(np.uint8)
            vertex_colors[:, :3] = rgb
            alpha_slice = layout.get("alpha")
            if isinstance(alpha_slice, slice):
                alpha = attrs[:, alpha_slice].to(torch.float32).clamp(0.0, 1.0)
                if alpha.ndim == 2 and alpha.shape[1] >= 1:
                    vertex_colors[:, 3] = (
                        (alpha[:, 0].numpy() * 255.0).round().astype(np.uint8)
                    )
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.warning(
                "Vertex-color extraction failed ({}). Trying nearest-voxel color fallback.",
                exc,
            )
            try:
                from scipy.spatial import cKDTree

                coords = mesh.coords.detach().cpu().to(torch.float32).numpy()
                attrs = mesh.attrs.detach().cpu().to(torch.float32).numpy()
                if coords.ndim == 2 and coords.shape[1] >= 3 and attrs.ndim == 2:
                    origin = mesh.origin.detach().cpu().to(torch.float32).numpy()
                    voxel_size = float(mesh.voxel_size)
                    voxel_xyz = origin[None, :3] + (coords[:, :3] + 0.5) * voxel_size
                    nn_indices = cKDTree(voxel_xyz).query(vertices, k=1)[1]
                    layout = dict(getattr(mesh, "layout", {}))
                    base_color_slice = layout.get("base_color", slice(0, 3))
                    base = attrs[nn_indices, base_color_slice]
                    base = np.clip(base, 0.0, 1.0)
                    if base.ndim == 2 and base.shape[1] >= 3:
                        vertex_colors[:, :3] = (base[:, :3] * 255.0).round().astype(
                            np.uint8
                        )
                    alpha_slice = layout.get("alpha")
                    if isinstance(alpha_slice, slice):
                        alpha = attrs[nn_indices, alpha_slice]
                        if alpha.ndim == 2 and alpha.shape[1] >= 1:
                            alpha = np.clip(alpha[:, 0], 0.0, 1.0)
                            vertex_colors[:, 3] = (
                                (alpha * 255.0).round().astype(np.uint8)
                            )
                else:
                    logger.warning("Nearest-voxel color fallback skipped due to invalid mesh attrs/coords.")
            except Exception as nn_exc:  # pragma: no cover
                logger.warning(
                    "Nearest-voxel color fallback failed ({}). Exporting white mesh.",
                    nn_exc,
                )

        mesh_glb = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        mesh_glb.export(output_path)
        return output_path

    def _run_chunk(
        self,
        *,
        example: StreamingExample,
        chunk_spec: dict[str, Any],
        images_by_index: dict[int, Image.Image],
        selection: Any,
        candidate_indices: list[int],
    ) -> Path:
        pipeline_cfg = self.config.pipeline
        pipeline = self._load_pipeline()

        chunk_name = str(chunk_spec["chunk_name"])
        chunk_index = int(chunk_spec["chunk_index"])
        chunk_dir = self.config.output_root / example.object_name / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)

        selected_indices = [int(index) for index in selection.selected_indices]
        selected_images = [images_by_index[index] for index in selected_indices]
        chunk_view_indices = [int(index) for index in chunk_spec["global_frame_indices"]]

        selected_input_paths = _save_input_views(
            chunk_dir,
            "selected_views",
            indices=selected_indices,
            images_by_index=images_by_index,
        )
        chunk_input_paths = _save_input_views(
            chunk_dir,
            "chunk_views",
            indices=chunk_view_indices,
            images_by_index=images_by_index,
        )

        start_time = perf_counter()
        generation_seed = int(self.config.seed) + chunk_index
        meshes, latent = pipeline.run_multi_image(
            selected_images,
            seed=generation_seed,
            preprocess_image=bool(pipeline_cfg["preprocess_image"]),
            mode=str(pipeline_cfg["mode"]),
            pipeline_type=(
                None
                if pipeline_cfg["pipeline_type"] is None
                else str(pipeline_cfg["pipeline_type"])
            ),
            return_latent=True,
            max_num_tokens=int(pipeline_cfg["max_num_tokens"]),
            fusion_config=self._fusion_config(),
        )
        _, _, resolution = latent
        mesh = meshes[0]
        glb_path = self._export_glb(
            pipeline=pipeline,
            mesh=mesh,
            output_path=chunk_dir / "result.glb",
            resolution=int(resolution),
        )

        native_state_path: Path | None = None
        if bool(pipeline_cfg["save_native_state"]):
            from trellis2.streaming3d.native_state import save_meshwithvoxel_state

            native_state_path = save_meshwithvoxel_state(
                mesh,
                chunk_dir / "result_native_meshwithvoxel.pt",
                resolution=int(resolution),
            )

        metadata = {
            "backend": {
                "name": "trellis2",
                "model_path": str(pipeline_cfg["model_path"]),
                "pipeline_type": pipeline_cfg["pipeline_type"],
                "device": str(pipeline_cfg["device"]),
                "attn_backend": pipeline_cfg["attn_backend"],
            },
            "object_name": example.object_name,
            "chunk": {
                "index": chunk_index,
                "name": chunk_name,
                "global_frame_indices": chunk_view_indices,
                "candidate_global_frame_indices": candidate_indices,
                "num_views": len(chunk_view_indices),
            },
            "selection": {
                "selected_indices": selected_indices,
                "candidate_indices": [int(index) for index in selection.candidate_indices],
                "scores": _selection_scores_to_payload(selection.scores),
                **_to_jsonable(selection.metadata),
            },
            "fusion": _to_jsonable(self._fusion_config()),
            "generation": {
                "elapsed_seconds": perf_counter() - start_time,
                "resolution": int(resolution),
                "num_selected_views": len(selected_indices),
                "mode": str(pipeline_cfg["mode"]),
                "preprocess_image": bool(pipeline_cfg["preprocess_image"]),
            },
            "outputs": {
                "glb": str(glb_path),
                "native_state": (
                    None if native_state_path is None else str(native_state_path)
                ),
                "metadata": str(chunk_dir / "result_metadata.json"),
                "selected_input_views": selected_input_paths,
                "chunk_input_views": chunk_input_paths,
            },
        }
        (chunk_dir / "result_metadata.json").write_text(
            json.dumps(_to_jsonable(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        del meshes
        del latent
        del mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return chunk_dir

    def run_example(self, example: StreamingExample) -> Path | None:
        pipeline = self._load_pipeline()
        selection_config = self._selection_config()

        images_by_index: dict[int, Image.Image] = {}
        for global_index, image_path in enumerate(example.image_files):
            mask_path = example.mask_root / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask file for {image_path.name}: {mask_path}")
            images_by_index[int(global_index)] = _load_masked_rgba(image_path, mask_path)

        reconstruction_indices = {
            int(chunk["chunk_index"])
            for chunk in example.execution_plan.reconstruction_plan
        }
        final_result: Path | None = None

        from trellis2.streaming3d.selection import (
            SelectionMethod,
            TokenVoteMemory,
            select_views,
            token_vote_report,
            update_token_vote_selection,
        )

        if selection_config.method == SelectionMethod.TOKEN_VOTE:
            token_memory = TokenVoteMemory()
            for chunk_spec in example.execution_plan.warmup_plan:
                chunk_index = int(chunk_spec["chunk_index"])
                warmup_indices = [
                    int(index) for index in chunk_spec["global_frame_indices"]
                ]
                warmup_images = [images_by_index[index] for index in warmup_indices]
                token_memory, selection = update_token_vote_selection(
                    pipeline,
                    warmup_indices,
                    warmup_images,
                    selection_config,
                    token_memory,
                )
                logger.info(
                    "TRELLIS2 token-vote warmup chunk {}: memory_views={} selected={}",
                    chunk_index,
                    len(token_vote_report(token_memory)),
                    selection.selected_indices,
                )
                if chunk_index not in reconstruction_indices:
                    continue
                candidate_end = max(int(index) for index in chunk_spec["global_frame_indices"]) + 1
                candidate_indices = list(range(candidate_end))
                final_result = self._run_chunk(
                    example=example,
                    chunk_spec=chunk_spec,
                    images_by_index=images_by_index,
                    selection=selection,
                    candidate_indices=candidate_indices,
                )
            return final_result

        for chunk_spec in example.execution_plan.reconstruction_plan:
            candidate_end = max(int(index) for index in chunk_spec["global_frame_indices"]) + 1
            candidate_indices = list(range(candidate_end))
            candidate_images = [images_by_index[index] for index in candidate_indices]
            selection = select_views(
                pipeline,
                candidate_indices,
                candidate_images,
                selection_config,
            )
            final_result = self._run_chunk(
                example=example,
                chunk_spec=chunk_spec,
                images_by_index=images_by_index,
                selection=selection,
                candidate_indices=candidate_indices,
            )
        return final_result
