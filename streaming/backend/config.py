from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from .selector.selector import Stage2SelectionConfig, ViewConditionCacheConfig
from ..data import StreamingExample


@dataclass(frozen=True)
class StreamingBackendConfig:
    model_config_path: Path
    output_root: Path
    pipeline: Dict[str, Any]
    cache: ViewConditionCacheConfig
    stage2_selection: Stage2SelectionConfig
    camera_pose_source: str
    dataset_camera_path: Path | None
    seed: int
    shared_chunk_rng_stream: bool


def build_streaming_backend_config(args: DictConfig) -> StreamingBackendConfig:
    streaming_config = dict(OmegaConf.to_container(args.streaming, resolve=True))
    stage2_selection = dict(streaming_config.pop("stage2_selection"))

    cache_config = ViewConditionCacheConfig(**streaming_config)
    stage2_selection_config = Stage2SelectionConfig(**stage2_selection)

    pipeline_config = dict(OmegaConf.to_container(args.pipeline, resolve=True))
    return StreamingBackendConfig(
        model_config_path=Path(args.model_config_path),
        output_root=Path(args.output_root),
        pipeline=pipeline_config,
        cache=cache_config,
        stage2_selection=stage2_selection_config,
        camera_pose_source=str(args.camera_pose_source),
        dataset_camera_path=(
            None
            if args.dataset_camera_path is None
            else Path(args.dataset_camera_path)
        ),
        seed=int(args.seed),
        shared_chunk_rng_stream=bool(args.shared_chunk_rng_stream),
    )


@dataclass
class SelectedChunkPlanContext:
    args: StreamingBackendConfig
    example: StreamingExample
    execution_plan: Any
    image_files: list[Path]
    mask_root: Path
    scene_da3: Dict[str, Any]
    pipeline: Any
    cache_config: Any
    stage2_selection_config: Stage2SelectionConfig
    cache_state: Any
    view_condition_selector: Any
    reconstruction_indices: set[int]
    warmup_prefix: list[Dict[str, Any]] = field(default_factory=list)
    prev_loaded_image_names: Optional[list[str]] = None
    prev_reconstructed_chunk_index: Optional[int] = None
    final_result_dir: Optional[Path] = None
    reconstructed_count: int = 0
    seen_chunk_specs: list[Dict[str, Any]] = field(default_factory=list)
    evaluated_warmup_global_indices: set[int] = field(default_factory=set)


@dataclass
class WarmupResult:
    chunk_name: str
    chunk_index: int
    warmup_chunk_spec: Dict[str, Any]
    warmup_frame_keys: list[str]
    selected_views: list[Dict[str, Any]]
    selection_warnings: list[str]
    selection_metadata: Dict[str, Any]
    warmup: Dict[str, Any] | None
    warmup_profile: Any
    attention_count: int
    cache_warnings: list[str]
    skipped_duplicate_global_indices: list[int]
    source_chunk: Dict[str, Any]
    prepared_warmup: Any | None = None
    warmup_images: Any | None = None
    warmup_masks: Any | None = None
    warmup_da3: Any | None = None

    def release(self) -> None:
        self.prepared_warmup = None
        self.warmup_images = None
        self.warmup_masks = None
        self.warmup_da3 = None
        self.warmup = None
        self.warmup_profile = None


@dataclass
class SelectedRuntime:
    chunk_name: str
    chunk_index: int
    selected_views: list[Dict[str, Any]]
    runtime_chunk_spec: Dict[str, Any]
    loaded_image_names: list[str]
    runtime_frame_keys: list[str]
    view_images: list[Any]
    view_masks: list[Any]
    chunk_da3: Dict[str, Any]
    crop_views: list[Dict[str, Any]]
    prev_view_index_map: Optional[Dict[int, int]]
    output_dir: Path
    pipeline_kwargs: Dict[str, Any]
    stage2_weighting: Dict[str, Any]


@dataclass
class ChunkResultArtifacts:
    outputs: Dict[str, Any]
    stage2_weighting_metadata: Dict[str, Any]
    stage2_selection_metadata: Optional[Dict[str, Any]]
    stage1_sparse_files: list[str]
