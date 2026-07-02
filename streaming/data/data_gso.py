from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DataGSOConfig:
    name: str
    roots: list[Path]
    render_split: str
    image_dir_name: str
    mask_dir_name: str
    da3_dir_name: str | None
    chunk_size: int
    chunk_overlap: int
    chunk_indices: list[int] | None

@dataclass(frozen=True)
class ChunkExecutionPlan:
    warmup_plan: list[dict[str, Any]]
    reconstruction_plan: list[dict[str, Any]]
    fast_single_chunk: bool
    requested_chunk_indices: list[int] | None
    resolved_chunk_indices: list[int]


@dataclass(frozen=True)
class StreamingExample:
    object_name: str
    root: Path
    render_root: Path
    image_dir: Path
    mask_root: Path
    da3_root: Path | None
    image_files: list[Path]
    chunk_plan: list[dict[str, Any]]
    execution_plan: ChunkExecutionPlan


class DataGSO(Iterable[StreamingExample]):
    def __init__(self, cfg: DataGSOConfig) -> None:
        self.cfg = cfg

    def __iter__(self):
        for root in self.cfg.roots:
            # GSO roots are object-level directories: GSO30/<object_id>/render_spiral_100.
            root = Path(root).resolve()
            render_root = root / self.cfg.render_split
            image_dir = render_root / self.cfg.image_dir_name
            mask_root = render_root / self.cfg.mask_dir_name

            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not mask_root.is_dir():
                raise FileNotFoundError(f"Mask directory not found: {mask_root}")

            image_files = sorted(
                list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            if not image_files:
                raise FileNotFoundError(f"No PNG/JPG images found in: {image_dir}")
            da3_root = None
            if self.cfg.da3_dir_name is not None:
                da3_root = render_root / self.cfg.da3_dir_name

            chunk_plan = self.build_chunk_plan(image_files)
            execution_plan = self.resolve_chunk_execution_plan(chunk_plan)

            yield StreamingExample(
                object_name=root.name,
                root=root,
                render_root=render_root,
                image_dir=image_dir,
                mask_root=mask_root,
                da3_root=da3_root,
                image_files=image_files,
                chunk_plan=chunk_plan,
                execution_plan=execution_plan,
            )

    def build_chunk_plan(self, image_files: list[Path]) -> list[dict[str, Any]]:
        chunk_size = int(self.cfg.chunk_size)
        chunk_overlap = int(self.cfg.chunk_overlap)
        if chunk_size <= 0:
            raise ValueError("data.chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("data.chunk_overlap must be non-negative and smaller than data.chunk_size")
        if len(image_files) < chunk_size:
            raise ValueError(
                f"Need at least {chunk_size} images to build chunks, found {len(image_files)}."
            )
        step = chunk_size - chunk_overlap
        chunk_starts = list(range(0, len(image_files) - chunk_size + 1, step))
        final_start = len(image_files) - chunk_size
        if chunk_starts[-1] != final_start:
            chunk_starts.append(final_start)

        plan: list[dict[str, Any]] = []
        for chunk_idx, start in enumerate(chunk_starts):
            frame_paths = list(image_files[start:start + chunk_size])
            global_frame_indices = list(range(start, start + chunk_size))
            plan.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_name": f"chunk_{chunk_idx:04d}",
                    "frame_paths": frame_paths,
                    "stems": [frame_path.stem for frame_path in frame_paths],
                    "names": [frame_path.name for frame_path in frame_paths],
                    "global_frame_indices": global_frame_indices,
                    "num_views": len(frame_paths),
                }
            )

        return plan

    def resolve_chunk_execution_plan(self, plan: list[dict[str, Any]]) -> ChunkExecutionPlan:
        if self.cfg.chunk_indices is None:
            return ChunkExecutionPlan(
                warmup_plan=plan,
                reconstruction_plan=plan,
                fast_single_chunk=False,
                requested_chunk_indices=None,
                resolved_chunk_indices=[
                    int(item["chunk_index"]) for item in plan
                ],
            )

        requested = [int(index) for index in self.cfg.chunk_indices]

        if not requested:
            return ChunkExecutionPlan(
                warmup_plan=plan,
                reconstruction_plan=plan,
                fast_single_chunk=False,
                requested_chunk_indices=requested,
                resolved_chunk_indices=[
                    int(item["chunk_index"]) for item in plan
                ],
            )

        if len(requested) == 1:
            requested_index = int(requested[0])
            target_position = requested_index
            if requested_index < 0:
                target_position = len(plan) + requested_index

            target = plan[target_position]
            resolved_index = int(target["chunk_index"])
            return ChunkExecutionPlan(
                warmup_plan=plan[: target_position + 1],
                reconstruction_plan=[target],
                fast_single_chunk=True,
                requested_chunk_indices=requested,
                resolved_chunk_indices=[resolved_index],
            )

        lookup = {int(item["chunk_index"]): item for item in plan}
        reconstruction_plan = [lookup[int(idx)] for idx in requested]

        return ChunkExecutionPlan(
            warmup_plan=reconstruction_plan,
            reconstruction_plan=reconstruction_plan,
            fast_single_chunk=False,
            requested_chunk_indices=requested,
            resolved_chunk_indices=[
                int(item["chunk_index"]) for item in reconstruction_plan
            ],
        )
