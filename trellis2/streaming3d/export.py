from __future__ import annotations

from pathlib import Path
from typing import Any
import sys


def export_glb(
    pipeline: Any,
    mesh: Any,
    output_path: Path,
    resolution: int | None = None,
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
) -> Path:
    try:
        import o_voxel
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "o-voxel"))
        import o_voxel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_size = resolution if resolution is not None else round(1.0 / float(mesh.voxel_size))
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.export(str(output_path), extension_webp=True)
    return output_path
