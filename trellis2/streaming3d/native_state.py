from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1
REPRESENTATION = "MeshWithVoxel"


def _layout_to_payload(layout: dict[str, slice]) -> dict[str, tuple[int | None, int | None, int | None]]:
    return {str(name): (value.start, value.stop, value.step) for name, value in layout.items()}


def _layout_from_payload(payload: dict[str, tuple[int | None, int | None, int | None]]) -> dict[str, slice]:
    return {str(name): slice(*value) for name, value in payload.items()}


def save_meshwithvoxel_state(mesh: Any, path: Path, *, resolution: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "representation": REPRESENTATION,
        "resolution": int(resolution),
        "vertices": mesh.vertices.detach().cpu(),
        "faces": mesh.faces.detach().cpu(),
        "origin": mesh.origin.detach().cpu(),
        "voxel_size": float(mesh.voxel_size),
        "coords": mesh.coords.detach().cpu(),
        "attrs": mesh.attrs.detach().cpu(),
        "voxel_shape": list(mesh.voxel_shape),
        "layout": _layout_to_payload(mesh.layout),
    }
    torch.save(payload, path)
    return path


def load_meshwithvoxel_state(path: Path, *, device: torch.device | str = "cuda") -> Any:
    from trellis2.representations.mesh import MeshWithVoxel

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if int(payload["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"Unsupported TRELLIS native state format_version={payload['format_version']}")
    if str(payload["representation"]) != REPRESENTATION:
        raise ValueError(f"Unsupported TRELLIS native representation={payload['representation']!r}")
    target = torch.device(device)
    return MeshWithVoxel(
        vertices=payload["vertices"].to(target),
        faces=payload["faces"].to(target),
        origin=payload["origin"].tolist(),
        voxel_size=float(payload["voxel_size"]),
        coords=payload["coords"].to(target),
        attrs=payload["attrs"].to(target),
        voxel_shape=torch.Size(payload["voxel_shape"]),
        layout=_layout_from_payload(payload["layout"]),
    )
