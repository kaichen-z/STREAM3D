from __future__ import annotations

from typing import Any, Protocol

from omegaconf import DictConfig

from .backend import StreamingSam3DBackend
from .backend_trellis2 import StreamingTrellis2Backend, build_trellis2_backend_config
from .config import build_streaming_backend_config


class StreamingBackend(Protocol):
    def run_example(self, example: Any) -> Any:
        ...


def build_backend(args: DictConfig) -> StreamingBackend:
    backend_name = str(args.backend.name)
    if backend_name == "sam3d":
        return StreamingSam3DBackend(build_streaming_backend_config(args))
    if backend_name == "trellis2":
        return StreamingTrellis2Backend(build_trellis2_backend_config(args))
    raise ValueError(f"Unsupported backend.name: {backend_name}")

