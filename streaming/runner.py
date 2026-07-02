#!/usr/bin/env python3
from pathlib import Path

import hydra
from omegaconf import DictConfig

from streaming.backend import build_backend
from streaming.data import DataGSO


def normalize_runtime_args(args: DictConfig) -> None:
    for field in ("chunk_size", "chunk_overlap", "chunk_indices"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args.data, field, value)

    for path_field in ("output_root", "dataset_camera_path"):
        value = getattr(args, path_field, None)
        if isinstance(value, str):
            setattr(args, path_field, Path(value))

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base="1.3", config_path="../configs/", config_name="base")
def main(args: DictConfig) -> None:
    normalize_runtime_args(args)
    backend = build_backend(args)

    for example in DataGSO(args.data):
        backend.run_example(example)


if __name__ == "__main__":
    main()
