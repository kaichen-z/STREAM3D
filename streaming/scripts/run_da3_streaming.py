#!/usr/bin/env python3
"""
run DA3-Streaming on an image sequence directory, with optional overrides for chunk size, overlap, and checkpoint path.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DA3_ROOT = PROJECT_ROOT / "third_party" / "Depth-Anything-3"
DA3_STREAMING_ROOT = DA3_ROOT / "da3_streaming"

def build_parser():
    parser = argparse.ArgumentParser(description="Run DA3-Streaming on an image sequence directory.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Input image directory.")
    parser.add_argument("--chunk_size", type=int, default=64, help="DA3-Streaming chunk size override.")
    parser.add_argument("--chunk_overlap", type=int, default=32, help="DA3-Streaming overlap override.")
    parser.add_argument("--ckpt_path", type=Path, default=None, help="Optional DA3 checkpoint override.")
    return parser.parse_args()

if __name__ == "__main__":
    args = build_parser()

    image_dir: Path = args.image_dir.resolve()

    def collect_image_files(image_dir: Path) -> List[Path]:
        """
        Collect image files from the directory,
        sorted in natural order based on the numeric suffix in the filename.
        """
        def natural_sort_key(path: Path) -> Tuple[int, int, str]:
            stem = path.stem
            try:
                return (0, int(stem.split("_")[-1]), stem)
            except ValueError:
                return (1, 0, stem)
        return sorted(list(image_dir.glob("*.png")), key=natural_sort_key)

    image_files: List[Path] = collect_image_files(image_dir)

    output_root = image_dir.parent / "da3"
    output_root.mkdir(parents=True, exist_ok=True)
    # Load the base config and override with command-line arguments
    config_path = DA3_STREAMING_ROOT / "configs" / "base_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["Model"]["chunk_size"] = int(args.chunk_size)
    config["Model"]["overlap"] = int(args.chunk_overlap)
    if args.ckpt_path is not None:
        config["Weights"]["DA3"] = str(args.ckpt_path.expanduser())

    resolved_config_path = output_root / "da3_streaming_config_resolved.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    # Run the DA3-Streaming script with the resolved config
    command: List[str] = [
        sys.executable, "da3_streaming.py",
        "--image_dir", str(image_dir),
        "--config", str(resolved_config_path),
        "--output_dir", str(output_root),
    ]
    # Ensure the DA3 source directory is in PYTHONPATH for the subprocess
    env = os.environ.copy()
    pythonpath_parts = [str((DA3_ROOT / "src").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    subprocess.run(
        command,
        cwd=str(DA3_STREAMING_ROOT),
        check=True,
        env=env,
        text=True,
    )
