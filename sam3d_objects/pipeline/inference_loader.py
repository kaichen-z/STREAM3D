from __future__ import annotations

import builtins
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List

from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


def check_target(
    target: str,
    whitelist_filters: List[Callable[[str], bool]],
    blacklist_filters: List[Callable[[str], bool]],
) -> None:
    if any(filter_fn(target) for filter_fn in whitelist_filters):
        if not any(filter_fn(target) for filter_fn in blacklist_filters):
            return
    raise RuntimeError(
        f"Hydra target '{target}' is not allowed. Update inference_loader.py if this target is expected."
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable[[str], bool]] | None = None,
    blacklist_filters: List[Callable[[str], bool]] | None = None,
) -> None:
    whitelist_filters = WHITELIST_FILTERS if whitelist_filters is None else whitelist_filters
    blacklist_filters = BLACKLIST_FILTERS if blacklist_filters is None else blacklist_filters
    to_check = [config]
    while to_check:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(str(node["_target_"]), whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_pipeline_from_config(config_file: str | Path, *, compile: bool = False):
    config_path = Path(config_file)
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = bool(compile)
    config.workspace_dir = str(config_path.parent)
    check_hydra_safety(config)
    return instantiate(config)
