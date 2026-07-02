# Copyright (c) Meta Platforms, Inc. and affiliates.
import importlib.util
from typing import *
from loguru import logger

BACKEND = "sdpa"
DEBUG = False
VALID_BACKENDS = {
    "xformers",
    "flash_attn",
    "torch_flash_attn",
    "sdpa",
    "naive",
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _supports_torch_flash_attn() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if not hasattr(torch.nn, "attention") or not hasattr(
            torch.nn.attention, "sdpa_kernel"
        ):
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def _auto_backend() -> str:
    if _has_module("flash_attn"):
        return "flash_attn"
    if _supports_torch_flash_attn():
        return "torch_flash_attn"
    if _has_module("xformers"):
        return "xformers"
    return "sdpa"


def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get("ATTN_BACKEND")
    env_sttn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is None or env_attn_backend == "auto":
        BACKEND = _auto_backend()
    elif env_attn_backend in VALID_BACKENDS:
        BACKEND = env_attn_backend
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == "1"

    logger.info(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(
    backend: Literal["xformers", "flash_attn", "torch_flash_attn", "sdpa", "naive"]
):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
