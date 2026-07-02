# Copyright (c) Meta Platforms, Inc. and affiliates.
import importlib.util
from typing import *
from loguru import logger

BACKEND = "spconv"
# BACKEND = "torchsparse"
DEBUG = False
ATTN = "sdpa"
VALID_ATTN_BACKENDS = {
    "xformers",
    "flash_attn",
    "sdpa",
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _auto_sparse_attn() -> str:
    if _has_module("flash_attn"):
        return "flash_attn"
    if _has_module("xformers"):
        return "xformers"
    return "sdpa"


def __from_env():
    import os

    global BACKEND
    global DEBUG
    global ATTN

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in [
        "spconv",
        "torchsparse",
    ]:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    if env_sparse_attn is None or env_sparse_attn == "auto":
        ATTN = _auto_sparse_attn()
    elif env_sparse_attn in VALID_ATTN_BACKENDS:
        ATTN = env_sparse_attn

    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")


__from_env()


def set_backend(backend: Literal["spconv", "torchsparse"]):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn(attn: Literal["xformers", "flash_attn", "sdpa"]):
    global ATTN
    ATTN = attn


from .basic import *
from .norm import *
from .nonlinearity import *
from .linear import *
from .attention import *
from .conv import *
from .spatial import *
from . import transformer
