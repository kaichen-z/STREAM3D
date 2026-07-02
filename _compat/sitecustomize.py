"""Python 3.10 compatibility shim for a repo that targets 3.11.

The streaming3d-us code_final tree uses `from enum import StrEnum`, which only
exists in Python 3.11+. The dependency stack in this workspace is installed for
system python3.10, so we backfill StrEnum to run the code unchanged.

Auto-imported by CPython at startup when this directory is on PYTHONPATH.
"""
import enum

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        """Minimal backport of Python 3.11's enum.StrEnum."""

        def __str__(self):  # behave like the 3.11 version
            return str(self.value)

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

    enum.StrEnum = StrEnum


# ---------------------------------------------------------------------------
# utils3d device-hygiene shim.
#
# The installed utils3d (1.3) converts python-float arguments to CPU float32
# tensors via its @totensor decorator, even when sibling arguments are CUDA
# tensors. That triggers "Expected all tensors to be on the same device"
# inside geometry helpers such as intrinsics_from_focal_center. Moving a scalar
# constant from CPU to CUDA is numerically identical, so we wrap utils3d.torch
# callables to pre-promote python-float scalars onto the device of any sibling
# CUDA tensor. Ints (used as image sizes etc.) and existing tensors are left
# untouched, so this changes device placement only -- never values.
# ---------------------------------------------------------------------------
def _install_utils3d_device_shim():
    try:
        import torch
        import utils3d.torch as u3t
    except Exception:
        return

    if getattr(u3t, "_device_shim_installed", False):
        return

    import functools

    def _wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            dev = None
            for a in list(args) + list(kwargs.values()):
                if torch.is_tensor(a) and a.is_cuda:
                    dev = a.device
                    break
            if dev is None:
                return fn(*args, **kwargs)

            def promote(a):
                # bool is a subclass of int -- exclude both; only real floats.
                if isinstance(a, float):
                    return torch.tensor(a, dtype=torch.float32, device=dev)
                return a

            args = tuple(promote(a) for a in args)
            kwargs = {k: promote(v) for k, v in kwargs.items()}
            return fn(*args, **kwargs)

        return wrapper

    # utils3d.torch exposes geometry helpers lazily via module __getattr__, so
    # iterating dir(utils3d.torch) misses them. Patch the defining submodules
    # in place instead; the package __getattr__ then returns the wrapped object.
    import importlib
    import inspect

    for sub in ("transforms", "maps", "mesh", "rasterization", "utils"):
        try:
            mod = importlib.import_module(f"utils3d.torch.{sub}")
        except Exception:
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if (inspect.isfunction(obj) or inspect.isbuiltin(obj)) \
                    and not getattr(obj, "_u3d_dev_wrapped", False):
                try:
                    wrapped = _wrap(obj)
                    wrapped._u3d_dev_wrapped = True
                    setattr(mod, name, wrapped)
                except Exception:
                    pass

    u3t._device_shim_installed = True


_install_utils3d_device_shim()
