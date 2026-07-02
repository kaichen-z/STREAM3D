from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from torch.utils._pytree import tree_map_only


__all__ = [
    "_cached_velocity_entry_count",
    "_canonicalize_slat_decode_feats",
    "_chain_step_recorders",
    "_detach_cpu_tree",
    "_inject_fixed_initial_noise",
    "_inject_generate_iter_latent_recorder",
    "_make_step_edited_state_recorder",
    "_make_step_latent_recorder",
    "_make_step_target_fused_velocity_recorder",
    "_make_step_target_latent_recorder",
    "_make_step_velocity_recorder",
    "_stack_view_weight_dict",
    "_summarize_weight_structure",
    "_uniform_slat_local_weights",
    "_uniform_ss_local_weights",
    "_unstack_view_weight_tensor",
]


def _detach_cpu_tree(tree):
    return tree_map_only(
        torch.Tensor,
        lambda x: x.detach().cpu().to(torch.float32).contiguous(),
        tree,
    )


def _make_step_velocity_recorder(storage: Dict[int, Any]):
    def _record(*, step_idx: int, view_predictions: List[Any], fused_prediction=None, target_only_phase: bool = False):
        cache_entry = fused_prediction if fused_prediction is not None else view_predictions
        storage[int(step_idx)] = _detach_cpu_tree(cache_entry)
    return _record


def _make_step_target_fused_velocity_recorder(storage: Dict[int, Any]):
    def _record(*, step_idx: int, fused_prediction=None, view_predictions: Optional[List[Any]] = None, **kwargs):
        cache_entry = fused_prediction if fused_prediction is not None else view_predictions
        storage[int(step_idx)] = _detach_cpu_tree(cache_entry)
    return _record


def _chain_step_recorders(*recorders):
    active_recorders = [recorder for recorder in recorders if recorder is not None]

    def _record(**kwargs):
        for recorder in active_recorders:
            recorder(**kwargs)

    return _record


def _make_step_latent_recorder(storage: Dict[int, Any]):
    def _record(*, step_idx: int, x_t=None, z_t=None, **kwargs):
        state = x_t if x_t is not None else z_t
        storage[int(step_idx)] = _detach_cpu_tree(state)
    return _record


def _make_step_target_latent_recorder(storage: Dict[int, Any]):
    def _record(*, step_idx: int, z_t, **kwargs):
        storage[int(step_idx)] = _detach_cpu_tree(z_t)
    return _record


def _make_step_edited_state_recorder(storage: Dict[int, Any]):
    def _record(*, step_idx: int, z_edit, **kwargs):
        storage[int(step_idx)] = _detach_cpu_tree(z_edit)
    return _record


@contextmanager
def _inject_generate_iter_latent_recorder(generator, latent_recorder=None):
    if latent_recorder is None:
        yield
        return

    original_generate_iter = generator.generate_iter

    def wrapped_generate_iter(*args, **kwargs):
        for step_idx, (t, x_t, extra) in enumerate(original_generate_iter(*args, **kwargs)):
            latent_recorder(step_idx=int(step_idx), x_t=x_t, t=t, extra=extra)
            yield t, x_t, extra

    generator.generate_iter = wrapped_generate_iter
    try:
        yield
    finally:
        generator.generate_iter = original_generate_iter


@contextmanager
def _inject_fixed_initial_noise(generator, fixed_noise=None):
    if fixed_noise is None:
        yield
        return

    original_generate_noise = getattr(generator, "_generate_noise", None)
    if original_generate_noise is None:
        raise AttributeError("Generator does not expose _generate_noise for fixed-noise injection.")

    injected = False

    def wrapped_generate_noise(shape, device):
        nonlocal injected
        if not injected:
            injected = True
            return tree_map_only(torch.Tensor, lambda x: x.to(device), fixed_noise)
        return original_generate_noise(shape, device)

    generator._generate_noise = wrapped_generate_noise
    try:
        yield
    finally:
        generator._generate_noise = original_generate_noise


def _cached_velocity_entry_count(cache: Dict[int, Any]) -> int:
    if not cache:
        return 0
    first_entry = next(iter(cache.values()))
    return len(first_entry) if isinstance(first_entry, (list, tuple)) else 1


def _canonicalize_slat_decode_feats(feats: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(feats):
        raise TypeError(f"Expected torch.Tensor for SLAT decode feats, got {type(feats)!r}")
    if feats.dim() == 2:
        return feats.contiguous()
    if feats.dim() == 3 and feats.shape[0] == 1:
        return feats[0].contiguous()
    raise ValueError(f"Unexpected SLAT decode feat shape {tuple(feats.shape)}; expected [N,C] or [1,N,C].")


def _summarize_weight_structure(weights: Any) -> Dict[str, Any]:
    if weights is None:
        return {"kind": "none", "shape": None, "view_means": None}
    if isinstance(weights, dict):
        return {
            "kind": "dict",
            "shape": {int(k): list(v.shape) for k, v in weights.items()},
            "view_means": {int(k): float(v.detach().cpu().to(torch.float32).mean().item()) for k, v in weights.items()},
        }
    if torch.is_tensor(weights):
        if weights.dim() == 1:
            view_means = {int(idx): float(value) for idx, value in enumerate(weights.detach().cpu().to(torch.float32).tolist())}
        else:
            view_means = {
                int(idx): float(weights[idx].detach().cpu().to(torch.float32).mean().item())
                for idx in range(int(weights.shape[0]))
            }
        return {
            "kind": "tensor",
            "shape": list(weights.shape),
            "view_means": view_means,
        }
    if isinstance(weights, (list, tuple)):
        return {
            "kind": type(weights).__name__,
            "shape": [None if not torch.is_tensor(value) else list(value.shape) for value in weights],
            "view_means": {
                int(idx): None if not torch.is_tensor(value) else float(value.detach().cpu().to(torch.float32).mean().item())
                for idx, value in enumerate(weights)
            },
        }
    return {"kind": type(weights).__name__, "shape": None, "view_means": None}


def _uniform_ss_local_weights(num_views: int, num_latents: int = 4096) -> torch.Tensor:
    return torch.full(
        (int(num_views), int(num_latents)),
        1.0 / float(num_views),
        dtype=torch.float32,
    )


def _uniform_slat_local_weights(num_views: int, num_rows: int) -> Dict[int, torch.Tensor]:
    weight = 1.0 / float(num_views)
    return {
        int(view_idx): torch.full((int(num_rows),), weight, dtype=torch.float32)
        for view_idx in range(int(num_views))
    }


def _stack_view_weight_dict(weights: Dict[int, torch.Tensor]) -> torch.Tensor:
    view_indices = sorted(int(view_idx) for view_idx in weights.keys())
    if not view_indices:
        raise ValueError("Cannot stack an empty view-weight dict.")
    return torch.stack(
        [weights[int(view_idx)].detach().cpu().to(torch.float32).contiguous() for view_idx in view_indices],
        dim=0,
    )


def _unstack_view_weight_tensor(weights: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {
        int(view_idx): weights[int(view_idx)].detach().cpu().to(torch.float32).contiguous()
        for view_idx in range(int(weights.shape[0]))
    }
