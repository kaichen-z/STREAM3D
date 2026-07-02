# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-view multidiffusion utilities for SAM 3D Objects
Adapted from TRELLIS implementation, adapted for SAM 3D Objects' two-stage structure
"""

from contextlib import contextmanager
from typing import Literal, Optional
import torch
from loguru import logger

# Pose-related keys; these should not be averaged
POSE_KEYS = {
    "translation",
    "rotation",
    "scale",
    "translation_scale",
    "6drotation",
    "6drotation_normalized",
    "quaternion",
}


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    shape_weights: Optional[torch.Tensor] = None,
    step_recorder=None,
    target_latent_recorder=None,
    timestep_entropy_recorder=None,
    dynamic_jam_alpha: Optional[float] = None,
):
    """
    Inject multi-view support into generator.

    Args:
        generator: SAM 3D Objects generator (ss_generator or slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        mode: 'stochastic' or 'multidiffusion'
        shape_weights: Optional weights for shape fusion
            - If None: use simple average
            - If [num_views]: use per-view weights (same weight for all latent points)
            - If [num_views, num_latent_points]: use per-view-per-latent weights

    Yields:
        None (kept for API compatibility)

    Multi-view Iteration Strategy:
    ------------------------------
    - Shape: Weighted average (or simple average if no weights)
    - Pose: Only View 0's velocity is used (other views' pose velocity ignored)
    - Output: shape + View 0's pose
    """
    all_view_states_storage = None

    original_dynamics = generator._generate_dynamics

    if mode == "stochastic":
        # Stochastic mode: randomly select one view at each step
        if num_views > num_steps:
            logger.warning(
                f"Warning: number of views ({num_views}) is greater than number of steps ({num_steps}). "
                "This may lead to performance degradation."
            )

        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        cond_idx_counter = [0]

        def _new_dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            """Stochastic mode: select one view per time step"""
            step_idx = int(cond_idx_counter[0])
            cond_idx = cond_indices[step_idx % len(cond_indices)]
            cond_idx_counter[0] += 1

            if len(args_conditionals) > 0:
                cond_tokens = args_conditionals[0]
                if isinstance(cond_tokens, (list, tuple)):
                    cond_i = (
                        cond_tokens[cond_idx : cond_idx + 1]
                        if isinstance(cond_tokens[0], torch.Tensor)
                        else [cond_tokens[cond_idx]]
                    )
                    new_args = (cond_i,) + args_conditionals[1:]
                elif (
                    isinstance(cond_tokens, torch.Tensor)
                    and cond_tokens.shape[0] == num_views
                ):
                    cond_i = cond_tokens[cond_idx : cond_idx + 1]
                    new_args = (cond_i,) + args_conditionals[1:]
                else:
                    new_args = args_conditionals
            else:
                new_args = args_conditionals

            if target_latent_recorder is not None:
                target_latent_recorder(
                    step_idx=step_idx, z_t=x_t, target_only_phase=False
                )
            if timestep_entropy_recorder is not None:
                timestep_entropy_recorder.begin_view(
                    step_idx=step_idx, view_idx=cond_idx
                )
            try:
                return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
            finally:
                if timestep_entropy_recorder is not None:
                    timestep_entropy_recorder.end_view()

        generator._generate_dynamics = _new_dynamics_stochastic

    elif mode == "multidiffusion":
        # Multidiffusion mode: fuse predictions from all views at each step
        dt = 1.0 / num_steps
        state = {"step_idx": 0}

        def _new_dynamics_multidiffusion(
            x_t, t, *args_conditionals, **kwargs_conditionals
        ):
            """
            Multidiffusion mode: fuse predictions from all views.

            Shape: use averaged velocity to update shape
            Pose:
                - Default mode: only use View 0's velocity
                - Per-view mode: each view uses its own velocity to update its own pose
            """
            nonlocal all_view_states_storage

            # Find the position of condition tokens in args
            cond_idx = 0
            if len(args_conditionals) > 0:
                if isinstance(args_conditionals[0], (int, float)) or (
                    isinstance(args_conditionals[0], torch.Tensor)
                    and args_conditionals[0].numel() == 1
                ):
                    cond_idx = 1

            if len(args_conditionals) <= cond_idx:
                return original_dynamics(
                    x_t, t, *args_conditionals, **kwargs_conditionals
                )

            cond_tokens = args_conditionals[cond_idx]

            # Logging (only once)
            if not hasattr(_new_dynamics_multidiffusion, "_logged_cond_shape"):
                logger.info(
                    f"[Multidiffusion] num_views: {num_views}, cond_idx: {cond_idx}"
                )
                if isinstance(cond_tokens, torch.Tensor):
                    logger.info(
                        f"[Multidiffusion] Condition tokens shape: {cond_tokens.shape}"
                    )
                elif isinstance(cond_tokens, (list, tuple)):
                    logger.info(
                        f"[Multidiffusion] Condition tokens: list/tuple, length={len(cond_tokens)}"
                    )
                _new_dynamics_multidiffusion._logged_cond_shape = True

            # Parse the condition for each view
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif (
                isinstance(cond_tokens, torch.Tensor)
                and cond_tokens.shape[0] == num_views
            ):
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(
                    f"Condition tokens not organized by views, using same condition for all views"
                )
                view_conditions = [cond_tokens] * num_views

            # Fuse predictions from all views
            # Shape: averaged, Pose: View 0 only
            preds = []
            step_idx = int(state["step_idx"])
            state["step_idx"] += 1
            if target_latent_recorder is not None:
                target_latent_recorder(
                    step_idx=step_idx, z_t=x_t, target_only_phase=False
                )
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = (
                        args_conditionals[:cond_idx]
                        + (view_cond,)
                        + args_conditionals[cond_idx + 1 :]
                    )
                else:
                    new_args = args_conditionals + (view_cond,)
                if timestep_entropy_recorder is not None:
                    timestep_entropy_recorder.begin_view(
                        step_idx=step_idx, view_idx=view_idx
                    )
                try:
                    pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                finally:
                    if timestep_entropy_recorder is not None:
                        timestep_entropy_recorder.end_view()
                preds.append(pred)

            # Log (only once)
            if not hasattr(_new_dynamics_multidiffusion, "_logged_shape"):
                if isinstance(x_t, dict):
                    logger.info(f"[Multidiffusion] x_t keys: {list(x_t.keys())}")
                if isinstance(preds[0], dict):
                    logger.info(f"[Multidiffusion] pred keys: {list(preds[0].keys())}")
                if shape_weights is not None:
                    logger.info(
                        f"[Multidiffusion] Using weighted fusion: weights shape = {shape_weights.shape}"
                    )
                else:
                    logger.info(f"[Multidiffusion] Using simple average (no weights)")
                logger.info(
                    f"[Multidiffusion] Default mode: Shape=weighted/avg, Pose=View0"
                )
                _new_dynamics_multidiffusion._logged_shape = True

            active_weights = shape_weights
            if timestep_entropy_recorder is not None and dynamic_jam_alpha is not None:
                step_weights = timestep_entropy_recorder.get_step_view_weights(
                    step_idx,
                    alpha=float(dynamic_jam_alpha),
                )
                if step_weights is not None:
                    active_weights = step_weights

            from sam3d_objects.pipeline.multi_view_weighting import fuse_predictions

            fused_pred = fuse_predictions(
                preds, weights=active_weights, pose_keys=POSE_KEYS
            )
            if step_recorder is not None:
                step_recorder(
                    step_idx=step_idx,
                    view_predictions=preds,
                    fused_prediction=fused_pred,
                    target_only_phase=False,
                )
            return fused_pred

        generator._generate_dynamics = _new_dynamics_multidiffusion

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        yield all_view_states_storage
    finally:
        generator._generate_dynamics = original_dynamics
