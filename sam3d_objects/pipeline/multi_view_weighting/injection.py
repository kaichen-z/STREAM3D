"""
Multi-view weighted fusion utilities.

This module provides weighted multidiffusion fusion based on attention entropy.
It extends the basic multidiffusion to support per-latent weighting.

Key Design (Two-Pass):
    1. Warmup Pass: Run step 0 with simple averaging to collect attention
    2. Compute weights from attention entropy
    3. Main Pass: Run full generation from step 0 with weighted fusion

This ensures ALL steps benefit from weighted fusion.
"""
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import torch
from loguru import logger

from sam3d_objects.pipeline.multi_view_weighting.attention import (
    AttentionCollector,
    SSAttentionCollector,
    _compute_attention_scores,
    _compute_ss_attention_scores,
)
from sam3d_objects.pipeline.multi_view_weighting.fusion import (
    fuse_predictions,
)

@contextmanager
def inject_ss_generator_with_collector(
    generator,
    num_views: int,
    num_steps: int,
    attention_collector: SSAttentionCollector,
):
    """
    Inject multi-view support with attention collection for SS (Stage 1).

    This is similar to inject_generator_multi_view_with_collector but for SS generator
    which uses MM-DiT architecture and dense latent (4096 voxels).

    Args:
        generator: SS generator (ss_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        attention_collector: SSAttentionCollector instance

    Yields:
        None
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules.attention import MultiHeadAttention

    original_dynamics = generator._generate_dynamics

    # Hook into cross-attention to collect attention
    hooks = []
    cfg_wrapper = getattr(generator, "reverse_fn", None)
    backbone = getattr(cfg_wrapper, "backbone", None)

    if backbone is not None:
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx != attention_collector.target_layer:
                    continue
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is None:
                    continue

                # MM-DiT: cross_attn is a ModuleDict, we only care about 'shape'
                import torch.nn as nn
                if isinstance(cross_attn, nn.ModuleDict):
                    shape_attn = cross_attn["shape"] if "shape" in cross_attn else None
                    if shape_attn is not None and isinstance(shape_attn, MultiHeadAttention):
                        def make_hook(layer_idx):
                            def hook(module, inputs, outputs):
                                if len(inputs) < 2:
                                    return
                                query, context = inputs[0], inputs[1]

                                # Compute attention weights
                                with torch.no_grad():
                                    scores = _compute_ss_attention_scores(module, query, context)
                                    attn = torch.softmax(scores, dim=-1).mean(dim=1) if scores is not None else None
                                    if attn is not None:
                                        attention_collector.collect(
                                            layer_idx,
                                            attn,
                                            attention_scores=scores,
                                            query_input=query,
                                            context_input=context,
                                            attn_module=module,
                                        )
                            return hook

                        handle = shape_attn.register_forward_hook(make_hook(idx))
                        hooks.append(handle)
                        logger.info(f"[SSAttentionCollector] Hooked layer {idx} for shape attention collection")
                else:
                    # Non-MM-DiT fallback
                    if isinstance(cross_attn, MultiHeadAttention):
                        def make_hook(layer_idx):
                            def hook(module, inputs, outputs):
                                if len(inputs) < 2:
                                    return
                                query, context = inputs[0], inputs[1]
                                with torch.no_grad():
                                    scores = _compute_ss_attention_scores(module, query, context)
                                    attn = torch.softmax(scores, dim=-1).mean(dim=1) if scores is not None else None
                                    if attn is not None:
                                        attention_collector.collect(
                                            layer_idx,
                                            attn,
                                            attention_scores=scores,
                                            query_input=query,
                                            context_input=context,
                                            attn_module=module,
                                        )
                            return hook

                        handle = cross_attn.register_forward_hook(make_hook(idx))
                        hooks.append(handle)
                        logger.info(f"[SSAttentionCollector] Hooked layer {idx} for attention collection")

    # Import POSE_KEYS from multi_view_utils
    from sam3d_objects.pipeline.utils.multi_view_utils import POSE_KEYS

    def _new_dynamics_with_collection(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with attention collection for SS."""
        # Mark new step for attention collector (so it keeps only the last step's attention)
        attention_collector.new_step()

        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1

        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]

            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                view_conditions = [cond_tokens] * num_views

            # Collect predictions from all views
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)

                # Set current view for attention collection
                attention_collector.set_view(view_idx)

                pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                preds.append(pred)

            # Simple average for warmup pass (with POSE_KEYS handling)
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    stacked = torch.stack([p[key] for p in preds])
                    if key in POSE_KEYS:
                        fused_pred[key] = preds[0][key]
                    else:
                        fused_pred[key] = stacked.mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)

    generator._generate_dynamics = _new_dynamics_with_collection

    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
        # Remove hooks
        for handle in hooks:
            handle.remove()


@contextmanager
def inject_generator_multi_view_with_collector(
    generator,
    num_views: int,
    num_steps: int,
    attention_collector: AttentionCollector,
):
    """
    Inject multi-view support with attention collection.

    This is similar to inject_generator_multi_view but also collects attention
    weights into memory for weight computation.

    Args:
        generator: SAM 3D Objects generator (slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        attention_collector: AttentionCollector instance

    Yields:
        None
    """
    original_dynamics = generator._generate_dynamics

    # Also hook into cross-attention to collect attention
    hooks = []
    cfg_wrapper = getattr(generator, "reverse_fn", None)
    backbone = getattr(cfg_wrapper, "backbone", None)

    if backbone is not None:
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx != attention_collector.target_layer:
                    continue
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is None:
                    continue

                # Create hook to collect attention and idx mapping
                def make_hook(layer_idx):
                    def hook(module, inputs, outputs):
                        if len(inputs) < 2:
                            return
                        query, context = inputs[0], inputs[1]

                        # Handle multi-view context tensor
                        # context shape could be [num_views, B, L, C]
                        if torch.is_tensor(context) and context.dim() == 4:
                            view_idx = attention_collector._current_view
                            if 0 <= view_idx < context.shape[0]:
                                context = context[view_idx]
                            else:
                                context = context[0]

                        # Compute attention weights
                        with torch.no_grad():
                            scores = _compute_attention_scores(module, query, context)
                            attn = torch.softmax(scores, dim=-1).mean(dim=1) if scores is not None else None
                            score_summary = scores.mean(dim=1) if scores is not None and scores.dim() == 4 else scores
                            if attn is not None:
                                attention_collector.collect(
                                    layer_idx,
                                    attn,
                                    attention_scores=score_summary,
                                    query_sparse=query,
                                    query_input=query,
                                    context_input=context,
                                    attn_module=module,
                                )
                    return hook

                handle = cross_attn.register_forward_hook(make_hook(idx))
                hooks.append(handle)
                logger.info(f"[AttentionCollector] Hooked layer {idx} for attention collection")

    def _new_dynamics_with_collection(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with attention collection."""
        attention_collector.new_step()
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1

        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]

            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                view_conditions = [cond_tokens] * num_views

            # Collect predictions from all views
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)

                # Set current view for attention collection
                attention_collector.set_view(view_idx)

                pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                preds.append(pred)

            # Simple average for warmup pass
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    fused_pred[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)

    generator._generate_dynamics = _new_dynamics_with_collection

    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
        # Remove hooks
        for handle in hooks:
            handle.remove()


def _build_view_condition_args(args_conditionals, cond_idx: int, view_cond: Any):
    if cond_idx < len(args_conditionals):
        return args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx + 1 :]
    return args_conditionals + (view_cond,)


@contextmanager
def inject_weighted_multi_view_with_precomputed_weights(
    generator,
    num_views: int,
    num_steps: int,
    precomputed_weights: Optional[Dict[int, torch.Tensor]],
    step_recorder=None,
    target_latent_recorder=None,
    timestep_entropy_recorder=None,
    dynamic_jam_alpha: Optional[float] = None,
):
    """
    Inject weighted multi-view support with precomputed weights.

    This is used in the second pass of the two-pass approach,
    where weights have already been computed from the warmup pass.

    Args:
        generator: SAM 3D Objects generator (slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        precomputed_weights: Dict mapping view_idx -> [L_latent] weight tensor

    Yields:
        None
    """
    original_dynamics = generator._generate_dynamics

    # Check if we have valid weights
    use_weighted = precomputed_weights is not None and len(precomputed_weights) == num_views

    if use_weighted:
        logger.info(f"[WeightedMultidiffusion] Using precomputed weights for {num_views} views")
    else:
        logger.warning("[WeightedMultidiffusion] No valid precomputed weights, using simple average")
    state = {"step_idx": 0}

    def _new_dynamics_with_weights(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with precomputed weights."""
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1

        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]
            step_idx = int(state["step_idx"])
            state["step_idx"] += 1
            if target_latent_recorder is not None:
                target_latent_recorder(step_idx=step_idx, z_t=x_t, target_only_phase=False)

            # Log shape once
            if not hasattr(_new_dynamics_with_weights, '_logged_cond_shape'):
                logger.info(f"[WeightedMultidiffusion] args_conditionals length: {len(args_conditionals)}")
                logger.info(f"[WeightedMultidiffusion] cond_idx: {cond_idx}")
                if isinstance(cond_tokens, torch.Tensor):
                    logger.info(f"[WeightedMultidiffusion] Condition tokens shape: {cond_tokens.shape}")
                elif isinstance(cond_tokens, (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Condition tokens type: {type(cond_tokens)}, length: {len(cond_tokens)}")
                _new_dynamics_with_weights._logged_cond_shape = True

            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(
                    f"Condition tokens shape {cond_tokens.shape if isinstance(cond_tokens, torch.Tensor) else type(cond_tokens)} "
                    "not organized by views, using same condition for all views"
                )
                view_conditions = [cond_tokens] * num_views

            # Collect predictions from all views
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)

                if timestep_entropy_recorder is not None:
                    timestep_entropy_recorder.begin_view(step_idx=step_idx, view_idx=view_idx)
                try:
                    pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                finally:
                    if timestep_entropy_recorder is not None:
                        timestep_entropy_recorder.end_view()
                preds.append(pred)

            # Log shapes once
            if not hasattr(_new_dynamics_with_weights, '_logged_shape'):
                if isinstance(x_t, dict):
                    logger.info(f"[WeightedMultidiffusion] Latent shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in x_t.items()]}")
                elif isinstance(x_t, (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Latent shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in x_t]}")
                else:
                    logger.info(f"[WeightedMultidiffusion] Latent shape: {x_t.shape if isinstance(x_t, torch.Tensor) else type(x_t)}")

                if isinstance(preds[0], dict):
                    logger.info(f"[WeightedMultidiffusion] Pred shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in preds[0].items()]}")
                elif isinstance(preds[0], (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Pred shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in preds[0]]}")
                else:
                    logger.info(f"[WeightedMultidiffusion] Pred shape: {preds[0].shape if isinstance(preds[0], torch.Tensor) else type(preds[0])}")
                logger.info(f"[WeightedMultidiffusion] Number of views: {num_views}, using_weights: {use_weighted}")
                _new_dynamics_with_weights._logged_shape = True

            active_weights = precomputed_weights if use_weighted else None
            if timestep_entropy_recorder is not None and dynamic_jam_alpha is not None:
                step_weights = timestep_entropy_recorder.get_step_view_weights(
                    step_idx,
                    alpha=float(dynamic_jam_alpha),
                )
                if step_weights is not None:
                    active_weights = step_weights

            fused_pred = fuse_predictions(preds, weights=active_weights)
            if step_recorder is not None:
                step_recorder(
                    step_idx=step_idx,
                    view_predictions=preds,
                    fused_prediction=fused_pred,
                    target_only_phase=False,
                )
            return fused_pred
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)

    generator._generate_dynamics = _new_dynamics_with_weights

    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
