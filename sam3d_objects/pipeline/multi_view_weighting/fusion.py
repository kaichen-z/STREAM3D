"""
Multi-view weighted fusion utilities.

This module provides weighted multidiffusion fusion based on joint attention mass.
It extends the basic multidiffusion to support per-latent weighting.

Key Design (Two-Pass):
    1. Warmup Pass: Run step 0 with simple averaging to collect attention
    2. Compute weights from joint attention mass
    3. Main Pass: Run full generation from step 0 with weighted fusion

This ensures ALL steps benefit from weighted fusion.
"""
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from streaming.backend.attention_metric import (
    AttentionMetricFactory,
    AttentionWeightManager,
    AttentionWeightingConfig,
    ConditionMetricInput,
    ConditionMetricMode,
)


def _lookup_weight_for_view(weights: Any, view_idx: int):
    if weights is None:
        return None
    if isinstance(weights, dict):
        return weights.get(int(view_idx))
    if torch.is_tensor(weights):
        if weights.dim() == 0:
            return weights
        if weights.shape[0] != 0:
            return weights[view_idx]
    if isinstance(weights, (list, tuple)):
        return weights[view_idx]
    return None


def _apply_weight_to_prediction(
    pred: torch.Tensor,
    weight: torch.Tensor | float | None,
) -> torch.Tensor:
    if weight is None:
        return pred
    if not torch.is_tensor(weight):
        weight = torch.tensor(float(weight), device=pred.device, dtype=pred.dtype)
    else:
        weight = weight.to(device=pred.device, dtype=pred.dtype)

    if weight.dim() == 0:
        return pred * weight
    if pred.dim() == 3 and weight.dim() == 1:
        return pred * weight.unsqueeze(0).unsqueeze(-1)
    if pred.dim() == 2 and weight.dim() == 1:
        return pred * weight.unsqueeze(-1)
    return pred * weight


def fuse_predictions(
    predictions: List[Any],
    *,
    weights: Any = None,
    pose_keys: Optional[set[str]] = None,
) -> Any:
    if not predictions:
        raise ValueError("Empty predictions list")

    if isinstance(predictions[0], dict):
        fused = {}
        for key in predictions[0].keys():
            if pose_keys is not None and key in pose_keys:
                fused[key] = predictions[0][key]
                continue
            fused[key] = fuse_predictions(
                [pred[key] for pred in predictions],
                weights=weights,
                pose_keys=None,
            )
        return fused

    if isinstance(predictions[0], (list, tuple)):
        values = [
            fuse_predictions([pred[i] for pred in predictions], weights=weights, pose_keys=None)
            for i in range(len(predictions[0]))
        ]
        return type(predictions[0])(values)

    if weights is None:
        return torch.stack(predictions).mean(dim=0)

    fused = torch.zeros_like(predictions[0])
    weight_sum = None
    for view_idx, pred in enumerate(predictions):
        weight = _lookup_weight_for_view(weights, view_idx)
        fused = fused + _apply_weight_to_prediction(pred, weight)
        if weight is not None:
            weight_tensor = (
                weight
                if torch.is_tensor(weight)
                else torch.tensor(float(weight), dtype=pred.dtype, device=pred.device)
            )
            weight_sum = weight_tensor if weight_sum is None else weight_sum + weight_tensor

    if weight_sum is None:
        return torch.stack(predictions).mean(dim=0)
    return fused


def compute_ss_joint_attention_mass_vector_from_scores(
    scores: torch.Tensor,
    *,
    patch_start: int,
    patch_end: int,
) -> torch.Tensor:
    if scores.dim() not in {3, 4}:
        raise ValueError(
            f"Expected SS attention scores [B,4096,T] or [B,H,4096,T], got {tuple(scores.shape)}."
        )
    if int(scores.shape[-2]) != 4096:
        raise ValueError(f"Expected 4096 SS query tokens, got {int(scores.shape[-2])}.")

    return AttentionMetricFactory.build(ConditionMetricMode.JOINT_ATTENTION_MASS)(
        ConditionMetricInput(
            attention=scores,
            patch_start=int(patch_start),
            patch_end=min(int(patch_end), int(scores.shape[-1])),
        )
    )


def compute_ss_joint_attention_mass_weights(
    attention_scores: Dict[int, torch.Tensor],
    *,
    weight_source: str,
    jam_alpha: float,
    jam_kappa: float,
    uniform_blend: float,
    min_weight: float,
    patch_start: int,
    patch_end: int,
) -> torch.Tensor:
    """Compute Stage 1 fusion weights from normalized joint attention mass."""
    views = sorted(attention_scores.keys())
    num_views = len(views)

    if num_views == 0:
        return None
    if num_views == 1:
        return torch.ones(1, attention_scores[views[0]].shape[-2])

    metric = ConditionMetricMode(weight_source)
    metric_impl = AttentionMetricFactory.build(metric)
    if metric is ConditionMetricMode.MASS_RELATIVE:
        evidence_by_view = metric_impl.score_by_view(
            attention_scores,
            patch_start=patch_start,
            patch_end=patch_end,
            kappa=float(jam_kappa),
        )
        log_prefix = "SS Variant=mass_relative"
    elif metric is ConditionMetricMode.JOINT_ATTENTION_MASS:
        evidence_by_view = {
            view: compute_ss_joint_attention_mass_vector_from_scores(
                attention_scores[view],
                patch_start=patch_start,
                patch_end=patch_end,
            )
            for view in views
        }
        log_prefix = "SS JAM"
    else:
        raise ValueError(f"Unsupported SS weight_source: {metric.value}")

    evidences = []
    for v in views:
        evidence = evidence_by_view[v]
        evidences.append(evidence)
        logger.info(
            f"[{log_prefix}] View {v}: mean={evidence.mean():.4f}, std={evidence.std():.4f}, "
            f"min={evidence.min():.4f}, max={evidence.max():.4f}"
        )

    evidence_stack = torch.stack(evidences, dim=0)
    evidence_mean_per_view = evidence_stack.mean(dim=1)
    evidence_std_across_views = evidence_stack.std(dim=0).mean()
    logger.info(f"[{log_prefix}] Cross-view statistics:")
    logger.info(f"  Per-view mean evidence: {evidence_mean_per_view.tolist()}")
    logger.info(f"  Cross-view std (avg over latents): {evidence_std_across_views:.4f}")

    weights = metric_impl.normalize_score_stack(
        evidence_stack,
        exponent=float(jam_alpha),
        uniform_blend=float(uniform_blend),
        min_weight=float(min_weight),
    )
    logger.info(
        f"[{log_prefix} Weights] Evidence range: min={evidence_stack.min():.4f}, "
        f"max={evidence_stack.max():.4f}, spread={evidence_stack.max()-evidence_stack.min():.4f}, "
        f"exponent={float(jam_alpha):.3f}, uniform_blend={float(uniform_blend):.3f}, "
        f"kappa={float(jam_kappa):.3f}"
    )

    logger.info(
        f"[{log_prefix} Weights] Computed weights: shape={weights.shape}, "
        f"mean per view: {[f'{weights[i].mean():.4f}' for i in range(num_views)]}"
    )

    best_views = weights.argmax(dim=0)
    view_counts = [(best_views == v).sum().item() for v in range(num_views)]
    logger.info(f"[{log_prefix} Weights] Best view distribution (per latent): {view_counts}")

    max_weights = weights.max(dim=0)[0]
    logger.info(
        f"[{log_prefix} Weights] Max weight per latent: mean={max_weights.mean():.4f}, "
        f"min={max_weights.min():.4f}, max={max_weights.max():.4f}"
    )

    return weights


def weighted_fusion_sparse(
    predictions: List[torch.Tensor],
    weights: Dict[int, torch.Tensor],
    num_views: int,
) -> torch.Tensor:
    """
    Perform weighted fusion of sparse predictions.

    Args:
        predictions: List of [B, L_latent, C] or [L_latent, C] tensors
        weights: Dict mapping view_idx -> [L_latent] weight tensor
                 The weights should be expanded to match prediction's L_latent dimension
        num_views: Number of views

    Returns:
        fused: Weighted sum of predictions
    """
    if not predictions:
        raise ValueError("Empty predictions list")

    device = predictions[0].device
    pred_shape = predictions[0].shape

    # Determine the latent dimension
    # predictions can be [B, L_latent, C] or [L_latent, C]
    if len(pred_shape) == 3:
        L_pred = pred_shape[1]
    elif len(pred_shape) == 2:
        L_pred = pred_shape[0]
    else:
        L_pred = pred_shape[-2] if len(pred_shape) > 1 else pred_shape[0]

    # Check if weights match prediction dimension
    sample_weight = list(weights.values())[0] if weights else None
    if sample_weight is not None:
        L_weight = sample_weight.shape[0]
        if L_weight != L_pred:
            logger.warning(
                f"[weighted_fusion_sparse] Dimension mismatch: "
                f"prediction L={L_pred}, weight L={L_weight}. "
                f"This should not happen if weights were properly expanded!"
            )
            # Fallback to simple average
            return torch.stack(predictions).mean(dim=0)

    fused = torch.zeros_like(predictions[0])

    for view_idx, pred in enumerate(predictions):
        if view_idx in weights:
            w = weights[view_idx].to(device)

            # Expand weight to match prediction shape
            # pred: [B, L_latent, C] or [L_latent, C]
            # w: [L_latent]
            if pred.dim() == 3:
                # [B, L_latent, C] -> w needs to be [1, L_latent, 1]
                w = w.unsqueeze(0).unsqueeze(-1)
            elif pred.dim() == 2:
                # [L_latent, C] -> w needs to be [L_latent, 1]
                w = w.unsqueeze(-1)

            fused = fused + pred * w
        else:
            # Fallback to equal weight
            fused = fused + pred / num_views

    return fused


class WeightedMultiViewFusion:
    """
    Helper class to manage weighted multi-view fusion during inference.

    This class coordinates:
    1. Attention collection during step 0
    2. Weight computation from joint attention mass
    3. Weighted fusion application
    """

    def __init__(
        self,
        config: AttentionWeightingConfig,
        visualize: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.weight_manager = AttentionWeightManager(self.config)
        self.visualize = visualize
        self.output_dir = output_dir

        # State
        self._attention_collected = False
        self._current_step = -1

    def reset(self):
        """Reset for new inference."""
        self.weight_manager.reset()
        self._attention_collected = False
        self._current_step = -1

    def on_attention(
        self,
        view_idx: int,
        attention: torch.Tensor,
        step: int,
        layer: int,
    ):
        """
        Callback when attention is computed.

        Args:
            view_idx: View index
            attention: [B, L_latent, L_cond] attention weights
            step: Current diffusion step
            layer: Layer index
        """
        # Only collect attention at the configured step and layer
        if step != self.config.attention_step:
            return
        if layer != self.config.attention_layer:
            return

        self.weight_manager.add_view_attention(view_idx, attention, step)
        logger.debug(f"[WeightedMultiViewFusion] Collected attention for view {view_idx}, step {step}")

    def compute_weights(self) -> Dict[int, torch.Tensor]:
        """Compute fusion weights from collected attention."""
        return self.weight_manager.compute_weights()

    def get_analysis_data(self) -> Dict:
        """Get analysis data for visualization."""
        return self.weight_manager.get_analysis_data()

    def save_visualization(self, coords: Optional[torch.Tensor] = None):
        """
        Save weight visualizations.

        Args:
            coords: [L_latent, 4] spatial coordinates (batch, x, y, z)
        """
        if not self.visualize or not self.output_dir:
            return

        from pathlib import Path
        import numpy as np

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis = self.get_analysis_data()
        weights = analysis.get("weights", {})
        jam_per_view = analysis.get("joint_attention_mass_per_view", {})

        if not weights:
            logger.warning("[WeightedMultiViewFusion] No weights to visualize")
            return

        # Save weights as .pt file
        torch.save({
            "weights": {k: v.cpu() for k, v in weights.items()},
            "joint_attention_mass": {k: v.cpu() for k, v in jam_per_view.items()},
            "config": {
                "jam_alpha": self.config.jam_alpha,
                "attention_layer": self.config.attention_layer,
                "attention_step": self.config.attention_step,
            },
            "coords": coords.cpu() if coords is not None else None,
        }, output_dir / "fusion_weights.pt")

        logger.info(f"[WeightedMultiViewFusion] Saved weights to {output_dir / 'fusion_weights.pt'}")

        # Generate visualizations if matplotlib available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            self._plot_weight_distribution(weights, output_dir)
            self._plot_metric_distribution(jam_per_view, output_dir)

            if coords is not None:
                self._plot_3d_weights(weights, coords, output_dir)
                self._plot_3d_metric(jam_per_view, coords, output_dir)

        except ImportError:
            logger.warning("[WeightedMultiViewFusion] matplotlib not available, skipping plots")

    def _plot_weight_distribution(self, weights: Dict[int, torch.Tensor], output_dir):
        """Plot weight distribution histogram."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 4))
        if len(weights) == 1:
            axes = [axes]

        for ax, (view_idx, w) in zip(axes, sorted(weights.items())):
            w_np = w.cpu().numpy()
            ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Weight')
            ax.set_ylabel('Count')
            ax.set_title(f'View {view_idx}\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')

        plt.tight_layout()
        plt.savefig(output_dir / 'weight_distribution.png', dpi=150)
        plt.close()
        logger.info(f"[WeightedMultiViewFusion] Saved weight distribution plot")

    def _plot_metric_distribution(self, metric_per_view: Dict[int, torch.Tensor], output_dir):
        """Plot JAM distribution histogram."""
        import matplotlib.pyplot as plt

        if not metric_per_view:
            return

        fig, axes = plt.subplots(1, len(metric_per_view), figsize=(4 * len(metric_per_view), 4))
        if len(metric_per_view) == 1:
            axes = [axes]

        for ax, (view_idx, metric) in zip(axes, sorted(metric_per_view.items())):
            metric_np = metric.cpu().numpy()
            ax.hist(metric_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.set_xlabel('Joint Attention Mass')
            ax.set_ylabel('Count')
            ax.set_title(f'View {view_idx}\nmean={metric_np.mean():.4f}, std={metric_np.std():.4f}')

        plt.tight_layout()
        plt.savefig(output_dir / 'joint_attention_mass_distribution.png', dpi=150)
        plt.close()
        logger.info(f"[WeightedMultiViewFusion] Saved JAM distribution plot")

    def _plot_3d_weights(self, weights: Dict[int, torch.Tensor], coords: torch.Tensor, output_dir):
        """Plot 3D weight visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        coords_np = coords.cpu().numpy()
        # coords: [N, 4] where columns are (batch, x, y, z)
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]

        # Normalize coordinates
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        z = (z - z.min()) / (z.max() - z.min() + 1e-6)

        for view_idx, w in sorted(weights.items()):
            w_np = w.cpu().numpy()

            # Robust normalization
            vmin, vmax = np.percentile(w_np, [2, 98])
            w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)

            # Sort by depth for better visualization
            order = np.argsort(z)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                x[order], y[order], z[order],
                c=w_norm[order],
                cmap='viridis',
                s=1,
                alpha=0.6,
            )

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {view_idx} Weight')

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Weight')

            plt.savefig(output_dir / f'weight_3d_view{view_idx:02d}.png', dpi=150)
            plt.close()

        logger.info(f"[WeightedMultiViewFusion] Saved 3D weight plots")

    def _plot_3d_metric(self, metric_per_view: Dict[int, torch.Tensor], coords: torch.Tensor, output_dir):
        """Plot 3D JAM visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        if not metric_per_view:
            return

        coords_np = coords.cpu().numpy()
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]

        # Normalize coordinates
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        z = (z - z.min()) / (z.max() - z.min() + 1e-6)

        for view_idx, metric in sorted(metric_per_view.items()):
            metric_np = metric.cpu().numpy()

            # Robust normalization
            vmin, vmax = np.percentile(metric_np, [2, 98])
            metric_norm = np.clip((metric_np - vmin) / (vmax - vmin + 1e-6), 0, 1)

            order = np.argsort(z)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                x[order], y[order], z[order],
                c=metric_norm[order],
                cmap='hot',
                s=1,
                alpha=0.6,
            )

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {view_idx} JAM')

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Joint Attention Mass')

            plt.savefig(output_dir / f'joint_attention_mass_3d_view{view_idx:02d}.png', dpi=150)
            plt.close()

        logger.info(f"[WeightedMultiViewFusion] Saved 3D JAM plots")
