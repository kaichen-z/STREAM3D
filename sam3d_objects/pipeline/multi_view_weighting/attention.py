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
from typing import Any, Dict, List, Literal, Optional
import math
import torch
from loguru import logger
from torch.utils._pytree import tree_map_only

from sam3d_objects.data.utils import tree_tensor_map
from streaming.backend.attention_metric import SLAT_CONDITION_LAYOUT


def _compute_ss_patch_entropy_vector(
    attention: torch.Tensor,
    *,
    patch_start: int,
    patch_end: int,
) -> torch.Tensor:
    actual_end = min(patch_end, attention.shape[-1])
    patch_attn = attention[..., patch_start:actual_end]
    patch_sum = patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    patch_attn_norm = patch_attn / patch_sum
    log_attn = torch.log(patch_attn_norm + 1e-10)
    entropy = -(patch_attn_norm * log_attn).sum(dim=-1)
    if entropy.dim() == 3:
        entropy = entropy.mean(dim=(0, 1))
    elif entropy.dim() == 2:
        entropy = entropy.mean(dim=0)
    else:
        entropy = entropy.reshape(-1)
    num_patches = patch_attn.shape[-1]
    max_entropy = math.log(num_patches)
    return entropy / max_entropy


def _compute_slat_patch_entropy_vector(
    attention: torch.Tensor,
    *,
    patch_start: int,
    patch_end: int,
) -> torch.Tensor:
    region_name = "image_cropped"
    region_start, _ = SLAT_CONDITION_LAYOUT[region_name]
    patch_start = region_start + int(patch_start)
    patch_end = min(region_start + int(patch_end), attention.shape[-1])
    patch_attn = attention[..., patch_start:patch_end]
    patch_sum = patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    patch_attn_norm = patch_attn / patch_sum
    log_attn = torch.log(patch_attn_norm + 1e-10)
    entropy = -(patch_attn_norm * log_attn).sum(dim=-1)
    if entropy.dim() == 3:
        entropy = entropy.mean(dim=(0, 1))
    elif entropy.dim() == 2:
        entropy = entropy.mean(dim=0)
    else:
        entropy = entropy.reshape(-1)
    num_patches = patch_attn.shape[-1]
    max_entropy = math.log(num_patches)
    return entropy / max_entropy


def _reduce_attention_entropy_scalar(
    attention: torch.Tensor,
    *,
    stage: str,
    patch_start: int,
    patch_end: int,
) -> float:
    if str(stage) == "ss":
        entropy = _compute_ss_patch_entropy_vector(
            attention,
            patch_start=patch_start,
            patch_end=patch_end,
        )
    else:
        entropy = _compute_slat_patch_entropy_vector(
            attention,
            patch_start=patch_start,
            patch_end=patch_end,
        )
    return float(entropy.mean().item())


class AttentionCollector:
    """
    Collects attention weights during warmup pass for weight computation.

    This is used to collect attention in memory (not files) during the warmup pass,
    so we can compute weights before the main pass.

    Also collects the idx mapping from SparseDownsample to expand weights
    from downsampled dimension (e.g., 4369) to original dimension (e.g., 21411).

    IMPORTANT: Due to CFG (Classifier-Free Guidance), the cross-attention is called
    twice per view: once for cond branch and once for uncond branch. We only want
    the cond branch attention (which has meaningful attention patterns), not the
    uncond branch (which has uniform attention due to zeroed conditions).
    """

    def __init__(
        self,
        num_views: int,
        target_layer: int = 6,
        target_step: int = 0,
        *,
        patch_start: int,
        patch_end: int,
        prev_query_latent: Optional[torch.Tensor] = None,
        prev_query_latents: Optional[Dict[int, torch.Tensor]] = None,
        prev_query_pool: Optional[List[torch.Tensor]] = None,
        prev_query_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.num_views = num_views
        self.target_layer = target_layer
        self.target_step = target_step
        self.patch_start = int(patch_start)
        self.patch_end = int(patch_end)
        self._attentions: Dict[int, torch.Tensor] = {}
        self._attention_scores: Dict[int, torch.Tensor] = {}
        self._global_attentions: Dict[int, torch.Tensor] = {}
        self._global_entropies: Dict[int, torch.Tensor] = {}
        self._query_latents: Dict[int, torch.Tensor] = {}
        self._global_attention_statuses: Dict[int, str] = {}
        self._current_view: int = 0
        self._current_step: int = -1
        # Track which views have already been collected in the current step (to skip uncond branch)
        self._step_collected_views: set = set()
        # idx mapping: maps original points to downsampled points
        # idx[i] = j means original point i maps to downsampled point j
        self._downsample_idx: Optional[torch.Tensor] = None
        # Original coords before downsampling
        self._original_coords: Optional[torch.Tensor] = None
        # Downsampled coords (where attention is computed)
        self._downsampled_coords: Optional[torch.Tensor] = None
        self._prev_query_latents: Dict[int, torch.Tensor] = {}
        self._prev_query_pool: List[torch.Tensor] = []
        self._prev_query_entries: List[Dict[str, Any]] = []
        if prev_query_latent is not None:
            dense_query = self._to_dense_query_latent(prev_query_latent)
            if dense_query is not None:
                for view_idx in range(num_views):
                    self._prev_query_latents[int(view_idx)] = dense_query
        if prev_query_latents is not None:
            for view_idx, query in prev_query_latents.items():
                dense_query = self._to_dense_query_latent(query)
                if dense_query is not None:
                    self._prev_query_latents[int(view_idx)] = dense_query
        if prev_query_pool is not None:
            for query in prev_query_pool:
                dense_query = self._to_dense_query_latent(query)
                if dense_query is not None:
                    self._prev_query_pool.append(dense_query)
        if prev_query_entries is not None:
            for entry in prev_query_entries:
                dense_query = self._to_dense_query_latent(entry.get("query"))
                if dense_query is None:
                    continue
                coords = entry.get("hook_coords")
                if torch.is_tensor(coords):
                    coords = coords.detach().cpu().to(torch.int64).contiguous()
                else:
                    coords = None
                self._prev_query_entries.append(
                    {
                        "query": dense_query,
                        "hook_coords": coords,
                        "runtime_frame_key": entry.get("runtime_frame_key"),
                    }
                )

    def set_view(self, view_idx: int):
        """Set current view being processed."""
        self._current_view = view_idx

    def new_step(self):
        """Called at the start of each new step to reset per-step tracking."""
        self._current_step += 1
        self._step_collected_views.clear()

    @staticmethod
    def _to_dense_query_latent(query_input, *, to_cpu: bool = True):
        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import (
            SparseTensor,
        )

        if query_input is None:
            return None
        if isinstance(query_input, SparseTensor):
            if not query_input.layout:
                return None
            slices = [query_input.feats[slc].unsqueeze(0) for slc in query_input.layout]
            tensor = torch.cat(slices, dim=0).detach()
            if to_cpu:
                tensor = tensor.cpu().to(torch.float32)
            return tensor.contiguous()
        if torch.is_tensor(query_input):
            tensor = query_input.detach()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if to_cpu:
                tensor = tensor.cpu().to(torch.float32)
            return tensor.contiguous()
        return None

    @staticmethod
    def _build_overlap_indices(
        prev_coords: Optional[torch.Tensor],
        current_coords: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prev_coords is None or current_coords is None:
            return None, None
        prev_lookup = {
            tuple(int(v) for v in row.tolist()): idx
            for idx, row in enumerate(prev_coords)
        }
        prev_indices: List[int] = []
        current_indices: List[int] = []
        for current_idx, row in enumerate(current_coords):
            match_idx = prev_lookup.get(tuple(int(v) for v in row.tolist()))
            if match_idx is None:
                continue
            prev_indices.append(int(match_idx))
            current_indices.append(int(current_idx))
        if not prev_indices:
            return None, None
        return (
            torch.tensor(prev_indices, dtype=torch.long),
            torch.tensor(current_indices, dtype=torch.long),
        )

    def _compute_global_attention(
        self,
        *,
        view_idx: int,
        attn_module,
        current_query_latent,
        context_input,
        local_attention,
        current_hook_coords: Optional[torch.Tensor],
    ):
        if attn_module is None or context_input is None or current_query_latent is None:
            return None, None, "attention_failure"

        if self._prev_query_entries:
            if current_hook_coords is None:
                return None, None, "query_unavailable"
            accum_entropy = torch.zeros(
                current_hook_coords.shape[0],
                dtype=torch.float32,
            )
            accum_count = torch.zeros(
                current_hook_coords.shape[0],
                dtype=torch.float32,
            )
            overlap_candidate_count = 0
            attention_failure_count = 0
            for entry in self._prev_query_entries:
                prev_query_latent = entry["query"]
                prev_coords = entry.get("hook_coords")
                prev_indices, current_indices = self._build_overlap_indices(
                    prev_coords, current_hook_coords
                )
                if prev_indices is None or current_indices is None:
                    continue
                overlap_candidate_count += 1
                prev_query = prev_query_latent[:, prev_indices, :].to(
                    device=current_query_latent.device,
                    dtype=current_query_latent.dtype,
                )
                global_attention = _compute_attention_weights(
                    attn_module, prev_query, context_input
                )
                if global_attention is None:
                    attention_failure_count += 1
                    continue
                global_entropy = _compute_slat_patch_entropy_vector(
                    global_attention,
                    patch_start=self.patch_start,
                    patch_end=self.patch_end,
                )
                current_indices_cpu = current_indices.to(torch.long)
                accum_entropy[current_indices_cpu] += (
                    global_entropy.detach().cpu().to(torch.float32)
                )
                accum_count[current_indices_cpu] += 1.0

            valid_mask = accum_count > 0
            if valid_mask.any():
                aligned_entropy = torch.zeros_like(accum_entropy)
                aligned_entropy[valid_mask] = (
                    accum_entropy[valid_mask] / accum_count[valid_mask]
                )
                return None, aligned_entropy, "applied"
            if attention_failure_count > 0:
                logger.warning(
                    f"[AttentionCollector] View {view_idx}: failed to compute previous-query attention for "
                    f"{attention_failure_count}/{len(self._prev_query_entries)} overlapping previous queries. "
                    "Using local-only entropy for this view."
                )
                return None, None, "attention_failure"
            if overlap_candidate_count == 0:
                logger.info(
                    f"[AttentionCollector] View {view_idx}: no overlapping SLAT hook coordinates found "
                    "with previous queries. Using local-only entropy for this view."
                )
                return None, None, "no_overlap"
            return None, None, "attention_failure"

        prev_query_candidates: List[torch.Tensor] = []
        if self._prev_query_pool:
            prev_query_candidates.extend(self._prev_query_pool)
        else:
            prev_query_latent = self._prev_query_latents.get(int(view_idx))
            if prev_query_latent is not None:
                prev_query_candidates.append(prev_query_latent)
        if not prev_query_candidates:
            return None, None, "no_match"
        global_attentions: List[torch.Tensor] = []
        shape_mismatch_count = 0
        attention_failure_count = 0
        for prev_query_latent in prev_query_candidates:
            if tuple(prev_query_latent.shape) != tuple(current_query_latent.shape):
                shape_mismatch_count += 1
                continue
            prev_query = prev_query_latent.to(
                device=current_query_latent.device, dtype=current_query_latent.dtype
            )
            global_attention = _compute_attention_weights(
                attn_module, prev_query, context_input
            )
            if global_attention is None:
                attention_failure_count += 1
                continue
            if tuple(global_attention.shape[:2]) != tuple(local_attention.shape[:2]):
                shape_mismatch_count += 1
                continue
            global_attentions.append(global_attention.detach().cpu().clone())

        if global_attentions:
            return torch.cat(global_attentions, dim=0), None, "applied"
        if shape_mismatch_count > 0:
            logger.warning(
                f"[AttentionCollector] View {view_idx}: no previous-query attention survived shape checks. "
                f"{shape_mismatch_count}/{len(prev_query_candidates)} previous queries mismatched current query "
                f"shape {tuple(current_query_latent.shape)} or local attention shape {tuple(local_attention.shape)}. "
                "Using local-only entropy for this view."
            )
            return None, None, "shape_mismatch"
        if attention_failure_count > 0:
            logger.warning(
                f"[AttentionCollector] View {view_idx}: failed to compute previous-query attention for "
                f"{attention_failure_count}/{len(prev_query_candidates)} previous queries. "
                "Using local-only entropy for this view."
            )
            return None, None, "attention_failure"
        return None, None, "no_match"

    def collect(
        self,
        layer_idx: int,
        attention: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        query_sparse=None,
        query_input=None,
        context_input=None,
        attn_module=None,
    ):
        """
        Collect attention for the current view.

        Only collects the FIRST attention for each view (cond branch).
        Skips subsequent collections for the same view (uncond branch).

        Args:
            layer_idx: Layer index
            attention: [B, L_latent, L_cond] attention weights
            query_sparse: SparseTensor containing spatial cache with idx mapping
        """
        if layer_idx != self.target_layer or self._current_step != self.target_step:
            return

        # Skip if already collected for this view in the current step (this is the uncond branch)
        if self._current_view in self._step_collected_views:
            logger.debug(
                f"[AttentionCollector] Skipping uncond branch for view {self._current_view}"
            )
            return

        # Store attention (cond branch)
        self._attentions[self._current_view] = attention.detach().cpu().clone()
        if attention_scores is not None:
            self._attention_scores[self._current_view] = (
                attention_scores.detach().cpu().clone()
            )
        self._step_collected_views.add(self._current_view)

        # Try to extract idx mapping from SparseTensor (only need to do once)
        if self._downsample_idx is None and query_sparse is not None:
            self._extract_downsample_info(query_sparse)

        query_latent = self._to_dense_query_latent(query_input)
        runtime_query_latent = self._to_dense_query_latent(query_input, to_cpu=False)
        if query_latent is not None:
            self._query_latents[self._current_view] = query_latent
            global_attention, global_entropy, status = self._compute_global_attention(
                view_idx=self._current_view,
                attn_module=attn_module,
                current_query_latent=runtime_query_latent,
                context_input=context_input,
                local_attention=attention,
                current_hook_coords=self._downsampled_coords,
            )
            self._global_attention_statuses[self._current_view] = status
            if global_attention is not None:
                self._global_attentions[self._current_view] = global_attention
            if global_entropy is not None:
                self._global_entropies[self._current_view] = global_entropy
        else:
            self._global_attention_statuses[self._current_view] = "query_unavailable"

        logger.info(
            f"[AttentionCollector] Step {self._current_step}: collected COND attention for view {self._current_view}, "
            f"shape={attention.shape}, min={attention.min():.4f}, max={attention.max():.4f}"
        )

    def _extract_downsample_info(self, query_sparse):
        """Extract downsample idx and coords from SparseTensor's spatial cache."""
        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import (
            SparseTensor,
        )

        if not isinstance(query_sparse, SparseTensor):
            return

        # The downsampled coords are the current coords of query_sparse
        self._downsampled_coords = query_sparse.coords.detach().cpu().clone()

        # Try to get idx from spatial cache
        # The key format is "upsample_{factor}_idx" where factor is (2, 2, 2) for 3D
        spatial_cache = query_sparse._spatial_cache

        if not spatial_cache:
            logger.warning(
                "[AttentionCollector] No spatial cache found in query_sparse"
            )
            return

        # Look for the idx in any scale's cache
        for scale_key, cache in spatial_cache.items():
            # Try common factor formats
            for factor_key in ["upsample_(2, 2, 2)_idx", "upsample_2_idx"]:
                if factor_key in cache:
                    self._downsample_idx = cache[factor_key].detach().cpu().clone()
                    logger.info(
                        f"[AttentionCollector] Found downsample idx: shape={self._downsample_idx.shape}"
                    )
                    break

            # Also try to get original coords
            for factor_key in ["upsample_(2, 2, 2)_coords", "upsample_2_coords"]:
                if factor_key in cache:
                    self._original_coords = cache[factor_key].detach().cpu().clone()
                    logger.info(
                        f"[AttentionCollector] Found original coords: shape={self._original_coords.shape}"
                    )
                    break

            if self._downsample_idx is not None:
                break

        if self._downsample_idx is not None and self._original_coords is not None:
            logger.info(
                f"[AttentionCollector] Downsample mapping: "
                f"original {self._original_coords.shape[0]} -> downsampled {self._downsampled_coords.shape[0]}"
            )

    def get_attentions(self) -> Dict[int, torch.Tensor]:
        """Get all collected attentions."""
        return self._attentions

    def get_attention_scores(self) -> Dict[int, torch.Tensor]:
        """Get all collected pre-softmax attention scores."""
        return self._attention_scores

    def get_global_attentions(self) -> Dict[int, torch.Tensor]:
        """Get previous-chunk global attentions computed with cached query latents."""
        return self._global_attentions

    def get_global_entropies(self) -> Dict[int, torch.Tensor]:
        """Get previous-chunk global entropy vectors aligned to current latent layout."""
        return self._global_entropies

    def get_query_latents(self) -> Dict[int, torch.Tensor]:
        """Get current query latents keyed by runtime view index."""
        return self._query_latents

    def get_global_attention_statuses(self) -> Dict[int, str]:
        """Get per-view status for previous-query cross attention."""
        return self._global_attention_statuses

    def get_fused_query_latent(self) -> Optional[torch.Tensor]:
        """Get the current chunk query latent averaged over views."""
        if not self._query_latents:
            return None
        latents = [self._query_latents[v] for v in sorted(self._query_latents.keys())]
        first_shape = tuple(latents[0].shape)
        for latent in latents[1:]:
            if tuple(latent.shape) != first_shape:
                logger.warning(
                    f"[AttentionCollector] Cannot fuse query latents with mismatched shapes: "
                    f"{first_shape} vs {tuple(latent.shape)}"
                )
                return latents[0]
        return torch.stack(latents, dim=0).mean(dim=0)

    def get_downsample_idx(self) -> Optional[torch.Tensor]:
        """Get the downsample idx mapping."""
        return self._downsample_idx

    def get_original_coords(self) -> Optional[torch.Tensor]:
        """Get the original coords before downsampling."""
        return self._original_coords

    def get_downsampled_coords(self) -> Optional[torch.Tensor]:
        """Get the downsampled coords where attention is computed."""
        return self._downsampled_coords

    def reset(self):
        """Reset collected data."""
        self._attentions.clear()
        self._attention_scores.clear()
        self._global_attentions.clear()
        self._global_entropies.clear()
        self._query_latents.clear()
        self._global_attention_statuses.clear()
        self._step_collected_views.clear()
        self._downsample_idx = None
        self._original_coords = None
        self._downsampled_coords = None
        self._current_step = -1


# ============================================================================
# SS (Stage 1) Attention Collector - for Dense Latent (4096 voxels)
# ============================================================================


class SSAttentionCollector:
    """
    Collects attention weights during SS (Stage 1) warmup pass for weight computation.

    Unlike SLAT, SS uses dense latent (4096 voxels), so no downsample mapping is needed.
    This collector specifically targets the 'shape' latent in MM-DiT architecture.

    Strategy: Keep the LAST step's attention (closer to t=1, more stable patterns).
    For each step, we collect cond branch attention and overwrite previous step's data.

    Attention format: [bs, 4096, num_cond_tokens]
    """

    def __init__(
        self,
        num_views: int,
        target_layer: int = 9,
        prev_query_latents: Optional[Dict[int, torch.Tensor]] = None,
        prev_query_pool: Optional[List[torch.Tensor]] = None,
    ):
        self.num_views = num_views
        self.target_layer = target_layer
        self._attentions: Dict[int, torch.Tensor] = {}
        self._attention_scores: Dict[int, torch.Tensor] = {}
        self._global_attentions: Dict[int, torch.Tensor] = {}
        self._query_latents: Dict[int, torch.Tensor] = {}
        self._global_attention_statuses: Dict[int, str] = {}
        self._current_view: int = 0
        self._current_step: int = 0
        # Track which views have been collected in THIS step (to skip uncond branch)
        self._step_collected_views: set = set()
        self._prev_query_latents: Dict[int, torch.Tensor] = {}
        self._prev_query_pool: List[torch.Tensor] = []
        if prev_query_latents is not None:
            for view_idx, query in prev_query_latents.items():
                dense_query = AttentionCollector._to_dense_query_latent(query)
                if dense_query is not None:
                    self._prev_query_latents[int(view_idx)] = dense_query
        if prev_query_pool is not None:
            for query in prev_query_pool:
                dense_query = AttentionCollector._to_dense_query_latent(query)
                if dense_query is not None:
                    self._prev_query_pool.append(dense_query)

    def set_view(self, view_idx: int):
        """Set current view being processed."""
        self._current_view = view_idx

    def new_step(self):
        """Called at the start of each new step to reset per-step tracking."""
        self._current_step += 1
        self._step_collected_views.clear()

    def collect(
        self,
        layer_idx: int,
        attention: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        query_input=None,
        context_input=None,
        attn_module=None,
    ):
        """
        Collect attention for the current view.

        Only collects cond branch (first call for each view in each step).
        Overwrites previous step's attention to keep only the latest.

        Args:
            layer_idx: Layer index
            attention: [B, 4096, L_cond] attention weights
        """
        if layer_idx != self.target_layer:
            return

        # Skip if already collected for this view in THIS step (this is the uncond branch)
        if self._current_view in self._step_collected_views:
            return

        # Store attention (cond branch), overwriting previous step's data
        self._attentions[self._current_view] = attention.detach().cpu().clone()
        if attention_scores is not None:
            self._attention_scores[self._current_view] = (
                attention_scores.detach().cpu().clone()
            )
        self._step_collected_views.add(self._current_view)
        query_latent = AttentionCollector._to_dense_query_latent(query_input)
        runtime_query_latent = AttentionCollector._to_dense_query_latent(
            query_input, to_cpu=False
        )
        if query_latent is not None:
            self._query_latents[self._current_view] = query_latent
            prev_query_candidates: List[torch.Tensor] = []
            if self._prev_query_pool:
                prev_query_candidates.extend(self._prev_query_pool)
            else:
                prev_query_latent = self._prev_query_latents.get(self._current_view)
                if prev_query_latent is not None:
                    prev_query_candidates.append(prev_query_latent)
            if not prev_query_candidates:
                self._global_attention_statuses[self._current_view] = "no_match"
            elif (
                runtime_query_latent is None
                or attn_module is None
                or context_input is None
            ):
                self._global_attention_statuses[self._current_view] = (
                    "attention_failure"
                )
            else:
                global_attentions: List[torch.Tensor] = []
                shape_mismatch_count = 0
                attention_failure_count = 0
                for prev_query_latent in prev_query_candidates:
                    if tuple(prev_query_latent.shape) != tuple(
                        runtime_query_latent.shape
                    ):
                        shape_mismatch_count += 1
                        continue
                    prev_query = prev_query_latent.to(
                        device=runtime_query_latent.device,
                        dtype=runtime_query_latent.dtype,
                    )
                    global_attention = _compute_ss_attention_weights(
                        attn_module, prev_query, context_input
                    )
                    if global_attention is None:
                        attention_failure_count += 1
                        continue
                    if tuple(global_attention.shape[:2]) != tuple(attention.shape[:2]):
                        shape_mismatch_count += 1
                        continue
                    global_attentions.append(global_attention.detach().cpu().clone())
                if global_attentions:
                    self._global_attentions[self._current_view] = torch.cat(
                        global_attentions, dim=0
                    )
                    self._global_attention_statuses[self._current_view] = "applied"
                elif shape_mismatch_count > 0:
                    self._global_attention_statuses[self._current_view] = (
                        "shape_mismatch"
                    )
                elif attention_failure_count > 0:
                    self._global_attention_statuses[self._current_view] = (
                        "attention_failure"
                    )
                else:
                    self._global_attention_statuses[self._current_view] = "no_match"
        else:
            self._global_attention_statuses[self._current_view] = "query_unavailable"
        logger.debug(
            f"[SSAttentionCollector] Step {self._current_step}: Collected attention for view {self._current_view}"
        )

    def get_attentions(self) -> Dict[int, torch.Tensor]:
        """Get all collected attentions."""
        return self._attentions

    def get_attention_scores(self) -> Dict[int, torch.Tensor]:
        """Get all collected pre-softmax attention scores."""
        return self._attention_scores

    def get_global_attentions(self) -> Dict[int, torch.Tensor]:
        return self._global_attentions

    def get_query_latents(self) -> Dict[int, torch.Tensor]:
        return self._query_latents

    def get_global_attention_statuses(self) -> Dict[int, str]:
        return self._global_attention_statuses

    def reset(self):
        """Reset collected data."""
        self._attentions.clear()
        self._attention_scores.clear()
        self._global_attentions.clear()
        self._query_latents.clear()
        self._global_attention_statuses.clear()
        self._step_collected_views.clear()
        self._current_step = 0


def _compute_ss_attention_scores(
    module, query, context, *, average_heads: bool = False
):
    """
    Compute pre-softmax cross-attention scores for SS (Stage 1).

    Args:
        module: MultiHeadAttention module
        query: Query tensor [B, L_latent, C]
        context: Context tensor [B, L_cond, C]

    Returns:
        scores: [B, H, L_latent, L_cond] if average_heads=False, else [B, L_latent, L_cond]
    """
    if query is None or context is None:
        return None

    # For dense tensor
    if not torch.is_tensor(query) or not torch.is_tensor(context):
        return None

    try:
        B, L_q, C = query.shape
        _, L_c, _ = context.shape

        # Get head dim and num_heads
        head_dim = (
            module.head_dim if hasattr(module, "head_dim") else C // module.num_heads
        )
        num_heads = module.num_heads

        # Project to Q
        q = module.to_q(query)  # [B, L_q, C]

        # Project to K, V (they are combined in to_kv)
        # to_kv outputs [B, L_c, C * 2] which contains both K and V
        kv = module.to_kv(context)  # [B, L_c, C * 2]
        k, v = kv.chunk(2, dim=-1)  # Each is [B, L_c, C]

        # Reshape for multi-head
        q = q.view(B, L_q, num_heads, head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = k.view(B, L_c, num_heads, head_dim).transpose(1, 2)  # [B, H, L_c, D]

        scale = head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L_q, L_c]

        if average_heads:
            scores = scores.mean(dim=1)  # [B, L_q, L_c]

        return scores
    except Exception as e:
        logger.warning(f"[SS Attention] Failed to compute scores: {e}")
        return None


def _compute_ss_attention_weights(
    module, query, context, *, average_heads: bool = True
):
    """
    Compute attention weights for SS (Stage 1).

    Args:
        module: MultiHeadAttention module
        query: Query tensor [B, L_latent, C]
        context: Context tensor [B, L_cond, C]

    Returns:
        attention: [B, L_latent, L_cond] if average_heads else [B, H, L_latent, L_cond]
    """
    scores = _compute_ss_attention_scores(module, query, context, average_heads=False)
    if scores is None:
        return None
    attention = torch.softmax(scores, dim=-1)
    if average_heads:
        attention = attention.mean(dim=1)
    return attention


def _compute_attention_weights(module, query, context, *, average_heads: bool = True):
    """
    Compute attention weights from query and context.

    This is a simplified version of the attention computation for collection.
    """
    scores = _compute_attention_scores(module, query, context, average_heads=False)
    if scores is None:
        return None
    weights = torch.softmax(scores, dim=-1)
    if average_heads:
        weights = weights.mean(dim=1)
    return weights


def _compute_attention_scores(module, query, context, *, average_heads: bool = False):
    """
    Compute pre-softmax attention scores from query and context.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor

    if query is None or context is None:
        return None

    # Handle SparseTensor
    if isinstance(query, SparseTensor):
        layouts = query.layout
        feats = query.feats
        batch = len(layouts)
        results = []

        for batch_idx in range(batch):
            slc = layouts[batch_idx]
            q_slice = feats[slc].unsqueeze(0)  # [1, L, C]
            ctx = context
            if torch.is_tensor(ctx) and ctx.shape[0] > batch:
                ctx_slice = ctx[batch_idx : batch_idx + 1]
            else:
                ctx_slice = ctx
            if ctx_slice is None:
                continue
            if torch.is_tensor(ctx_slice) and ctx_slice.dim() == 2:
                ctx_slice = ctx_slice.unsqueeze(0)

            dense_scores = _compute_dense_attention_scores(
                module, q_slice, ctx_slice, average_heads=average_heads
            )
            if dense_scores is not None:
                results.append(dense_scores)

        if not results:
            return None
        return torch.cat(results, dim=0)

    elif torch.is_tensor(query):
        return _compute_dense_attention_scores(
            module, query, context, average_heads=average_heads
        )

    return None


def _compute_dense_attention(module, query, context, *, average_heads: bool = True):
    """Compute dense attention weights."""
    scores = _compute_dense_attention_scores(
        module, query, context, average_heads=False
    )
    if scores is None:
        return None
    weights = torch.softmax(scores, dim=-1)
    if average_heads:
        weights = weights.mean(dim=1)
    return weights


def _compute_dense_attention_scores(
    module, query, context, *, average_heads: bool = False
):
    """Compute dense pre-softmax attention scores."""
    if not (torch.is_tensor(query) and torch.is_tensor(context)):
        return None
    if query.dim() == 2:
        query = query.unsqueeze(0)
    if context.dim() == 2:
        context = context.unsqueeze(0)
    if query.shape[0] != context.shape[0]:
        if context.shape[0] == 1 and query.shape[0] > 1:
            context = context.expand(query.shape[0], -1, -1)
        else:
            return None

    q = module.to_q(query)
    kv = module.to_kv(context)
    num_heads = module.num_heads
    head_dim = module.channels // num_heads
    q = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    kv = kv.reshape(kv.shape[0], kv.shape[1], 2, num_heads, head_dim)
    k = kv[:, :, 0].permute(0, 2, 1, 3)

    if hasattr(module, "qk_rms_norm") and module.qk_rms_norm:
        q = module.q_rms_norm(q)
        k = module.k_rms_norm(k)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if average_heads:
        scores = scores.mean(dim=1)
    return scores
