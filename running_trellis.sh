#!/usr/bin/env bash
# =============================================================================
# running_trellis.sh — ORIGINAL TRELLIS2 baseline (random view selection)
# =============================================================================
# The TRELLIS2 reconstruction backbone with the ORIGINAL random multi-view setup:
# 8 RANDOM seen views fused with UNIFORM weighting (NO VA view selection, NO VA
# weighting). This is the TRELLIS2 analogue of the MVSAM3D random-8 baseline.
#
# Setting (vs the two Stream3D-on-TRELLIS2 scripts):
#   * backend = trellis2                       (the ONLY change vs the SAM3D trio)
#   * selection_strategy = random, topk = 8    -> random 8 views (NO VA selection)
#   * VA weighting OFF                          -> uniform multi-view fusion
#   * selection_random_seed = $SEED            -> the random-8 set depends on the seed
#
# Same structure/interface/style as running_sam3d.sh; only backend + model differ.
# Self-contained: no code edits needed; everything is a Hydra override below.
#
# Usage:
#   bash running_trellis.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]
# Example:
#   bash running_trellis.sh 3 alarm /tmp/out_trellis "[0,4,8,12]" 0
# =============================================================================
# TRELLIS2 accepts selection.method=random for this baseline. The uniform fusion
# setting is treated as the average-fusion baseline.
set -uo pipefail

# ---- environment (override via env if your layout differs) -------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"                 # NOTE: TRELLIS2 needs python>=3.11 + o_voxel (see README)
GSO="${GSO:-/workspace/data/kaichen/sam3d/data/GSO30}"
MODEL_PATH="${MODEL_PATH:-microsoft/TRELLIS.2-4B}"   # TRELLIS.2-4B (HF id or local snapshot dir)

# ---- args -------------------------------------------------------------------
GPU="${1:?usage: running_trellis.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]}"
OBJS_CSV="${2:?need objects_csv, e.g. alarm or alarm,bell}"
OUT="${3:?need output_dir}"
CHUNKS="${4:-[0,1,2,3,4,5,6,7,8,9,10,11,12]}"
SEED="${5:-0}"

cd "$REPO"
IFS=',' read -ra OBJS <<< "$OBJS_CSV"
ROOTS=""; for o in "${OBJS[@]}"; do ROOTS="$ROOTS${ROOTS:+,}$GSO/$o"; done
FIRST="${OBJS[0]}"
if [ -d "$GSO/$FIRST/render_spiral_100/da3_full100_da3chunk8_overlap2/results_output" ]; then
  DA3NAME=da3_full100_da3chunk8_overlap2; else DA3NAME=da3; fi

exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/_compat" \
  HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 ATTN_BACKEND=sdpa \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m streaming.runner backend=trellis2 \
  data.roots=[$ROOTS] data.da3_dir_name=$DA3NAME camera_pose_source=da3 \
  pipeline.selection.method=random pipeline.selection.topk=8 \
  ++pipeline.selection.random_seed=$SEED \
  pipeline.fusion.weight_source=uniform \
  pipeline.decode_formats=[gaussian,mesh] pipeline.with_texture_baking=false \
  output_root="$OUT" pipeline.model_path="$MODEL_PATH" \
  chunk_size=8 chunk_overlap=2 chunk_indices=$CHUNKS seed=$SEED \
  hydra.run.dir=/tmp/running_trellis/$FIRST hydra.output_subdir=.hydra
