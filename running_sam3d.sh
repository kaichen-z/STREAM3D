#!/usr/bin/env bash
# =============================================================================
# running_sam3d.sh — ORIGINAL SAM3D baseline (NOT MVSAM3D)
# =============================================================================
# Reconstructs each object from a SINGLE randomly-picked view (SAM3D = "random pick 1").
# This is the original single-view SAM3D baseline used in the Stream3D paper comparisons
# (cf. plan Task-14: "sam3d (randomly pick 1)"), as opposed to MVSAM3D which fuses 8 views.
#
# Setting (vs the two Stream3D scripts):
#   * selection_strategy = random, topk = 1   -> exactly ONE random seen view
#   * VA weighting OFF (irrelevant with a single view)
#   * selection_random_seed = $SEED           -> the random view depends on the seed
#
# Self-contained: no code edits needed; everything is a Hydra override below.
#
# Usage:
#   bash running_sam3d.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]
# Examples:
#   bash running_sam3d.sh 3 alarm        /tmp/out_sam3d                 # 1 object, all 13 chunks, seed 0
#   bash running_sam3d.sh 3 alarm,bell   /tmp/out_sam3d "[0,4,8,12]" 1  # 2 objects, 4 horizons, seed 1
# =============================================================================
set -uo pipefail

# ---- environment (override via env if your layout differs) -------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"                 # torch 2.4.1+cu121 + pytorch3d stack
GSO="${GSO:-/workspace/data/kaichen/sam3d/data/GSO30}"
CKPT="${CKPT:-checkpoints/hf/pipeline.yaml}"

# ---- args -------------------------------------------------------------------
GPU="${1:?usage: running_sam3d.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]}"
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
  "$PYTHON" -m streaming.runner backend=sam3d \
  data.roots=[$ROOTS] data.da3_dir_name=$DA3NAME camera_pose_source=da3 \
  streaming.topk=1 streaming.stage2_selection.topk=1 \
  ++streaming.selection_strategy=random ++streaming.selection_random_seed=$SEED \
  pipeline.ss_weighting=false pipeline.stage2_weighting.enabled=false \
  pipeline.decode_formats=[gaussian,mesh] pipeline.with_texture_baking=false \
  output_root="$OUT" model_config_path="$CKPT" \
  chunk_size=8 chunk_overlap=2 chunk_indices=$CHUNKS seed=$SEED \
  hydra.run.dir=/tmp/running_sam3d/$FIRST hydra.output_subdir=.hydra
