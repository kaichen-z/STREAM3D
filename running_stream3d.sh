#!/usr/bin/env bash
# =============================================================================
# running_stream3d.sh — Stream3D FULL pipeline (VA view selection + VA weighting)
# =============================================================================
# The complete Stream3D method: VA attention-based view SELECTION (va_div, choosing 8
# informative views) PLUS VA WEIGHTING (mass_relative, jam_kappa K=8) of the fused views.
# View selection is deterministic (attention-based, not random).
#
# Setting (vs the other two scripts):
#   * selection_strategy = va_div, topk = 8   -> 8 VA-selected views (diversity lambda=0.1)
#   * VA weighting ON: weight_source = mass_relative, jam_kappa = 8, uniform_blend = 0
#   * seed only labels the run (selection is deterministic)
#
# Self-contained: no code edits needed; everything is a Hydra override below.
#
# Usage:
#   bash running_stream3d.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]
# Example:
#   bash running_stream3d.sh 5 alarm /tmp/out_full "[0,4,8,12]" 0
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"
GSO="${GSO:-/workspace/data/kaichen/sam3d/data/GSO30}"
CKPT="${CKPT:-checkpoints/hf/pipeline.yaml}"
KAPPA="${KAPPA:-8}"                                  # VA weighting sharpness (jam_kappa)
DIV="${DIV:-0.1}"                                    # VA selection diversity lambda

GPU="${1:?usage: running_stream3d.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]}"
OBJS_CSV="${2:?need objects_csv}"
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
  streaming.topk=8 streaming.stage2_selection.topk=8 \
  ++streaming.selection_strategy=va_div ++streaming.selection_div_lambda=$DIV \
  pipeline.ss_weight_source=mass_relative pipeline.stage2_weighting.weight_source=mass_relative \
  pipeline.ss_jam_kappa=$KAPPA pipeline.stage2_weighting.jam_kappa=$KAPPA \
  pipeline.ss_uniform_blend=0 pipeline.stage2_weighting.uniform_blend=0 \
  pipeline.decode_formats=[gaussian,mesh] pipeline.with_texture_baking=false \
  output_root="$OUT" model_config_path="$CKPT" \
  chunk_size=8 chunk_overlap=2 chunk_indices=$CHUNKS seed=$SEED \
  hydra.run.dir=/tmp/running_stream3d/$FIRST hydra.output_subdir=.hydra
