#!/usr/bin/env bash
# =============================================================================
# running_stream3d_fast.sh — Stream3D FULL pipeline, FAST (shortcut) inference
# =============================================================================
# Same method as running_stream3d.sh (VA view selection va_div + VA weighting),
# but with the shortcut launch settings from 0_task/stream3d/shortcut_model_runs.md:
#
#   Stage 1 sparse-structure generation:
#     shortcut/distilled model ENABLED (use_stage1_distillation=true -> no_shortcut=False,
#     cfg strength forced to 0), 4 inference steps
#   Stage 2 structured-latent / texture-refinement generation:
#     shortcut/distillation DISABLED (use_stage2_distillation=false), 4 inference steps
#
# i.e. "Stage-1 shortcut + Stage-2 fewer-step inference", not full 2-stage distillation.
# Everything else (selection, weighting, decoding) is identical to running_stream3d.sh.
#
# Setting (vs the other scripts):
#   * selection_strategy = va_div, topk = 8   -> 8 VA-selected views (diversity lambda=0.1)
#   * VA weighting ON: weight_source = mass_relative, jam_kappa = 8, uniform_blend = 0
#   * pipeline.use_stage1_distillation=true, pipeline.stage1_inference_steps=4
#   * pipeline.use_stage2_distillation=false, pipeline.stage2_inference_steps=4
#   * seed only labels the run (selection is deterministic)
#
# Self-contained: no code edits needed; everything is a Hydra override below.
# Step counts are overridable via env: STAGE1_STEPS / STAGE2_STEPS (default 4).
#
# Usage:
#   bash running_stream3d_fast.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]
# Example:
#   bash running_stream3d_fast.sh 5 alarm /tmp/out_fast "[0,4,8,12]" 0
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"
GSO="${GSO:-/workspace/data/kaichen/sam3d/data/GSO30}"
CKPT="${CKPT:-checkpoints/hf/pipeline.yaml}"
KAPPA="${KAPPA:-8}"                                  # VA weighting sharpness (jam_kappa)
DIV="${DIV:-0.1}"                                    # VA selection diversity lambda
STAGE1_STEPS="${STAGE1_STEPS:-4}"                    # Stage-1 shortcut inference steps
STAGE2_STEPS="${STAGE2_STEPS:-4}"                    # Stage-2 (no shortcut) inference steps

GPU="${1:?usage: running_stream3d_fast.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]}"
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
  pipeline.use_stage1_distillation=true pipeline.stage1_inference_steps=$STAGE1_STEPS \
  pipeline.use_stage2_distillation=false pipeline.stage2_inference_steps=$STAGE2_STEPS \
  pipeline.decode_formats=[gaussian,mesh] pipeline.with_texture_baking=false \
  output_root="$OUT" model_config_path="$CKPT" \
  chunk_size=8 chunk_overlap=2 chunk_indices=$CHUNKS seed=$SEED \
  hydra.run.dir=/tmp/running_stream3d_fast/$FIRST hydra.output_subdir=.hydra
