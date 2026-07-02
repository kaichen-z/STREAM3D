#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — GSO30 metric evaluation for Stream3D reconstructions
# =============================================================================
# Scores the reconstructions produced by running_stream3d.sh / running_sam3d.sh
# (or the TRELLIS.2 variants) against the GSO `render_mvs_25` ground truth and
# reports the paper metrics:
#
#     CD (Chamfer)   IoU   PSNR   SSIM   LPIPS   Image-FID   P-FID
#
# It drives the repo's two-step GSO evaluator (no code edits needed):
#   1) manual_register_sam3d_gso.py estimate
#        per scene/chunk: fit a global Sim(3) (24-init GICP + scale) from the
#        predicted Gaussian (result.ply) to render_mvs_25 GT, then render the 25
#        GT camera views. Writes: alignment_state.json, renders/<v>_pred.png,
#        gt/<v>_gt.png, summary.json  →  <eval_root>/<variant>/<scene>/<chunk>/
#   2) evaluate_manual_registration.py
#        aggregates those renders + GT into the metric table (CSV + summary.json)
#        →  <eval_root>/evaluation/
#
# Self-contained: every path/knob is an argument or an env override.
#
# Usage:
#   bash evaluate.sh <gpu> <recon_output_dir> <objects_csv> <eval_root> [variant]
#
#   recon_output_dir : the OUT you passed to running_stream3d.sh, laid out as
#                      <recon_output_dir>/<scene>/chunk_*/{result.ply,params.npz}
#   objects_csv      : comma-separated GSO30 scene names (e.g. alarm,shoe2)
#   eval_root        : where registration renders + metrics are written
#   variant          : label for this run's column (default: basename of recon dir)
#
# Example (after: bash running_stream3d.sh 5 alarm /tmp/out_full):
#   bash evaluate.sh 5 /tmp/out_full alarm /tmp/eval_full
#
# Optional env:
#   PYTHON            python interpreter                (default /usr/bin/python3)
#   GSO               GSO30 GT root (has <scene>/render_mvs_25)  (default matches runner)
#   SAMPLE_POINTS     Chamfer surface samples           (default 4096)
#   INITIAL_COUNT     Sim(3) initialisations for GICP   (default 24)
#   IMAGE_FID_BACKEND torchvision_inception|timm_inception|skip (default torchvision_inception)
#   PFID_CKPT         PointNet++ checkpoint for P-FID   (default: unset -> P-FID skipped)
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"
GSO="${GSO:-/workspace/data/kaichen/sam3d/data/GSO30}"
SAMPLE_POINTS="${SAMPLE_POINTS:-4096}"
INITIAL_COUNT="${INITIAL_COUNT:-24}"
IMAGE_FID_BACKEND="${IMAGE_FID_BACKEND:-torchvision_inception}"

GPU="${1:?usage: evaluate.sh <gpu> <recon_output_dir> <objects_csv> <eval_root> [variant]}"
RECON="${2:?need recon_output_dir (the OUT from running_stream3d.sh)}"
OBJS_CSV="${3:?need objects_csv}"
EVAL_ROOT="${4:?need eval_root}"
VARIANT="${5:-$(basename "$RECON")}"

cd "$REPO"
SCRIPTS="$REPO/docs/skills/sam3d-gso-manual-registration/scripts"
REG="$SCRIPTS/manual_register_sam3d_gso.py"
EVAL="$SCRIPTS/evaluate_manual_registration.py"
IFS=',' read -ra OBJS <<< "$OBJS_CSV"

# --- Step 1: per scene/chunk global-Sim(3) registration + render_mvs_25 render ---
for scene in "${OBJS[@]}"; do
  scene_dir="$RECON/$scene"
  if [ ! -d "$scene_dir" ]; then
    echo "[evaluate] WARN: no reconstruction dir for scene '$scene' ($scene_dir); skipping." >&2
    continue
  fi
  for chunk_dir in "$scene_dir"/chunk_*; do
    [ -d "$chunk_dir" ] || continue
    chunk="$(basename "$chunk_dir")"
    out="$EVAL_ROOT/$VARIANT/$scene/$chunk"
    mkdir -p "$out"
    echo "[evaluate] register+render  $VARIANT / $scene / $chunk"
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/_compat" HYDRA_FULL_ERROR=1 \
      "$PYTHON" "$REG" estimate \
        --chunk-root "$chunk_dir" \
        --scene "$scene" \
        --variant "$VARIANT" \
        --gt-root "$GSO/$scene" \
        --output-dir "$out" \
        --sample-points "$SAMPLE_POINTS" \
        --initial-count "$INITIAL_COUNT" \
        --views all
  done
done

# --- Step 2: aggregate metrics (CD / IoU / PSNR / SSIM / LPIPS / Image-FID / P-FID) ---
PFID_ARGS=()
if [ -n "${PFID_CKPT:-}" ]; then
  PFID_ARGS=(--point-feature-backend torchscript --pointnet-checkpoint "$PFID_CKPT")
fi

echo "[evaluate] aggregating metrics -> $EVAL_ROOT/evaluation"
env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/_compat" \
  "$PYTHON" "$EVAL" \
    --output-root "$EVAL_ROOT" \
    --state-index "$EVAL_ROOT/final_state_index.json" \
    --variants "$VARIANT" \
    --eval-dir "$EVAL_ROOT/evaluation" \
    --sample-points "$SAMPLE_POINTS" \
    --image-fid-backend "$IMAGE_FID_BACKEND" \
    "${PFID_ARGS[@]}"

echo "[evaluate] done. Metrics table + summary.json under: $EVAL_ROOT/evaluation"
