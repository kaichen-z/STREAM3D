---
name: sam3d-gso-manual-registration
description: Use when a Streaming SAM3D GSO result needs a stable manual or agent-in-the-loop global Sim(3) registration against raw GSO render_mvs_25 assets.
---

# SAM3D GSO Manual Registration

This skill provides a direct source-geometry global Sim(3) workflow for GSO30 cases where the `result_pose.npz` camera chain is not trusted enough for evaluation.  It is an auxiliary diagnostic loop, not a replacement for the formal automatic benchmark.

The script is:

```bash
docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py
```

The constrained Sim(3) adjustment helper is:

```bash
docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py
```

The skill scripts are intentionally independent from other skill directories.  They may use repository code under `streaming/` and `sam3d_objects/`, but they must not import scripts from `docs/skills/*`.

## Metric Reporting Rule

When reporting Chamfer Distance in paper tables, summaries, or user-facing comparisons, use `cd_l2_mean` by default.  This is the unsquared symmetric nearest-neighbor L2 diagnostic and is the column used for comparing this skill's outputs with prior GSO manual-registration tables.

The final evaluator also writes `cd_mean`, which is PyTorch3D `chamfer_distance(..., norm=2)` on 4096 sampled pred/GT surface points.  Treat `cd_mean` as squared-L2 Chamfer.  Do not label it simply as `CD` in paper tables unless the table explicitly says squared Chamfer.  If both are shown, use names such as:

```text
CD-L2 = cd_l2_mean
CD-sq = cd_mean
```

The render-time `summary.json` field `active_mesh_diagnostics.cd` is also diagnostic-only and should not be used as the paper CD unless explicitly requested.

## Inputs

Predicted chunk:

```text
tmp/gso30-streaming-ablation/<variant>/<scene>/chunk_*/
  result.glb
  result.ply
  params.npz
```

Ground truth scene:

```text
data/raw/gso/GSO30/<scene>/render_mvs_25/
  model_norm.glb
  model/{000..024}.png
  model/{000..024}.npy
```

`result_pose.npz` may exist and is recorded as provenance only.  It is not used to estimate the global Sim(3).

When `result.glb` is missing, the official workflow allows `params.npz` to provide a sparse registration source:

1. load `coords`;
2. drop the batch column if present;
3. reorder latent `[D,H,W]` indices to local mesh-style `XYZ` with `coords[:, [2, 0, 1]]`;
4. convert to canonical local coordinates with `((xyz + 0.5) / 64) - 0.5`;
5. fit a 24-way signed-permutation `centroid + uniform-scale + translation` Sim(3) from `params` points to downsampled `result.ply` xyz;
6. use that locally aligned sparse cloud for global registration against `render_mvs_25/model_norm.glb`.

Final rendering still uses the chunk's `result.ply` Gaussian with the estimated global Sim(3).  The sparse `params.npz` source is registration-only.

## State File

Each run writes:

```text
scratch/sam3d-gso-manual-registration/<variant>/<scene>/<chunk_name>/alignment_state.json
```

The stable fields are:

```text
format_version
scene
chunk_root
gt_root
registration_space
source_mesh
source_geometry_kind
source_geometry_path
source_local_alignment
target_mesh
target_bbox_diagonal
needs_manual_review
auto_initial_sim3
active_sim3
registration_diagnostics
render_eval
artifacts
```

Both `auto_initial_sim3` and `active_sim3` contain:

```text
scale
rotation
translation
matrix
source_space
target_space
```

`active_sim3` is the only transform used by `render`.  Agent or human edits should change `active_sim3`, leaving `auto_initial_sim3` intact.

## Workflow

First estimate the automatic initialization and render all 25 views:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py \
  estimate \
  --scene alarm \
  --variant k2_4 \
  --chunk-root tmp/gso30-streaming-ablation/k2_4/alarm/chunk_0016 \
  --source-geometry auto
```

Inspect:

```text
scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/
  comparisons/{000..024}_gt_pred.png
  comparison_contact_sheet.png
  per_view_metrics.csv
  summary.json
  alignment_state.json
```

For vision-model diagnosis, do not concatenate all 25 comparison images into one image.  Review comparison pairs one at a time.  The recommended first pass is 3 representative pairs, for example:

```text
comparisons/000_gt_pred.png
comparisons/012_gt_pred.png
comparisons/024_gt_pred.png
```

Use the contact sheet only as a human index for choosing which individual pairs to inspect next.

If the visual overlay is not acceptable, use the constrained controller instead of raw 4x4 matrix editing.  First create orientation candidates:

```bash
./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  propose \
  --state-file scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/alignment_state.json \
  --round-name orientation_001 \
  --preset orientation
```

Render candidate comparison triples:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  render-candidates \
  --round-dir scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/manual_adjustment_rounds/orientation_001 \
  --views 0,12,24
```

Inspect each candidate's individual `comparisons/000_gt_pred.png`, `012_gt_pred.png`, and `024_gt_pred.png`.  Apply only a visually selected candidate:

```bash
./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  apply-candidate \
  --state-file scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/alignment_state.json \
  --round-dir scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/manual_adjustment_rounds/orientation_001 \
  --candidate-id rot_z_180 \
  --reason "front/back orientation corrected in 000/012/024"
```

Then rerender the canonical state:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py \
  render \
  --state-file scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/alignment_state.json
```

If orientation is now correct but a small residual remains, run a micro round:

```bash
./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  propose \
  --state-file scratch/sam3d-gso-manual-registration/k2_4/alarm/chunk_0016/alignment_state.json \
  --round-name micro_001 \
  --preset micro \
  --rotation-step-deg 4 \
  --translation-step-fraction 0.01 \
  --scale-step 0.02
```

Repeat until the contact sheet and per-view comparisons are visually aligned.  The final `alignment_state.json` is the reusable global Sim(3) record for that scene/chunk.

## Configurable Initial GICP Variant

For scenes where TEASER++ chooses the wrong semantic orientation, use the pre-matched variant:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py \
  estimate \
  --scene alarm \
  --variant k2_12 \
  --chunk-root tmp/gso30-streaming-ablation/k2_12/alarm/chunk_0016 \
  --registration-method initial_gicp_scale \
  --initial-count 60
```

This always evaluates the 24 proper signed-permutation rotations first.  When `--initial-count` is larger than 24, it appends a deterministic Hopf-grid SO(3) rotation set until the requested count is reached.  Typical values are 24, 36, 48, and 60.  The legacy method name `initial24_gicp_scale` is still accepted; use `--initial-count` to choose the actual number of initial rotations.

Each initial is centroid/scale matched, refined with Open3D Generalized ICP for rotation/translation at fixed scale, then refit with a trimmed nearest-neighbor uniform scale/translation solve.  The best candidate becomes both `auto_initial_sim3` and `active_sim3`, and all candidate `final_sim3` records are saved under `registration_diagnostics.candidates`.

For the ablation1 GSO past-solution outputs, most chunks contain `params.npz` and `result.ply` but no `result.glb`.  Use:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py \
  estimate \
  --scene alarm \
  --variant mvsam3d_flowedit \
  --chunk-root outputs/ablation1_gso_past_solution/mvsam3d_flowedit/alarm/chunk_0000 \
  --output-dir scratch/sam3d-gso-manual-registration-ablation1/mvsam3d_flowedit/alarm/chunk_0000 \
  --source-geometry auto \
  --registration-method initial_gicp_scale \
  --initial-count 60 \
  --views all
```

This automatically falls back from `result.glb` to `params.npz` for registration, while still rendering the transformed `result.ply` Gaussian.

When a scene from this variant is visually rejected, first export and render the saved registration candidates:

```bash
./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  export-registration-candidates \
  --state-file scratch/sam3d-gso-manual-registration-initial-gicp/k2_12/alarm/chunk_0016/alignment_state.json \
  --round-name registration_candidates \
  --top-k 24

CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  render-candidates \
  --round-dir scratch/sam3d-gso-manual-registration-initial-gicp/k2_12/alarm/chunk_0016/manual_adjustment_rounds/registration_candidates \
  --views 0,12,24
```

Inspect candidate comparison triples one candidate at a time.  Prefer a candidate that fixes semantic orientation even if its metrics are not rank 0.  Apply the chosen candidate, then rerender the canonical state:

```bash
./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py \
  apply-candidate \
  --state-file scratch/sam3d-gso-manual-registration-initial-gicp/k2_12/alarm/chunk_0016/alignment_state.json \
  --round-dir scratch/sam3d-gso-manual-registration-initial-gicp/k2_12/alarm/chunk_0016/manual_adjustment_rounds/registration_candidates \
  --candidate-id rank_03_perm_xyz_sign_mmp \
  --reason "candidate fixes top/bottom orientation in individual 000/012/024 comparisons"

CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py \
  render \
  --state-file scratch/sam3d-gso-manual-registration-initial-gicp/k2_12/alarm/chunk_0016/alignment_state.json
```

Only after the best saved initial is chosen should subagents use `propose --preset micro` for small residual translation, rotation, or scale corrections.


## Final Evaluation

After manual states and final 25-view renders are in `outputs/sam3d-gso-manual-registration`, run the evaluation stage:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/evaluate_manual_registration.py \
  --variants k2_4,k2_8,k2_12,k2_16 \
  --sample-points 4096 \
  --eval-dir outputs/sam3d-gso-manual-registration/evaluation
```

On an 8-GPU machine, it is fine to expose all cards:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/evaluate_manual_registration.py \
  --variants k2_4,k2_8,k2_12,k2_16 \
  --sample-points 4096 \
  --eval-dir outputs/sam3d-gso-manual-registration/evaluation \
  --image-fid-backend timm_inception \
  --image-fid-batch-size 64 \
  --lpips-batch-size 16 \
  --point-feature-backend pointnet2_ssg \
  --pointnet2-repo scratch/external/Pointnet_Pointnet2_pytorch \
  --pointnet-checkpoint scratch/external/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth \
  --point-feature-batch-size 8
```

Outputs:

```text
outputs/sam3d-gso-manual-registration/evaluation/
  scene_metrics.csv
  scene_metrics.json
  metrics_summary.csv
  metrics_summary.json
  evaluation_run.json
```

Metric definitions:

- `cd`: PyTorch3D Chamfer Distance on exactly 4096 uniformly sampled pred and GT surface points.  The pred points are sampled from the active-Sim(3)-aligned pred mesh, not raw vertices.  Sampling uses `trimesh.sample.sample_surface`.
- `cd_l2`: diagnostic unsquared nearest-neighbor Chamfer using the same sampled point sets.
- `psnr`, `ssim`, `lpips`: recomputed by the final evaluation script with the same photometric protocol as `docs/skills/sam3d-gso-render`: pred and GT RGBA are hard-alpha composited onto white, PSNR/SSIM use the full RGB image, and LPIPS is LPIPS-VGG on the same full white-background RGB image pair.  Do not report the masked values stored by older manual-render summaries as final photometric metrics.
- `image_fid`: Fréchet distance between Inception features of the same full white-background pred/GT `render_mvs_25` RGB images used by PSNR/SSIM/LPIPS.  If Inception weights are unavailable, use `--image-fid-backend skip` to still produce the core table.
- `p_fid`: Fréchet distance between PointNet++ features of aligned pred/GT 4096-point samples.  The preferred deadline-safe backend is the pretrained ModelNet40 PointNet++ SSG classifier from `yanx27/Pointnet_Pointnet2_pytorch`, using its 1024D global feature before the classifier head.  A custom TorchScript PointNet++ feature extractor is also supported.

Formal final-table protocol:

- render source: existing `outputs/sam3d-gso-manual-registration/<variant>/<scene>/chunk_*/renders/{000..024}_pred.png`;
- GT source: `data/raw/gso/GSO30/<scene>/render_mvs_25/model/{000..024}.png`;
- alpha handling: use hard alpha from each RGBA image, composite each image independently onto white, then compute metrics on RGB;
- PSNR: full-image RGB MSE, `-10 * log10(mse)`;
- SSIM: full-image RGB `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=1.0)`;
- LPIPS: `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)`, never the torchmetrics default `alex`; resize only when max side exceeds 512 px;
- FID: InceptionV3 ImageNet features on the same white-background RGB image tensors; use one feature per rendered view and compute Fréchet distance per variant and for `ALL`;
- geometry CD: recompute from aligned meshes with uniform surface samples, never from raw mesh vertices;
- P-FID: report only when a real pretrained PointNet++ feature extractor is provided.  Leave blank/skipped rather than using random features.  Mesh sources are already sampled to 4096 points; sparse Gaussian/params point sources are deterministically re-sampled to the same count before feature extraction.

P-FID is intentionally opt-in.  Do not report P-FID from random or untrained features.  To enable it:

```bash
git clone --depth 1 https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git \
  scratch/external/Pointnet_Pointnet2_pytorch

CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/evaluate_manual_registration.py \
  --point-feature-backend pointnet2_ssg \
  --pointnet2-repo scratch/external/Pointnet_Pointnet2_pytorch \
  --pointnet-checkpoint scratch/external/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth \
  --point-feature-normalization unit_sphere \
  --strict-distribution-metrics
```

For a custom extractor, use `--point-feature-backend torchscript`.
The TorchScript point feature model should accept either `[B, N, 3]` (`bnc`) or `[B, 3, N]` (`bcn`) float32 tensors and return a `[B, D]` global feature tensor.  Dict outputs may use `features`, `feat`, `embedding`, or `global_feat`.

Fast deadline-safe mode, when only core metrics are needed:

```bash
CUDA_VISIBLE_DEVICES=0 ./.env/bin/python \
  docs/skills/sam3d-gso-manual-registration/scripts/evaluate_manual_registration.py \
  --variants k2_4,k2_8,k2_12,k2_16 \
  --sample-points 4096 \
  --image-fid-backend skip \
  --point-feature-backend skip
```

## Subagent-In-The-Loop Work

Use subagents for batches of rejected scenes, not for one-shot visual labeling.  A subagent owns a small, disjoint scene set and must run the full loop:

```text
inspect individual comparison pairs -> propose candidates -> render candidate triples -> select/apply one candidate -> render canonical state -> inspect again
```

Do not let subagents report `render` completion as success.  Success means the current `active_sim3` has been visually accepted from individual comparison pairs.  If the reconstruction itself is visibly poor, use rough acceptance: direction, scale, and center should be broadly correct, while fine geometry/texture mismatches can remain documented.  If the overlay is still directionally wrong, the status is `improved_not_accepted` or `not_solved`.

Recommended supervision pattern:

- keep at most 5 subagents active at the same time across all variants;
- assign 1-3 scenes per subagent;
- give each subagent exclusive write ownership of only those scene scratch directories;
- require `alignment_state.before_manual_adjustment.json` before the first edit;
- require `manual_adjustment_notes.md` in each scene directory;
- require a final status per scene: `accepted`, `accepted_rough`, `improved_not_accepted`, or `not_solved`;
- require evidence listing exactly which individual comparison files were inspected after the final render.
- do not allow raw 4x4 matrix edits unless the controller cannot express the needed candidate and the reason is documented.

Use this bundled prompt template when delegating the work: `references/subagent_worker_prompt.md`.

## Manual Adjustment Heuristics

Keep the search low-dimensional and hypothesis-driven.  Subagents should use `sim3_adjustment_controller.py` rather than directly editing matrix fields.  Start with orientation mode errors before micro-adjustments:

- front/back reversed: try a 180 degree rotation around the target up axis;
- upside-down object: try 180 degree rotations around the two horizontal target axes;
- left/right mirror-like view error: try 90/180/270 degree rotations around the target up axis before touching translation;
- persistent silhouette offset after orientation is correct: adjust translation in small steps in target coordinates;
- object consistently too large/small: adjust uniform scale by small multiplicative factors, then rerender;
- after any rotation or scale change, recompute translation to keep the transformed source centroid near the previous transformed centroid unless the visual evidence says the object must move.

The controller composes deltas in target space around a pivot, so rotations do not accidentally fling the object around the world origin.  The default pivot is the transformed source bbox center.

Use metrics only as secondary evidence.  A lower LPIPS or Chamfer score is not enough to accept a front/back, top/bottom, or semantic orientation error.

For each iteration, write down:

```text
hypothesis
active_sim3 edit
render command
comparison files inspected one by one
visual verdict
next action
```

Stop when the first-pass views (`000`, `012`, `024`) are visually close enough for the scene quality.  For good reconstructions, this means small residual pose error.  For failed or partial reconstructions, this means the semantic direction, object scale, and center are broadly correct; document the remaining geometry/appearance failure as reconstruction quality, not registration failure.  If more than a few structured candidates fail, mark the scene `not_solved` and preserve the best state plus notes rather than pretending it passed.

## Default Registration Contract

The default automatic estimate is:

- sample 4096 surface points from `result.glb`;
- sample 4096 surface points from `render_mvs_25/model_norm.glb`;
- use TEASER++ with FPFH mutual correspondences and `estimate_scaling=True`;
- follow with Open3D point-to-point rigid ICP on TEASER-aligned points;
- keep the TEASER scale fixed during ICP;
- record `cd_before`, `cd_after_teaser`, `cd_after`, correspondence count, ICP `fitness`, and ICP `inlier_rmse`.

The source and target are both converted into the skill's direct mesh registration basis before solving.  There is no 24-init search and no manual initialization in the default automatic pass.

The `initial_gicp_scale` variant intentionally replaces TEASER++ with the configurable initial candidate pool described above.  Use it when saved initial candidates are more valuable than a single global TEASER++ answer.

## Render Contract

`render` reads `active_sim3` from `alignment_state.json`.

- Predicted mesh is only conceptually transformed for diagnostics; no official aligned mesh copy is exported.
- Predicted Gaussian centers are transformed by `p' = scale * (p @ R.T) + t`.
- Gaussian scales are multiplied by the uniform Sim(3) scale.
- Gaussian rotations are left-multiplied by the global rotation.
- Original `result.ply` and `result.glb` are never modified.

The camera and target protocol is fixed to raw `render_mvs_25`:

- camera extrinsics: `render_mvs_25/model/{000..024}.npy`;
- GT RGBA target: `render_mvs_25/model/{000..024}.png`;
- focal: one `focal_norm` fitted from GT mesh projection bbox and alpha-mask bbox;
- default views: all 25.

## Notes

Use `scratch/` for all exploratory outputs.  Move only confirmed final records to `outputs/` after the manual loop is accepted by the user.

For difficult scenes such as `grandfather` or `elephant`, use the contact sheet first.  Metrics can be numerically plausible even when a front/back or left/right mode is visually wrong.

When asking an Agent or vision model to judge alignment, pass individual `comparisons/*_gt_pred.png` files, not a 25-view collage.  Start with three separated views, then add more pairs only when the failure mode is ambiguous.
