# Stream3D: Sequential Multi-View 3D Generation via Evidential Memory

[**Project Page**](https://stream-3d.github.io/stream3d.github.io/) &nbsp;•&nbsp; [**Paper (arXiv)**](https://arxiv.org/pdf/2605.21472)

Official inference code for **Stream3D**. Stream3D turns a stream of posed views into a single 3D
asset (Gaussians + mesh) by keeping an evidential memory of each view and using it to (1) **select**
the most informative views and (2) **weight** them during fusion. Two reconstruction backbones are
supported: **SAM3D** (default) and **TRELLIS.2**.

## Installation

Always use the `streaming3d` conda environment.

```bash
conda env create -f environment.yml
conda activate streaming3d

# PyTorch — pick the CUDA wheel for your machine (CUDA 12.1 shown)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Dependencies + local package
pip install -r requirements.txt \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
pip install --no-build-isolation -r requirements.inference.txt \
  --extra-index-url https://pypi.ngc.nvidia.com
pip install -e .

# Optional: TRELLIS.2 backend (needs Python >= 3.11)
pip install -r requirements.trellis2-backend.txt
```

Place a SAM3D-compatible pipeline config at `checkpoints/hf/pipeline.yaml` (or pass
`model_config_path=...`). The TRELLIS.2 backend loads `microsoft/TRELLIS.2-4B`. Each scene root
should contain a render split with `images/`, `masks/`, and (for SAM3D) `da3/` pose outputs.

## Data

The evaluation set is **GSO30** — 30 objects from Google Scanned Objects — used for both
reconstruction (scene names like `alarm`) and evaluation (`render_mvs_25` ground truth). Download it
from:

- **GSO30:** http://huggingface.co/datasets/WalkerCH/Streaming3D/tree/main/GSO30

```bash
# e.g. with the HuggingFace CLI
huggingface-cli download WalkerCH/Streaming3D --repo-type dataset \
  --include "GSO30/*" --local-dir ./data
```

Point the `GSO` environment variable (used by the run and eval scripts as the data / GT root) at the
downloaded `GSO30` folder:

```bash
export GSO=./data/GSO30
```

## Usage

All scripts share the same call signature:

```bash
bash <script>.sh <gpu> <objects_csv> <output_dir> [chunk_indices] [seed]
```

| Script | Backbone | Method |
|--------|----------|--------|
| `running_sam3d.sh` | SAM3D | baseline (random views, uniform fusion) |
| `running_stream3d.sh` | SAM3D | **Stream3D** (VA selection + VA weighting) |
| `running_stream3d_fast.sh` | SAM3D | **Stream3D, fast inference** (same method, shortcut sampling) |
| `running_trellis.sh` | TRELLIS.2 | baseline (random views, uniform fusion) |
| `running_stream3d_trellis.sh` | TRELLIS.2 | **Stream3D** (VA selection + VA weighting) |

Run the full Stream3D method on GSO30 scene `alarm`, GPU 5, reconstructing chunks `[0,4,8,12]`:

```bash
bash running_stream3d.sh 5 alarm /tmp/out_full "[0,4,8,12]" 0          # SAM3D backbone
bash running_stream3d_trellis.sh 5 alarm /tmp/out_full_trellis "[0,4,8,12]" 0   # TRELLIS.2 backbone
```

### Fast inference

`running_stream3d_fast.sh` runs the identical Stream3D method (same selection, weighting,
outputs) with shortcut sampling: the stage-1 sparse-structure generator uses the distilled
shortcut model at 4 inference steps, and stage 2 runs plain 4-step sampling (defaults are
25/25). Same call signature:

```bash
bash running_stream3d_fast.sh 5 alarm /tmp/out_fast "[0,4,8,12]" 0
```

End-to-end it is ~1.5× faster per chunk (≈85 s → ≈55 s steady-state on an idle H100) at a
small quality cost (CD-L2 +0.003–0.005, PSNR −0.3–0.6 dB on GSO30); output formats and sizes
are unchanged. Use it for previews and iteration; use `running_stream3d.sh` for final numbers.
Step counts are overridable via `STAGE1_STEPS` / `STAGE2_STEPS` (default 4).

Reconstructions are written to `<output_dir>/<scene>/chunk_*/` (`result.ply`, `result.glb`,
`params.npz`). Useful env overrides: `GSO` (data root), `KAPPA` (weighting sharpness, default 8),
`DIV` (selection diversity, default 0.1); TRELLIS.2 also honours `MODEL_PATH`.

## Evaluation

`evaluate.sh` scores the reconstructions against the GSO `render_mvs_25` ground truth and reports
**CD, IoU, PSNR, SSIM, LPIPS, Image-FID, P-FID**:

```bash
bash evaluate.sh <gpu> <recon_output_dir> <objects_csv> <eval_root> [variant]
```

Point it at the `output_dir` you passed to `running_stream3d.sh`:

```bash
bash running_stream3d.sh 5 alarm /tmp/out_full
bash evaluate.sh        5 /tmp/out_full alarm /tmp/eval_full
```

It runs the two-step GSO evaluator — global Sim(3) registration + render of the 25 GT views, then
metric aggregation — and writes the metric table (CSV + `summary.json`) to
`<eval_root>/evaluation/`. Env overrides: `GSO` (GT root), `SAMPLE_POINTS` (Chamfer samples,
default 4096), `IMAGE_FID_BACKEND`, and `PFID_CKPT` (PointNet++ checkpoint; P-FID is skipped if
unset).

## Reference

```bibtex
@article{zhou2026stream3d,
  title={Stream3D: Sequential Multi-View 3D Generation via Evidential Memory},
  author={Zhou, Kaichen and Bai, Zeyang and Chang, Xinhai and Wang, Mengyu and Liang, Paul and Zhan, Fangneng},
  journal={arXiv preprint arXiv:2605.21472},
  year={2026}
}
```
