# Depth Estimation for Short-Long Focal Binocular Cameras

Official implementation of **Depth Estimation for Short-Long Focal Binocular
Cameras via LUT-Based Pseudo-3D Fusion**. DEforSLFBC combines the wide field of
view of a short-focal camera with accurate central depth cues from a long-focal
camera. It accepts one rectified `1280x768` left RGB image and one rectified
`1920x1152` right RGB image, and produces a full-frame metric depth map at the
left-image resolution.

The released inference pipeline works as follows:

1. The Depth Anything V2 monocular branch produces full-frame depth and
   quarter-resolution features.
2. The geometry-corrected IGEV-Stereo branch produces center-view disparity
   and probability features.
3. Disparity is converted to depth using `fb=146.98806411139032 px*m`.
4. The LUT-based pseudo-3D fusion network places the `640x384` stereo ROI at
   left-image coordinate `(x, y)=(320, 192)` and produces a `1280x768`
   full-frame depth map.

## Visual Overview

### Acquisition Platform

The CHU-LSStereo platform combines a short-focal camera, a long-focal camera,
and LiDAR for metric-depth supervision.

<p align="center">
  <img src="docs/assets/acquisition-platform.png" width="720" alt="CHU-LSStereo acquisition platform">
</p>

### Captured and Rectified Image Pairs

The two cameras are hardware-synchronized and their original sensor frames are
captured at the **same native image dimensions**. Their focal lengths and
fields of view differ; the short-focal image covers the wider scene, while the
long-focal image resolves the central overlapping region more densely.

The different `1280x768` left and `1920x1152` right dimensions used by this
release are produced by adaptive rectification and resampling, not by unequal
native capture sizes. Rectification aligns the long-focal view with the
corresponding crop of the short-focal image.

<p align="center">
  <img src="docs/assets/captured-image-pair.png" width="1000" alt="Raw short- and long-focal images">
</p>

<p align="center">
  <img src="docs/assets/rectified-image-pair.png" width="1000" alt="Rectified short- and long-focal images">
</p>

### Evaluation Example

Bundled sample `000030` is shown as a 2x2 grid: RGB input (top left),
semi-dense LiDAR ground truth (top right), estimated depth (bottom left), and
relative-error map (bottom right). Ground-truth pixels are evaluated over
`0 < depth < 100 m`; pixels without valid LiDAR supervision are dark gray. The
displayed estimated depth has `sky_mask.png` pixels set to zero. Both depth
maps use the fixed `0-100 m` scale (near is warm, far is cool), and relative
error is clipped at `0.25`. This frame has `38.7%` valid ground-truth coverage
and a valid-pixel AbsRel of `0.0215`.

<p align="center">
  <img src="docs/assets/evaluation-example-000030.png" width="1200" alt="Sample 000030 RGB, ground truth, estimated depth, and relative-error map">
</p>

## Repository Layout

```text
checkpoints/                 Three independent inference checkpoints
docs/assets/                 Permanent README figures
src/deforslfbc/              End-to-end inference and fusion implementation
third_party/                 Monocular and stereo model implementations
test_data/                   100 low-similarity release samples
infer.py                     Single-sample end-to-end inference
evaluate.py                  Release-set inference and metric evaluation
scripts/verify_release.py    Release integrity verifier
```

## Release Data

All collection-date directories have been removed. The selected samples from
the source groups are interleaved into a single sequential namespace from
`000000` through `000099`. Each directory contains one complete, aligned
sample:

```text
test_data/000000/
|-- left.png
|-- right.png
|-- ground_truth.png
`-- sky_mask.png
```

The inference code uses the same right-image processing as the current dataset
generator. It keeps the width at `1920`, resizes only the height from `1152` to
`384` with OpenCV bilinear interpolation, and then performs the horizontal
`phase::3` split. This produces three ordered `640x384` phase images.

Ground truth is stored as uint16 PNG and decoded with:

```text
depth_m = pixel / 65535 * 300
```

Official metrics use pixels satisfying `0 < depth < 100 m`. `sky_mask.png` is
a uint8 full-frame mask aligned with the left image. Its resolution is
`1280x768`, and its values are 0 or 255.

The 100 release samples were selected from 200 candidates. Collection type was
not restricted to outdoor scenes. The retained subset excludes directly
adjacent frames and favors scene diversity. Collection dates and source
sequence identifiers are intentionally omitted from the release layout and
manifest.

## Checkpoints

The three checkpoints remain separate and are not merged:

| File | Branch | Purpose |
|---|---|---|
| `checkpoints/mono.pth` | Monocular | Full-frame monocular depth and features |
| `checkpoints/stereo.pth` | Stereo | Central disparity and probability features |
| `checkpoints/fusion.pth` | Fusion | Full-frame LUT-based depth fusion |

The checkpoint files contain only model states, architecture configuration,
and calibration metadata required for inference. They do not contain optimizer,
scheduler, or gradient-scaler state. Exact sizes and hashes are recorded in
`checkpoints/manifest.json`.

The original monocular checkpoint exceeds the 100 MB upload limit, so it is
distributed as a standard four-volume split ZIP:

| Volume | Size (bytes) |
|---|---:|
| `checkpoints/mono.pth.z01` | 99,614,720 |
| `checkpoints/mono.pth.z02` | 99,614,720 |
| `checkpoints/mono.pth.z03` | 99,614,720 |
| `checkpoints/mono.pth.zip` | 63,589,718 |

Keep all four volumes together in `checkpoints/` and do not rename them. With
7-Zip, WinRAR, Bandizip, or another archive manager that supports split ZIP
archives, double-click `checkpoints/mono.pth.zip` and select **Extract**. The
archive manager automatically reads the three `.z01`-`.z03` volumes. Select
`checkpoints/` as the destination so that the restored file is placed at:

```text
checkpoints/mono.pth
```

The operating system's built-in ZIP viewer may not support split archives. A
compatible archive manager is therefore recommended for double-click
extraction. From a terminal, 7-Zip can restore the checkpoint directly:

```bash
7z x checkpoints/mono.pth.zip -ocheckpoints
```

If only Info-ZIP is available, first join the volumes into a temporary ZIP and
then extract it into `checkpoints/`:

```bash
cd checkpoints
zip -s 0 mono.pth.zip --out /tmp/mono.pth.full.zip
unzip /tmp/mono.pth.full.zip -d .
rm /tmp/mono.pth.full.zip
cd ..
```

Verify the reconstructed checkpoint before inference:

```bash
sha256sum checkpoints/mono.pth
# 9322c7a4eab0ae7cb8121216a406c8f5bcefed0f120e224febc518403c771c5d
```

The reconstructed `checkpoints/mono.pth` is intentionally ignored by Git. The
other checkpoints and test PNG files remain configured for Git LFS in
`.gitattributes`. Enable Git LFS before staging the release:

```bash
git lfs install
git add .gitignore .gitattributes checkpoints test_data
```

Run these commands before the general `git add` step. Each committed monocular
checkpoint volume remains below 100 MB.

## Environment

The release was verified with Python 3.11, PyTorch 2.10, and CUDA 12.8. The
recommended installation uses Conda:

```bash
conda env create -f environment.yml
conda activate deforslfbc
```

If a different CUDA build is required, first install matching versions of
`torch` and `torchvision` using the official PyTorch instructions, then run:

```bash
pip install -r requirements.txt
```

CUDA is recommended for end-to-end testing. CPU mode can be used for model-load
checks, but full inference is slow.

## Single-Sample Inference

Run inference on a bundled sample:

```bash
python infer.py \
  --sample-dir test_data/000000 \
  --output outputs/demo \
  --device cuda:0
```

Custom inputs can also be provided explicitly:

```bash
python infer.py \
  --left /path/to/left.png \
  --right /path/to/right.png \
  --output outputs/custom \
  --device cuda:0
```

The output directory contains:

- `depth_m.npy`: float32 metric depth for downstream computation.
- `depth_m.pt`: the same depth map as a PyTorch tensor.
- `depth_color.png`: a quick preview over the 0-100 m range.
- `prediction.json`: input paths, output shape and range, and checkpoint
  provenance.

## Release-Set Evaluation

Run a one-sample smoke test:

```bash
python evaluate.py --limit 1 --output outputs/smoke --device cuda:0
```

Evaluate all 100 bundled samples:

```bash
python evaluate.py --output outputs/test100 --device cuda:0
```

Add `--no-save-predictions` to retain only `metrics.json` instead of per-sample
depth outputs. The JSON report contains both aggregate and per-sample metrics.

## Release Verification

```bash
python scripts/verify_release.py
python scripts/verify_release.py --load-models --device cpu
python scripts/verify_release.py --run-inference --device cuda:0
```

The first command validates file hashes, image dimensions, sample uniqueness,
the date-free merged layout, sky masks, generated right phases, and checkpoint
contents. The second command also loads all three models strictly. The third
command performs a real end-to-end inference. Results are printed to stdout;
use `--report /path/to/report.json` only when a persistent JSON report is
needed.

## Input Constraints

The released weights are trained for fixed calibration, resolution, and ROI.
Custom inputs must use the same rectification and dimensions as the bundled
samples. The right input must be one unsplit `1920x1152` image; the inference
code performs the three-phase split. Ordinary unrectified stereo images cannot
be used directly.

## Citation

The paper was accepted by *IEEE Robotics and Automation Letters* in September
2026. The DOI, volume, issue, and page range are not yet available; update the
placeholder note below when the final bibliographic record is published.

```bibtex
@article{wang2026deforslfbc,
  title   = {Depth Estimation for Short-Long Focal Binocular Cameras via
             LUT-Based Pseudo-3D Fusion},
  author  = {Wang, Ke and Feng, Yaohui and Wu, Bingcao and Liu, Zhanwen and
             Shen, Shaojie},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Accepted September 2026; DOI and final publication details to appear}
}
```
