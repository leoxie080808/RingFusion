# ICRAE 2026 captures: 2026-11-17 session

**Presentation only. Not in the paper.** Recorded on the AGX Orin for the ICRAE 2026 talk (Nagoya,
Nov 20 to 22). Procedure: [`docs/ICRAE2026_CAPTURE.md`](../ICRAE2026_CAPTURE.md). Every slide that
uses something from this folder carries a "Not in the paper" tag.

This folder is the light copy. The full session, 51 GB including the ROS bags, raw frame folders
and both map recordings, is on the Seagate drive under
`RingFusion_ICRAE2026_capture_2026-11-17/icrae_rerun/` (5,885 files, verified byte for byte).

## Conditions

| | |
|---|---|
| Build | Not a clean freeze: local commits and uncommitted edits beyond `3e592d8`. Under `ros2_ws`, only `robot_geometry.yaml` differs. See `logs/freeze.txt` |
| Engines | `student_v4_heldout_fp16`, `residual_v7_fov73_fp16` (sha256 values in `logs/freeze.txt`) |
| Settings | Arbitration on, ridge 0, ROI on (`logs/perception_params.yaml`) |
| Power | MAXN, `jetson_clocks` applied, leroi stack stopped |
| Rate | `/depth` **9.57 Hz**, `/cloud` 9.56, `/tof` 16.0, `/image` 30.2; depth age median 140 ms (`logs/rate_check.json`) |

## What's here

| Shot | Slide | Files | Use |
|---|---|---|---|
| C1, σ clip | 20 | `clips/c1_take{1,2,3}.mp4`, `c1_take1_static.mp4` | **Take 2** is calmest: far field at about 8 to 23 s. Take 3 swings between near and far twice. `static` is a backup still |
| C1 as the ceiling | 24 | `clips/c1_take2.mp4` (about 6.9 s and 20.9 s), `clips/c3_take1.mp4` (about 7 s) | Far wall inside the dToF view reads about 4 m; the same wall outside it reads about 1.5 m. Replaces C2 |
| C3, demo drive | 8 | `clips/c3_take{1,2}.mp4` | Take 1 is clean. **Take 2: avoid about 10 to 16 s** (another robot at 0.06 m, panels go empty) |
| C4, point-cloud source | 9 | `clips/c4_cloud.mp4`, `c4_cloud_meta.json` | Left: colour. Right: depth as luma over 0.15 to 6 m, with 0 = no data. 12 fps. No σ saved |
| Z1, ZJU-L5 | 19 | `zjul5/*_sheet.png` and per-panel PNGs | See below |
| H1, held-out frame | 13 | `heldout/*_sheet.png` | `gap_000265`: copy-nearest 22.1 cm against ours 2.5 cm. `median_000054`: 3.2 against 3.1 cm |
| M1, map frames | 25 | Seagate only (`map2/`, 194 MB) | **Use `map2`** (the loop closed: 402 frames, 0 dropped, 1 unpaired). `map1` did not close and is a backup only. Timing in `logs/map*_frames.csv` |
| Check strips | — | `stills/` | Six moments per take, plus start-position previews, the C4 3D test and the σ-filter test |

Not captured: C2 (replaced by C1 and C3, as above), F1, and P1 (photos to come).

## Z1 results

Same ViT-S weights as DEPTHOR, ZJU-L5 test, 517 of 527 images scored:

| | pooled AbsRel | pooled RMSE | δ1 | per-image median AbsRel |
|---|---|---|---|---|
| Ours (closed form, no learned fusion) | **0.094** (paper 0.094) | 1.056 | 0.902 | 0.073 |
| DEPTHOR-Small (+6 M learned completion) | **0.079** (paper 0.079) | 0.867 | 0.925 | 0.067 |

Ours is better on **226 of 517** images (`zjul5/per_image.json`).

| Pick | Image | AbsRel ours / DEPTHOR | GT beyond 5 m |
|---|---|---|---|
| match | 252 `classroom` | 0.027 / 0.027 | 0 % |
| median | 391 `library1` | 0.073 / 0.028 | 0 % |
| far | 386 `library1` | 0.210 / 0.227 | 40 % |

Caption it as "competitive without learned fusion", not "better than DEPTHOR".

⚠️ **`zjul5_mislabelled/` is the first run. Don't use its captions.** `zjul5_qual.py` picks by position in
the list of scored images, then renders that position as a dataset index. Once an image is skipped,
the two disagree, so those three images are real, and their AbsRel is correct, but they aren't the
match, median and far images. `zjul5/` was re-run through `logs/z1_fixed_picks.py`, which maps
each position back to its dataset index without editing the repo script. The pooled numbers are the
same in both runs.

⚠️ **RMSE is worth checking before the paper uses it.** With the same pooled metric code, DEPTHOR-Small
scores 0.867 against our 1.056, about 1.2×. The README's "RMSE 2.8× better" compares our 1.056
with DEPTHOR's 0.371, a figure from DEPTHOR's own evaluation, which may average RMSE image by image.
This hasn't been verified, and nothing in the paper has been changed.

## Caveats for the slides

- `record_clip.py` prints the scale as "0.1–5.0 m" (one decimal). The real scale is 0.15 to 5 m.
  Fix it in the caption.
- C1 and C3: "centre zones" and "same zones" can disagree mid-turn. The depth map is about 140 ms older
  than the newest ToF frame. Pick moments where the two agree, or crop that line.
- C1 takes are 35 s (5 s lead-in plus 30 s of driving). M1 was recorded with `--max-skew 0.12`.
- About 13 % of M1 frames have no σ file (σ arrived late). They're scattered single frames.
- A hard σ cut keeps only 13 to 17 % of points, mostly near floor (`stills/sigma_filter_test.png`). Fade
  by σ instead of cutting.
- `logs/loop_watch.py` stopped map2 when the view matched the start frame (mean difference under 9
  for 6 frames, after 30 s).
