# Demo captures

On-robot captures backing the numbers in [`ros2_ws/README.md`](../../ros2_ws/README.md).
Everything here was recorded live on the AGX Orin with `jetson_clocks` on.

> **Dated 2026-07-28 or later = valid.** Anything earlier predates the ToF↔camera projection
> fix (mirrored grid + swapped FOV) and was measured through a broken correspondence. Those
> files are kept for provenance, not as evidence.

## Rendered analytics page

| File | What |
|---|---|
| `network_b_v1_vs_v2.html` | Self-contained analytics page: interactive 3D depth-cloud flythrough, 4-panel clip, calibration-fix and Network B v3 results. Open directly in a browser — no server, no external assets. |
| `pipeline_view.html` | Earlier page (pre-fix). Superseded. |

Rebuild with `python3 tools/diagnostics/build_page_v3.py`.

## Clips — `clips/`

All from the same 34 s drive on 2026-07-28 unless noted.

| File | What |
|---|---|
| `drive_4panel.mp4` | Camera │ ToF │ fused depth │ top-down, the headline clip |
| `drive_camera.mp4` | Rectified camera only |
| `drive_tof_heatmap.mp4` | 32×32 ToF only (un-mirrored) |
| `drive_fused_depth.mp4` | Fused metric depth only |
| `drive_topdown.mp4` | Bird's-eye obstacle map only |
| `drive_cloudseq_colordepth.mp4` + `drive_cloudseq_meta.json` | Colour \| depth-as-luma pair that drives the page's 3D viewer. Depth is 8-bit over 0.15–4.0 m with valid levels starting at 16 (the gap keeps codec ringing off the 0 = no-data sentinel). `meta.json` carries the intrinsics needed to unproject it. |
| `drive_frames_raw.tar.gz` | Source frames, 46 MB. **Regenerable from the mp4s — consider not committing.** |
| `static_4panel.mp4` | A stationary capture; useful as the "no motion" baseline (0.66 % frame-to-frame vs 9 % driving) |
| `drive1_3panel_prefix.mp4` | First drive, **pre-calibration-fix**. Provenance only. |

## Calibration fix — `calibration_fix/`

Visual proof of the projection bug. ToF anchors splatted onto the camera image, coloured by
measured distance.

| File | What |
|---|---|
| `anchors_deployed_broken.png` | Old config (as-is, 61×45): anchors scrambled, ρ 0.737 |
| `anchors_fixed.png` | Fixed (mirrored, 45×61): far/red on the back wall, near/blue on the floor, clean top-to-bottom gradient, ρ 0.914 |
| `scene.png` | The same frame with no overlay |

## Network B v3 under motion — `v3_moving/`

Three moments from the drive, each rendered by all three configurations on the **same** input
frame, so any difference is the residual alone.

`t{0,1,2}_cam.png` · `_closedform.png` (no residual) · `_residual_v2.png` · `_residual_v3.png`

## Stats

| File | What |
|---|---|
| `moving_ab_v3.json` | The headline moving run: 105 frames, closed-form vs v2 vs v3, per-frame + summary |
| `ab_v1_v3.json` | Static A/B, v1 vs v3 |
| `stream_v2.json` | 5 s stationary hold (drift / flicker baseline) |
| `moving.json` | First moving run, **pre-fix**. Provenance only. |
| `stats.json`, `scene2_stats.json` | 2026-07-25 single-frame captures, **pre-fix** |

## Reproducing any of this

The capture and analysis tools live in [`tools/diagnostics/`](../../tools/diagnostics/) — see
the docstring at the top of each. They lived in a scratch directory for most of their life and
were lost to a reboot twice; they are in the repo now for that reason.
