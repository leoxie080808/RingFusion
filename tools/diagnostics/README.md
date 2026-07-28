# Diagnostics & capture tools

On-robot measurement tools. These found the ToF↔camera projection bug and produced every
number in the 2026-07-28 sections of [`ros2_ws/README.md`](../../ros2_ws/README.md).

They are in the repo because they were originally written to a scratch directory and lost to
a reboot **twice** — re-deriving them cost more than keeping them.

All of them need a running pipeline and a sourced overlay:

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
CAL=$(ros2 pkg prefix ringfusion_bringup)/share/ringfusion_bringup/config/calibration.yaml
```

## Evaluating a residual engine

| Tool | Answers |
|---|---|
| `moving_ab.py` | **The decisive one.** Runs two residual engines off one backbone pass per frame while the robot drives, so A, the anchors and the affine fit are byte-identical between arms. Reports far-field blowup, clamp hits, anchor MAE and frame-to-frame jitter. |
| `ab_residual.py` | Same idea on a still scene, plus banding power at a given spatial period. |
| `sigma_cal.py` | Is `/depth_var` meaningful? Coverage at ±1σ/±2σ against live ToF, and `corr(σ, \|error\|)` — the metric that catches "confident in exactly the wrong place". Takes several engines at once. |
| `stream_test.py` | Temporal stability of the deployed `/depth` over N seconds. Note a **stationary** scene makes this trivially easy — 0.66 % frame-to-frame vs ~9 % driving — so it is a baseline, not a test. |

```bash
python3 tools/diagnostics/moving_ab.py --calib $CAL \
  --backbone-engine $HOME/RingFusion/student_v3_fp16.engine \
  --engine-v1 $HOME/RingFusion/residual_v2_fp16.engine \
  --engine-v2 $HOME/RingFusion/residual_v3_last_fp16.engine \
  --secs 30 --out /tmp/moving.json --save-prefix /tmp/mv
```

## Diagnosing geometry

Run these before blaming a network — a projection error looks exactly like a bad model.

| Tool | Answers |
|---|---|
| `orient_full.py` | **Start here.** Searches every ToF grid orientation × FOV assignment, scoring each by how well backbone disparity tracks true inverse depth. Writes a before/after anchor overlay. If the deployed config is not the winner, the calibration is wrong. |
| `proj_sweep.py` | Continuous sweep of FOV and the ToF→camera offset. Use when `orient_full` says the config is right but you suspect a smaller error. |
| `bias_diag.py` | Is a depth error scale, offset, or geometric? Regresses depth against ToF, splits by distance and by image row, and fits an oracle affine to bound what re-fitting could ever recover. |
| `centre_diag.py` | Reconciles the ToF panel's centre reading with the depth panel's. They sample different solid angles ~50 px apart; this measures the matched-region error and prints the ROI to use. |
| `teacher_vs_student.py` | Is the backbone the bottleneck? Scores the distilled student and the Depth Anything V2 teacher against the same anchors. **If both score alike and both score poorly, suspect the projection, not the model** — that is how the mirror bug hid. |

## Capture

| Tool | Produces |
|---|---|
| `record_clip.py` | The 4-panel clip (camera │ ToF │ depth │ top-down). Colour scales are fixed, not per-frame, so a colour means the same distance throughout. `--roi` keeps the two centre readouts sampling the same patch of world — re-measure it with `centre_diag.py` after any geometry change. |
| `record_cloudseq.py` | Colour+depth pair for the 3D viewer, straight off `/depth` at full float32 precision. **Prefer this** over `extract_from_clip.py`. |
| `extract_from_clip.py` | Retrofits an already-recorded clip into the same format by inverting the TURBO colourisation (~8 mm). Only for reusing a drive you cannot repeat. |
| `grab_cloud.py` | One colour point cloud as quantised binary. |

## Page building

`build_page_v3.py` renders the analytics page; `viewer_snippet.py` is the self-contained
WebGL2 point-cloud viewer it embeds (no external libraries — the artifact CSP blocks CDNs).

## Gotchas these tools encode

- **Depth-as-luma needs sentinel headroom.** Valid levels start at 16, not 1. Video
  compression pushes near-range values down, and with `0` = no-data adjacent to valid levels
  it destroyed 23 % of points (84 % → 61 % valid).
- **`int32` for colour distances.** Inverting a colormap by nearest-LUT with `int16`
  silently overflows at 255² and produces garbage that looks like a real result.
- **Check the room is lit.** A dark room yields black frames, the affine fit goes degenerate,
  and Network A reports 10 000 m maxima. `sigma_cal.py` and `moving_ab.py` print mean frame
  brightness for this reason — below ~15 the run is worthless.
- **Verify motion actually happened.** Compare consecutive camera frames: ~1.8 grey levels
  means stationary, ~10 means driving.
