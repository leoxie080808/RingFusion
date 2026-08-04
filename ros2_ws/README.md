# RingFusion ROS 2 Workspace

ROS 2 Humble workspace for the RingFusion project (ToF hub + camera driver, perception, bringup, and message definitions).

## Project status (handoff)

Snapshot for anyone picking this up, current as of **2026-08-04**.

**The whole stack runs live on the Orin with both real networks** — no mocks in the loop.
Recommended launch:

```bash
ros2 launch ringfusion_bringup single_module.launch.py port:=/dev/ttyACM1 \
    backbone_engine:=$HOME/RingFusion/student_v4_heldout_fp16.engine \
    residual_engine:=$HOME/RingFusion/residual_v7_fov73_fp16.engine
```

`residual_v7_fov73` is the recommended residual (trained 2026-08-04 at the corrected
`fov_h`, 110 epochs; supersedes `residual_v6_fov73`, which stopped at 40 epochs mid-descent).
`residual_v4_last` is mismatched to the current calibration — see
[Network B v7](#network-b-v7--v6-was-stopped-mid-descent).

> **The backbone line changed too.** This example previously read
> `student_v3_fp16.engine`, which contradicted the recommended configuration further down
> *and* mismatched the backbone Network B was actually trained against
> (`--student-ckpt runs/student_v4_heldout/student_best.pth`). Copy-pasting the old line
> produced a genuine train/deploy mismatch. `student_v4_heldout` is what has been running.

Stage 7c blend and the
ROI/σ stages are on by default and cost 19.8 ms together at full resolution *offline* — but
measured on the robot they first added 70.7 ms and dropped the deployed rate to 7.2 Hz, below
the ToF's own 8.3 Hz. Two GPU rewrites later the default configuration measures **10.3 Hz ✅**.
See [the deployed rate, measured](#the-deployed-rate-measured-2026-07-30).
`blend:=false roi_enable:=false` restores the pre-2026-07-30 behaviour and **13.3 Hz**.

*(10.29 Hz on the current `residual_v7` build, 2026-08-04. The **10.4 Hz** figures elsewhere in
this file are the 2026-07-30 session and are left as measured on that date; the 1 % spread
across v4/v6/v7 is run-to-run noise — same architecture, same engine size.)*

Two things changed on 2026-07-28 and they dominate everything measured before that date.

**1. A calibration bug was found and fixed.** The ToF grid was **horizontally mirrored**
relative to the camera *and* `fov_h`/`fov_v` were **swapped**. Every anchor had been landing
on the wrong pixel, which corrupted the affine fit and, through it, every number in this
file predating the fix. Correcting it moved the backbone's agreement with real ToF from
**ρ 0.737 → 0.917** and best-case depth MAE from **0.319 → 0.251 m**. See
[ToF↔camera projection fix](#tofcamera-projection-fix-found--applied-2026-07-28).

**2. Network B was retrained on the corrected geometry** (`residual_v3`, `--struct-weight
0.15`):

| | closed-form | v1 | v2 | **v3** |
|---|---|---|---|---|
| anchor MAE, moving ***(in-sample — see below)*** | 0.294 m | — | 0.247 m | **0.199 m** |
| corr(σ, \|error\|) | — | 0.490 | 0.913 | **0.943** |
| frames at the 20 m clamp | 0 | 47/102 *(pre-fix)* | 0 | **0** |

> ### ⚠ These numbers are IN-SAMPLE. Corrected 2026-07-28.
>
> `moving_ab.py` and `sigma_cal.py` both call `solve_robust` on the in-bounds anchor set
> and then score at **that same set** (`ys, xs = np.nonzero(am > 0)` is the splat of the
> anchors that determined the fit). So the row above measures how well the fit reproduces
> the points it was fitted to — not generalisation. Training *does* hold out 25%
> (`anchoring_bridge.build_real_supervision`); the on-robot harnesses never did.
>
> The tell: under that protocol a nearest-neighbour lookup scores **exactly 0.000 m**,
> because it predicts each zone from itself. Any metric a trivial method aces is not
> measuring depth estimation.
>
> The claim "better than the pure closed-form path" also does not survive: it compared an
> **unclamped** closed-form output against a residual path that caps at `MAX_DEPTH_M = 20`.
> See [Benchmarks](#benchmarks-vs-trivial-baselines).

The uncertainty channel — previously saturated and *most confident where it was most
wrong* — now tracks actual error at **ρ 0.943**, and σ is small at the anchors (0.078 m)
while staying large where the net extrapolates. That is the correct shape.

**What is still not trustworthy.** The ToF covers **12.4% of the frame**, so the other
87.6% of every depth map is monocular extrapolation carrying the affine fit — a geometry
limit, not a bug, but it bounds what can be believed. Every number in this file is also
**scored against the ToF we anchor to**, a closed loop that cannot detect a ToF bias and
says nothing about the 87.6%. Design details live in
`RingFusion_technical_reference_updateP2.md`.

> ⚠ **Every benchmark below predates the 2026-08-03 ToF field-of-view correction** — see
> [ToF field of view, measured against tape](#tof-field-of-view-measured-against-tape).
> The horizontal FOV was wrong by a factor of ~1.6, so anchors were paired with the wrong
> disparities in every fit. The numbers below are not wrong *as recorded*, but they describe
> a rig that no longer exists. See [what needs re-running](#what-needs-re-running).

## Reading the benchmarks — what each one asks, and what it answered

Seven separate evaluations appear below. They are **not interchangeable**: each asks a
different question, and several disagree with each other *because* of that. This table is the
key to the rest.

| evaluation | the question it answers | judged against | what it found |
|---|---|---|---|
| **`insample`** | How well does the fit reproduce the very points it was fitted to? | the same ToF zones used for the fit | **Not an evaluation.** A nearest-neighbour lookup scores 0.000 m. Every number published before 2026-07-28 used this. |
| **`random` hold-out** | Can it interpolate *between* adjacent ToF zones? | 25 % of zones held out at random | **Raw ToF wins** (0.010 vs 0.044 medAE). 99.6 % of these targets have a real measurement ~1.7 cm away, so this is a lookup task, not depth estimation. |
| **`center` hold-out** | Can it extrapolate *away* from the ToF, which is what deployment requires? | zones outside a central 16×16 anchor island | **The architecture's case.** Nearest-zone degrades 9× with distance; the camera path stays flat and wins 3–5× past 10°. |
| **ZJU-L5** | Is the depth actually *right*, judged by a different device? | dense RealSense GT — **not our ToF** | **The only open-loop result.** Competitive on Rel/δ₁ with published methods; **2.7× worse on RMSE** — a far-field tail, not a typical-pixel problem. |
| **DEPTHOR head-to-head** | How do we compare to published SOTA on identical data, identical metric code, identical regions? | dense RealSense GT | **4–8 % behind DEPTHOR-Small on Rel, uniformly across regions**, with *zero learned fusion* against their 6 M head. |
| **Held-out re-distill** | Were the absolute numbers inflated because Network A trained on the eval set? | held-out ToF zones, on 200 frames Network A never saw | **Yes, by ~5 %.** Rankings unchanged. Numbers now clean. |
| **Timing** | Does it run in real time, and how does that compare fairly? | wall clock on this Orin | **10.4 Hz measured on the robot**, 14.1 Hz for `pipeline.run()` alone, both at 1640×1232 with every stage on; **3.1× faster than DEPTHOR-Small** at their resolution, our whole pipeline against their network alone. |
| **Blend A/B, driving** | Does Stage 7c help on a *moving* robot, or only on saved frames? | held-out ToF zones, 600 frames | **5.4 % better MAE against 5.8 % predicted offline.** It improves the tail, not the median. No added flicker. |
| **Tape ground truth** | Is the **ToF itself** right? | a tape measure | ✅ **Pilot done 2026-08-03, and it found a calibration error.** The ToF is accurate (1 cm) but its horizontal field of view was wrong by 1.6×, so readings were attributed to the wrong pixels. Correcting it cut pipeline MAE **0.427 → 0.192 m** and the median error **9.8×**. See below. |

### ToF field of view, measured against tape

The first open-loop measurement in this project immediately found something no
ToF-scored benchmark could see: **`fov_h` was 45.0° and the real value is ~73.5°.**

Seven markers were placed spanning −43° to +48° off the optical axis and hand-measured with
a tape. Scoring by *direction lookup* — for each marker, ask which ToF zone points that way
and compare that zone's reading to the tape — gives:

| | fov_h 45.0° | **fov_h 73.5°** |
|---|---|---|
| MAE, ToF zone vs tape | 0.472 m | **0.010 m** |
| markers the cone reaches | 3 of 7 | 5 of 7 |

The clearest single case is the marker on the red pole at +8.1°, tape **2.03 m**. At 45° the
direction lookup lands on zone (10,21), which reads **0.64 m** — that is the water bottle, a
different object. At 73.5° it lands on (10,19), reading **2.03 m**.

**End-to-end effect.** Same scene, same markers, same tape numbers, only the config changed:

| # | on | tape | before | after |
|---|---|---|---|---|
| 1 | wall, −43° | 1.80 | −0.43 | −0.33 |
| 2 | yellow discs, −33° | 0.76 | +0.75 | **−0.01** |
| 3 | white barrier, −0.5° | 1.66 | −0.01 | −0.01 |
| 4 | red pole, +8° | 2.03 | +0.75 | +0.78 |
| 5 | water bottle, +12° | 0.63 | +0.79 | **−0.01** |
| 6 | blue block, +36° | 0.72 | +0.10 | **−0.02** |
| 7 | red stool, +48° | 0.89 | −0.16 | −0.19 |

| | before | after |
|---|---|---|
| **all 7** | 0.427 m | **0.193 m** |
| **inside the ToF cone** | 0.479 m | **0.166 m** (2.9×) |
| outside the cone | 0.297 m | 0.260 m |

**How well pinned:** anything in **66–82°** scores ~1 cm, because the outer markers span
several ToF columns. Above 84° is excluded. 73.5° sits mid-band. `fov_v` is **not**
independently confirmed — every marker sat within −10..+9° vertically.

**What the correction does not fix.** #4 (the red pole) is unchanged at +0.78 m. Its own zone
reads 2.03 m correctly, but neighbouring zones see the banner 2.78 m behind and the mix wins.
That is a thin-object partial-fill failure, not a calibration one. #1 and #7 sit outside the
cone even at 73.5° and remain monocular extrapolation.

**σ did not improve.** At #3 it reports 0.649 m for a 7 mm error; at #1, 0.094 m for a 0.335 m
error. Still uncorrelated with actual error — see the σ discussion below.

Data: [`docs/demo/benchmarks/live3_pilot_2026-08-03.json`](../docs/demo/benchmarks/live3_pilot_2026-08-03.json).
Pilot scale — 14 tape points over two sessions, 0.39–2.03 m, no far field. Directional, not
a certified number.

### What needs re-running

The FOV controls **where each ToF zone lands in the image**. Anything downstream of that is
invalidated; anything that never touches it is fine. The raw logs are all still good — no
data needs re-collecting, only re-processing.

**Tier 1 — re-score offline. Cheap, no retraining, do first.**

Every evaluation scored at ToF zone locations, because the zones moved:

| file | what it is |
|---|---|
| `baselines.json`, `baselines_valsplit.json` | B1–B5 baseline table |
| `score_v3_last`, `score_v4_*`, `score_v5_*` | residual scoring, all protocols |
| `heldout_v4.json`, `heldout_v4_bothclean.json` | the clean re-distill numbers |
| `ceiling_ours.json` | ceiling analysis |

The `insample` / `random` / `center` hold-out protocols all pick zones and score there, so
every number in [Reading the benchmarks](#reading-the-benchmarks--what-each-one-asks-and-what-it-answered)
that judges "against ToF zones" moves.

**Tier 2 — retrain. This is the expensive one.**

**Network B (the residual) must be re-trained.** [`train_residual.py`](../training/train_residual.py)
builds its supervision through `anchoring_bridge.build_real_supervision`, which calls
`geo.zone_directions(cols, rows, calib['fov_h'], calib['fov_v'])` — so `anchor_depth` and
`anchor_mask`, two of its four inputs, were generated at 45°. It now receives anchors placed
at 73.5° at inference. That is a **train/deploy mismatch**, not just a stale number, and it
applies to the variance head too — which may be part of why σ is uninformative.

> Note: `anchoring_bridge.default_calib` still defaults to `tof_fov=(61.0, 45.0)`, the old
> reversed guess. It only affects synthetic/non-`--real` runs, but it should be updated.

**Tier 3 — re-measure on the robot.**

- **Blend A/B** (`blend_ab_live.json`) — the blend acts near anchors, and the anchors moved.
- **LIVE-3 / LIVE-4 proper.** Now worth doing: the pilot was measuring a rig with a broken
  extrinsic. A full session should wait for the marker redesign (see below) so the far field
  is reachable.

**Tier 4 — unaffected. Do not re-run.**

- **ZJU-L5** (`zjul5_*.json`) and the **DEPTHOR head-to-head** (`depthor_*.json`) — external
  datasets with their own rigs and dense RealSense GT. `calibration.yaml` never enters.
  These stay valid, which matters: they are the comparisons against published work.
- **Camera intrinsics** — a separate calibration, untouched.
- **Timing / throughput** (`profile_node_*`, `rate_live_*`, `timing_*`) — stage structure is
  unchanged. The ToF box is larger so blend and ROI cover more area; worth one cheap
  confirmation run, but no result is expected to move much.

**Not a benchmark, but blocking a real LIVE-3:** the printed markers are unreadable past
~2 m. The X stroke is 1.43 mm, sub-pixel beyond 0.5 m, and the border is 0.25 mm, invisible
everywhere — so four-corner clicking never works and the 3–4.5 m band cannot be measured at
all. `make_markers.py` needs a pattern whose features scale with the marker (a 2×2
checkerboard) before the far field is reachable.

#### Status as of 2026-08-04

| | item | state |
|---|---|---|
| ✅ | Tier 1 — offline re-score at the corrected FOV | done |
| ✅ | Tier 2 — Network B retrained (`residual_v6`, then `residual_v7`) | done |
| ✅ | Tape re-measure, 7 markers, **v6** | done — MAE 0.427 → 0.192 m |
| ✅ | Rate re-confirmation | done — 10.29 Hz, unchanged |
| ✅ | Resource utilisation (CPU/GPU/RAM/power) | done — first ever measurement |
| ✅ | FP16 engine parity vs checkpoint | done — σ within 0.42 % |
| ⬜ | **Tape re-measure on v7** | **needs lights** — the key open number |
| ⬜ | **`fov_v` verification** | **needs lights** — highest priority; see below |
| ⬜ | Blend A/B re-run | needs lights **+ driving** |
| ⬜ | LIVE-4 σ evaluation, n > 7 | needs lights |
| ⬜ | 4-panel demo clip re-record | needs lights — `live2_4panel.mp4` still shows the 45° cone |
| ⬜ | Marker redesign (checkerboard) | blocks the 3–4.5 m band |
| ⬜ | EMC / memory-bandwidth measurement | this `tegrastats` build does not emit `EMC_FREQ` |
| ⬜ | `anchoring_bridge.default_calib` still `(61.0, 45.0)` | synthetic runs only, but stale |
| ⬜ | `B3_medscale` MAE 64.5 m; `B2_bilinear` NaN under `center` | pre-existing, appears in baseline tables |

> **`fov_v` is the one that could invalidate the others.** It has never been checked against
> tape: every marker in the 2026-08-03 session sat within −10.9°…+9.4° vertically, out of a
> claimed 60.5°, so there is almost no leverage on the vertical angle. If `fov_v` is also
> wrong, the v7 tape re-measure and the blend A/B both have to be redone afterwards — so it
> should go **first** once the lights are on. It needs markers placed high and low in the
> frame rather than the middle third, plus the camera height above the floor (which makes the
> floor itself a free dense vertical target).

### Re-run results — where the correction actually landed

Done 2026-08-03/04. Every config was run **twice with everything else identical**, so each
delta isolates one change. 1,234 recorded frames, no new data collected.

**Stage 1 — calibration only** (`fov_h` 45.0 → 73.5, residual untouched):

| method | before | after | |
|---|---|---|---|
| **B4 affine** (2-param fit, no learning) | 0.068 m | **0.055 m** | −19 % |
| B1 nearest-zone | 0.010 m | 0.018 m | worse |
| B5 ringfusion (`center`) | 0.047 m | 0.055 m | worse |

The geometry improved and everything else got worse, for two separate reasons. **B4 improves**
because anchors now span a wider field, so the 2-parameter fit is better conditioned.
**B1/B2/B6 degrade** because the same 1,024 zones now cover 12.4 % of the frame instead of
7 % — anchors are *sparser per unit image area*, so local interpolation has a harder job.
That is a harder task, not a regression. **B5 degrades** because Network B was trained against
anchors at the old projection: a train/deploy mismatch.

**Stage 2 — retrain Network B** on the re-projected supervision (`residual_v6_fov73`, same
architecture at 463,971 params, same hyperparameters, 40 epochs):

| medAE, 1234 frames | before | after |
|---|---|---|
| B4 affine, `random` | 0.068 m | **0.055 m** |
| **B5 ringfusion, `random`** | **0.101 m** | **0.050 m** |
| B5 ringfusion, `center` | 0.047 m | **0.045 m** |
| B6 blend, `center` | 0.043 m | **0.042 m** |

**The finding that matters is not the size of the gain — it is a sign flip.** Network B's whole
justification is that B5 beats B4. On the `random` protocol it did not:

```
BEFORE:  B5 0.101  vs  B4 0.068   ->  the 0.46M network HURT by 0.033 m
AFTER :  B5 0.050  vs  B4 0.055   ->  it HELPS by 0.005 m
```

A 463,971-parameter refiner was making depth **48 % worse** than the 2-parameter closed form
it exists to improve, on the interpolation regime, and shipped that way. It had learned to
correct a fit against supervision that did not line up with the image. On `center` — the
extrapolation protocol that reflects deployment — B5 always helped, so the architecture's
core claim was never wrong; it was being undermined near the anchors.

Training also improved on its own terms: best val NLL **−0.1183** vs v4's −0.1055, and it was
**still improving at epoch 39** where v4 had plateaued — 40 epochs may now be under-trained.
*(Confirmed 2026-08-04: it was. See [Network B v7](#network-b-v7--v6-was-stopped-mid-descent).)*

> **These are all ToF-scored, and the ToF projection is exactly what changed** — anchors and
> evaluation targets moved together, so this is a closed loop and cannot certify the
> improvement. The non-circular evidence is the tape: **0.427 → 0.193 m**, measured before
> the retrain.

Files: `baselines.json`, `baselines_valsplit.json`, `heldout_v6.json` hold the deployed
configuration. `*_pre_fov_fix.json` are the originals; `*_fov73_residualv4.json` are the
intermediate stage-1 runs. Training log: `train_v6_fov73.log`.

**RMSE moves the same way**, which matters because it is tail-sensitive — a gain that only
improved the median while growing outliers would show up here as a regression:

| | before | after | |
|---|---|---|---|
| **tape, 7 markers** (non-circular) | 0.530 m | **0.321 m** | −39 % |
| B5 ringfusion, `random` | 0.895 m | **0.413 m** | −54 % |
| B5 ringfusion, `center` | 0.503 m | **0.451 m** | −10 % |
| B6 blend, `center` | 0.486 m | **0.439 m** | −10 % |

Max error is the exception: 0.788 → 0.774 m, essentially unchanged, because it is marker #4
(the thin pole) which neither fix addresses.

#### Deployed system vs tape — the headline table

Same seven markers, same tape numbers, scene never moved. Old = `fov_h 45.0 +
residual_v4_last`. New = `fov_h 73.5 + residual_v6_fov73`.

> **This table is a v6 measurement.** `residual_v7_fov73` now supersedes v6 but the markers have
> **not** been re-measured against it — that needs the lights on. v7's offline gain is
> concentrated in extrapolation, which is where markers #4 and #7 fail, so this is the table
> most likely to move. Treat the numbers below as the current *confirmed* ground-truth result
> and v7's tape performance as unknown.

| # | marker | **measured (tape)** | old system | old err | **new system** | **new err** | change |
|---|---|---|---|---|---|---|---|
| 1 | leftmost wall, −43° | 1.80 m | 1.37 m | −0.43 | **1.60 m** | **−0.20** | −0.23 ✅ |
| 2 | yellow discs, −33° | 0.76 m | 1.51 m | +0.75 | **0.75 m** | **−0.01** | −0.74 ✅ |
| 3 | white barrier, −0.5° | 1.66 m | 1.65 m | −0.01 | 1.64 m | −0.02 | +0.01 — |
| 4 | red pole, +8° | 2.03 m | 2.78 m | +0.75 | 2.80 m | +0.77 | +0.03 ❌ |
| 5 | water bottle, +12° | 0.63 m | 1.42 m | +0.79 | **0.61 m** | **−0.02** | −0.77 ✅ |
| 6 | blue block, +36° | 0.72 m | 0.82 m | +0.10 | **0.68 m** | **−0.04** | −0.06 ✅ |
| 7 | red stool, +48° | 0.89 m | 0.73 m | −0.16 | 1.17 m | +0.28 | +0.12 ❌ |

| | old | new | |
|---|---|---|---|
| **MAE** | 0.427 m | **0.192 m** | −2.2× |
| **median \|err\|** | 0.431 m | **0.044 m** | **−9.8×** |
| **RMSE** | 0.530 m | **0.321 m** | −1.7× |
| max \|err\| | 0.788 m | 0.774 m | unchanged |

Four of seven markers went from 0.10–0.79 m of error to **within 4 cm**. The median moves
~10× and the mean only 2.2×, because two markers still carry large errors: **#4** (thin pole,
partial fill — untouched by either fix) and **#7** (outside the cone at +48°, unsupported
extrapolation, and it got worse). Max error is unchanged for the same reason.

> **Almost all of this came from the calibration, not the retrain.** Calibration alone gave
> 0.427 → 0.193 m; adding the retrained Network B gave 0.193 → **0.192 m**, i.e. nothing.
> The retrain is a large win on the ToF-scored benchmarks (B5 medAE halved) and ~zero against
> independent ground truth. That gap is the entire argument for tape measurement.

**σ became informative for the first time.** `corr(|error|, σ)` rose from **+0.14 to +0.755**
with the retrain. Still uncalibrated — 19× too wide at #3, 3× too narrow at #7 — but the
*ordering* is now broadly right, which it never was. n=7, so this is a lead for LIVE-4 rather
than a result.

**Throughput is unaffected — measured, not assumed** (`rate_live_fov73_v6.json`):

| | before | after |
|---|---|---|
| `/depth` | 10.37 Hz | **10.31 Hz** |
| `/cloud` | 10.37 Hz | 10.28 Hz |
| latency | 128.5 ms | 132.9 ms |

Within noise. The residual is the same architecture and engine size, and the wider footprint
puts more anchors in frame without changing the stage structure. **The timing session does not
need re-running.** The blend A/B *does* — it acts near anchors, the anchors moved, and there
are now more of them in frame.

### Network B v7 — v6 was stopped mid-descent

The v6 note above flagged that 40 epochs "may now be under-trained." It was. `residual_v7_fov73`
is the same 463,971-parameter architecture, same data, same hyperparameters — **only the epoch
budget changed**, from 40 to 110.

`train_residual.py` gained `--patience` / `--min-delta` so a run can stop itself when validation
stops moving. The best checkpoint always tracks the true minimum; only an improvement larger
than `--min-delta` resets the patience counter, so a run creeping along at 1e-6 per epoch still
terminates. Launched with `--epochs 200 --patience 8 --min-delta 1e-4`, **patience never fired** —
v7 was still setting new bests when it was stopped by hand at epoch 110.

| | v6 | **v7** |
|---|---|---|
| epochs run | 40 | **110** (stopped by hand; patience 8 never fired) |
| best epoch | 39 (the last one) | **108** |
| best val NLL | −0.1183 | **−0.2328** |
| val coverage @1σ | — | 0.71 (target 0.683) |

v6's best landed on its *final* epoch — the signature of a run cut off rather than converged.
v7 tracked v6 to within 0.012 through the whole overlap and only diverged after epoch 39, which
confirms the two runs are the same experiment at different lengths.

**Offline scoring, v7 vs v6** — identical script, stems, and calibration; the only variable is
the residual engine:

| split | protocol | method | v6 MAE | **v7 MAE** | Δ | v6 RMSE | **v7 RMSE** | δ<1.25 |
|---|---|---|---|---|---|---|---|---|
| held-out (200) | `random` | B5 ringfusion | 0.1906 | **0.1789** | −6.1 % | 0.4296 | **0.4087** | .777→.798 |
| held-out (200) | `random` | B6 blend | 0.0605 | **0.0591** | −2.3 % | 0.2151 | **0.2089** | .958→.958 |
| held-out (200) | **`center`** | **B5 ringfusion** | 0.2005 | **0.1627** | **−18.9 %** | 0.4699 | **0.4124** | .746→**.811** |
| held-out (200) | **`center`** | **B6 blend** | 0.1904 | **0.1563** | **−17.9 %** | 0.4576 | **0.4041** | .762→**.821** |
| val split (61) | `random` | B5 ringfusion | 0.1792 | **0.1696** | −5.4 % | 0.4186 | **0.3991** | .782→.807 |
| val split (61) | **`center`** | **B5 ringfusion** | 0.1988 | **0.1605** | **−19.3 %** | 0.4679 | **0.4121** | .748→**.816** |
| all frames (1234) | `random` | B6 blend | 0.0595 | **0.0580** | −2.5 % | 0.2090 | **0.2033** | — |
| all frames (1234) | **`center`** | **B6 blend** | 0.1878 | **0.1491** | **−20.6 %** | 0.4394 | **0.3890** | — |

Medians tell the same story on the full 1,234 frames — **B6 blend `center` medAE 0.0425 →
0.0319 (−25 %)**, B5 `center` 0.0445 → 0.0326 (−27 %) — while `random` barely moves
(B6 0.0148 → 0.0144). Per-angle, the retrain pushed the crossover where the network overtakes
the raw ToF from **6° down to 3°**:

| medAE, `center` | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| nearest-zone ToF | **0.027** | 0.063 | 0.104 | 0.163 | 0.204 |
| Network B v6 | 0.046 | 0.042 | 0.046 | 0.045 | 0.044 |
| **Network B v7** | 0.033 | **0.028** | **0.031** | **0.032** | **0.036** |
| **blend (v7)** | **0.026** | **0.027** | **0.031** | **0.032** | **0.036** |

**The gain is concentrated in extrapolation.** `center` confines anchors to the middle and scores
the periphery; `random` scatters them and scores the gaps. v7 gains ~19 % on `center` and ~6 % on
`random`, because the interpolation regime was already near the ToF's own noise floor and had
little left to give. RMSE moves with MAE in every row, so this is not a median gain paid for
with a fatter tail.

**Clean control:** B0–B4 are **bit-identical** between v6 and v7 (deltas exactly ±0.0000). Those
baselines never touch the residual, so the only thing that changed is Network B — no calibration
drift, no data drift, no scoring nondeterminism.

> **Still ToF-scored, so still circular.** Anchors and evaluation targets come from the same
> sensor, and this repo has already been burned by that once: v6 *halved* ToF-scored B5 medAE
> and then moved the tape MAE by **1 mm**. **v7 is confirmed better at agreeing with the ToF; it
> is not yet confirmed better at measuring the world.** Given the v6 precedent, expect the tape
> gain to be well under 19 %. The re-measure is pending — see
> [What needs re-running](#what-needs-re-running).

**The FP16 engine is a faithful build of the checkpoint.** `export_onnx.py` checks PyTorch vs
ONNXRuntime in FP32 (max abs diff **7.15e-06**), which proves the graph but says nothing about
the precision loss from the FP16 build. Driving the checkpoint and the engine with matched
inputs and comparing each output field against its own spread:

| field | p99 abs err | ref std | relative |
|---|---|---|---|
| `da` | 1.50e-04 | 1.34e-02 | 1.1 % |
| `db` | 2.05e-03 | 2.42e-01 | 0.85 % |
| `log_tau2` | 8.37e-03 | 1.24 | 0.67 % |

σ is off by **+0.42 %** at p99. Any v7 result is attributable to training, not quantisation.

**Throughput is unchanged** (`rate_live_v7.json`): `/depth` **10.29 Hz** vs v6's 10.31, `/cloud`
10.29 vs 10.28, latency 137.3 ms vs 132.9. v7 is a better-trained network, not a bigger one —
it costs nothing at runtime.

Files: `heldout_v7.json`, `baselines_valsplit_v7.json`, `baselines_v7.json`. Training log:
`train_v7_long.log`. The unsuffixed `baselines.json` / `baselines_valsplit.json` / `heldout_v6.json`
are the **v6** run.

### Resource utilisation on the Orin — measured 2026-08-04

Previously unmeasured: every performance number in this file was wall-clock, so "does it fit on
the Orin" had never actually been answered. `tegrastats` at 500 ms, idle vs the full pipeline
publishing at 10.29 Hz (`utilisation_v7.json`):

| resource | idle | **under load** | Δ |
|---|---|---|---|
| CPU (12 cores) | 0.2 % | **32.4 %** (p90 33.6, max 35.7) | +32 % |
| — busy cores | 0.02 | **3.89 of 12** | +3.9 |
| GPU | 0 % | **25.5 %** (p90 41, max 46) | +26 % |
| RAM | 11.7 GB | **13.4 GB** | +1.7 GB |
| Power (total rails) | 10.1 W | **21.7 W** | +11.6 W |

**Neither CPU nor GPU is saturated.** Load spreads evenly across all 12 cores (15–26 % each, no
single pegged thread), and the GPU peaks at 46 %. There is real headroom for a second camera
module. The 46 % peak against only 10.3 Hz output suggests the pipeline is latency-bound between
stages rather than throughput-bound — the ring architecture's per-stage handoffs, not the kernels.

> **Memory bandwidth is NOT measured.** This `tegrastats` build does not emit `EMC_FREQ`. On
> Orin, EMC is the usual first ceiling for multi-camera scale-out, so the headroom claim above
> covers compute only and should not be extended to bandwidth without a separate measurement.

### What "good" looks like for each one

Direction alone is not enough — for a comparison it matters *whose* number should move which
way. Every metric table below carries ↓ / ↑ on its headers; this is the target for each
evaluation as a whole.

| evaluation | what we are trying to achieve | how to tell if it worked |
|---|---|---|
| `insample` | **nothing** — it exists only to reproduce the pre-2026-07-28 figures and show they were optimistic | ignore the value; the point is that a trivial method scores 0.000 |
| `random` hold-out | **not to win.** Raw ToF *should* beat us here — the answer is 1.7 cm away | our medAE close to raw ToF's is fine; being far worse would point to an actual defect. We are **0.010 vs 0.010** ✅ |
| `center` hold-out | **to win, and to stay flat with distance** | our medAE **below** nearest-zone's, and roughly constant across the 0–3° → 15–30° columns while nearest-zone's rises ✅ |
| ZJU-L5 vs published | **our Rel ↓ and RMSE ↓ below theirs, our δ₁ ↑ above theirs** | we beat CFPNet and PENet on Rel/δ₁, sit 4–8 % behind DEPTHOR-Small, and are **2.7× worse on RMSE** ❌ — the one target clearly missed |
| DEPTHOR by region | **the gap should not depend on region.** A gap that shrinks inside the ToF footprint would mean we only look good where we have measurements | gap is −4 % inside, −8 % outside, −6 % overall → **uniform**, so no region-picking advantage ✅ (and none to claim) |
| Held-out re-distill | **the clean numbers should be close to the contaminated ones.** A large gap would mean the published figures were fiction | +5 %, rankings unchanged ✅ |
| Timing | **Hz ↑, ms ↓**, and stay above the ToF's 8.3 Hz complete-map rate | **10.4 Hz measured on the robot ✅** with every stage on (7.2 Hz as first shipped — fixed by two GPU rewrites). 13.3 Hz with blend+ROI off. See [the deployed rate, measured](#the-deployed-rate-measured-2026-07-30) |
| σ calibration | **coverage at ±1σ should hit 0.683 exactly** — this is a target, not a direction. Too low = overconfident, too high = σ inflated and useless | analytic σ in the far field: **0.229** ❌ badly overconfident |
| Blend A/B, driving | **the offline prediction should transfer.** A large miss either way would mean the offline bench does not describe the robot | 5.4 % measured against 5.8 % predicted ✅ — the first offline prediction in this project to survive contact unchanged |
| Tape ground truth | **our depth ↔ tape agreement**, especially outside the ToF cone. The headline is the **outside/inside AbsRel ratio** — near 1.0 means the extrapolation holds up | ⬜ not collected; tooling built and tested, see [below](#tape-ground-truth-tools-built-and-tested-measurements-not-yet-taken) |

**Scorecard as it stands: seven targets met, two missed (far-field RMSE and σ calibration —
both the same root cause), one unmeasured (tape).**

The deployed rate went "unmeasured → missed → met" on 2026-07-30. The extrapolation said
9.9 Hz, the robot said 7.2 Hz, and two GPU rewrites took it to 10.4 Hz. It is the reason
extrapolated figures are now labelled as such throughout: the estimate was not merely
imprecise, it was optimistic in a way that hid a failed target.

### The single most important thing to understand

**Six of those seven score against the ToF we anchor to.** That is a closed loop: it cannot
detect a bias in the ToF, and it says nothing about the 87.6 % of the frame the ToF never sees.
Only ZJU-L5 (and, when it exists, the tape) breaks that loop.

This is why `random` and `center` disagree about whether the neural stack is worth having, and
why both are reported. Neither is "the" number.

### What none of these answer

- **Is our ToF biased?** Nothing here can tell. Needs the tape.
- **Does the far field work?** No — it demonstrably does not
  ([the ceiling](#the-far-field-ceiling--mechanism-and-what-does-and-does-not-fix-it)).
  Depth beyond ~1.4 m on our sensor is bounded by the model, not measured.
- **Does deployed σ flag its own errors in the far field?** Untested. The σ result below is the
  *analytic* term only, which is ~0.1 % of what `/depth_var` actually publishes.
- ~~**Does the deployed node still hit its rate with the new stages?**~~ **ANSWERED
  2026-07-30.** Initially no — 7.2 Hz against 8.3 Hz needed. Now **10.4 Hz ✅** after moving
  the blend expansion and the whole ROI mask onto the GPU. See
  [the deployed rate, measured](#the-deployed-rate-measured-2026-07-30).
- ~~**Latency: depth maps arrive 385 ms stale.**~~ **FIXED 2026-07-30** — queue depth 5 → 1
  plus lazy rectification took it to **128.5 ms** with no change in throughput.

## Metric reference

One definition for all of it, in [`tools/diagnostics/metrics.py`](../tools/diagnostics/metrics.py).
Before this existed the harnesses reported MAE only, which appears in no depth paper as a
primary number — so nothing here could be placed beside a published result.

| metric | direction | ideal | meaning | what it catches |
|---|---|---|---|---|
| `MAE` | ↓ lower | 0 | mean \|error\|, metres | overall error, skewed by outliers |
| `medAE` | ↓ lower | 0 | *typical* miss; half beat it | everyday accuracy |
| `p95` | ↓ lower | 0 | 95 % of pixels beat this | bad-but-not-worst case |
| `RMSE` | ↓ lower | 0 | errors squared | **the tail** — rare catastrophic pixels |
| `AbsRel`/`Rel` | ↓ lower | 0 | \|error\|/truth, a fraction | scale-free → **comparable across papers** |
| `δ<1.25` | ↑ higher | 1.0 | fraction within 25 % of truth | **the literature standard** |
| `δ<1.25²` | ↑ higher | 1.0 | fraction within 56 % | looser form of the same |
| `coverage` | ↑ higher | 1.0 | fraction of pixels answered at all | methods that decline to predict |
| `bias` | → toward 0 | 0 | mean *signed* error | systematic over/under-reading |
| `ρ` | ↑ higher | 1.0 | does the backbone track real depth | a **gate**, not a score |

**Always report `MAE`, `medAE` and `RMSE` together.** Depth error here is heavy-tailed and the
gap between them is the diagnostic: medAE 0.066 with MAE 18.092 is a handful of pixels
predicting absurd distances, and either number alone misleads.

**σ coverage is a target, not a direction.** Coverage at ±1σ should hit **0.683**; below is
overconfident (v3: 0.590), above means σ is inflated and carries no information (v4: 0.73–0.75).
Both directions are failures.

## Benchmarks vs. trivial baselines

Measured 2026-07-28 over all 1,234 logged pairs with
[`tools/diagnostics/baselines.py`](../tools/diagnostics/baselines.py); raw numbers in
[`docs/demo/benchmarks/baselines.json`](../docs/demo/benchmarks/baselines.json). Every method sees the **same
anchors** and is scored at the **same held-out zones**.

Two hold-out protocols, because the choice dominates the conclusion:

- **`random`** — 25 % of zones held out at random. This is what training and every
  published number above used. 99.6 % of the held-out zones have an anchor within one
  zone, a **median of 1.7 cm away**, so it measures interpolation between adjacent
  samples.
- **`center`** — anchor on a central 16×16 island, predict everything outside it. Mirrors
  the deployed geometry (ToF in the middle, extrapolate outward): median **8.4°** from the
  nearest anchor, ~30 cm at 2 m.

| | `random` MAE ↓ | `random` medAE ↓ | `center` MAE ↓ | `center` medAE ↓ |
|---|---|---|---|---|
| B0 constant (median of anchors) | 0.527 m | 0.254 m | 0.633 m | 0.441 m |
| **B1 nearest zone** *(0 params, no camera)* | **0.056 m** | **0.010 m** | **0.232 m** | 0.108 m |
| B2 bilinear upsample *(0 params)* | 0.058 m | 0.007 m | *coverage 0.00* | — |
| B3 mono + median scale *(1 param)* | 35.488 m | 0.136 m | 33.468 m | 0.169 m |
| B4 closed-form, as deployed *(2 params)* | 0.283 m | 0.068 m | **18.092 m** | **0.066 m** |
| B4c closed-form + the 20 m clamp | 0.283 m | 0.068 m | 0.359 m | 0.066 m |
| B5 RingFusion v3 *(~0.46 M params)* | **0.148 m** | **0.026 m** | 0.331 m | 0.067 m |

> **What this says:** on `random` a zero-parameter lookup beats the whole pipeline, because the
> answer is 1.7 cm away. On `center` that reverses. The protocol, not the method, decides the
> winner — which is why a single headline MAE was misleading.

### What this changes

- **A zero-parameter lookup table beats the full pipeline on `random`** — 0.056 m vs
  0.148 m. Unsurprising once you know the nearest real measurement is 1.7 cm away, but it
  means the published protocol systematically *understates* the architecture.
- **`B4` vs `B4c` is a clamping artefact, not a model difference.** `anc.to_metric_depth`
  bounds inverse depth at `min_disp=1e-4`, so the closed-form path can emit **10 000 m**,
  while `ResidualRefiner.refine` caps at 20 m. One `np.clip` moves it 18.092 → 0.359 m and
  recovers ~98 % of what looked like Network B's contribution. **Deployment bug:** the
  Network-B-off fallback path is genuinely unclamped and can publish absurd far-field
  points into `/cloud`.
- **Network B is a local refiner, not an extrapolator.** Against a fairly-clamped
  baseline it wins big where anchors are near (`random`, −48 % MAE) and almost nothing
  where they are not (`center`, −8 % MAE, medians tied at 0.066/0.067). Consistent with
  its training: held-out zones were always ~1.7 cm from an anchor, so it never had to
  learn extrapolation.
- **Not contaminated.** `residual_v3` trained on these pairs, so B5 was re-scored on the
  61 frames from `train_residual.py`'s `random_split(..., manual_seed(0))` validation
  split: `center` MAE 0.296 m vs 0.331 m on the full set. No overfitting to these scenes.

### The result that justifies the camera

Median error by angular distance from the nearest anchor, `center` protocol:

| medAE ↓ *(lower is better everywhere)* | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| *n* | 103 586 | 157 446 | 216 632 | 185 856 | 72 335 |
| B1 nearest zone | **0.027** | 0.072 | 0.122 | 0.180 | 0.239 |
| B4c closed-form | 0.073 | 0.071 | 0.078 | **0.061** | **0.047** |
| B5 RingFusion v3 | 0.064 | 0.073 | 0.078 | 0.062 | 0.047 |

> **What this says:** read left to right, this is "how does each method decay as it leaves the
> measurements". It is the most informative table in the file — the ToF collapses, the camera
> path does not, and that flatness *is* the argument for having a camera at all.

Nearest-neighbour degrades **9×** across the range; the camera path is **flat**. They
cross at 3–6°, and past 10° the camera wins 3–5×. Since the ToF covers 12.4 % of the frame
and the rest is far outside 30°, the right-hand columns are the ones that predict
deployment behaviour. This is the first direct measurement of that claim.

## Network B v4 — the training protocol *was* the limitation

`build_real_supervision` gained a `holdout=` mode. `random` (v1–v3) hides 25 % of zones at
random, which leaves an anchor ~1.7 cm from 99.6 % of the supervision targets — the net was
only ever graded on interpolation, so near-identity was the correct thing to learn.
`island` anchors on a central 16×16 block and supervises everything outside it.

`residual_v4_island`: identical architecture, hyperparameters, data and student backbone as
v3. **One changed argument.** Scored on the 61 frames of `train_residual.py`'s
`random_split(..., manual_seed(0))` validation split, which no version trained on:

| `center` (extrapolation) | MAE ↓ | medAE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| B4c closed-form + clamp | 0.316 m | 0.064 m | 0.616 |
| v3 (random hold-out) | 0.296 m | 0.066 m | 0.628 |
| **v4 (island hold-out)** | **0.206 m** | **0.045 m** | **0.746** |

v3 was **statistically tied with having no network at all** (0.066 vs 0.064). v4 beats the
closed-form by 30 %. A third run, `residual_v5_mixed` (alternating per sample), lands
between the two on both protocols — a genuine compromise, not a free lunch.

There is a measurable cost: v4 gives up interpolation, `random` medAE 0.026 → 0.091. Which is what
Stage 7c exists to recover.

## Stage 7c — the ToF/network blend

Neither source wins everywhere and the crossover is sharp, so
[`blend.py`](src/ringfusion_perception/ringfusion_perception/blend.py) takes the ToF near
the anchors and the network far from them, smoothstepping between (2°→5°, angle-driven so
it follows the optics rather than the resolution).

| medAE ↓, `center` | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| nearest-zone ToF | **0.025** | 0.070 | 0.123 | 0.188 | 0.244 |
| v4 network | 0.048 | 0.047 | 0.052 | 0.042 | 0.038 |
| **blend** | **0.028** | **0.046** | **0.052** | **0.042** | **0.038** |

It tracks whichever source is better in every bin. Headline, same 61-frame split:

| | `random` MAE ↓ / medAE ↓ | `center` MAE ↓ / medAE ↓ |
|---|---|---|
| nearest-zone ToF | 0.056 / 0.010 | 0.240 / 0.108 |
| v4 network alone | 0.261 / 0.091 | 0.206 / 0.045 |
| **v4 + blend** | **0.045 / 0.009** | **0.194 / 0.042** |

The blend beats **both** sources it mixes, including raw ToF on interpolation
(δ<1.25 0.969 vs 0.956) — averaging two partly-independent estimates near the crossover
cancels noise neither cancels alone. It also settles v4-vs-v5: with the blend, interpolation
is *identical* between them (the ToF supplies it), and v4 wins extrapolation by 19 %.

**Recommended configuration: `student_v4_heldout` + `residual_v7_fov73` + blend.**
*(Was `residual_v6_fov73` until 2026-08-04, and `student_v3` + `residual_v4_last` before the
2026-08-03 field-of-view fix.)*

> **Confirmed live, 2026-07-30 — and the offline number held.** 600 frames while driving,
> both arms computed from **one** backbone+residual pass per frame so the only difference is
> the blend, and the blend fed **only** the central island so it never saw a scored zone:
>
> | `center`, held-out zones | MAE ↓ | medAE ↓ | p95 ↓ | δ<1.25 ↑ |
> |---|---|---|---|---|
> | network alone | 0.168 m | 0.045 m | 0.782 m | 0.775 |
> | **+ blend** | **0.159 m** | **0.044 m** | **0.761 m** | **0.790** |
>
> **5.4 % better MAE live against 5.8 % predicted offline** — the offline figure transferred
> almost exactly, which is the first time in this project an offline prediction has survived
> contact with the robot unchanged.
>
> Two things offline could not test, both now measured: **temporal stability is identical**
> (median frame-to-frame |Δdepth| 0.0027 m blended vs 0.0028 m not — the blend adds no
> flicker under motion), and the **far field does not blow up** (max depth 2.93 m blended vs
> 3.08 m, nothing beyond the ToF range in either arm).
>
> **What it actually improves — the medians say it plainly.** The *typical* error barely
> moves (medAE 0.045 → 0.044 m) while the *mean* moves ten times as far (MAE 0.168 → 0.159 m)
> and the tail moves with it (p95 0.782 → 0.761 m). The blend is not making ordinary pixels
> better; it is making **bad** pixels less bad. Where the network is already roughly right it
> contributes nothing, and where the network is badly wrong near an anchor it catches the
> error. That is worth knowing before anyone tries to tune it: optimising it against a median
> would show no signal at all.
>
> **Scope: this measured extrapolation only.** The `center` protocol holds out everything
> outside the island, so every scored zone is in the outer ring. The blend's *large* win is
> on `random` — interpolation *between* zones inside the cone, where it takes MAE 0.261 →
> 0.045 — and no zone inside the cone is scored here. That benefit remains offline-only.
>
> **The trade, now that both sides are measured on the robot:** the blend costs **15.35 ms of
> a 96.1 ms frame**. Turning it off gives **12.4 Hz instead of 10.4 Hz (+19 %)** for **5.4 %
> worse MAE**.
>
> **Decision (2026-07-30): the blend stays ON by default.** A 19 % rate gain is not worth a
> 5 % accuracy loss while the pipeline already clears its 8.3 Hz real-time target with room
> to spare. `blend:=false` remains available for a consumer that would rather have the rate.
>
> [`blend_ab_live.py`](../tools/diagnostics/blend_ab_live.py),
> `docs/demo/benchmarks/blend_ab_live.json`, and a visual readout at
> [`docs/demo/live2_readout.html`](../docs/demo/live2_readout.html).

#### Intent: the blend is architecture, not a patch

Worth stating explicitly, because a 5 % number invites the wrong reading. Stage 7c is **not a
correction bolted on to compensate for a weak network** — it is the part of the design that
decides *which source to believe where*, and it is meant to stay.

The system has two estimators with opposite failure modes. The ToF measures true distance but only
at ~1024 points covering 12.4 % of the frame. The network covers every pixel but knows only
relative geometry and inherits whatever the affine fit gets wrong. Neither is right everywhere,
so something has to arbitrate — and doing that by distance to the nearest real measurement is
the principled rule, not a heuristic.

Three consequences that follow from treating it that way rather than as a patch:

- **It should get *better* as the sensor improves, not redundant.** More ToF zones, or a
  longer range, widen the region where a real measurement is nearby. The blend is the
  mechanism that converts extra sensor coverage into map accuracy.
- **It is where per-pixel provenance belongs.** The blend already computes a weight per pixel
  — how much of this depth came from measurement versus inference. That is exactly the signal
  a downstream consumer needs and `/depth_var` currently approximates. Exposing it is a
  natural extension; deleting the blend would remove the only place it exists.
- **Judge it on the tail, not the average.** Per the medians above it targets the worst
  pixels, which is what "arbitrate between two sources" is supposed to do. A future version
  should be tuned against p95 and RMSE, and against the tape ground truth once that exists —
  not against a headline mean that averages its effect away.

Enabled by default; `blend:=false`, `blend_near_deg`, `blend_far_deg` are `perception` node
parameters for live A/B.

> **Caveat.** Every evaluation point is a ToF **zone centre**, which is exactly where
> nearest-neighbour looks best. Across a depth discontinuity *between* zones the nearest
> anchor may sit on a different surface while the network reads the edge correctly. These
> numbers are structurally blind to that, so the blend may smear object boundaries inside
> the cone. Independent tape ground truth is the way to check.

Raw results, both training logs and the val-split stem list:
[`docs/demo/benchmarks/`](../docs/demo/benchmarks/).

## Train/test contamination, quantified

`training/data.py:24` globs `**` with `recursive=True`, so distilling over `data/rect` swept up
`data/rect/paired/` — **1,228 frames byte-identical to `ros2_ws/data/real/rgb`, i.e. every
frame the benchmarks score on.** Network A had trained on the entire evaluation set. The
61-frame split recovered from `train_residual.py` was clean for Network *B* only.

`DistillDataset(exclude_stems=...)` and `distill_backbone.py --exclude-stems-file` now hold
frames out. `student_v4_heldout` was re-distilled with **200 stems** held out (a superset of
Network B's 61-frame validation split, so one set is unseen by both), reaching
**val_ssi 3.3984** against v3's 3.51 on 6 % less data.

### The effect was small and bounded — as predicted, now measured

`center` protocol, median AE:

| medAE ↓ | contaminated *(v3, 61 frames)* | **clean** *(v4, 200 frames)* | change |
|---|---|---|---|
| B4c closed-form | 0.064 m | 0.067 m | +5 % |
| B5 Network B v4 | 0.045 m | 0.048 m | +7 % |
| **B6 + blend** | **0.042 m** | **0.044 m** | **+5 %** |

**Rankings are unchanged and the correction is ~5 %.** That matches the a-priori bound: the
distillation target is the *teacher's disparity*, not ToF depth, and the student scores *below*
its teacher even on frames it trained on (ρ 0.737 vs 0.750), so it had not memorised them.

Scored with `student_v4_heldout` + `residual_v4_last` + blend *(pre-2026-08-03 calibration — superseded)*:

| protocol *(all metrics ↓ lower is better)* | B1 nearest-zone | B4c closed-form | B5 Network B | **B6 + blend** |
|---|---|---|---|---|
| `random` MAE / medAE | **0.059 / 0.010** | 0.296 / 0.066 | 0.309 / 0.098 | 0.059 / 0.010 |
| `center` MAE / medAE | 0.243 / 0.109 | 0.357 / 0.067 | 0.213 / 0.048 | **0.198 / 0.044** |

medAE by angular distance from the nearest anchor, `center`:

| medAE ↓ | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| B1 nearest-zone | **0.026** | 0.071 | 0.126 | 0.183 | 0.239 |
| B4c closed-form | 0.073 | 0.072 | 0.079 | 0.062 | 0.048 |
| B5 Network B v4 | 0.057 | 0.051 | 0.054 | 0.045 | **0.035** |
| **B6 + blend** | **0.030** | **0.049** | **0.054** | **0.045** | **0.035** |

The blend tracks whichever source is better in every bin, and on `random` it now matches raw
ToF exactly (0.059 / 0.010) — correctly deferring to the sensor where the sensor wins.

> **Partial caveat on the B5/B6 rows.** Of the 200 stems, 139 were in `residual_v4`'s training
> set (only the 61-frame intersection is clean for *both* networks). Re-scored on those 61:
> B4c 0.063, B5 0.043, **B6 0.041** medAE — marginally *better* than the 200-frame figures, so
> Network B's own overlap is not inflating anything here.
>
> *(Superseded 2026-08-04: Network B was retrained as `residual_v6_fov73` after the
> field-of-view fix, then as `residual_v7_fov73` at a 110-epoch budget. The reasoning below
> explains why v4 was reused at the time.)*
>
> **Why `residual_v4_last` is reused unchanged with the new backbone:** it was trained against
> `student_v3` disparity, but corr(v3, v4) over the held-out frames is **0.99944** (min 0.98963).
> There is a ~21 % relative magnitude shift, which the per-frame affine fit absorbs and which
> `refine()` normalises away anyway (it feeds `log(D0/median)`). Retraining B on v4 was therefore
> not required for these numbers. Raw output:
> [`heldout_v4.json`](../docs/demo/benchmarks/heldout_v4.json),
> [`heldout_v4_bothclean.json`](../docs/demo/benchmarks/heldout_v4_bothclean.json).

### A dead-gradient trap in the distillation loss, found doing this

The first re-distill attempt **froze for 39 epochs** at val_ssi 167.24 (v3's reference: 3.51),
identically for AMP on/off and three learning rates — the giveaway that the loss had stopped
depending on the prediction.

`losses.align_ssi` clamped the SSI alignment scale to `a >= 0`. At exactly zero,
`aligned = 0·pred + mean(target)` is a **constant**, so `∂ssi/∂pred` is identically zero and
the model can never recover. A random init is mildly anti-correlated with the teacher
(corr −0.12, chance), least squares asks for `a = −48`, the clamp sets 0, and training dies.
Only `gradient_loss` retained a path, which is why the total moved briefly then locked.

The clamp was added to fix a specific failure — its docstring records a from-scratch re-distill
converging *sign-inverted* at ρ −0.998 — but it makes ~half of random seeds unrecoverable, and
it is in committed code (`936775f`), so it would break any re-distill, not just ours.

Fixed by flooring `a` at `A_FLOOR_FRAC = 0.01` of the std-matching scale: the aligned output
still depends on `pred`, so the gradient survives and still pushes toward positive correlation,
while the sign-inverted basin stays excluded. Verified — ssi-only gradient **0 → 21,518**, and
a single-batch overfit reaches **corr +0.9939** with the teacher.

## ZJU-L5 — the first open-loop evaluation, and what it exposed

*Full 527-sample test split. Raw output:
[`zjul5_teacher.json`](../docs/demo/benchmarks/zjul5_teacher.json),
[`zjul5_student.json`](../docs/demo/benchmarks/zjul5_student.json).*

Every other number in this project is scored against the ToF we anchor to. [ZJU-L5](https://zju3dv.github.io/deltar/)
(from DELTAR, GPL-3.0) ships **dense ground truth from a RealSense 435i** rigged with a
VL53L5CX, so the thing measured and the thing measuring are finally different devices.
Harness: [`tools/diagnostics/zjul5_eval.py`](../tools/diagnostics/zjul5_eval.py).

The dataset supplies each zone's pixel rectangle in `fr`, so **their** ToF→camera projection
is used as-is. Nothing is re-derived — which matters, because a new rig is precisely where a
mirrored or transposed grid would recur.

| | ZJU-L5 | ours |
|---|---|---|
| zones | 8×8, median **41 valid**/frame | 32×32, ~830 valid |
| frame coverage | **~50 %** (zones tile it) | 12.4 % |
| depth range | 0.33–3.24 m | 0.15–6.5 m |
| ground truth | dense, RealSense 435i | sparse, the same ToF we anchor to |

### The finding: the *fit* transfers, the *backbone* does not

`zjul5_eval.py` prints ρ(disparity, 1/z) at the anchors as a gate, and it fired at once:

| backbone | params | ρ at anchors ↑ | Rel, all px ↓ | δ₁, all px ↑ |
|---|---|---|---|---|
| **`student_v3`** — deployed | 3.66 M | **0.417** | 0.185 | 0.716 |
| **Depth Anything V2 Large** — its teacher | **335.3 M** | **0.867** | **0.086** | **0.908** |

On our own sensor ρ is 0.917, so 0.417 is a serious domain failure — and with it the pipeline
**loses to every trivial baseline on this dataset** (nearest-zone scores Rel 0.122).

So "the closed-form path has no learned parameters, therefore it transfers" is **true of the
affine fit and false of the pipeline**: the pipeline still contains Network A, which is
learned. It was distilled on **~2 000 of our own rectified-fisheye frames**, so it learned to
be the teacher *on one lens in one building*. Consistent with `teacher_vs_student.py` on our
data — teacher ρ 0.750 vs student 0.737 at `corr 0.989`: the student matches the teacher
exactly where it was trained, and collapses elsewhere. That is what distillation does, not a
defect.

**This is scoped, not fixed.** The robot has one camera and Network A only ever sees it;
re-distilling on more of *our* images would not change this. ZJU-L5 is therefore reported
with a domain-general MDE in place of the student — which is also architecturally what
DEPTHOR does (24 M of its 30 M is a monocular model), so it is the like-for-like
configuration rather than a dodge. Any speed comparison must state which backbone was timed.

### Head-to-head against published ZJU-L5 results

Their metrics are computed over all pixels with valid ground truth, which matches our `all`
region exactly. Published figures from the [DEPTHOR paper](https://arxiv.org/abs/2504.01596),
Table 2:

| method | params | δ₁ ↑ | δ₂ ↑ | Rel ↓ | RMSE ↓ |
|---|---|---|---|---|---|
| CFPNet | — | 0.883 | 0.949 | 0.103 | **0.431** |
| PENet\* | — | 0.889 | 0.949 | 0.093 | **0.447** |
| **ours — closed-form, no learned fusion** | 335 M backbone | **0.908** | **0.954** | **0.086** | 0.984 |
| **ours — + Stage 7c blend** | 335 M backbone | 0.907 | 0.952 | **0.085** | 0.986 |
| DEPTHOR-Small | 30 M | 0.921 | 0.963 | 0.080 | 0.379 |
| DEPTHOR-Large | 36 M | 0.933 | 0.972 | 0.075 | 0.350 |
| *ours — deployed `student_v3`* | *4.1 M* | *0.716* | *0.875* | *0.185* | *1.174* |

\* asterisk is the DEPTHOR authors' own — the PENet number is reported by them, not by
PENet's authors. Check provenance before leaning on it.

**Three of four metrics are competitive; the fourth says we have a tail problem.** δ₁, δ₂ and
Rel all beat CFPNet and PENet and sit just under DEPTHOR-Small — with **zero learned fusion
parameters**. RMSE is 2.2–2.8× worse than every published method.

That combination has one explanation: δ₁ and Rel are dominated by *typical* pixels, RMSE
squares errors and is dominated by *extreme* ones. Typical pixels are genuinely good; a small
minority are catastrophically wrong. medAE 0.049 alone would have hidden this completely —
the same signature that caught the missing clamp (MAE 18.092 vs medAE 0.066).

> **The competitive row is not the robot.** It uses Depth Anything V2 **Large — 335.3 M
> parameters, measured, not quoted** — which is 11× DEPTHOR-Small's *entire* model. The stack
> that runs on the Jetson at 13.7 Hz is the 4.1 M configuration scoring δ₁ 0.716. Claiming
> DEPTHOR-class accuracy from a 335 M backbone while claiming speed from a 4.1 M one, without
> separating them, is exactly the sleight of hand a reviewer looks for. A DAv2-Small (~25 M)
> run is the compute-fair comparison and is **not yet done**.

### Where the error actually is

| region | method | Rel ↓ | RMSE ↓ | δ₁ ↑ | medAE ↓ | cov ↑ |
|---|---|---|---|---|---|---|
| **inside** ToF footprint | nearest-zone | 0.058 | 0.445 | 0.949 | 0.033 | 1.00 |
| | bilinear | 0.053 | **0.302** | 0.958 | 0.031 | 0.75 |
| | closed-form | 0.052 | 0.418 | **0.968** | 0.035 | 1.00 |
| | **+ blend** | **0.050** | 0.426 | 0.964 | **0.031** | 1.00 |
| **outside** ToF footprint | nearest-zone | 0.201 | 1.642 | 0.696 | 0.123 | 1.00 |
| | bilinear | 0.163 | **0.697** | 0.760 | 0.148 | **0.18** |
| | closed-form | **0.128** | 1.398 | **0.835** | **0.079** | 1.00 |
| | **+ blend** | **0.128** | 1.398 | **0.835** | **0.079** | 1.00 |

Two things fall out of this:

- **Inside the footprint we are already best on the literature metrics** — δ₁ 0.968 and
  Rel 0.050 beat both trivial baselines, and the blend improves Rel and medAE over the raw
  closed-form. Bilinear wins RMSE but only answers for 75 % of those pixels.
- **The RMSE tail is an OUTSIDE problem**: 1.398 outside vs 0.418 inside. Inside, we are near
  bilinear; outside, we are 2× worse than it. So the blow-ups are specifically far-field
  extrapolation, which is what a scene-bounded far-field clamp would target.

Caveats that must travel with these rows:

- **Bilinear covers only 18 % of the outside region.** It cannot extrapolate past the convex
  hull of ~41 points, so its RMSE 0.697 is measured on a favourable subset and is *not*
  comparable to the 100 %-coverage rows.
- **Network B is out of domain here** — trained on 32×32 anchors at our intrinsics, facing
  8×8 over half a frame. Rel 1.819. Reported for completeness only. A Network B trained on
  ZJU-L5's 483-frame train split would be the meaningful head-to-head, and is not built.
- **Blending cannot repair a bad source.** Fed the out-of-domain residual instead of `D0`,
  the blend scored Rel 1.252 — it mixes, it does not fix. `--blend-over` selects which.

## DEPTHOR — reproduced on our hardware, and timed

DEPTHOR (ICCV 2025) is the current state of the art on ZJU-L5 and the method we are most
often compared against. We ran **their** released weights on **our** Orin so the comparison
measures their model rather than our re-implementation of it.

Setup lives outside this repo at `~/external/` — DEPTHOR ships **no LICENSE** (all rights
reserved) and BP-Net/DELTAR are GPL-3.0, so none of it is vendored here. Four fixes were
needed and are recorded in
[`depthor_small_zjul5.json`](../docs/demo/benchmarks/depthor_small_zjul5.json):

| problem | fix |
|---|---|
| `BpOps` CUDA extension (BP-Net) fails to compile on torch 2.11 | `tensor.type()` → `scalar_type()`, `.data<T>()` → `.data_ptr<T>()` |
| `pip` build isolation hides torch from `setup.py` | `--no-build-isolation`, `TORCH_CUDA_ARCH_LIST=8.7` |
| `src/utils/set_mde.py` hardcodes the author's home dir | redirected, `DAV2_CKPT_DIR` override |
| variant selected by commenting imports in/out | `DEPTHOR_VARIANT=small\|large` |

### Harness validation

| DEPTHOR-Small | our run | their paper | *(agreement is the point, not the direction)* |
|---|---|---|
| δ₁ | **0.923** | 0.921 |
| δ₂ | **0.968** | 0.963 |
| Rel | **0.079** | 0.080 |
| RMSE | **0.371 m** | 0.379 |
| params | **30.2 M** (24.8 M MDE) | 30 M (24 M MDE) |

Within 0.002–0.008 on every metric, parameter counts matching. **So the ZJU-L5 comparison is
sound** — if we could not reproduce their number, any comparison would be measuring our bugs.

> **Metric-name trap.** Their `compute_errors` returns a key called `mae` computed as
> `mean(|pred-gt|/gt)` — that is **AbsRel, not MAE**. Their `rmse` *is* in metres. Comparing
> our MAE against their `mae` would be comparing metres against a ratio.

### The speed number the authors did not publish

**7.84 it/s = 128 ms/frame on the Orin.** The paper states "not fast enough for real-time
inference is the main limitation of our method" and reports no timing at all, so this fills a
gap they left open. Two caveats that must travel with it:

- **Resolution differs.** They run 480×640 (0.31 MP); our 13.7 Hz is at 1640×1232 (2.0 MP),
  **6.5× the pixels**. We are faster *and* denser, but a fair claim needs our pipeline timed
  at their resolution too.
- **Their 128 ms includes h5 dataloading** (it is tqdm over the dataloader), so it is not a
  pure network-forward figure.
- **Both variants share the same DAv2 ViT-S backbone (24.8 M)**; only the completion head
  differs (6 M vs 12 M).

### Compute-fair comparison: our analytic path on DEPTHOR's own backbone

Same DAv2 ViT-S weights DEPTHOR uses, so the only difference is 6 M of learned completion
versus **zero learned fusion**. ZJU-L5 test, all valid GT pixels:

| method | backbone | learned fusion | Rel ↓ | RMSE ↓ | δ₁ ↑ |
|---|---|---|---|---|---|
| CFPNet | — | yes | 0.103 | **0.431** | 0.883 |
| PENet\* | — | yes | 0.093 | **0.447** | 0.889 |
| ours, affine | ViT-S 24.8 M | **none** | 0.094 | 1.056 | 0.902 |
| **ours, + `b_prior` 0.003** | **ViT-S 24.8 M** | **none** | **0.091** | 1.031 | **0.908** |
| ours, affine | ViT-L 335 M | none | 0.086 | 0.984 | 0.908 |
| ours, + `b_prior` 0.003 | ViT-L 335 M | none | **0.083** | 0.953 | **0.913** |
| DEPTHOR-Small | ViT-S 24.8 M | 6 M | 0.079 | **0.371** | 0.923 |
| DEPTHOR-Large | ViT-S 24.8 M | 12 M | 0.075 | **0.350** | 0.933 |

**Dropping 335 M → 24.8 M costs almost nothing** (Rel 0.086 → 0.091, δ₁ unchanged at 0.908,
ρ 0.867 → 0.873). So the earlier "competitive" row was not being carried by the large
backbone, and the compute-fair version still beats CFPNet on both metrics and matches PENet on
Rel while beating it on δ₁ — with no learned fusion at all.

**`b_prior = 0.003`, chosen on the train split, generalises to test** on both backbones
(Rel −3 %, RMSE −2 %, δ₁ +0.005, bias toward zero). Small, consistent, free — and *not* yet
adopted as a deployed default, see the ceiling section.

**What DEPTHOR's completion head buys**: Rel −13 %, δ₁ +0.015, and **RMSE 2.8× better**. The
accuracy gap is modest; the **tail** gap is the one that matters, consistent with every other
measurement here.

### Speed, on the same hardware at the same resolution

Network-only, batch 1, 480×640, CUDA-synced, nothing else on the GPU:

| | latency ↓ | Hz ↑ |
|---|---|---|
| DEPTHOR-Small | 79.4 ms | 12.6 |
| DEPTHOR-Large | 183.8 ms | 5.4 |
| ours, `pipeline.run()`, blend + ROI **off** | 20.0 ms | 50.1 |
| **ours, `pipeline.run()`, blend + ROI on (deployed)** | **25.8 ms** | **38.7** |

**3.1× faster than DEPTHOR-Small in the deployed configuration — and that is our *entire*
pipeline against their network alone** (4.0× with the two optional stages off).

> Quote the **25.8 ms** row. An earlier revision of this section reported 21.3 ms / 3.7×
> against a "full" pipeline that in fact had blend and ROI off, while the summary table at
> the top of this file quoted 3.0× from the everything-on number — the same comparison
> appearing as two different ratios. Both are now the everything-on figure, re-measured
> 2026-07-30.

At our deployed 1640×1232 (6.5× the pixels) `pipeline.run()` is **70.7 ms / 14.1 Hz** with
every stage on, 50.9 ms / 19.6 Hz with blend and ROI off. The **deployed node**, which also
pays ROS transport and rectification, measures **10.4 Hz** — see
[the deployed rate, measured](#the-deployed-rate-measured-2026-07-30). Quote the deployed
figure when comparing whole systems; quote `pipeline.run()` when comparing to a network.

> Two integrity notes. Their `evaluate.py` tqdm rate (7.84 it/s = 128 ms) **includes h5
> dataloading** — use the network-only figures for model-vs-model. And a background GPU job
> silently inflated an earlier DEPTHOR-Small measurement to 166 ms, 2× the clean 79.4 ms;
> network-only cannot exceed end-to-end, which is what exposed it. Check the GPU is idle.

### Does restricting to the ToF region flatter us? No — the gap is uniform

Our numbers are reported split by whether a pixel lies inside the ToF footprint; the published
DEPTHOR numbers are whole-frame. Putting our inside-footprint row beside their whole-frame row
compares **different evaluation sets** — and it makes us look far better than we are.

Since their weights are public the objection can be removed rather than caveated.
[`depthor_regions.py`](../tools/diagnostics/depthor_regions.py) drives their model and scores it
through **our** `metrics.py`, with **our** coverage masks, GT gate and regions — identical code
on both sides. (Their own `evaluate.py` cannot do this: its prediction-saving block is commented
out, and its `compute_errors` key named `mae` is actually AbsRel.)

Harness check first — our re-score of their model against their paper:

| Rel ↓ / δ₁ ↑ | our re-score | published |
|---|---|---|
| DEPTHOR-Small Rel | 0.081 | 0.079 |
| DEPTHOR-Large Rel | 0.077 | 0.075 |
| DEPTHOR-Large δ₁ | 0.929 | 0.933 |

Within 0.002–0.004 despite a different GT gate, so the region numbers can be trusted:

| region | ours *(DAv2-L, no learned fusion)* | DEPTHOR-Small | DEPTHOR-Large | our gap vs Small |
|---|---|---|---|---|
| all valid GT | Rel 0.086 / δ₁ 0.908 | 0.081 / 0.920 | 0.077 / 0.929 | −6 % |
| **inside footprint** | Rel 0.052 / δ₁ 0.968 | 0.050 / 0.973 | 0.048 / 0.975 | −4 % |
| **outside footprint** | Rel 0.128 / δ₁ 0.835 | 0.119 / 0.857 | 0.113 / 0.874 | −8 % |

**The gap is nearly uniform across regions.** Restricting to the ToF footprint improves our
absolute numbers a lot (Rel 0.086 → 0.052) but improves theirs by the same proportion. Earlier
it looked as though we were close to DEPTHOR inside the footprint (0.052 against their *published*
0.079) — that was **entirely the region mismatch** — an artefact of scoring the two methods on different sets of pixels, not an advantage of ours.

So the defensible claim is: consistently **4–8 % behind DEPTHOR-Small on Rel in every region,
with zero learned fusion** against their 6 M completion head — and that this is *not*
region-dependent, which the paper must not imply.

One asymmetry that does **not** apply: their dataloader feeds `sparse_depth` as **64 single
pixels** (nonzero fraction 1e-4), the same sparse-point form we splat — not filled zone
rectangles. The ~50 %-of-frame footprint is purely an evaluation region derived from the
dataset's `fr`, and it applies identically to both methods.

RMSE remains the one metric where the gap is large rather than marginal — ours 0.984 against
their 0.371 whole-frame, 2.7× — consistent with
[the far-field ceiling](#the-far-field-ceiling--mechanism-and-what-does-and-does-not-fix-it)
rather than with typical-pixel accuracy.

## The far-field ceiling — mechanism, and what does and does not fix it

The single largest error source in the system, found 2026-07-29. Root cause of two separate
failures that had looked unrelated.

### Mechanism

`D = 1/(a·disp + b)`, so as disparity → 0 the depth asymptotes to **`1/b`**. That is a hard
ceiling on the deepest value the model can express, and `b` is fitted only from ToF anchors,
which cover near range. Measured with
[`ceiling_diag.py`](../tools/diagnostics/ceiling_diag.py) on 300 of our own logs — **no
ground truth needed, the ceiling falls out of the fit**:

| | ours | ZJU-L5 |
|---|---|---|
| ceiling `1/b`, median | **1.43 m** | 2.04 m |
| furthest ToF anchor, **median per frame** | 4.16 m | 1.59 m |
| ratio ceiling / anchor_max | **0.40×** | 1.28× |
| anchors above their frame's ceiling | **14.4 %** | — |

**Our ceiling sits below the ToF's own furthest reading**, so the pipeline cannot express the
depth of anchors it is being fitted to.

> Three different ToF-range figures appear in this file and they are **not** interchangeable:
> **4.16 m** is the median *per-frame* furthest anchor over 300 logs (`ceiling_diag.py`);
> **5.74 m** is the max in one particular moving run; **6.11 m** is the farthest return in the
> whole 1,234-log set. The ceiling-vs-range gap is therefore run-dependent and the 0.40× ratio
> is a median, **not** a worst case.

**A second, independent far-field constraint.** Network B's supervision carries a target range
gate, raised 5.0 → 6.5 m (task 7b), so **the residual has never been trained against a target
beyond 6.5 m** regardless of what the fit can express. That is separate from the `1/b` ceiling
and compounds with it: even a lifted ceiling leaves Network B extrapolating past 6.5 m with no
supervision it has ever seen. Widening the gate further is bounded by the sensor — only 0.23 %
of zones sit beyond 5 m and the farthest return in the set is 6.11 m.

### What it costs, on our own sensor

Anchor error binned by true anchor depth (200 frames, 167,651 anchors) — the ToF is the
reference here, so this is closed-loop, but it is enough to show the shape:

| anchor depth | n | mean signed error | share of total error |
|---|---|---|---|
| 0–0.5 m | 85,974 *(51 %)* | +0.059 m | 12.5 % |
| 0.5–1.0 m | 43,895 | +0.055 m | 12.7 % |
| 1.5–2.0 m | 6,751 | −0.746 m | 9.3 % |
| 2.0–3.0 m | 13,920 | −1.259 m | **31.9 %** |
| 4.0–6.5 m | 2,397 | −3.264 m | 14.2 % |

**69 % of the error comes from the 16 % of anchors beyond 1.5 m.** The anchor distribution is
median **0.49 m** with 51 % under half a metre, which is precisely why a single pooled MAE
looked healthy: it is diluted by tens of thousands of easy near anchors. We had never binned
by depth.

> This under-read is **documented and deliberate** — see `anchoring.py` ("A/ToF ~0.65 on the
> farthest quartile") and `roi.py` ("the far wall ... is not what the depth map is for").
> What was *not* documented is everything below.

### The ANALYTIC σ does not flag it — it is confident exactly where it is wrong

> **⚠ Scope, and it is narrow.** [`sigma_zjul5.py`](../tools/diagnostics/sigma_zjul5.py)
> measures the **analytic** variance alone, because Network B's learned head is out of domain
> on an 8×8 grid (its depth scores Rel 1.819 there, so its σ would be meaningless).
> But [the variance decomposition](#variance-decomposition) shows the analytic term is only
> **~0.1 %** of deployed `/depth_var` — the learned head is **~99.9 %**. So the numbers below
> describe the analytic term in the far field; they do **not** establish that deployed σ
> behaves this way. **Deployed far-field σ remains untested**, and testing it needs either an
> in-domain Network B or far-field ground truth on our own sensor.
>
> Likewise `corr(σ,|error|) = 0.943` from `sigma_cal.py` is the *deployed* (analytic+learned)
> σ at *anchor pixels only* — near-range, in-cone. The two figures differ in **both** which
> variance term and which regime, so they are not comparable. An earlier version of this
> section presented them as if they were.

Analytic variance against ZJU-L5's dense independent GT:

| true depth | median \|error\| ↓ | median σ | σ / \|error\| *(want ≈1)* | share of RMSE² |
|---|---|---|---|---|
| 0–1 m | 0.029 m | 0.011 m | 0.40 | 0.5 % |
| 4–6 m | 2.402 m | 0.401 m | 0.17 | 12.8 % |
| 6–10 m | 5.517 m | 0.259 m | 0.05 | 28.6 % |
| **10–20 m** | **11.845 m** | **0.079 m** | **0.01** | **50.1 %** |

> **What this says:** the system is most confident exactly where it is most wrong, and half of
> all squared error comes from 0.3 % of pixels. But this is the *analytic* variance only — see
> the scope warning above before reading it as a statement about `/depth_var`.

Analytic σ is **150× too small** at range and *decreases* past 4–6 m, because
`Var[D] = D⁴ · jᵀ Cov j` and `D` is itself capped by the same `1/b` ceiling. **One mechanism,
two failures.**

| analytic σ, ZJU-L5 far field | measured | ideal |
|---|---|---|
| `corr(σ, \|error\|)` | **0.196** | 1.0 |
| coverage \|e\| ≤ 1σ | **0.229** | 0.683 |
| coverage \|e\| ≤ 2σ | **0.429** | 0.955 |

**Analytic σ is also useless as a filter here**: dropping the top 1 % of pixels by σ removes
only **2.0 %** of squared error; dropping 10 % removes 38.9 %.

### Three candidate fixes, tested. Two falsified.

**Scene-bounded clamp** (cap depth at `k × max anchor depth`) — **falsified.** Swept on
ZJU-L5 train: tighter `k` is monotonically *worse* (MAE 0.231 → 0.261 at k=1.25) and bias
grows more negative. Capping only reduces depth, and the system already under-reads. Kept in
[`blend.py`](src/ringfusion_perception/ringfusion_perception/blend.py) as
`apply_scene_cap()`, documented as a dead end so it is not retried.

**σ as a filter** — falsified above.

**A ridge on `b`** (`solve_scale_shift(..., b_prior=)`; `lam = b_prior · Σw`, so 0 reproduces
the plain affine fit and ∞ is exact scale-only) — **a genuine but modest win.** Bracketed on
ZJU-L5 train, all pixels:

| `b_prior` | RMSE ↓ | MAE ↓ | Rel ↓ | δ₁ ↑ | bias → 0 |
|---|---|---|---|---|---|
| 0 *(current default)* | 1.012 | 0.261 | 0.105 | 0.886 | −0.135 |
| **0.003** | **0.962** | **0.246** | **0.097** | 0.890 | −0.091 |
| 0.01 | 1.085 | 0.278 | 0.103 | **0.896** | **−0.014** |
| 0.03 | 1.302 | 0.362 | 0.139 | 0.858 | +0.105 |
| ∞ *(scale-only)* | **3.006** | 1.155 | 0.529 | 0.569 | +0.927 |

At `b_prior = 0.003`, Rel improves 8 % and RMSE 5 % **with no near-field cost** (inside-ROI
medAE 0.031 → 0.030). Expected a trade; there isn't one at that strength, which says the
unridged fit was mildly mis-fitting `b` rather than that near and far are in tension.
Past 0.01 the trade appears and it is bad.

### The ceiling is partly PROTECTIVE — do not remove it

**Scale-only triples RMSE** (1.012 → 3.006) and flips bias to **+0.927 m**, a large
over-read. At far range disparity → small, so `D = 1/(a·disp)` becomes extremely sensitive to
disparity noise: **the `b` term regularises against that noise.** Removing it swaps a bounded
bias for unbounded variance.

So "remove the pathology structurally" (log-depth reparameterisation, unconstrained `b`) is
refuted by measurement, not just untested. **`1/b` is a bias/variance knob, not a bug.**

### What actually shipped

- **`roi.py` is wired into `pipeline.py`** — it had been written, measured, and then imported
  by *nothing*, while range weighting was disabled on the grounds that ROI replaced it. The
  pipeline was running with **neither** mitigation.
- **ROI fit weighting is OFF by default.** A/B'd on 200 logs: it moved the ceiling 1.33 →
  1.11 m and degraded the far field, buying only 0.109 → 0.101 m at 0.5–1.5 m. It narrows
  scope; it does not fix the under-read.
- **Stage 7d floors σ outside the ROI** at 100 % relative uncertainty and returns `roi_mask`,
  so `/depth_var` no longer reports ±8 cm on pixels that are metres wrong.
- **`to_metric_depth_valid()`** marks pixels clipped at the `1e-4` singularity as invalid
  rather than emitting an arbitrary 10,000 m the fit has no information about.
- **`b_prior` stays 0 in deployment — validated, and it does NOT transfer.** Swept on 300 of
  our own logs with [`ceiling_diag.py`](../tools/diagnostics/ceiling_diag.py):

  | `b_prior` | ceiling ↑ | 0.5–1.5 m ↓ | 3–6.5 m ↓ | MAE ↓ | medAE ↓ |
  |---|---|---|---|---|---|
  | **0** *(default)* | 1.43 m | **0.104 m** | 2.355 m | 0.301 m | **0.060 m** |
  | 0.003 *(ZJU-L5's pick)* | 1.45 m | 0.105 m | 2.340 m | 0.301 m | 0.060 m |
  | 0.01 | 1.50 m | 0.106 m | 2.308 m | 0.300 m | 0.061 m |
  | 0.03 | 1.64 m | 0.112 m | **2.223 m** | **0.299 m** | 0.063 m |

  At 0.003 the ceiling moves 1.43 → 1.45 m and nothing else changes. Even at 0.03 the −5.6 %
  far-field gain is paid for in the near field and medAE gets worse. The cause is the anchor
  distribution: ours is median **0.49 m** with 51 % under half a metre, so `b` is dominated by
  near anchors and the same ridge strength barely registers, where ZJU-L5's median 1.59 m
  anchors let it bite. A hyperparameter tuned on one sensor's range distribution does not
  carry to another's — which is why it was swept here before being adopted.

### The honest limitation to state in the paper

None of these makes range extrapolation *work* — they make it degrade more gracefully. The
affine form bounds expressible depth at `1/b`; we measured where that bites, on two sensors;
the σ channel does not report it, and now floors itself outside the ROI instead. Wider ToF
coverage addresses angular reach, **not** the range ceiling, so it is the wrong axis for this
problem. Temporal fusion is the only candidate that acquires the missing information.

### Done

- **Sensors, MCU → ROS.** ESP32-C6 firmware streams TMF8829 ToF frames; `tof_driver`
  (serial → `ToFFrame`) and `camera` (Arducam IMX219 CSI) nodes publish live, plus
  `tof_heatmap` + `dual_view` for inspection.
- **Perception pipeline, structurally complete** (stages 2–8): backbone → zone
  projection → closed-form anchoring → analytic per-pixel variance → residual →
  unprojection. Publishes `/cloud`, `/depth`, `/depth_var`. The pure-numpy core
  ([src/ringfusion_perception/ringfusion_perception/pipeline.py](src/ringfusion_perception/ringfusion_perception/pipeline.py))
  has a PC test suite ([src/ringfusion_perception/test/](src/ringfusion_perception/test/), 5/5 passing — no ROS/CUDA needed). See [Perception](#perception).
- **Both networks are pluggable and default to mocks**, so the pipeline runs now and
  swaps to real engines with a launch arg (`backbone_engine:=…`, `residual_engine:=…`) —
  no code change. `MockResidual` is the exact identity, so output = the closed-form fit.
- **Both real engines run live on the Orin** (2026-07-25). `student_v3_fp16.engine` +
  `residual_fp16.engine` load in `perception_node` and hold ~13–14 Hz end-to-end. Two
  independent code paths (`perception_node` and a standalone A-vs-B renderer) agree on
  centre depth to **2 mm**, so `residual.py` and `gpu_ops.py` are consistent.
- **Anchor re-splatting** (`gpu_ops.resplat_anchors`). Resizing the sparse anchor maps from
  2 MP down to the 288×384 engine input dropped ~95% of the ToF anchors; re-projecting each
  nonzero pixel onto the engine grid instead keeps **~97%** (991/1024 measured). Network B
  was trained at full anchor density, so this removes a mismatch between how the network was trained and how it is used.
- **Training + export code written** ([../training/](../training/), [../tools/](../tools/)): Network A
  (student backbone), Network B (residual, measured ~0.46M params), distillation + NLL
  losses, datasets, teacher caching, ONNX + TensorRT build scripts. Torch-only parts
  smoke-tested; the residual reuses the deployed anchoring math so training matches inference.

### Not done / blocked

- **Full training set** — only a **2000-image pilot** has been collected so far. The
  full ~15–20k (deployment environment, via `collect_frames`) still needs gathering for
  the final-quality backbone; the pilot is enough to prove the pipeline, not to ship.
- **v3 under-corrects** — mean |B−A| is only **0.019 m**. Dropping `--struct-weight` from 0.3
  to 0.15 improved accuracy, so the optimum is probably lower still. Sweep it.
- **The v2→v3 win is not attributable** — v3 changed the geometry *and* the structure weight
  together. A control run at `--struct-weight 0.3` on corrected geometry would separate them.
- **ToF covers only 12.4 % of the frame** — the 32×32 zone grid projects into a 567×443 px box
  in the middle of the 1640×1232 rectified image. The remaining 87.6% has no metric support,
  so it is monocular extrapolation carrying the affine fit. This is a **geometry/FOV
  limitation, not a bug**, but it bounds how much of the depth map can be believed.
- **Checkpoint selection is noisy** — `--val-frac` defaults to 0.05, i.e. ~62 samples against
  a heavy-tailed NLL loss. Validation loss swung 1.30 → −0.34 across the v3 run while the
  training curve descended smoothly. `best` and `last` happened to land statistically
  identical (MAE 0.198 both), so it did not bite this time; widen the split before it does.
- **Uncertainty is much better but still mildly overconfident** — coverage at ±1σ is **0.590**
  against the 0.683 ideal (v1 was 0.612, v2 a badly overconfident 0.264). The *shape* is now
  right — `corr(σ, |error|)` 0.943, σ tight at anchors and wide where extrapolating — but the
  absolute width wants recalibration.
- **Network B's supervision is sparse and range-gated, not absent.** B *is* trained on real
  ToF via the held-out-anchor split in
  [`anchoring_bridge.build_real_supervision`](../training/anchoring_bridge.py) — 75% of each
  frame's valid zones drive the fit and feed B's input channels, the other 25% are withheld
  from input and splatted as sparse targets, so B is scored where a real measurement exists
  but it got no input. The design is sound; its *shape* is the problem:

  | Limitation | Consequence measured on-robot |
  |---|---|
  | hold-out is ~175 zones/frame, all inside the **12.4% ToF box** | 87.6% of each frame is never supervised → far field runs free |
  | `max_range = 5.0 m` gate | B has **never seen a target beyond 5 m** → 6.29 m and 20 m-clamp pixels are pure extrapolation |
  | `min_range = 0.15 m` gate | near floor below 15 cm unsupervised → ground-plane under-read |
  | targets land on the **13 px anchor lattice** | plausibly the 13 px component B injects and A lacks |
  | variance head sees only those few hundred pixels, all <5 m, all central | σ saturates near ~1 m, no spatial signal |

  So B is well-constrained inside the central box under 5 m and unconstrained everywhere
  else — which is exactly the map of where it misbehaves. Dense GT (synthetic/LiDAR/OAK-D)
  would help but is **not** the only lever; see steps 7–8.
- **All paper numbers are placeholders** — FPS, accuracy, and param counts (student measured
  **3.66M**, residual measured 0.46M; the doc's "6.1M" student was a placeholder). Nothing is
  submittable until measured on real hardware.
- The dev PC (Windows) **cannot build the ROS workspace** (`rclpy` is Linux/ROS) — build and
  train on the Jetson or a Linux GPU box. The pure-numpy pipeline + training code do run on the PC.

### Next steps (in dependency order)

| # | Step | Needs | Note |
|---|---|---|---|
| 0 | ~~Run Depth Anything V2 on real rectified Arducam frames (sanity)~~ | — | **DONE** (B1 passed → `step0.png`) |
| 1 | ~~Fisheye calibration → real intrinsics in `calibration.yaml`~~ | — | **DONE** (RMS 0.5406 px, `identity=False` verified) |
| 2 | Collect rectified images (`collect_frames`) | camera | **pilot done (2000)**; full ~15–20k still to gather |
| 3 | ~~`cache_teacher.py` — cache DA V2 disparity targets~~ | GPU | **DONE** (2000 cached) |
| 4 | ~~`distill_backbone.py` → student, validate vs teacher~~ | 3 | **DONE** (pilot: val_ssi 3.51, ρ 0.9962) |
| 5 | ~~`export_onnx` → `build_engine`, **measure Orin FPS**~~ | 4, Jetson | **DONE** (~13–14 Hz with A+B live) |
| 6 | Run DEPTHOR-Small on the same Orin | Jetson | efficiency claim |
| 7a | ~~Structure loss tying B to A's relative geometry off-anchor~~ | — | **IMPLEMENTED** — `losses.structure_loss`, `--struct-weight` (default 0.3) |
| 7b | ~~Widen B's supervision (no new hardware)~~ | — | **IMPLEMENTED** — `max_range` 5.0 → 6.5. Small gain (0.23% of zones); the sensor, not the gate, is what limits this |
| 7c | ~~Re-weight the closed-form fit~~ | — | **IMPLEMENTED** — `anchoring.range_weights`, `w = z^1`. Note: **theory said z², measurement said z¹** |
| 8 | ~~Re-train B with the new objective~~ | 7 | **DONE** — `residual_v2`, measured 2026-07-28: blowup + lattice fixed, but over-regularised |
| 8b | ~~Sweep `--struct-weight` down from 0.3~~ | 8 | **DONE at 0.15** → `residual_v3`, the first B that beats the closed-form path |
| 8c | ~~Fix the ToF↔camera projection~~ | — | **DONE 2026-07-28** — mirror + FOV swap; ρ 0.737 → 0.917 |
| 8d | **Sweep `--struct-weight` below 0.15 + control run at 0.3 on fixed geometry** ← **you are here** | 8b, 8c | v3 still only deviates 0.019 m from A; and its win confounds two changes |
| 8e | Widen `--val-frac`, then recalibrate σ | 8d | ~62 val samples is too few to select on; coverage 0.590 vs 0.683 ideal |
| 9 | Ground-truth collection (synthetic/LiDAR/OAK-D) | — | only for the 87.6 % outside the ToF box; **not** a prerequisite for 7–8 |
| 10 | ~~Re-test on a non-repeating scene to separate texture banding from anchor-pitch artefacts~~ | — | **DONE** (two-scene control, below — both effects confirmed and separated) |
| 11 | Re-distill Network A on the full ~15–20k set | 2 | now known **not** to be the bottleneck — the 0.75 teacher ceiling was a projection artefact |

**Steps 0–6 need no ground truth and deliver the two headline results** (Orin throughput and
the DEPTHOR comparison) — do them first. Because the residual is zero-initialized to the
identity, the system ships and produces the closed-form result before Network B exists.

## Packages

| Package | Type | Purpose |
|---|---|---|
| `ringfusion_msgs` | ament_cmake | Custom message definitions (`ToFFrame.msg`) |
| `ringfusion_drivers` | ament_python | ToF hub (`tof_driver`), Arducam camera (`camera`), ToF heatmap colorizer (`tof_heatmap`), local combined viewer (`dual_view`) |
| `ringfusion_perception` | ament_python | Mono depth + ToF anchoring perception node (see [Perception](#perception)) |
| `ringfusion_bringup` | ament_cmake | Launch files and extrinsic calibration config |

> `ringfusion_msgs` must declare `<export><build_type>ament_cmake</build_type></export>` in its `package.xml`, or colcon misidentifies it as a plain `catkin` package and skips it during the build.

## First-time setup

```bash
source /opt/ros/humble/setup.bash
cd ~/RingFusion/ros2_ws
rosdep install --from-paths src --ignore-src -r -y   # install missing dependencies
```

## Building

```bash
# Build everything
colcon build --symlink-install

# Build a single package (and nothing else)
colcon build --symlink-install --packages-select ringfusion_msgs

# Build a package and everything that depends on it
colcon build --symlink-install --packages-up-to ringfusion_bringup

# Clean build (wipe generated artifacts, not src/)
rm -rf build/ install/ log/
colcon build --symlink-install
```

After building, source the overlay in every new terminal:

```bash
source install/setup.bash
```

## Running

```bash
# Just view both feeds (camera + ToF heatmap) — see "Viewing both feeds" below
ros2 launch ringfusion_bringup feeds.launch.py

# Full fusion module: ToF hub + camera + perception (-> /cloud, /depth)
ros2 launch ringfusion_bringup single_module.launch.py port:=/dev/ttyACM1

# Same, but feed a still image instead of the live CSI camera
ros2 launch ringfusion_bringup single_module.launch.py image:=/path/to/shot.jpg

# Run a single node directly
ros2 run ringfusion_drivers tof_driver
ros2 run ringfusion_drivers camera
ros2 run ringfusion_drivers tof_heatmap
ros2 run ringfusion_perception perception
ros2 run ringfusion_perception collect_frames   # collect rectified training images (see below)
```

View the output point cloud in `rviz2`: add a `PointCloud2` display on `/cloud` with fixed frame `cam_0`.

## Perception

`perception_node` caches the latest camera frame + ToF frame and runs the pure-numpy
pipeline (`pipeline.run`) whenever a ToF frame arrives (~15 Hz; the limit is perception,
not the ToF, since 2026-07-28 — see [Throughput, reconciled](#throughput-reconciled) and
Performance notes). Heavy per-pixel math is GPU-offloaded via `gpu_ops.py`. It publishes:

| Topic | Type | Contents |
|---|---|---|
| `/cloud` | `sensor_msgs/PointCloud2` | metric point cloud (the goal) |
| `/depth` | `sensor_msgs/Image` `32FC1` | metric depth map |
| `/depth_var` | `sensor_msgs/Image` `32FC1` | per-pixel depth variance (calibrated uncertainty) |

The pipeline runs two neural networks, both **pluggable** so the workspace runs today
with mocks and swaps to real engines on the Jetson with no other changes:

- **Backbone** (Network A, `backbone.py`) — monocular relative disparity. `MockBackbone`
  by default; pass `backbone_engine:=student_int8.engine` to use `TensorRTBackbone`.
- **Residual** (Network B, `residual.py`) — per-pixel correction to the affine fit plus
  extra variance. `MockResidual` (the exact identity → output equals the closed-form
  fit) by default; pass `residual_engine:=residual_fp16.engine` to use `ResidualRefiner`.

The math between them (zone projection, closed-form anchoring, analytic covariance,
unprojection) has no learned parameters. See `RingFusion_technical_reference_updateP2.md`.

**Running the real TensorRT backbone (retires the mock).** Once an engine is built on
the Orin (see [../training/README.md](../training/README.md) §3), pass it in and the
node loads `TensorRTBackbone` instead of `MockBackbone` — no code change:

```bash
ros2 launch ringfusion_bringup single_module.launch.py \
    backbone_engine:=$HOME/RingFusion/student_int8.engine port:=/dev/ttyACM1
ros2 topic hz /depth      # headline FPS; compare student_int8.engine vs student_fp16.engine
```

The runtime (`trt_util.TRTRunner`) uses **torch** for device memory + the CUDA stream,
**not pycuda** (pycuda isn't installed on the Jetson and is painful to build). So the
only runtime deps are `tensorrt` (JetPack) + `torch` (both present). Before the full
launch you can smoke-test the runtime in isolation: instantiate `TRTRunner` with an
engine and run one inference — a clear error beats a buried ROS failure.

**Stage 1 rectification.** The lens is a ~155° fisheye, so `perception_node` remaps each
frame to a rectilinear (pinhole) image before the pipeline runs (`rectify.FisheyeRectifier`),
and everything downstream — zone projection *and* cloud unprojection — then uses one
consistent pinhole `K`. A zero-distortion fisheye is still an *equidistant* fisheye, and the
nominal focal length is close to its true value, so rectification is **active even with the
nominal calibration** (a rough but real de-warp); real calibration just refines the
lens-specific coefficients. It falls back to an identity passthrough only for a `pinhole`
model or if cv2 is missing.

**See it live:** `ros2 run ringfusion_perception rectify_view` publishes `/rectify_compare`
(raw | rectified, side by side). View it at
`http://<jetson-ip>:8080/stream?topic=/rectify_compare` (with `web_video_server` running).
Point at straight edges — door frames, floor tiles — to confirm the de-warp; tune
`rectify:` `fov_scale`/`balance` in `calibration.yaml` and relaunch to adjust the crop.

**Calibrating the lens** (fills in the real intrinsics that activate rectification):

```bash
# 1. Collect ~20 checkerboard views (headless auto-capture; move the board around,
#    especially into the corners where fisheye distortion is strongest)
PYTHONNOUSERSITE=1 python tools/calibrate_camera.py --capture calib_imgs --cols 9 --rows 6
# 2. Calibrate and print the yaml block to paste into calibration.yaml
python tools/calibrate_camera.py --images calib_imgs --cols 9 --rows 6 --square-mm 25
```

Paste the printed `camera:` + `rectify:` block into
[src/ringfusion_bringup/config/calibration.yaml](src/ringfusion_bringup/config/calibration.yaml).
Intrinsics are resolution-specific — calibrate at the resolution you deploy.

Parameters (`perception_node`): `calib`, `frame_id`, `backbone_engine`, `residual_engine`,
`min_confidence` (default `-1` = ignore ToF confidence and weight all zones equally; set
`>= 0` to reject weak zones and weight the fit by confidence).

**Testing on a dev PC (no ROS/CUDA/cv2 needed).** `pipeline.py`, `geometry.py`,
`anchoring.py`, `residual.MockResidual`, and `backbone.MockBackbone` are pure numpy, so
the whole pipeline runs and is unit-tested off-robot:

```bash
cd src/ringfusion_perception
python -m pytest test/ -v          # or: python test/test_pipeline.py
```

## Collecting training images (backbone distillation — B2)

The backbone (Network A) is trained by **distillation**: Depth Anything V2 (the
"teacher") auto-generates the depth targets, so collection needs **images only — no
measured depth**. The `collect_frames` node banks **rectified** frames off `/image`
through the exact `color-correct → rectify` path the robot runs at inference (the
camera node has already white-balanced/contrast-corrected the frame; this node then
rectifies it with the same `FisheyeRectifier` + `calibration.yaml` the pipeline uses),
straight into a folder that `training/cache_teacher.py` reads as-is.

**Why the DEPLOYMENT ENVIRONMENT, not a generic/online dataset.** Distillation makes
the student copy the teacher's depth *on whatever image distribution you show it*, so
the student ends up good at scenes that look like its training set. Your deployed
frames have a specific look — this rectified 155° fisheye crop, the IMX219's
color/noise, your scenes and lighting — that internet photos (different camera, lens,
projection, content) do not share, and you **cannot rectify a normal photo to mimic
your lens**. So the bulk of the data must be your own rectified frames. External
images can be a **minority supplement** for diversity if your environment is very
homogeneous, but they never substitute for in-domain data — and since your own frames
are free to label (the teacher does it), there's little reason to. To stretch a
smaller set, prefer **augmentation of your own frames** (the distill trainer already
augments) over importing foreign images. Rule of thumb: **diversity > volume** —
~15–20k varied in-domain frames is the target, but you can pilot with ~5k and re-run
(the cached teacher makes re-distilling cheap).

**Lock `fov_scale`/`balance` before collecting.** The whole dataset *and* the deployed
pipeline must use the same `rectify:` settings in `calibration.yaml`, or the training
frames won't match what the robot sees. (Eyeball the crop first with `rectify_view`.)

### Run it (two terminals, both sourced)

```bash
# one-time: build so ros2 run sees the node
cd ~/RingFusion/ros2_ws
colcon build --symlink-install --packages-select ringfusion_perception
source install/setup.bash
```

```bash
# Terminal 1 — camera (publishes the color-corrected /image)
cd ~/RingFusion/ros2_ws && source install/setup.bash
PYTHONNOUSERSITE=1 ros2 run ringfusion_drivers camera
```

```bash
# Terminal 2 — collector (live preview window on the Jetson's monitor)
cd ~/RingFusion/ros2_ws && source install/setup.bash
ros2 run ringfusion_perception collect_frames --ros-args \
  -p calib:=$HOME/RingFusion/ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
  -p out_dir:=$HOME/RingFusion/data/rect \
  -p target:=20000
```

Startup log should read `identity=False` (real de-warp active) and report how many
frames are already on disk. If `imshow` errors, prefix Terminal 2 with
`PYTHONNOUSERSITE=1` too (forces JetPack's GTK-enabled OpenCV). Needs the Jetson's
own monitor — not over SSH.

### Controls (shown in the preview window)

| Key | Action |
|---|---|
| **SPACE / y** | save the current rectified frame (manual) |
| **c** | toggle **continuous** auto-capture on/off |
| **q / ESC** | quit |

The overlay shows `[count/target]`, the mode, and a live `NEW`/`similar` + `sharp NNN`/`BLURRY NNN` status.

### Two automatic quality gates (continuous mode)

- **Dedup** — a frame counts as new only if it differs enough from the last *saved*
  frame (mean abs gray diff > `dedup_thresh`, default 8). Stops you banking 500 copies
  of the same wall as you stand still.
- **Sharpness / blur** — continuous mode saves only frames with variance-of-Laplacian
  ≥ `blur_thresh` (default 60), so **motion-blurred frames from walking are rejected**.
  Watch the live `sharp NNN` readout: stand still to see the sharp value, wave the
  camera to see it drop, set `blur_thresh` between the two (e.g. `-p blur_thresh:=100`).
  **Manual `y`-save always honours your keypress** (shows the BLURRY tag as a warning).

### Parameters

`out_dir` (default `data/rect`), `calib`, `target` (default 20000), `dedup_thresh`
(default 8.0), `min_interval` (seconds between auto-saves, default 0.3), `blur_thresh`
(default 60.0).

### Field-session tips

- **Cover the variety axes:** different areas/rooms, distances (close *and* far),
  angles/heights, lighting conditions. Diversity is what matters, not raw count.
- **Beat blur physically:** walk slowly or **pause-step** (a half-second stop lets a
  clean frame land, and continuous mode grabs it). Motion blur comes from long
  exposure, which comes from dim scenes — **brighter areas → sharper frames**, so move
  slowest where it's dark.
- **Manual for deliberate shots, continuous for bulk.** Hold `c` on while walking a
  route; tap it off and use `y` for specific poses you care about.
- **Stop and resume anytime:** quit with `q`; re-running **appends** (it continues
  numbering after the frames already in `out_dir`), so collect across several sessions.

## Camera hardware (Arducam IMX219 on the B0472 CSI adapter)

The `camera` node (`ringfusion_drivers/camera.py`) drives the Arducam IMX219 wide-angle
module through `nvarguscamerasrc`, the native Jetson ISP path — this requires the Arducam
kernel driver to be installed first (it does not come with JetPack by default).

**One-time driver install** (Jetson AGX Orin, JetPack 6 / L4T 36.5):

```bash
cd ~
wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
chmod +x install_full.sh
./install_full.sh -m imx219      # downloads the .deb matching your exact kernel build
sudo reboot
```

Verify after reboot:

```bash
dpkg -l | grep arducam                 # arducam-nvidia-l4t-kernel should be listed
ls /dev/video0                         # should exist
v4l2-ctl --list-devices                # should show "imx219 ..." on /dev/video0
media-ctl -p                           # should show an imx219 sensor entity, not an empty topology
```

**Raw pipeline smoke test** (bypasses ROS entirely — good first check):

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=21/1" ! \
  nvvidconv ! fakesink
```

If this reports `Argus Correctable Error Status` / `CANCELLED` at the very end after
`Got EOS`, that's normal teardown noise for a fixed-buffer-count pipeline — not a failure.

**Rotation and color:** the module is mounted upside down on the ring, so `camera_node`
rotates 180° by default (`flip` param, `nvvidconv flip-method=2`; pass `flip:=0` to disable).
The IMX219 has no ISP color-tuning profile installed, so raw frames come out with a strong
color cast and crushed contrast — `camera.py`'s `ArducamCSI.read()` applies a gray-world
white-balance + contrast-stretch correction (via a LUT, ~8ms/frame) before publishing.

## Viewing both feeds (camera + ToF heatmap)

One command brings up the whole viewing stack — camera, ToF driver, heatmap colorizer,
the local on-monitor window, and the browser server:

```bash
sudo apt-get install -y ros-humble-web-video-server   # one-time

ros2 launch ringfusion_bringup feeds.launch.py
```

Arguments: `port` (ToF serial, default `/dev/ttyACM1`), `fps` (camera capture, default 30),
`view` (local window, default true), `web` (browser server, default true).

**Local viewing — lowest latency, recommended.** Run the launch from a terminal *inside the
Jetson's desktop session* (monitor plugged in) and the `dual_view` window opens automatically,
showing camera + ToF side by side with live Hz. It subscribes to the ROS topics directly —
no network hop, no JPEG re-encode, which are the two things that add lag. Over SSH with no
display, pass `view:=false`.

**Browser viewing — from another machine on the network:**

```
http://<jetson-ip>:8080/stream?topic=/image&quality=60
http://<jetson-ip>:8080/stream?topic=/tof_heatmap
```

(find `<jetson-ip>` with `hostname -I`). **On the `quality` param:** at the default quality
(95), a 1280x720 JPEG stream needs ~20Mbps sustained. If the viewing device's WiFi can't hold
that, the server's write queue backs up into a growing multi-second lag even though capture
(~8ms) and ROS transport (~11ms) stay fast. `quality=60` cuts it to ~4Mbps with no visible
quality loss; drop to `30`-`40` if lag persists. Local viewing avoids this entirely.

### Performance notes (Jetson AGX Orin)

> **⚠ Superseded on the question of what the bottleneck is.** The 2026-07-23 table below
> reads the ToF as the limit and perception as having ~2× headroom. Re-profiled 2026-07-28/30
> that is **no longer true** — see [Throughput, reconciled](#throughput-reconciled) directly
> after it. The per-topic *rates* still stand; the *attribution* does not.

**Measured live** (2026-07-23), full pipeline, MAXN + `jetson_clocks`, backbone on TensorRT
FP16, **binary ToF firmware**:

| Topic | Component | Rate | Limited by |
|-------|-----------|------|------------|
| `/image` | camera | **~28.5 Hz** | IMX219 sensor mode (30 fps cap on 1640×1232) |
| `/tof` | ToF driver | **~16 Hz** *(subframes; see below)* | ToF sensor + USB delivery |
| `/depth` | perception (Network A + anchoring) | **~14.8 Hz** | *attribution superseded* |
| `/cloud` | perception unprojection | **~15.6 Hz** | *attribution superseded* |

**Fused pipeline: ~15 Hz** — up from ~8 Hz, roughly doubled by the binary ToF firmware +
persistent per-subframe assembler (see ToF note below).

### Throughput, reconciled

Four different rate figures appear in this file and they had been used interchangeably. They
measure different scopes, and the 2026-07-23 conclusion that "the ToF is the single
bottleneck ... perception has ~2× headroom" **does not survive re-profiling**:

| figure | what it actually measures | source |
|---|---|---|
| **~27 Hz** "backbone capable" | backbone **inference alone**, no anchoring, no residual, no rectification, no ROS | 2026-07-23 |
| **19.6 Hz** (50.9 ms) | `pipeline.run()` offline: backbone + residual + anchoring + variance + cloud, at 1640×1232, **no ROS, no rectification, blend/ROI off** | `time_pipeline.py`, 2026-07-30 |
| **14.1 Hz** (70.7 ms) | the same, **blend/ROI on** — the configuration that actually ships | `time_pipeline.py`, 2026-07-30 |
| **10.4 Hz** (96 ms) | the deployed `perception` node end-to-end, **including** ROS transport and CPU rectification, blend/ROI on | `rate_live.py`, 2026-07-30 |
| **13.7 Hz** (73 ms) | the same deployed node *before* blend/ROI existed — quoted historically, not a current figure | 2026-07-28 |
| **~16 Hz** `/tof` | ToF **subframe** arrivals. Complete 32×32 maps assemble at **8.3 Hz** (60.3 ms median inter-arrival, CV 0.142 = sensor-clocked, integration-bound) | `profile_stages.py` |

So: **perception is the constraint, not the ToF at 60.3 ms.** The "~2× headroom" claim
compared the backbone in isolation against the full ToF path — different scopes on either
side of the comparison. That remains true at 96 ms deployed: the pipeline is slower than the
ToF's 60.3 ms subframe cadence, though it now clears the 8.3 Hz *complete-map* rate.

The per-stage split is measured directly rather than estimated — see
[per-stage cost on the robot](#per-stage-cost-on-the-robot). The largest items are the
residual (28.7 ms) and the backbone (10.3 ms); rectification, once ~15 % of the frame, is now
4.4 ms and runs once per *used* frame rather than once per camera frame.

Consequence for the roadmap: **raising the ToF rate buys nothing until perception is faster**,
and switching ToF I²C→SPI is bounded to ~+31 % anyway because integration time is ~77 % of the
ToF frame period.

### Stage 7c/4b/7d cost, and the seven fixes that made them affordable

Measured with [`time_pipeline.py`](../tools/diagnostics/time_pipeline.py), `pipeline.run()`,
backbone + `residual_v4_last`, nothing else on the GPU.

**As first written the two stages cost 7.4× — unusable:**

| 1640×1232 | blend | ROI | median ms ↓ | Hz ↑ |
|---|---|---|---|---|
| before | off | off | 52.2 ms | 19.2 |
| before | on | off | 108.9 ms | 9.2 |
| before | **on** | **on** | **379.9 ms** | **2.6** |

Component profile located it: RANSAC plane fit **20.9 ms/frame**, `pixel_roi_mask` at full
resolution **283 ms**, the blend's 2 MP CPU `distanceTransformWithLabels` **56 ms**.

**Four fixes:**

1. **`roi.pixel_roi_mask(stride=)` was broken** — it wrote `out[::stride, ::stride] = m` and
   left the rest `False`, collapsing the inside-fraction 0.683 → 0.011 at stride 8. So the
   one knob intended for this was unusable. Fixed to upsample (`np.repeat`). Dead code until
   wired in, so never exercised. At stride 8: **283 → 5.6 ms**.
2. **Blend distance transform at 1/4 resolution** (`BLEND_SCALE`). The weight is a smoothstep
   over ~20 px and the nearest-anchor depth is a Voronoi diagram, so both survive it.
   Validated on real frames against full resolution: mean |diff| **0.0022 m**, p99 0.040 m.
   **56 → 33 ms.**
   *(A synthetic test with randomly scattered anchors suggested p99 1.7 m — misleading,
   because real anchors are spatially coherent and confined to the ToF box. Validate on real
   frames.)*
3. **GPU offload of the full-frame tails** — `gpu_ops.blend_apply` and
   `gpu_ops.roi_sigma_floor`. Those were ~8 and ~3 full-frame float passes respectively;
   matches numpy to 1e-6. Same problem and same remedy as `ResidualRefiner.refine`, which was
   ~70 ms on the CPU before it was offloaded.
4. **`PlaneTracker(refit_every=10)`** — the camera is rigidly mounted and the plane is
   near-constant (normal stable to 0.05, height to 0.02 m across three captures), so
   re-RANSACing every frame bought nothing. Cached EMA plane in between, which is already the
   documented behaviour for frames that cannot fit one. **20.9 → ~2 ms** amortised.
   *(Superseded 2026-07-30 — the fit itself is now 1.18 ms, so it refits every frame. See
   below.)*

### A fifth fix, and the bug that stopped the first four reaching the robot (2026-07-30)

**Background, if you are new to this part.** The pipeline fits a flat *ground plane* to the
ToF measurements each frame — essentially "where is the floor?" — so it can tell which pixels
are floor a robot could drive over and which are wall, shelving, or ceiling. It finds that
plane with an algorithm called **RANSAC**: guess a plane through 3 randomly chosen points,
count how many of the other points lie close to it, repeat 200 times, keep the best guess.
Fix 4 above sped this up by only re-running it every 10th frame and reusing the previous
answer in between (that reuse is what `PlaneTracker` does).

**The bug.** Checking the deployed code before the live timing run, `perception_node.py` — the
program that actually runs on the robot — never passed a `PlaneTracker` in. So on the robot,
RANSAC ran from scratch on **every single frame**, costing the full ~20 ms each time. Fix 4
worked, but only in the offline test harness. Nobody had checked that the two ran the same
way, so the published 12.4 Hz was measured on a faster configuration than the one shipping.

**Why that matters beyond speed.** Reusing the previous plane is not just a shortcut. Some
frames genuinely cannot find a floor — the robot is nose-in to a wall, or the floor is out of
view — and the reuse is what supplies a sensible answer on those frames. Without a tracker,
those frames had nothing to fall back on.

Fixed: the node now keeps one tracker for as long as it runs.

**Then the fifth fix — making RANSAC itself fast, so the shortcut is no longer needed:**

5. **`fit_ground_plane` rewritten to work on all 200 guesses at once.** The old code was a
   Python `for` loop testing one guess per pass. Python loops are slow; the arithmetic inside
   each pass is trivial. Rewriting it so all 200 guesses are scored in one batch of array
   maths (this is called **vectorising**) removed almost all of that overhead. The ToF is a
   32×32 grid, so there are at most **1024 points** — a tiny problem that was slow purely
   because of how it was written, not because of how much work it required.

   Measured over 120 logged frames: **19.6 ms → 1.18 ms** (p90 1.31, max 4.47), a **16.6×
   speedup**.

   *(19.6 ms here vs the 20.9 ms quoted above is the same code timed two ways — 20.9 ms
   measured inside the running pipeline, 19.6 ms measured calling the function directly.
   Not a contradiction, just two different harnesses.)*

   Two things came out of that rewrite:

   - **A hidden slowdown in the final step.** After picking the best guess, the code
     refines it using all the nearby points. It called `np.linalg.svd`, which by default
     also computes a large matrix the code never uses — at the typical 344 nearby points,
     a 344×344 one — just to read out three numbers. Swapped for `eigh` on a 3×3 matrix,
     which gives the identical answer (this is exact, not an approximation) and does not
     grow as more points are involved: **1.51 → 1.18 ms**. The 19.6 ms loop had been large
     enough to hide it.
   - **The every-10th-frame shortcut is no longer worth it**, so it now refits every frame.
     Two reasons. Skipping frames left a ~20 ms *spike* every 10th frame, which a real-time
     consumer feels more than a slightly higher average. And RANSAC's answer wobbles from
     frame to frame because it picks its guesses at random: over 5 different random seeds on
     60 frames, the plane's tilt moved by a median 4.0° but **as much as 21.6°** in the worst
     10%. Holding one unlucky answer for 10 frames therefore propagates that error; refitting
     every frame lets the smoothing average the wobble out instead.

> **The GPU was tried here and lost — worth recording so nobody retries it.** A CUDA version
> of the same calculation measured **2.6–2.9 ms against numpy's 1.18 ms**. The reason is size:
> at ~1.6 MB the job is too small to be worth shipping to the GPU, because copying data there
> and launching the work costs more than the arithmetic saves. Even if the GPU did its part
> instantly, the final refinement step stays on the CPU and would still cost ~0.4 ms.
>
> This is the **opposite** of the per-pixel work in `gpu_ops`, where the GPU wins easily —
> that operates on 2 million pixels, this on 1024 points. **Problem size decides, not how
> parallel the maths looks.** `roi.py` carries a comment saying so.

**After** (re-measured 2026-07-30 with fixes 1–5 and `refit_every=1`, 40 iters, clean GPU):

| resolution | blend | ROI | median ms ↓ | p90 ms ↓ | Hz ↑ |
|---|---|---|---|---|---|
| 1640×1232 | off | off | 50.9 ms | 52.7 | 19.6 |
| 1640×1232 | on | off | 64.8 ms | 65.0 | 15.4 |
| **1640×1232** | **on** | **on** | **70.7 ms** | **70.9** | **14.1** |
| 640×480 | off | off | 20.0 ms | 20.0 | 50.1 |
| 640×480 | on | off | 23.4 ms | 23.6 | 42.7 |
| **640×480** | **on** | **on** | **25.8 ms** | **26.0** | **38.7** |

Both stages cost **19.8 ms** together at full resolution, down from 328 ms.

*(Superseded values: before the two GPU rewrites described further below, the same table read
81.4 ms / 12.3 Hz at full resolution and 28.4 ms at 640×480. The narrative immediately below
was written against those numbers and is kept because its point — which configuration a
measurement describes — is what mattered.)*

**Why this table barely moved, and why that is still the point.** 81.4 ms is essentially the
same as the 80.7 ms measured before fix 5, which looks like the work achieved nothing. It did
not achieve much *here* — because this offline harness already used a `PlaneTracker`, so it
was already paying about 2 ms rather than 20 ms, and making a 20 ms step 16× faster cannot
save much when you were only running it every 10th frame anyway.

What changed is **which setup these numbers describe**. They now describe the program that
runs on the robot. Before, that program had no tracker and paid the full ~20 ms every frame,
so it was slower than this table ever showed. The p90 column is the other half of the story:
at 82.8 ms it sits just 1.4 ms above the median, meaning frames now take a consistent amount
of time instead of one in ten taking ~20 ms longer.

> **⚠️ The extrapolation below was WRONG. Superseded by measurement — see
> [the deployed rate, measured](#the-deployed-rate-measured-2026-07-30). Kept because the
> way it failed is the useful part.**
>
> The reasoning was: `pipeline.run()` is 50.3 ms here, the full program on the robot had
> measured 73 ms, so ROS transport and undistortion add ~20 ms. Adding that to the 81.4 ms
> above predicts ~101 ms ≈ **9.9 Hz**.
>
> The robot delivered **7.2 Hz (139 ms)** — 38 ms slower than predicted. The flaw was
> assuming the overhead is a *constant* you can add on. It is not: measured on the robot the
> non-pipeline overhead really is ~20 ms, but `pipeline.run` itself costs 120 ms there
> against 81 ms offline, because the extra full-frame passes contend for memory bandwidth
> that an isolated benchmark has entirely to itself. **Overhead on a shared memory system
> multiplies, it does not add.**
>
> Treat every offline figure in this file as a lower bound on deployed cost, not an estimate
> of it. `blend:=false roi_enable:=false` restores the older, faster behaviour — measured at
> **13.3 Hz** — and **both are now genuine launch arguments**; until 2026-07-30 typing them
> did nothing at all (see below).

All of it found and fixed offline on logged frames, which is the point of having the harness.
The next section is what happened when the same configuration was finally run on the robot.

### The deployed rate, measured (2026-07-30)

Everything above this point was measured offline, on logged frames, with no ROS and no
camera. This is the first measurement of the **actual robot**, and it is the worst result in
this file: the extrapolation said 9.9 Hz and the robot delivered **7.2 Hz**, below the ToF's
own 8.3 Hz complete-map rate.

| configuration | rate ↑ | per frame ↓ | verdict |
|---|---|---|---|
| blend + ROI **off** | **13.3 Hz** | 75 ms | ✅ above 8.3 Hz |
| blend + ROI on, as first shipped | 7.2 Hz | 139 ms | ❌ below 8.3 Hz |
| blend + ROI on, after the GPU-upsample fix | 7.4 Hz | 135 ms | ❌ still below |
| **blend + ROI on (the default), after the ROI-on-GPU rewrite** | **10.4 Hz** | **98 ms** | ✅ **above 8.3 Hz** |

Measured with [`rate_live.py`](../tools/diagnostics/rate_live.py), robot stationary, 30 s
windows, nothing else on the machine
([`preflight.py`](../tools/diagnostics/preflight.py) reporting CLEAR).

**Latency needed a separate fix, because it is a separate problem.** Rate and latency are
reported separately here for a reason: making the pipeline faster barely moved staleness.
Each depth map arrived a median **426 ms old**, and after both GPU rewrites — a 45 % speedup
— it was still **385 ms**. The cause was not speed but **queueing**: the subscriptions were
created with queue depth 5, so with the ToF publishing at 16 Hz into a ~10 Hz pipeline the
queues simply filled, and the pipeline spent its time working through a backlog of frames
that were already obsolete.

Two changes, neither of which touches the maths:

1. **Queue depth 5 → 1** on both subscriptions. With `KEEP_LAST` the middleware discards the
   backlog and hands over the *newest* frame whenever the callback comes free. Throughput is
   unaffected — those frames were never going to be processed anyway.
2. **Rectify lazily.** `on_image` used to rectify every camera frame, at ~30 Hz, but a frame
   is only consumed when a ToF frame arrives at ~10 Hz — so two thirds of a 4.4 ms 2 MP
   `cv2.remap` was discarded, ~130 ms of CPU per second. Worse, both callbacks share one
   single-threaded executor, so every wasted remap directly delayed the next pipeline run.
   The node now stores the raw frame and rectifies the one it actually uses.

| | rate ↑ | ms/frame ↓ | **latency ↓** |
|---|---|---|---|
| as shipped | 7.16 Hz | 138.8 | 426.5 ms |
| + blend on GPU | 7.41 Hz | 133.1 | 420.5 ms |
| + ROI on GPU | 10.36 Hz | 97.8 | 385.1 ms |
| **+ queue depth 1, lazy rectify** | **10.37 Hz** | **96.1** | **128.5 ms** ✅ |

**Latency fell 67 % with no change in throughput** — which is the signature of a queueing fix
rather than a speed fix, and confirms the diagnosis. Depth maps are now ~1.3 publish periods
old instead of ~4.

The lesson worth keeping: **a rate measurement cannot detect this.** Throughput looked
identical at 385 ms and at 128 ms of staleness. Any consumer that acts on the world — obstacle
avoidance, SLAM — cares about the second number, so `rate_live.py` reports both and flags the
ratio between them.

#### Per-stage cost on the robot

From [`profile_node.py`](../tools/diagnostics/profile_node.py), which replicates the node
exactly — same rectification, same `pipeline.run`, same publishing — with `pipeline.run`
instrumented via its `timings=` argument. Its 140.12 ms total predicts 7.14 Hz against the
7.16 Hz `rate_live` measured independently, which is what makes the breakdown trustworthy.

| stage | ON ms ↓ | OFF ms ↓ | attributable to blend+ROI |
|---|---|---|---|
| residual (Network B) | 37.98 | 28.22 | **+9.8** ← *collateral* |
| **ROI σ floor (7d)** | **23.50** | 0.00 | **+23.5** |
| **blend (7c)** | **23.20** | 0.01 | **+23.2** |
| backbone | 12.57 | 10.35 | +2.2 ← *collateral* |
| publish (~17 MB/frame) | 8.34 | 8.29 | — |
| analytic variance | 6.40 | 6.37 | — |
| rectify (2 MP CPU remap) | 5.81 | 4.80 | +1.0 |
| far-field clamp | 3.81 | 1.06 | +2.8 |
| closed-form fit | 3.76 | 3.72 | — |
| cloud unprojection | 2.68 | 2.82 | — |
| zone projection + pairing | 0.87 | 0.88 | — |
| ground-plane RANSAC (4b) | **0.10** | 0.00 | — |
| **FRAME TOTAL** | **140.12** | **69.38** | **+70.7** |
| implied rate | 7.14 Hz | 14.41 Hz | |

Two things to read off this table:

- **The RANSAC fix worked.** The ground-plane fit is **0.10 ms**, down from ~20 ms. It was
  the largest ROI component before; it is now the smallest line in the table.
- **The two stages cost 70.7 ms of *added* time, not the 46.7 ms they spend directly.**
  (Unrelated to the 70.7 ms offline *total* quoted elsewhere — an unfortunate coincidence of
  the same number meaning two things.) The extra ~24 ms is
  *other* stages slowing down — the residual alone goes 28 → 38 ms, and it has nothing to do
  with blend or ROI. The likely mechanism is memory bandwidth: every stage here moves 2 MP
  float arrays through the Orin's shared LPDDR5, and adding several more full-frame passes
  slows everything sharing it. The jitter agrees — rectify's p90 goes 5.5 → 16.8 ms while
  doing identical work.

This is also why the offline benchmark of the day predicted 31 ms for the two stages while
the robot charged 70.7 ms of added time: an isolated benchmark has the whole memory system to
itself. (Both figures are from before the GPU rewrites; offline now measures 19.8 ms for the
two stages.)

#### The GPU-upsample fix: half of it worked

Both stages built a full 2 MP array **on the CPU** and then copied all 2 MP to the GPU —
`blend` expanded a ¼-resolution distance map, `ROI` expanded a stride-8 mask. Sending the
small array and expanding it device-side moves 16× less data for blend and 63× less for ROI.
Implemented as `gpu_ops.blend_apply_lowres` / `roi_sigma_floor_lowres`, both verified
**bit-identical** to the originals (max |diff| 0.0e+00).

| stage | before | after | |
|---|---|---|---|
| blend (7c) | 23.20 ms | **15.26 ms** | ✅ −34 % |
| ROI σ floor (7d) | 23.50 ms | **24.05 ms** | ❌ unchanged |
| deployed rate | 7.16 Hz | **7.41 Hz** | +3.5 % |

**The blend diagnosis was right and the ROI diagnosis was wrong.** ROI's cost is not the
expansion — it is `pixel_roi_mask` itself. The fix is kept regardless: bit-identical output,
strictly less work.

#### The ROI-on-GPU rewrite: this is the one that fixed it

Profiling the CPU mask turned up three separate problems, the first of which dominated:

1. **`np.asarray(depth, float)` promoted the entire 2 MP depth map to float64** — a 16 MB
   allocation — in order to read ~31 k strided samples out of it. 63 out of every 64 values
   converted were never looked at.
2. The back-projection and the height/reach geometry then ran in float64 on the CPU.
3. `metric` and `var` were being copied to the GPU for the σ floor *anyway*, so the mask was
   computed on one device from data that had to reach the other regardless.

`gpu_ops.roi_mask_and_sigma_floor` does the whole stage on device — strided subsample,
back-projection, plane geometry, mask, and floor — from the single copy the floor already
needed. Verified against the CPU path on 189 420 mask cells over 6 scenes: **zero
disagreeing cells**, variance bit-identical.

| stage | as shipped | +GPU upsample | **+ROI on GPU** |
|---|---|---|---|
| ROI σ floor (7d) | 23.50 ms | 24.05 ms | **6.16 ms** ✅ −74 % |
| blend (7c) | 23.20 ms | 15.26 ms | 15.35 ms |
| residual (collateral) | 37.98 ms | 29.42 ms | 28.68 ms |
| rectify (collateral) | 5.81 ms | 7.33 ms | 4.44 ms |
| frame total | 140.12 ms | 121.34 ms | **90.43 ms** |
| **deployed rate, measured** | **7.16 Hz** ❌ | **7.41 Hz** ❌ | **10.36 Hz** ✅ |

**The default configuration now clears the 8.3 Hz ToF rate for the first time**, at +45 % over
where it started. The collateral stages recovered too — the residual is back to 28.7 ms and
rectify to 4.4 ms without either being touched, which is the memory-bandwidth explanation
above behaving as predicted in reverse.

Both remaining costs are now the two networks (residual 28.7 ms, backbone 10.3 ms) plus the
blend at 15.4 ms. There is no longer a stage whose cost is an artefact of how it was written.

> **A note on which measurements to believe here.** Attempts to micro-benchmark *inside* the
> ROI stage produced `inside_mask` at 2.61 ms and `height_and_reach` — **which `inside_mask`
> calls** — at 7.11 ms. A function cannot be faster than its own callee, so those numbers are
> noise, not signal. (It is not clock throttling; the CPUs were pinned at 2201 MHz.) The same
> pattern earlier produced a bogus 13.3 ms RANSAC figure that a direct measurement put at
> 1.18 ms.
>
> Only the end-to-end numbers in this section are trustworthy, and they are trustworthy
> *because* two independent tools agree: `profile_node` predicts 7.14 Hz from summed stages,
> `rate_live` measures 7.16 Hz at the topic. Prefer whole-system measurements on this
> hardware; treat sub-10 ms micro-benchmarks as unreliable unless corroborated.

### Tape ground truth: tools built and tested, measurements not yet taken

> **"Tape" means two different things and it is worth separating them once.** The **tape
> measure** is the *instrument* — the retractable ruler. "Tape ground truth" just means
> "distances measured by hand"; a laser rangefinder does the same job and the budget barely
> changes (±12.2 mm by tape against ±11.6 mm by laser — the optical-centre offset dominates
> both, so **no rangefinder is needed**). The instrument is never in shot: measure, get it out
> of the way, then capture. A **marker** is the small sticker left *on* the scene so you know
> which pixel a measurement belongs to — masking or painter's tape is ideal. Markers *do* stay
> in view; that is what they are for.

**Why this test exists.** Every other number in this file compares our depth output against
the ToF sensor. But the pipeline *uses* that same ToF sensor to calibrate itself, so it is
marking its own homework: if the ToF is wrong, our numbers would look fine anyway. The fix is
to measure some distances by hand with a laser rangefinder and compare against those. That is
the only check here that does not depend on the ToF being correct.

Two tools, built and tested 2026-07-30 *before* the measuring session, because a tool that
breaks halfway through costs the whole session and the room has to be booked again:

- [`tape_capture.py`](../tools/diagnostics/tape_capture.py) — freeze the live camera view,
  click the point you measured, type the distance. It saves the raw depth output, the camera
  image and the ToF reading for that instant, and writes the file to disk **after every
  point** rather than at the end, so a crash costs one measurement instead of all of them. It
  also reopens an existing session, since ~20 points is more than one sitting. Clicking opens
  a ×8 magnified inset, because being a few pixels off can land on a different surface
  entirely — the wall behind the box rather than the box.

  **The view stays frozen between points.** The scene has to be static for this exercise
  anyway, so one frozen frame supplies every marker in it: click, type, click, type. Depth,
  variance and rgb are stored **once per frozen frame** rather than once per point — a
  20-point session on one frame is ~16 MB instead of ~320 MB of identical arrays.

  **Several camera positions are allowed, and are easier than one.** Fitting 20 markers
  spanning 0.3–6 m into a single view is awkward — a marker at 0.3 m nearly touches the lens
  and occludes whatever is behind it. Instead treat each position as a self-contained
  mini-session: park, measure what is visible *from there*, freeze, click them all, ESC, drive
  on. Points record which frozen frame they came from and are scored against that frame's
  depth map; `tape_eval` pools the lot. Roughly **three positions of ~7 markers** is a
  comfortable session.

  > **If the robot moves, re-measure.** Ranges are measured *from the camera*, so a marker
  > 2.10 m from one parking spot is not 2.10 m from the next. The same physical markers can be
  > reused, but the numbers must be taken again. Within a position nothing may move between
  > measuring and capturing.
- [`tape_eval.py`](../tools/diagnostics/tape_eval.py) — compares the two and reports the
  score **split by whether the point was inside the ToF's field of view or outside it**.

That split is the whole point. The ToF only covers 12.4% of the image; the other 87.6% is the
camera guessing, stretched to fit the ToF's calibration. Nothing so far tells us how good
that guess is. **A large gap between inside and outside is a genuine result worth publishing,
not a failure** — it would put a number on a limitation this file currently only describes in
words.

**Two corrections happen at scoring time rather than when measuring**, so the handwritten
measurements stay exactly as taken and can be re-scored later without going back to the room:
the offset from the camera housing to the lens's true optical centre, and a conversion
explained next.

> **The conversion that would have silently ruined this.** A rangefinder measures distance
> *along the direction it is pointing*. The pipeline reports distance *straight ahead*, along
> the camera's centre line. For a point off to one side these differ, and the further off to
> the side, the bigger the difference. Converting between them takes one line of trigonometry
> (`z = r·cos θ`).
>
> To check we got it right, we built a fake dataset where the correct answer is known exactly,
> then deliberately deleted the conversion to see what a mistake would look like. The result:
> the point dead-centre stayed perfect at 0.000 m, while points near the edges were off by up
> to 62.5% — giving **0.000 m error inside the ToF's field of view and 3.362 m outside it**.
>
> That is the dangerous part. It does not look like a bug; it looks exactly like the thing
> this experiment was built to detect, so we would have "discovered" that our depth falls
> apart outside the ToF cone and published it. With the conversion in place, the same fake
> dataset scores exact zeros everywhere including the extreme corners.

**How accurate the hand measurements themselves are: ±12 mm.** That combines the rangefinder
(±3 mm), how well we know the lens's optical centre (±10 mm) and how precisely a marker can
be placed and clicked (±5 mm). `tape_eval` prints this and warns that any error smaller than
about 0.023 m is indistinguishable from measurement noise and must not be read as a result.
For context, the errors we expect to see are around 0.2 m — roughly 16× larger — so the hand
measurements are precise enough to judge the pipeline by.

### Recorded clips and result files

Every measurement on this page is backed by a file. Nothing here is quoted from memory.

**Clips** (`docs/demo/clips/`). All four-panel clips are *camera · ToF · fused depth ·
top-down*, recorded from the deployed node by
[`record_clip.py`](../tools/diagnostics/record_clip.py).

| file | what it shows |
|---|---|
| `live2_4panel.mp4` | **2026-07-30, current build** — 35 s driving, 421 frames, blend + ROI on, after all seven speed fixes. The clip that matches every number in this file. |
| `live2_4panel_small.mp4` | the same, downscaled for embedding |
| `drive_4panel.mp4` | 2026-07-28, **pre-dates** the blend, the ROI stage and the GPU rewrites |
| `static_4panel.mp4` | 2026-07-28, stationary |
| `drive_camera.mp4`, `drive_tof_heatmap.mp4`, `drive_fused_depth.mp4`, `drive_topdown.mp4` | the 2026-07-28 panels separately |
| `drive_cloudseq_colordepth.mp4` | point-cloud flythrough |

> **Raw frames are deliberately not in git.** `.gitignore` excludes
> `docs/demo/**/*_frames_raw.tar.gz`; `live2_frames_raw.tar.gz` (43 MB, all 421 source JPEGs)
> and the 2026-07-28 equivalent live on the Orin only. Re-encode from those if a different
> crop, frame rate or panel layout is ever needed — the mp4s are lossy derivatives.

**Result files** (`docs/demo/benchmarks/`, 35 JSONs). The ones behind the live work:

| file | measurement |
|---|---|
| `blend_ab_live.json` | LIVE-2 — blend A/B, 600 frames driving |
| `rate_live_on_verified.json` | deployed rate as first shipped — 7.16 Hz |
| `rate_live_after_fix.json` | after the blend GPU rewrite — 7.41 Hz |
| `rate_live_roigpu.json` | after the ROI GPU rewrite — 10.36 Hz |
| `rate_live_q1.json` | after the queue fix — 10.37 Hz, latency 128.5 ms |
| `rate_live_stagesoff.json` | blend + ROI off — 13.28 Hz |
| `profile_node_on.json` / `_gpuup` / `_roigpu` / `_off` | per-stage cost at each step |
| `timing_after_gpufixes.json` | offline `pipeline.run` — 70.7 ms / 14.1 Hz |

`rate_live_on_final.json` is retained but **labelled inside as mislabelled at capture**: it
was intended as a blend-on run and actually measured a stale blend-off launch. Kept rather
than deleted so the record of the mistake survives with it.

**Visual readout.** [`docs/demo/live2_readout.html`](../docs/demo/live2_readout.html) presents
the LIVE-2 result, the rate recovery and the per-stage breakdown alongside the embedded clip.

### Turning the optional stages on and off — and why each test needs a restart

Checking the same code path turned up two more places where this file described something the
code could not actually do. Both matter because the planned live comparison depends on being
able to switch these stages off.

A **launch argument** is an option you type on the command line when starting the robot
software, like `blend:=false`. For one to work, two separate things must be true: the program
has to know the option exists, *and* the startup file has to pass it along. Each of these was
missing one of the two halves, so typing the option had no effect and gave no error either.

| | before 2026-07-30 | now |
|---|---|---|
| `roi_enable` | the program did not know this option existed. The stage was on, because on is the built-in default — but `roi_enable:=false`, which this file recommended as the fallback, silently did nothing | works |
| `blend` | the program knew the option, but the startup file never passed it along, so `blend:=false` was not accepted | works |
| `plane_refit_every` | not available at all | works, default 1 |

```bash
# the deployed configuration
ros2 launch ringfusion_bringup single_module.launch.py \
     backbone_engine:=student_v4_heldout_fp16.engine \
     residual_engine:=residual_v7_fov73_fp16.engine

# the A/B arm, and the fallback if the live rate disappoints
ros2 launch ringfusion_bringup single_module.launch.py … blend:=false roi_enable:=false
```

> **You cannot flip these while the robot is running.** ROS has a command
> (`ros2 param set`) that changes a setting on a running program, and it will appear to
> succeed here — but the program reads these three settings **once, when it starts**, and
> never looks again. So the two halves of the live comparison have to be separate runs: start
> it, drive the route, stop it, restart with `blend:=false`, drive the same route.
>
> *(Implementation note: the launch file wraps these in `ParameterValue(..., value_type=bool)`.
> Without that, a command-line value arrives as text — `"false"`, the word — and the program
> rejects it because it expects a true/false value, not a word. This is a common ROS trap
> and the failure message, "Wrong parameter type", does not make the cause obvious.)*

> **With Network B enabled** (`residual_engine:=…`), `/depth` used to drop to **~7 Hz** —
> B's `refine()` upsampled/applied its 3 fields over 2 MP **on the CPU**. Now GPU-offloaded;
> **B costs almost nothing** (see the 2026-07-25 re-test below).

### ToF↔camera projection fix (found + applied 2026-07-28)

**Applied and verified live.** Re-running the sweep after the fix, the deployed configuration
is now the winner and every alternative scores worse — which is the check that matters:

| | before | after |
|---|---|---|
| ρ(disparity, 1/z) | 0.737 | **0.917** |
| best-case depth MAE | 0.319 m | **0.251 m** (−21%) |

Changes made:

| # | Change | File |
|---|---|---|
| 1 | `MIRROR_COLUMNS = True` — `np.fliplr` on the assembled map, so `/tof` is correct for *every* consumer | `ringfusion_drivers/tof_source.py` |
| 2 | `fov_h_deg` 61→**45**, `fov_v_deg` 45→**61**; stale `cols: 48`→32 *(the FOV half of this was superseded 2026-08-03 — see [ToF field of view, measured against tape](#tof-field-of-view-measured-against-tape). The column mirror stands.)* | `ringfusion_bringup/config/calibration.yaml` |
| 3 | One-time migration of the 1234 pre-fix paired logs (backed up, idempotent) | `tools/migrate_tof_logs.py` |

**No new data collection was needed.** `paired_logger` stores the *raw* 32×32 ToF map plus the
rectified RGB, and the projection is applied at training time from `calibration.yaml` — so the
geometry fix applies retroactively to everything already logged. Only the column mirror needed
a migration (change 3); the FOV swap needs none.

<details><summary>How it was found</summary>

Sweeping grid orientation × FOV assignment and scoring each by how well backbone disparity
tracks true inverse depth at the projected anchors:

| Grid orientation | fov h × v | ρ | best-case MAE |
|---|---|---|---|
| as-is (old deployed) | 61 × 45 | 0.737 | 0.319 m |
| `fliplr` | 61 × 45 | 0.879 | 0.265 m |
| as-is | 45 × 61 | 0.884 | 0.282 m |
| **`fliplr`** | **45 × 61** | **0.914** | **0.252 m** |

Two independent errors that compose. Confirmed three ways: the sweep, a visual overlay (with
the fix, far/red anchors land on the back wall and near/blue on the floor in a clean
top-to-bottom gradient; before, scrambled), and by eye on the recorded clip — *the ToF panel
panned left when the robot turned right*.

**This invalidated an earlier conclusion.** Teacher-vs-student was measured under the broken
projection: student ρ 0.737, Depth Anything V2 teacher ρ 0.750, corr(student, teacher) 0.989.
That read as "monocular depth is intrinsically hard here; the student is already at the
teacher's ceiling". Wrong — *both* were scored against misprojected anchors. With the
projection corrected, the sweep's winning row reaches ρ 0.914 (table above) and the
**deployed** configuration measures ρ 0.917 (`moving_ab.py`, separate run). Those are two
different measurements — the sweep's best candidate versus what actually shipped — and an
earlier version of this file conflated them. The backbone was never the bottleneck.
</details>

### Network B v3 — current (2026-07-28)

`training/runs/residual_v3`, retrained on the corrected geometry with `--struct-weight 0.15`
(down from v2's over-regularising 0.3). Deployed as `residual_v3_last_fp16.engine`.

**Moving run — 105 frames over 32 s of driving.** Both engines fed byte-identical frames off
one backbone pass, so every difference is the residual. Motion verified, not assumed:
consecutive camera frames differ by 10.75 grey levels against 1.76 for a stationary capture.

> **In-sample.** `moving_ab.py` fits on the anchor set and scores at that same set, so the
> anchor MAE row below is not a hold-out. A nearest-neighbour lookup scores 0.000 m under it.
> Held-out equivalents are in [Benchmarks](#benchmarks-vs-trivial-baselines). The comparison
> *between* columns is still valid — all three share byte-identical frames and anchors.

| Metric | closed-form | v2 | **v3** | want |
|---|---|---|---|---|
| **anchor MAE** (vs real ToF, *in-sample*) | 0.294 m | 0.247 m | **0.199 m** | lower |
| max depth, mean / worst | 1.52 / 2.47 m | 1.81 / 3.23 m | 1.91 / 3.77 m | ≤ this run's ToF max 5.74 m |
| frames at the 20 m clamp | 0 | 0 | **0** | zero |
| pixels > 5 m | 0 % | 0 % | **0 %** | zero |
| frame-to-frame jump | 0.0271 m | 0.0283 m | 0.0290 m | near baseline |

v3 was **32 % more accurate than the closed-form path** *in-sample* and 19 % better than v2 — the first
version to beat doing nothing at all. It costs ~7 % extra jitter over the raw backbone.

**Uncertainty — `sigma_cal.py`, 8 frames, ~940 anchors each:**

| Engine | MAE | median σ | cov ±1σ | cov ±2σ | **corr(σ, \|err\|)** |
|---|---|---|---|---|---|
| v1 | 0.179 m | 0.159 m | 0.612 | 0.958 | 0.490 |
| v2 | 0.239 m | 0.091 m | 0.264 | 0.895 | 0.913 |
| **v3** | 0.198 m | 0.079 m | **0.590** | 0.973 | **0.942** |
| *ideal* | — | — | *0.683* | *0.955* | *higher* |

`corr(σ, |error|)` is the one that matters: does stated confidence track where the model is
actually wrong? v1 scored 0.490 and was *most confident exactly where it was most wrong*. v3
reaches **0.943**, with σ at 0.078 m on the anchors while the whole-frame median stays near
0.84 m — confident where the ToF constrains it, uncertain where it extrapolates.

**Also measured:** `/depth` at **13.7 Hz** (unchanged from v1/v2 — the structure loss is free
at runtime); centre-depth bias against matched-region ToF improved from **−14.0 % to +6.5 %**;
5 s stationary hold shows ±1.2 mm drift and 0.72 % frame-to-frame change with zero clamp hits.

**Notable:** under corrected geometry **v2 no longer blows up either** — zero clamp hits where
it previously saturated on 46 % of moving frames. The old catastrophe was substantially a
*calibration* artefact, not purely a training failure.

Captures: [`docs/demo/`](../docs/demo/README.md) · rendered page:
[`docs/demo/network_b_v1_vs_v2.html`](../docs/demo/network_b_v1_vs_v2.html) ·
tools: [`tools/diagnostics/`](../tools/diagnostics/README.md)

---

## ⚠ Historical results — all measured BEFORE the projection fix

Everything from here to the end of the results sections was measured through the mirrored,
axis-swapped ToF projection. The *relative* comparisons in them are still meaningful (both
arms saw the same broken geometry), but **no absolute number below should be quoted** — see
[the fix](#tofcamera-projection-fix-found--applied-2026-07-28) and the
[current numbers](#network-b-v3--current-2026-07-28) instead. Kept for provenance and because
the reasoning trail explains how the bug was eventually found.

### ⚠ Superseded — original write-up of the projection bug

**The ToF grid is horizontally mirrored, and `fov_h`/`fov_v` are swapped.** Both were found
by sweeping the projection parameters and scoring each candidate by how well the backbone's
disparity tracks true inverse depth at the projected anchors:

| Grid orientation | fov h × v | ρ(disp, 1/z) | best-case depth MAE |
|---|---|---|---|
| as-is (**deployed**) | 61 × 45 | 0.737 | 0.319 m |
| `fliplr` | 61 × 45 | 0.879 | 0.265 m |
| as-is | 45 × 61 | 0.884 | 0.282 m |
| **`fliplr`** | **45 × 61** | **0.914** | **0.252 m** |

The two errors are independent and compose. Confirmed three ways: the ρ sweep above, a
visual overlay (with the fix, far/red anchors land on the back wall and near/blue on the
floor in a clean top-to-bottom gradient; deployed, they are scrambled), and by eye on the
recorded clip — *the ToF panel pans left when the robot turns right.*

**This invalidates an earlier conclusion.** Teacher-vs-student was measured under the broken
projection: student ρ 0.737, Depth Anything V2 teacher ρ 0.750, corr(student, teacher) 0.989.
That looked like "monocular depth is intrinsically hard here, the student is at the teacher's
ceiling". Wrong — *both* models were being scored against misprojected anchors. With the
projection corrected the sweep's winning row reaches ρ 0.914 — the deployed configuration
separately measures ρ 0.917. The backbone was never the bottleneck.

**The fix** (not yet applied):

1. `tof_source.py` — apply `np.fliplr` when the subframes are assembled, so `/tof` is correct
   for *every* consumer (heatmap, perception, paired_logger), not just perception.
2. `calibration.yaml` — `fov_h_deg: 45.0`, `fov_v_deg: 61.0` (currently 61.0 / 45.0).
   Also `tof.cols: 48` is stale; the flashed mode is 32×32 (harmless — the live packet
   overrides it — but misleading).
   > **Superseded 2026-08-03.** The 45.0 here was still wrong; tape measurement puts
   > `fov_h` at **73.5°**. See [ToF field of view, measured against tape](#tof-field-of-view-measured-against-tape).

**Consequence: Network B must be retrained after this.** B was trained on logged data pushed
through the *same* broken projection (`anchoring_bridge` reuses the deployed geometry), so its
learned corrections encode the mirrored anchor field. Until it is retrained, run with
`residual_engine:=` unset — the closed-form path is pure geometry and improves the moment the
calibration is fixed.

**Expected gain:** anchor MAE 0.319 → 0.252 m (−21%) from geometry alone, before any retraining.

### Network B v2 (structure loss) — measured 2026-07-28

`training/runs/residual_v2`, Network B retrained with `losses.structure_loss`
(`--struct-weight 0.3`). Exported clean (PyTorch↔ONNXRuntime max abs diff **2.86e-06**),
built to `residual_v2_fp16.engine` (1.17 MB vs 1.21 MB for v1), loads in `perception_node`,
and holds **13.4 Hz on `/depth` — identical to v1**, so the structure loss is free at runtime.

**Controlled A/B.** Both engines run off one backbone pass per frame, so Network A's output,
the anchors and the affine fit are byte-identical between arms — every difference is the
residual. 10 frames, lit room (mean frame brightness 90.4), 755 anchors, ToF centre 1.20 m,
ToF max 4.65 m.

| Metric | A (no residual) | **v1** | **v2** | verdict |
|---|---|---|---|---|
| max depth | 2.46 m | **14.40 m** (20 m clamp on 4/10 frames) | **3.55 m** | **v2 fixes it** — now below the ToF's own 4.65 m max |
| frac > 3 m | 0.00% | 1.41% | **0.13%** | 11× better |
| frac > 5 m | 0.00% | 0.08% | **0.00%** | gone |
| banding @ 13 px (anchor pitch) | 2.98 | 12.33 (4.1× over A) | **4.16** (1.4× over A) | **largely fixed** |
| banding @ 173 px (scene texture) | 38.1 | 85.1 (2.2× over A) | **64.3** (1.7× over A) | improved, not solved |
| mean \|B−A\| | — | 0.173 m | **0.026 m** | v2 barely corrects A at all |
| anchor MAE | 0.479 m | **0.236 m** | 0.392 m | **v2 regresses** toward A |
| anchor p90 | 1.665 m | **0.786 m** | 1.366 m | **v2 regresses** |
| centre depth (ToF says 1.20 m) | 0.710 m | 0.653 m | 0.558 m | all three under-read; **v2 worst** |
| median σ | — | 0.968 m | 0.859 m | still saturated |

**Moving-robot run — the decisive test.** A static scene cannot show flicker, so the above was
re-run over **102 frames of a 30 s drive**, both engines again fed identical frames
(`moving_ab.py`, ToF max seen 6.11 m, brightness 63.9):

| Metric | A | **v1** | **v2** | want |
|---|---|---|---|---|
| frames hitting the 20 m clamp | 0 | **47 / 102** | **0 / 102** | zero |
| frames with max > 10 m | 0 | **55 / 102** | **0 / 102** | zero |
| max depth (mean / worst) | 2.66 / 6.43 m | 13.07 / 20.00 m | **3.40 / 8.62 m** | lower |
| pixels > 5 m | 0.03% | 0.64% | **0.03%** | lower |
| frame-to-frame jump | 0.066 m | 0.129 m (**2× A**) | **0.069 m** (≈ A) | near A |
| anchor MAE | 0.552 m | **0.400 m** | 0.428 m | lower |

**Read: under motion v2 wins outright, and the accuracy penalty largely evaporates.** v1 saturates
the clamp on **46% of moving frames** — in a room the ToF never measures past 6.1 m — and doubles
frame-to-frame jitter over the raw backbone. v2 never clamps and adds essentially no jitter
(0.069 vs A's 0.066). The anchor-MAE gap that looked bad on a still frame (0.236 vs 0.392) shrinks
to 0.400 vs 0.428 once the robot moves, i.e. ~28 mm.

**The remaining concern is that v2 under-corrects.** On a still frame mean |B−A| collapsed from
0.173 m to **0.026 m** — B is close to a no-op — and centre depth moved *further* from the ToF.
The structure loss at weight 0.3 is doing its job but is set too strong.

**Next: sweep `--struct-weight` below 0.3** (try 0.1 / 0.15) to recover correction strength
while keeping the far field bounded. v1 corrects more but is unsafe; v2 is safe but barely
corrects. The useful operating point is between them.

**Ship-today recommendation: use v2.** It beats v1 on every safety metric under motion at
equal cost and near-equal accuracy. Launch with
`residual_engine:=$HOME/RingFusion/residual_v2_fp16.engine`.

**Also measured:** 5 s stationary hold on the shipped pipeline — 70 frames at **13.85 Hz**,
median depth drift **±1.2 mm**, frame-to-frame change **0.72%**, zero clamp hits. Confirms no
idle drift; the moving run is what separates the two engines.

Rendered comparison with every metric explained: [`docs/demo/network_b_v1_vs_v2.html`](../docs/demo/network_b_v1_vs_v2.html).

### Live re-test — 2026-07-25 (retrained Network A + real Network B)

First run of **`student_v3_fp16.engine`** (retrained backbone) with **`residual_fp16.engine`**,
`jetson_clocks` on, full `single_module.launch.py`. All four nodes came up clean and held
rate for the whole capture; no mocks in the loop.

| Topic | Rate | vs 2026-07-23 |
|---|---|---|
| `/image` | ~22 Hz | down from 28.5 — **cause unknown**; a later dark-room run hit 28.0 Hz, so it is *not* auto-exposure |
| `/tof` | ~16 Hz | unchanged *(attribution superseded — see [Throughput, reconciled](#throughput-reconciled))* |
| `/depth` (**A + B**) | **~13.4 Hz** | **up from ~7 Hz** — B's apply moved to `gpu_ops` |
| `/cloud` | ~14.4 Hz | unchanged |

**Anchor retention.** 991/1024 anchors (**96.8%**) reach Network B, against ~5% under the old
nearest-resize path. Affine fit stable across frames: *a* ≈ 0.197–0.200, *b* ≈ 0.129–0.142.
`perception_node` and the standalone renderer agree on centre depth to **2 mm** (0.700 vs
0.702 m), confirming the two code paths are consistent.

**Depth statistics** (one frame, 1640×1232):

| Quantity | Network A | Network B (A+residual) | ToF (measured) |
|---|---|---|---|
| median depth | 0.524 m | 0.592 m | 1.17 m |
| 2–98 pct range | 0.067 – 2.31 m | 0.097 – 3.03 m | — |
| max | 5.16 m | **20.00 m** (= `max_depth` clamp) | **2.80 m** |
| frac > 3 m | 0.04% | 2.52% | 0% |
| frac > 5 m | 0.00% | 0.30% | 0% |

Mean \|B−A\| **0.264 m**, max \|B−A\| **14.4 m**. B/A ratio: median 1.27×, p99 2.95×, max 17.4×.

**Three findings.**

1. **ToF footprint is small.** The 32×32 grid projects into a **567×443 px box** (x 616–1063,
   y 395–735) — **12.4 % of the frame**. Everything else is monocular extrapolation. Depth
   outside that box should not be treated as metric.
2. **B's far field is unbounded.** B drives 0.30% of pixels past 5 m and reaches the 20 m
   clamp in a room the ToF measures at ≤2.80 m. Because the residual is a *log-ratio*
   correction, a modest positive residual compounds multiplicatively where D₀ is already
   large. Worth a range prior or a scene-aware clamp.
3. **Banding is two separate effects.** See the controlled comparison below.

#### Variance decomposition

**Uncertainty is uninformative, and it is entirely Network B's.** `/depth_var` is
`analytic + learned` (`pipeline.py` Stage 6 + 7). Decomposed on the captured frames:

| Term | Scene 1 σ | Scene 2 σ | Share of total variance |
|---|---|---|---|
| analytic (delta method, `D⁴·jᵀCov(a,b)j`) | 0.032 m | 0.010 m | **~0.1%** |
| learned (B's `τ² = exp(logτ²)`) | 0.947 m | 0.977 m | **~99.9%** |

The principled analytic term is numerically irrelevant, so the published variance *is*
Network B's untrained variance head — pinned near a ~1 m ceiling in both scenes. As a
fraction of the depth it reports, σ/D is **1.02 (scene 1)** and **1.58 (scene 2)**: the node
states ±100–158% uncertainty on every pixel. σ is also *lowest* (0.495 m) exactly where B
extrapolates worst and *highest* (0.855 m) on the anchor-dense floor — confident in the wrong
places. Needs recalibration (step 8) before `/depth_var` means anything.

#### Fit quality, measured properly — the anchoring is **not** broken

An earlier version of these notes compared "ToF centre depth" against "Network A centre
depth" and read a 2.4× gap as evidence of broken anchoring. **That comparison was invalid**
and is retracted. The two statistics sample *different points in the scene* — the ToF's
central zones project to ≈(840, 565) while the A-centre patch sits at the image centre
(820, 616) — and the depth gradient there is steep enough (A runs 1.30 m → 0.51 m between
40% and 50% down the frame) that a ~50 px offset spans most of it.

The correct test is the residual **at the anchor pixels**, where ToF truth and prediction
describe the same point:

| | n | median abs err | median rel err | **bias** |
|---|---|---|---|---|
| Scene 1 — A (closed-form) | 995 | 0.259 m | 23.8% | **−0.005 m** |
| Scene 1 — B (A+residual) | 995 | 0.166 m | **16.1%** | +0.157 m |
| Scene 2 — A (closed-form) | 703 | 0.076 m | 10.9% | **+0.003 m** |
| Scene 2 — B (A+residual) | 703 | 0.032 m | **5.7%** | +0.006 m |

**Bias is ≈0 in both scenes** — millimetres on a metre-scale scene. A broken projection,
wrong intrinsics, slant-range-vs-z confusion or a mm/m units error would all produce a large
systematic bias. None is present, so none of those is the problem. **And Network B measurably
improves accuracy where it is supervised** — 23.8% → 16.1% and 10.9% → 5.7% relative error.
B is doing its job inside its domain.

#### The one systematic error: far-field under-read

Binning the anchor residuals by image row exposes a consistent structure in **both** scenes:

| Anchor rows | ToF median | A median | **A / ToF** |
|---|---|---|---|
| top (far) | 1.93 / 2.40 m | 1.31 / 1.67 m | **0.67 / 0.72** |
| upper-mid | 1.23 / 0.96 m | 1.31 / 1.13 m | 1.04 / 1.16 |
| lower-mid | 0.64 / 0.50 m | 0.86 / 0.55 m | 1.25 / 1.06 |
| bottom (near) | 0.34 / 0.31 m | 0.35 / 0.31 m | **1.03 / 0.99** |

The near field is essentially exact (0.99–1.03×); the **far field is under-read by ~30%**
(0.67–0.72×). This is expected behaviour, not a bug: the closed-form fit is least squares in
**inverse-depth** space, where a point at 0.3 m contributes 1/0.3 ≈ 3.3 and a point at 2.4 m
contributes 1/2.4 ≈ 0.4. Far points carry ~8× less weight, so the fit nails the near field
and is loose at range.

This also explains Network B's behaviour: B correctly learns *"push the far field out"*
(scene 1 bias +0.157 m) — the right direction. With no supervision beyond `max_range = 5 m`
it simply cannot learn where to stop, so the correction runs on to the 20 m clamp.

#### Fixes implemented 2026-07-25 (7a / 7b / 7c)

All three are training-side or inside an existing stage — **no pipeline change**: same
sensors, stages, order, topics, message types, launch args, engine files and runtime
(`/depth` still ~12.9 Hz after the change).

| | Change | Where | Runs on robot? |
|---|---|---|---|
| 7a | `structure_loss` — match `∇log D_B` to `∇log D_A` on the unsupervised region, multi-scale | `training/losses.py`, `--struct-weight` (default 0.3) | no, training only |
| 7b | `max_range` 5.0 → 6.5 m | `training/anchoring_bridge.py` | no, training only |
| 7c | `range_weights` — weight anchors by `z^1` | `anchoring.py` (stage 5) | **yes** |

`--struct-weight 0` reproduces the previous objective exactly; `structure_loss` is
verified to be identically 0 when `D_pred == D_base`, so a converged identity residual is
unpenalised. All 5 pipeline tests still pass. `training/` imports the workspace's
`anchoring.py` directly, so 7c is a single source of truth across training and deploy.

**7c: theory said z², measurement said z¹.** The derivation (relative depth error ⇒
`w = z²`) is in the `range_weights` docstring along with the sweep that refutes it —
z² overshoots into a far *over*-read and takes scene 2's median relative error from 10.9%
to 41.0%, because the affine model is misspecified and re-weighting trades one regime for
the other. z¹ was chosen empirically.

**Live result** (same view, before vs after, n≈700 anchors):

| | A far ratio | A near ratio | A med rel err |
|---|---|---|---|
| old fit (p=0) | **0.66** | 0.99 | 10.9% |
| new fit (p=1) | **1.01** | 1.00 | 11.5% |

The far-field under-read is **gone** (0.66 → 1.01) with the near field untouched, at the
cost of ~0.6 pp on the overall median.

> **Network B must be retrained before it is used again.** B was trained against the old
> under-reading baseline, so it learned to push the far field outward — now redundant. On
> the same live frame B's tail got worse (max depth 6.29 m → 8.20 m, pixels >3 m 0.80% →
> 2.06%) while A's far field became correct. **Until B is retrained, leave
> `residual_engine:=` unset** — A + the new closed-form fit is the better output.

#### Two-scene control: what actually causes B's banding

Scene 1 (shelving full of near-identical trophies) was a confound — a strongly periodic
texture. Scene 2 pointed the robot at a different view (open floor, blackboard wall) and the
capture was repeated with the pipeline untouched. Power is measured as a multiple of each
signal's own broadband median, along the image's horizontal axis.

| | Scene 1 (trophies) | Scene 2 (new view) |
|---|---|---|
| image's dominant period | 173 px | 253 px |
| B−A dominant period | **173 px** (matches) | **253 px** (matches) |
| power @ image period — A | 47.5× | 76.1× |
| power @ image period — B | 94.1× (**B/A 1.98**) | 315.1× (**B/A 4.14**) |
| power @ 13 px anchor pitch — A | 3.1× (noise floor) | 2.9× (noise floor) |
| power @ 13 px anchor pitch — B | 15.4× (**B/A 5.0**) | 23.2× (**B/A 8.0**) |

**Both effects are present, and they are now separated:**

- **Texture amplification (scene-dependent).** B locks onto whatever periodic texture the scene
  has — 173 px with the trophies, 253 px with the blackboards — and amplifies it **2–4×** over
  what Network A already produces. The period tracks the scene, so this is a response to image
  content, not an artefact of the anchors.
- **Anchor-pitch lattice (scene-independent).** A fine **13 px** component sits at the noise
  floor in Network A (≈3× in both scenes) but at **15–23×** in Network B, and it **survived a
  complete scene change at similar strength**. That is a genuine artefact B injects, not
  scene texture. High-passing B−A isolates it as a uniform dot grid
  ([`scene2_anchor_highpass.png`](../docs/demo/scene2_anchor_highpass.png)).

So the first read ("it's the anchor grid") and the trophy hypothesis were **both partly right**.
Fixing this needs two different things: a smoothness/neighbourhood term for the anchor lattice,
and frequency-domain regularisation (or more diverse training texture) for the amplification.

#### Scene 2 numbers

| Quantity | Network A | Network B | ToF |
|---|---|---|---|
| median depth | 0.364 m | 0.495 m | 0.784 m |
| centre depth | 0.722 m | 0.652 m | 1.195 m |
| max | 2.05 m | 6.29 m | 4.58 m |
| frac > 3 m | 0.00% | 0.80% | — |

- **Anchor re-splatting confirmed exactly.** Scene 2 had only **703/1024 valid ToF zones** (the
  open space beyond ToF range returns nothing) and produced **exactly 703 anchors** — as did
  scene 1 at 995/995. `resplat_anchors` preserves **100% of valid zones** in both scenes.
- **B's overshoot is milder here but still present** — max 6.29 m against a ToF max of 4.58 m,
  0.80% of pixels past 3 m where A has none. No 20 m clamp hit this time, so scene 1's blowup
  was worst-case (dark, low-texture shelving at range), not typical.
- **B did not move toward the ToF this time.** A 0.722 m → B 0.652 m against a ToF centre of
  1.195 m — the correction went the *wrong way*. In scene 1 it went the right way. B's
  correction direction is not reliably toward truth.
- **σ saturation is scene-independent** — median 0.977 m, p95 1.049 m here vs 0.947/0.985 in
  scene 1. The variance head is pinned near ~1 m regardless of content, confirming it is
  saturated rather than reflecting the scene.

Captured artefacts: [`docs/demo/`](../docs/demo/). Scene 1 is unprefixed (`quad.png`,
`network_a.png`, `network_b.png`, `pipeline_depth.png`, `pipeline_sigma.png`, `stats.json`);
scene 2 uses the `scene2_` prefix, plus `scene2_crop_a/b.png` and
`scene2_anchor_highpass.png`. Files from 2026-07-23 (`pipeline_demo.png`,
`four_view_A_vs_B.png`, `arducam_view.png`, `network_a_depth.png`, `tof_heatmap.png`) are kept
for comparison. Rendered summary:
[`docs/demo/pipeline_view.html`](../docs/demo/pipeline_view.html).

**Live accuracy check (green cube):** ToF **0.447 m** vs Network-A anchored depth **0.459 m**
— **~12 mm** apart, inside the ToF's own ~20 mm error band. Confirms the closed-form ToF→mono
anchoring is metrically correct. Demo montage: [`docs/demo/pipeline_demo.png`](../docs/demo/pipeline_demo.png).

- **Power mode.** Check with `nvpmodel -q`. If it's not MAXN, everything is throttled (fewer
  cores, lower clocks) — set it with `sudo nvpmodel -m 0` (applies immediately, no reboot;
  persists across reboots) then `sudo jetson_clocks` (re-run after each boot).

- **Camera rate.** Runs at **1640×1232** (full-FOV 2×2-binned mode) at **~28.5 Hz**, near the
  30 fps sensor ceiling. The node's serial tick is ~36 ms (`cap.read()` at the 30 fps cap +
  ~10 ms tone-correct). It was formerly throttled to 15 Hz by the `rate` launch param — now
  `30.0` in `single_module.launch.py`. Going past 30 Hz needs a different sensor mode + re-calibration.
  (Passive camera → needs light: a dark room yields black/noise frames, not an EMI fault.)

- **Perception is GPU-offloaded.** Profiling found the bottleneck was *not* the neural net —
  the backbone is ~17 ms and the GPU sat near-idle. It was the pure-numpy per-pixel 2 MP math
  on the CPU: analytic variance (`depth**4`, ~74 ms), cloud unprojection (~43 ms), metric depth
  (~13 ms). Those are embarrassingly parallel, so `gpu_ops.py` moves them to torch/CUDA (numpy
  fallback preserved for off-robot testing), and float64 waste was cut to float32. Perception
  went from ~5.6 Hz to ~27 Hz capable. `cloud_stride=4` in `pipeline.run` also thins the cloud.

- **ToF rate — binary firmware (live).** The ESP32-C6 now streams framed **binary** subframes
  (`MAGIC + LEN + BODY + CRC16`, `firmware-esp/BINARY_OUTPUT_HANDOFF.md`) instead of ASCII CSV
  — ~4× fewer USB bytes and a CRC that drops only *corrupted* frames. `tof_source.py`
  auto-detects ASCII vs binary and uses a **persistent per-subframe assembler**: each even/odd
  subframe updates its half of a kept 32×32 map and publishes immediately (per-subframe, not
  per-pair), carrying a missed half forward instead of dropping the whole map. Together these
  took `/tof` from ~8 Hz (ASCII, all-or-nothing pairing) to **~16 Hz**. Parsing was never the
  limit (Orin decodes a map in ~1.2 ms); the cap was USB delivery + assembly survival.
  (Firmware note: ToF resolution is set by the flashed focal-plane mode — `CMD_LOAD_CFG_32X32`
  for 32×32; `tof_source.ROWS,COLS` must match the flashed build.)

- **`nvargus-daemon`.** If the camera starts failing with `INVALID_SETTINGS` intermittently,
  the Argus daemon is wedged (often from `kill -9`-ing a camera process). Fix:
  `sudo systemctl restart nvargus-daemon`. Stop camera nodes with SIGTERM, not `kill -9`.
  Note the camera's separate EMI failure mode (`PD_CRC_ERR`): the ESP32's USB cable radiates
  into the CSI ribbon — keep them ~12–15 cm apart with ground shielding (binary ToF also helps).

## Sanity checks / useful `ros2` commands

```bash
# Confirm colcon sees all 4 packages with the right build type
colcon list

# Confirm packages are registered on the ROS 2 graph (after sourcing install/setup.bash)
ros2 pkg list | grep ringfusion
ros2 pkg prefix ringfusion_msgs        # should point into install/

# Confirm the custom message built correctly
ros2 interface show ringfusion_msgs/msg/ToFFrame

# While nodes are running:
ros2 node list                         # expect tof_driver, camera, perception, cam_to_tof
ros2 topic list                        # expect /cloud, plus driver/camera topics
ros2 topic hz /cloud                   # confirm perception is publishing
ros2 topic echo /cloud --once          # sanity-check one message
ros2 node info /perception             # see its subscriptions/publications/params
ros2 param list /tof_driver
ros2 run tf2_ros tf2_echo cam_0 tof_0  # confirm the static transform is being published

# Debugging
ros2 doctor                            # general environment/network health check
rqt_graph                              # visualize the node/topic graph
```

## Troubleshooting

- **A package is missing from `colcon list` / didn't build**: check its `package.xml` has the correct `<export><build_type>...</build_type></export>` tag (`ament_cmake` for CMake packages, `ament_python` for Python packages).
- **"failed to create symbolic link ... Is a directory"**: a stale `build/<pkg>` directory from an earlier failed build is conflicting with `--symlink-install`. Remove just that package's build folder and rebuild:
  ```bash
  rm -rf build/<pkg> install/<pkg>
  colcon build --symlink-install --packages-select <pkg>
  ```
- **Nodes can't find messages/executables after building**: make sure you `source install/setup.bash` in the terminal you're running from (each new terminal needs it).
- **`camera` node crashes with "Could not open CSI camera"**: check `dpkg -l | grep arducam` and `/dev/video0` exist first (see Camera hardware section above — the Arducam driver may not be installed). If those are fine, check `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer` — if it says `NO`, a pip-installed `opencv-python` in `~/.local` is shadowing JetPack's GStreamer-enabled system `python3-opencv`. `single_module.launch.py` already sets `PYTHONNOUSERSITE=1` to work around this; if you run `ros2 run ringfusion_drivers camera` directly outside the launch file, prefix it with `PYTHONNOUSERSITE=1` too.

## Task tracker

Living checklist of remaining work, in dependency order. Scratch items off (`[x]`)
as they land. Detail on each in the technical reference and the sections above.

### ▶ Do next (immediate action items)

1. **B2 — get the pilot engine live on the Orin + measure FPS (CURRENT).** Pilot done:
   2000 images → distilled (val_ssi **3.51**, **ρ 0.9962**). **Exported to ONNX; FP16
   engine built; INT8 building.** Remaining: (a) isolated `TRTRunner` smoke test against
   an engine, (b) `ros2 launch … single_module.launch.py backbone_engine:=…student_int8.engine`
   → `ros2 topic hz /depth` for the FPS number (retires `MockBackbone`). Export/engine
   gotchas (onnxscript, KeyError, pycuda→torch) are all fixed — see
   [../training/README.md](../training/README.md) §3.
2. **Then collect the full ~15–20k images and re-distill** for the deployable-quality
   student — the 2000-image pilot is intentionally small. Re-run `compare_student` /
   `eval_student` to quantify the gain (target ρ → ~0.998, d1.25 → >0.9).

**✅ Done recently:** GPU torch fixed — cuBLAS + cuDNN verified on the Orin (recipe in
[training/README.md](../training/README.md#gpu-torch-on-the-orin--working-recipe-resolved));
**B1 Step-0 passed** — Depth Anything V2 produces clean depth on our rectified fisheye
(`step0.png`), validating the distillation plan.

**A — Camera / Arducam pipeline**
- [x] A1. `tools/calibrate_camera.py` — fisheye (Kannala-Brandt) checkerboard calibration tool
- [x] A2. **DONE (2026-07-21).** Calibrated the real lens (cv2.fisheye, RMS **0.5406 px**); real intrinsics in `calibration.yaml`. Verified live: `rectify_view` logs `identity=False` and straight edges de-warp correctly.
- [x] A3. Stage 1 rectification (fisheye → rectilinear) wired into perception (`rectify.py`); **active now** with the nominal equidistant model (a real de-warp), refined once A2 lands. Confirm live: `rectify_view` → `/rectify_compare`
- [x] A4. Capture resolution = **1640×1232** (IMX219 full-sensor 2×2-binned — full fisheye FOV; the 16:9 modes crop it). Wired into `calibration.yaml`, both launch files, `camera_node`, and the calib tool. Runs **~26 Hz** (was throttled to 15 by the `rate` launch param, now `30.0`; 30 fps is the sensor ceiling at this resolution). Tone-correct is ~10 ms/frame — not the limiter, see C3 for the 4-camera scaling concern.

**B — Networks (need no ground truth for B1–B2)**
- [x] B1. Step 0 sanity — **PASSED**. Depth Anything V2 on our rectified fisheye produces clean depth (near/far correct, crisp edges, objects separated) → distillation plan validated. `training/step0_sanity.py --image step0_raw_frame.png --raw` runs in ~14 s on GPU; result in `step0.png`.
- [ ] B2. Distill backbone → ONNX → INT8 engine **on the Orin** → measure FPS (headline number). Backbone input size (default **384×288**, must be a **multiple of 32**) is the depth-detail lever, tunable here with a latency measurement — *not* the camera resolution. Built student is **3.66M params** (design doc's "6.1M" was a placeholder). **GPU torch fixed** (cuBLAS/cuDNN verified — recipe in training/README). **Pilot done (2000 imgs):** collected (`collect_frames`) → cached → distilled → **val_ssi 3.51, ρ 0.9962** vs teacher (validated via `compare_student.py` / `eval_student.py`). **Now exporting to TensorRT on the Orin** (`export_onnx` legacy exporter → `build_engine` FP16→INT8 → `backbone_engine:=…` → `ros2 topic hz /depth`); then re-collect the full ~15–20k and re-distill for final quality.
- [ ] B3. Train residual (Network B) with NLL → calibration coverage (→0.68). **Real-data path scaffolded & tested** (`training/residual_real_data.py` + `anchoring_bridge.build_real_supervision`): collect paired `(rectified image, 32×32 ToF)` logs, train via **held-out ToF anchors** (no dense GT, no fisheye domain gap) — `train_residual.py --real`. See training/README §2a. Remaining: collect the paired logs (needs the ToF binary firmware + a paired logger), then train/export FP16.

**C — Housekeeping**
- [ ] C1. `tof_driver` 500 Hz poll cleanup (threaded blocking read; verify ToF latency doesn't regress)
- [ ] C2. Fix stale "B0472 stitches 4 cameras into one frame" comment in `camera.py` (it's native per-camera)
- [ ] C3. Camera color-correction is CPU-bound (~one core/camera at 1640×1232) — won't scale to 4 cameras; move the LUT tone-correction to the GPU (nvvidconv/CUDA) or make it optional
