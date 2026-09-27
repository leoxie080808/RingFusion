# RingFusion r33 — answers available today, and three corrections to the question list

Reply to *RingFusion Revision: Open Items and Tests* (Sep 27, 2026) · @Leo

This covers what can be answered from the code, the config and the r31 measurement record
without any new experiment, and flags three places where the question list rests on a premise
the code does not support. Everything cited here is already in `PAPER-REVISION-RESULTS.md` or
in a committed JSON under `docs/demo/benchmarks/`.

Headline: **T1 is more urgent than the note suggests, and the reason is already visible in the
r31 data.** See §0.

---

## 0. T1 — the existing DEPTHOR timings are FP32 PyTorch, ours are FP16 TensorRT

The r31 round already re-timed both DEPTHOR variants on the Orin and reproduced their published
numbers to under 1.5 %:

| Model | r31 median | Published | Agreement |
|---|---|---|---|
| DEPTHOR-Small | 80.05 ms (12.49 Hz) | 79.4 ms / 12.6 Hz | ✅ 0.8 % |
| DEPTHOR-Large | 186.25 ms (5.37 Hz) | 183.8 ms / 5.4 Hz | ✅ 1.3 % |

But the embedded metadata in `depthor_small_timing_r31.json` records:

```json
"weights": "depthor_small.pt",
"param_dtypes": ["torch.float32"]
```

**PyTorch, FP32.** Our own rows are TensorRT **FP16** (`student_v4_heldout_fp16` +
`residual_v7_fov73_fp16`). So the speedup — already corrected from the paper's 3.1× down to
**2.83×** on the deployed configuration — compares an FP16 TensorRT numerator against an FP32
PyTorch denominator. Two advantages are stacked, runtime and precision, and only one of them is
a property of our method.

This is exactly reviewer comment 5, and it means the abstract's headline multiplier is not yet
defensible at any value. T1 must land before anything citing the speedup is finalised.

For reference, the numerator as re-measured at r31 (`timing_pipeline_r31.json`):

| Config @ 480×640 | r31 | Paper |
|---|---|---|
| optional stages off | 20.08 ms | 20.0 ms ✅ |
| blend only | 25.57 ms | — |
| **deployed (blend + ROI + σ)** | **28.16 ms** | 25.8 ms ❌ |

The paper's 25.8 ms matches the *pre-σ* figure exactly — it excludes the uncertainty terms the
deployed system runs. At the deployed 1640×1232 the same three configs are 50.33 / 73.34 /
**79.02 ms** (12.65 Hz).

---

## 1. Section 1 — every item, answered or accounted for

All thirty items below are resolved into one of three states: **answered** from code/config,
**false premise** (the thing asked about does not exist, so the paper text must change), or
**absent** (genuinely not recorded anywhere and requiring physical measurement).

### 1.1 Answered from the code and config

| # | Item | Value | Source |
|---|---|---|---|
| 1 | Validity filter | A zone enters the fit iff **all** of: the projection is valid (`proj['valid']`, which already folds in `tof_valid`); `u, v, z` all finite; the pixel lands in-image (`0 ≤ u < W`, `0 ≤ v < H`); and **`z > 0`**. Confidence is applied *only* if `min_confidence >= 0` — and the deployed value is **`-1`**, so **no confidence threshold is applied and all weights are uniform**. There is **no range gate in the pipeline**; the `[0.15, 6.5] m` gate is in the offline eval harness (`baselines_r31.json → config.range_gate_m`), not the deployed path | `pipeline.py:130-146` |
| 2 | Robust reweighting | **Huber**, tuning constant **c = 1.345**, scale from **MAD**: `1.4826·median(|r − median(r)|)`. `u = |r|/(c·scale)`; weight `1` if `u ≤ 1`, else `1/u`. **It down-weights; it never rejects.** One IRLS iteration (`iters=1`) as called by the pipeline. Measured over 1234 frames: **22.9 %** of anchors exceed the threshold (median; mean 23.3 %), but that removes only **6.9 %** of total weight mass | `anchoring.py:106-145`; `baselines_r31.json → protocols.random.robust_pass` |
| 6 | Arbitration limits | **2.0° – 5.0°**, smoothstep `1 − t²(3 − 2t)` confirmed | `blend.py:28-29`, `blend.py:143-160` |
| 7 | Residual variance `σ̂²` | **Weighted**: `σ̂² = n/(n−2) · Σ(w·r²)/Σw`, with `n = count_nonzero(w > 0)`. Degrees of freedom **n − 2**. Computed **after** the robust pass in the sense that `(a, b)` are the post-Huber estimates — but **with the pre-Huber weights**, since `pipeline.run` passes the original `weights` to `covariance()`, not `weights·hub`. The module's own docstring flags this as a correspondence requirement | `anchoring.py:148-169`; `pipeline.py:201` |
| 8 | Neighbour spread `ν(p)` | **Neither a standard deviation nor a range.** It is the **absolute deviation of the pixel's own nearest-anchor depth from the local mean of valid nearest-anchor depths**: `ν = \|tof_r − local\|` where `local = blur(tof_r·valid) / blur(valid)`. Window **11 × 11 cells on the reduced grid** (`SPREAD_WIN = 11` at `BLEND_SCALE = 4`), ≈ 2.5 zones. A max−min range was **tried and rejected** — it fires on any pixel merely near an edge and took a correct 2.02 m reading from σ 0.97 to 2.72 | `blend.py:71`, `blend.py:189-205` |
| 9 | `c_a` | **1.00** (`DISAGREE_K`), entering **squared** | `blend.py:62` |
| 10 | `c_alpha` | **0.35** (`SUPPORT_FRAC`), entering **squared** → 0.1225 | `blend.py:63` |
| 11 | `c_nu` | **1.00** (`SPREAD_K`), entering **squared** | `blend.py:64` |
| 12 | `alpha_0` | **5.0°** (`FAR_DEG`) | `blend.py:29` |
| 16 | Resolution chain | capture **1640 × 1232** (IMX219 full-sensor 2×2-binned) → rectification **1640 × 1232** (unchanged, full detail preserved) → backbone input **384 × 288** → backbone output 384 × 288, resized back to **1640 × 1232** → refiner input **384 × 288**, output resized to **1640 × 1232** → unprojection **stride 4** → cloud **410 × 308**. **"Full resolution" means 1640 × 1232.** Note the Table I timing rows at 480 × 640 are the *DEPTHOR comparison* resolution, not ours | `calibration.yaml:4-26`, `backbone.py:37,47,59`, `residual.py:51,99-101`, `pipeline.py:57,301` |
| 17 | Software versions | **L4T R36.5.0** (rev 5.0, GCID 43688277, Jan 2026), **TensorRT 10.3.0.30-1+cuda12.5**, **CUDA 12.6.68**, **cuDNN 9.3.0.75-1** | `env` block, every r31 JSON |
| 18 | Orin configuration | **AGX Orin Developer Kit, 64 GB module** (61 GB visible); **nvpmodel MAXN**; **`jetson_clocks` applied** for every r31 timing run | `/proc/device-tree/model`, `nvpmodel -q`, `clocks` block |
| 19 | DEPTHOR timing setup | **PyTorch, FP32** — see §0 | `param_dtypes`, `weights` in both timing JSONs |
| 20 | Input queue | **Confirmed bounded to one frame.** Both inputs subscribe at depth 1: `create_subscription(Image, 'image', …, 1)` and `create_subscription(ToFFrame, 'tof', …, 1)`. Publishers are depth 5 | `perception_node.py:138-142` |
| 21 | Data-age timestamp | Data age starts from the **dToF message header stamp**, which is `self.get_clock().now()` **taken in the ToF driver at publish time** — i.e. the Jetson ROS clock *after* serial read and subframe assembly. It is **not** camera capture time and **not** sensor exposure time. Every published depth, variance and cloud message carries this same dToF stamp | `tof_driver_node.py:59`; `perception_node.py:206-217` |
| 23 | Frame pairing | **The dToF frame drives the pipeline, and it pairs with whatever camera frame is most recently cached — there is no stamp matching of any kind.** `on_image` stores the raw buffer and does not retain its stamp; `on_tof` rectifies that buffer and runs. Camera runs ~28–30 Hz, dToF ~10 Hz. **Partial subframes ARE used**: the assembler holds a persistent 32 × 32 map, each subframe overwrites only its half (even or odd rows), and the map is republished **on every subframe** once both halves have been seen — so the publish rate roughly doubles. A half not refreshed within **`HALF_MAX_AGE_S` = 0.5 s** is expired to NaN. **Consequence: every published map is half-stale by construction**, one half fresh and one up to 0.5 s old | `perception_node.py:163-190`; `tof_source.py:203-241` |
| 24 | Planner rate / top speed | Planner rate **3.0 Hz** (`route_rate_hz` default). **Top speed was deferred by author decision on 2026-09-14** — §V-A's real-time definition is rate-versus-rate and is satisfied without it | `leroi_perception/route_planner_node.py:95` |
| 25 | Refiner training data | Real logs `data/real/rgb` + `data/real/tof`, backbone `student_v4_heldout/student_best.pth`, hold-out **island, `--island 16`, `--holdout-frac 0.25`**, val fraction **0.05**. Optimizer **AdamW**, lr **1e-3**, weight decay **1e-4**, batch **8**, input **288 × 384**, cosine LR schedule, grad clip 1.0, `nll_weight` 0.2, `struct_weight` 0.3. Budget `--epochs 200` with `--patience 8`; the shipped `residual_v7` ran **111 epochs** (best −0.2328 at epoch 108) and patience **never fired** | `training/README.md:178-199`, `train_residual.py:157-246`, `train_v7_long.log` |

### 1.2 False premises — five items that cannot be filled

See §2 for the three already known (items 3, 4, 5 fallback; item 13 `α_max`; item 14 NLL fit).
**Item 22 is a new one of the same kind, found during this pass:**

> **Item 22 — there is no ESP32-to-Jetson clock mapping.** No hardware timestamp is
> transmitted by the ESP32, parsed by the driver, or stored anywhere. The dToF frame's stamp is
> `self.get_clock().now()` taken **on the Jetson at driver publish time** (`tof_driver_node.py:59`),
> and the assembler's internal time is `time.monotonic()`, also Jetson (`tof_source.py:222`).
> There is no offset estimate and no drift correction because there is no second clock to align to.

This has a consequence for §V-A that should be settled before T4 runs: **`age_budget_r31.json`'s
`sensor_to_arrival_ms` (median 1.39 ms) is not sensor-to-arrival.** It is the gap between the
driver's own publish stamp and the perception node's receipt — essentially one DDS hop. The real
exposure → ESP32 → USB-serial transport interval is **unmeasured**, and cannot be measured
without adding a hardware timestamp to the firmware. T4's step 2 ("log the capture stamp") is
therefore not currently runnable as specified.

### 1.3 Absent — items 27 to 30

Not recorded anywhere in the repository. `calibration.yaml` gives the camera↔dToF extrinsic
(`translation_mm: [0.0, 20.195, 1.13]`, rotation identity, side offset assumed zero) and the
floor implies a lens height of 15.97 cm against a 16.2 cm tape — 2.3 mm agreement. It also
records **fov_h 73.5°, fov_v 60.5°, cone edge 30.25°**, and the dToF footprint as **567 × 443 px,
12.4 % of the 1640 × 1232 frame**.

But **item 27** (optical centre relative to the *robot centre*), **item 28** (optical centre depth
inside the lens) and **item 29** (pitch/yaw relative to the *robot body*) do not exist in any
config or document, and **item 30** follows from 27. These require the CAD assembly or fresh
measurement, and they gate T3.

### 1.4 Item 26 — answered, with a provenance caveat

ρ = **0.9962** (with `val_ssi` 3.51, δ1.25 0.89), student vs teacher, measured on the
**2000-image pilot distillation run**, validated via `compare_student.py` / `eval_student.py`.

⚠️ **That is the pilot student (3.66 M params), not the deployed backbone.** The shipped engine is
`student_v4_heldout_fp16`, trained later on the full set. The paper cites ρ = 0.996 in the V intro
without saying it came from the pilot, so either re-measure ρ for `student_v4_heldout` or state
explicitly that the figure describes the pilot run.

---

## 2. Four items whose premise the code does not support

These cannot be filled in. They need method text changed, not values supplied. A fifth
(item 22, the ESP32 clock) is in §1.2, and a sixth (item 26 provenance) in §1.4.

### Items 3, 4, 5 — there is no fallback

§IV-C says the system "falls back to a scale-only estimate"; limitation 4 repeats it. The code
(`pipeline.py:171-173`):

```python
fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
if fit is None:
    return {'ok': False, 'n_anchors': int(inb.sum())}
```

**The frame is dropped.** There is no scale-only path, no held shift `b̂`, and no threshold
constants — `N_min`, `v_min` and `κ_max` have no counterparts anywhere in the source. The only
guards are inside `solve_scale_shift`: `w.size < 2`, `w.sum() <= eps`, `|den| < eps`.

Two ways forward, and this is an author decision: rewrite §IV-C and limitation 4 to describe
frame-dropping, or implement the documented guards and re-run every number that depends on
them.

**Consequence for T6.** As written, T6 asks for "the fire rate per trigger" across triggers
that do not exist. It should become: frame-drop rate over all logs, plus the fire rate of the
three real guards, plus the frame counts with zero anchors, a stale map, or a fitted scale ≤ 0.
That version is runnable today and still answers reviewer comment 6.

### Item 13 — `alpha_max` is not an angular threshold

The "100 % floor" is `ROI_OUTSIDE_SIGMA_FRAC = 1.0` (`pipeline.py:31`), applied as a separate
clamp keyed to the **geometric ROI mask** (reach ≤ 3.0 m, height ≤ 0.6 m), not to angular
distance from the nearest anchor. It is visible in the n = 15 tape data: points 8 and 9 have σ
exactly equal to the prediction. This needs rewritten method text, not a different number.

### Item 14 — the constants were not fitted by NLL minimisation

The paper says they were chosen "by minimizing the Gaussian negative log likelihood of the tape
residuals". **No NLL fit exists anywhere in the record.** What actually happened, from
`blend.py`'s comments and `live4_sigma_after_2026-08-04.json`:

- the terms were **designed against two identified failure modes** observed in one live session
  (a mixed-return marker at 3.9σ, and an angle asymmetry) — not fitted;
- `DISAGREE_K` and `SPREAD_K` are both **1.0**, a natural unit scale rather than a fitted value;
- `SUPPORT_FRAC = 0.35` is the only non-trivial constant;
- `SPREAD_WIN = 11` is fixed by **geometry** (~17.7 px per zone → 11 cells ≈ 2.5 zones); a first
  attempt at 3 cells was smaller than one zone and had no effect;
- a single global multiplier rescale **was swept and rejected** — "no single scale serves both
  ends";
- scored against 11 tape points, with the contemporaneous note that "overfitting risk is real".

Suggested replacement wording: *designed against two identified failure modes, two constants at
unity, one at 0.35, window set by zone geometry, scored against eleven tape points; a global
rescale was swept and rejected.* This is both a stronger methodological story and an accurate
one.

---

### Item 15 — Eq. 8 is not what the code computes

Asked: confirm the code computes `V_t = V_fit + τ² + c_a Δ² + c_α(α/α_0)² D² + c_ν ν²`, floored
at `D²`. **It does not.** The assembled total is right, but three structural details differ.

What `pipeline.run` assembles:

```
var = V_fit            (delta method: D⁴ · jᵀ Cov(a,b) j)        pipeline.py:196-208
    + τ²               (refiner's learned variance)               pipeline.py:220-224
    + sigma_support_var(...)                                      pipeline.py:256-260
```

and `sigma_support_var` (`blend.py:183-211`) computes:

```python
disagree = disagree_k * wgt * abs(tof_r - net_r)
short    = clip((ang - far_deg) / far_deg, 0.0, None)
support  = support_frac * maximum(net_r, 0.0) * short
spread   = spread_k * wgt * abs(tof_r - local)
var_r    = disagree**2 + support**2 + spread**2
```

The three differences:

1. **Every term carries the blend weight `wgt`**, the smoothstep of angular distance to the
   nearest anchor. Both the disagreement and the spread terms are multiplied by it, and `wgt`
   does not appear in Eq. 8 at all. It is deliberate — far from any anchor `wgt → 0` so an
   irrelevant Voronoi neighbour's disagreement cannot inflate σ — but the equation as printed
   does not say so.
2. **The angle term is `clip((α − α_0)/α_0, 0, ∞)²`, not `(α/α_0)²`.** It is **exactly zero
   whenever an anchor is within `far_deg`**, so it never touches the well-supported interior of
   the cone. `(α/α_0)²` as printed is nonzero everywhere, which describes different behaviour.
3. **The constants enter squared**, because each term is formed and then squared. `c_a` and
   `c_ν` are 1.0 so squaring is numerically invisible, but `c_α` is `0.35² = 0.1225`. Whether
   the paper's symbols denote the pre- or post-square value should be stated.

**And the `D²` floor is not in this function.** It is `ROI_OUTSIDE_SIGMA_FRAC = 1.0` applied as a
separate clamp keyed to the geometric ROI mask (`pipeline.py:31`) — the same finding as item 13.

Suggested fix: print the equation as the code computes it, with `wgt` shown explicitly and the
angle term written as a clipped, shifted ratio. The current form is not a simplification of the
code; it is a different function.

---

## 3. Section 2 — question 3 is already settled

**Tape was measured as slant range, and the conversion to axial depth is already applied.**
Both `tools/diagnostics/tape_capture.py:177` and the saved `tape_r31/tape_gt.json` record the
convention explicitly:

> `range_m` is SLANT RANGE from the optical centre; `tape_eval.py` converts to axis depth via
> `z = r*cos(theta)`

So the failure mode the note worries about — comparing slant range against axial depth and
inflating off-axis error by ~1.29× at 38.9° — **did not occur**. The r31 tape numbers are
already axial.

One methodological consequence worth deciding before T3, though. The existing rig measures `r`
from the **optical centre** and derives `theta` from the marker's **pixel coordinate**. That is
why §3.12 found that on point 11, of a reported 2.071 m error, a pixel within 45 px errs by
only 1.306 m — **~0.77 m of the largest out-of-cone error is where the click landed**, not what
the pipeline predicted. Human pointing is a first-order error term in the current reference.

The three-leg scheme proposed in T3 measures from the robot centre, so `theta` comes from tape
rather than from pixels, which removes pointing error from the reference entirely. That is the
better design — but it requires items 27–30 first (see §1).

---

## 4. Section 3 — both blocking issues are already diagnosed

### Issue 1: Table IV — resolved, and the explanation is structural

The eleven are indeed a subset of the fifteen. Matching on pixel coordinates (≤ 45 px, ground
truth agreeing within 10 cm ⇒ the same marker re-measured):

| Role | n | Point IDs |
|---|---|---|
| Calibration marker re-measured | **8** | 3, 5, 6, 7, 8, 9, 10, 12 |
| Ambiguous — same pixel region, ground truth differs ≥ 0.1 m | 2 | 1, 11 |
| **Strictly held out** | **5** | 2, 4, 13, 14, 15 |

**The independent sample is seven points, not fifteen.** The paper's "tuned on eleven of the
points and scored here on all fifteen" reads as fifteen independent points and must be
corrected.

All three columns have been recomputed under one set of constants, with binomial (Wilson)
intervals rather than the repeat-capture spread that was there before — the old `lo`/`hi` were
pipeline noise across 5 repeats of the same point, not sampling uncertainty:

| Set | @1σ | 95 % Wilson | @2σ | 95 % Wilson |
|---|---|---|---|---|
| n = 11, before σ terms | 0.818 | [0.523, 0.949] | 0.818 | [0.523, 0.949] |
| n = 11, after σ terms | 0.909 | [0.623, 0.984] | 0.909 | [0.623, 0.984] |
| n = 15, all points | 0.800 | [0.548, 0.930] | 1.000 | [0.796, 1.000] |
| n = 15, held out only (n = 7) | 0.571 | [0.250, 0.842] | 1.000 | [0.646, 1.000] |
| *targets* | *0.683* | | *0.954* | |

**Every interval contains its target and every pair overlaps heavily.** The claims that
coverage "moves toward its target rather than away" and that the σ terms improved coverage
(0.818 → 0.909) are **not supported at this sample size** and should be withdrawn.

What the data does support, and it is a better claim: **σ ranks errors correctly even where it
was not tuned.** Rank correlation is +0.679 on the held-out points against +0.762 on the
calibration points — but σ is **under-sized in magnitude** there, and the worst residual
(1.724 m) comes entirely from the held-out side.

Coverage counts **points**, not point captures. → `tape_stats_r31.json`

### Issue 2: the out-of-cone example — found

The quoted "error reaches 1.07 and 1.52 m while σ is 2.10 and 1.84" is
`live4_sigma_2026-08-04.json`, **points 7 and 8** — which are *calibration-set* points. That is
why no held-out capture reproduces them. Held-out replacements carrying the same conclusion (σ
declares the extrapolation) are available now:

| id | Role | gt | error | σ | nσ |
|---|---|---|---|---|---|
| 1 | ambiguous | 1.70 | −0.265 | 1.932 | 0.14 |
| 11 | ambiguous | 4.03 | −2.071 | 1.962 | 1.05 |
| 13 | held out | 0.82 | +0.531 | 3.274 | 0.16 |
| 15 | held out | 1.86 | −0.526 | 0.468 | 1.12 |

Recommendation: re-source §V-E to these four. Same conclusion, independent data, no new session
required. Note the caveat on point 11 from §3 above.

---

## 5. Values pulled from the existing r31 files

Everything in this section is extracted from committed JSONs. No new runs.

### 5.1 Drive 2 — MAE with and without arbitration (§V-B)

From `blend_ab_live_r31_run2.json`, 600 frames:

| | Without arbitration | With arbitration | Paired difference |
|---|---|---|---|
| MAE | 0.19843 | **0.18503** | **−0.013397** [−0.013997, −0.012792], p = 0.000 |
| medAE | 0.04553 | **0.04432** | **−0.001213** [−0.001817, −0.000721], p = 0.000 |

B = 2000 resamples, `n_groups` = 600. For comparison, drive 1 (`blend_ab_live_r31.json`) gives
MAE 0.16100 → 0.15260, paired −0.008393 [−0.009331, −0.007514].

### 5.2 Is the 15–30° band value a median or a mean? — **a median**

The field is `binned_medae`, the **median** absolute error per angular band, computed alongside
(not from) the pooled `mae`. Band edges are `[0, 3, 6, 10, 15, 30, 90]` degrees, so the 15–30°
band is index 4:

| | Drive 1 | Drive 2 |
|---|---|---|
| 15–30°, no arbitration | 0.036851 | 0.073452 |
| 15–30°, with arbitration | **0.034244** | **0.055413** |
| 30–90°, both arms | 0.032632 | 0.037005 |

⚠️ Worth stating in the paper: the near bands are **empty**. Drive 1 has no evaluated pixels
below 10°, drive 2 none below 15°. The band table has fewer populated cells than it appears to.

### 5.3 Tape — rank correlation and worst residual (Table IV)

Recomputed directly from the 15 points in `live4_sigma_n15_2026-08-05.json` (Spearman of
predicted σ against |error|), reproducing `tape_stats_r31.json`:

| Set | n | Spearman(σ, \|err\|) | p | Worst residual (σ) |
|---|---|---|---|---|
| **All fifteen** | 15 | **+0.625** | 0.013 | **1.7236** |
| Calibration re-measured | 8 | +0.762 | 0.028 | 0.2147 |
| **Held out + ambiguous** | 7 | **+0.679** | 0.094 | **1.7236** |
| Strictly held out | 5 | +0.800 | 0.104 | 1.7236 |

The worst residual is **identical (1.7236 σ) on all fifteen and on the held-out seven**, and is
0.2147 on the calibration points — so the worst case comes **entirely from the held-out side**.
That is the quantitative basis for the replacement claim: σ **orders** errors about as well
out-of-sample as in-sample, but is **under-sized in magnitude** out-of-sample.

Coverage counts **points**, not point captures. For completeness, the earlier n = 11 sets:
worst residual **3.863 σ before** the σ terms and **2.304 σ after**.

### 5.4 Confirmation — points 1, 11, 13 and 15 are the four out-of-cone points

**Confirmed.** `tape_stats_r31.json → out_of_cone_claim.all_out_of_cone_n15` lists exactly these
four and no others:

| id | Role | What | gt (m) | error (m) | σ | nσ |
|---|---|---|---|---|---|---|
| 1 | ambiguous | thin X, band far left | 1.700 | −0.265 | 1.932 | 0.137 |
| 11 | ambiguous | above the cone, banner | 4.031 | −2.071 | 1.962 | 1.055 |
| 13 | held out | far right, on the blue rings | 0.815 | +0.531 | 3.274 | 0.162 |
| 15 | held out | last marker on the left wall | 1.860 | −0.526 | 0.468 | 1.123 |

Two are "ambiguous" in the provenance sense (same pixel region as an earlier marker but ground
truth differing by 0.67 m and 0.96 m, so they cannot be the same surface) and two are strictly
held out. **None is a calibration point**, which is what makes them a valid replacement for the
§V-E example.

Note on point 11: of its 2.071 m error, **0.765 m is attributable to pointing** — the best pixel
within 45 px errs by only 1.306 m.

### 5.5 Ablation rows already present in `baselines_r31.json`

Run over **1234 frames** at 1232 × 1640, seed 0, B = 1000, `island 16`, blend 2.0–5.0°,
clamp 20 m, range gate [0.15, 6.5] m. All eighteen rows exist across three protocols
(`random`, `center`, `edge`):

- **Weighting rows** — `W0_uniform_norobust`, `W0_uniform`, `W1_rangep1` (w ∝ z),
  `W2_rangep2` (w ∝ z²), `W3_roi`
- **Baselines** — `B0_const`, `B1_nearest`, `B2_bilinear`, `B3_medscale`, `B4_affine`, `B4c_affine_cl`
- **Refiners** — `B5_ringfusion` / `B6_blend` (v7, deployed), **`B5_scattered` / `B6_scattered`**,
  `B5_island` / `B6_island`, plus `B6_analytic`

So T5's scattered-refiner row and all four weighting rows are **already computed** — no re-run
is required for them.

### 5.6 Robust-pass fraction — T5's remaining gap is already closed

`baselines_r31.json → protocols.random.robust_pass`, over all 1234 frames:

| Quantity | Value |
|---|---|
| Median anchors per frame | **621.5** |
| Anchors past the Huber threshold | **22.9 %** median (23.3 % mean) |
| Total weight mass removed | **6.9 %** median |

The file carries its own note: *"Huber DOWNWEIGHTS, it does not reject. The paper says 'removes
X% of anchors', which names neither."* The paper should quote **22.9 % down-weighted** and
**6.9 % of weight mass removed**, and drop the word "removes".

### 5.7 ZJU-L5 footprint as a percentage of valid pixels

The in/out-footprint split already exists in the r31 ZJU-L5 files (`regions`: `all`, `in`, `out`).
Over 527 test frames and 153,640,646 evaluated pixels:

| Region | Evaluated pixels | Share of valid pixels |
|---|---|---|
| Inside the 8 × 8 footprint | 85,178,297 | **55.44 %** |
| Outside | 68,462,349 | **44.56 %** |

Deployed-row MAE by region, showing the separation the paper wants:

| Row | all | in | out |
|---|---|---|---|
| `B1_nearest` | 0.3171 | **0.0945** | **0.5939** |
| `B2_bilinear` | 0.1146 (cov 0.500) | 0.0775 (cov 0.753) | 0.3040 (cov 0.184) |
| `B4c_affine_cl` | 0.4027 | 0.1814 | 0.6781 |
| `B6_blend_over_D0` | 0.3678 | **0.1191** | 0.6772 |

For our own hardware the corresponding figure is in `calibration.yaml`: the dToF footprint is
**567 × 443 px, 12.4 % of the 1640 × 1232 frame**, with fov_h 73.5°, fov_v 60.5° and a cone edge
at 30.25°.

### 5.8 Warm-up and timed iteration counts (Table I note)

Every r31 timing JSON records the same procedure block:

```json
"warmup_iters": 100, "timed_iters": 500,
"cuda_synchronized": true, "dataloading_excluded": true,
"batch_reused": true, "gpu_otherwise_idle": "asserted by operator"
```

Batch 1 at 480 × 640 for the DEPTHOR rows. This matches the "100 warm-up and 500 timed
iterations" rule in the test plan, so Table I's note can state it as measured rather than intended.

---

## 6. Section 4 — what r31 already banked

Four of the nine tests are substantially complete, each with an embedded environment and
engine-SHA block.

| Test | Status | Evidence | Gap |
|---|---|---|---|
| T5 ablation | ✅ largely done | `baselines_r31.json` (1234 pairs), `baselines_valsplit_r31.json` (61 val) — all rows, both protocols, angle bands, bootstrap CIs, paired tests | the anchor fraction the robust pass down-weights/rejects |
| T7 driving per session | ✅ done | `blend_ab_live_r31.json` + `_run2.json`, two 600-frame drives, paired intervals: MAE −0.0084 [−0.0093, −0.0075] and −0.0134 [−0.0140, −0.0128], both p = 0.000 | **both drives topped out at 4.56 m**, under the dToF's own 6.5 m gate — the beyond-range regime has never been measured under motion |
| T8 ZJU-L5 | ✅ largely done | `zjul5_deployed_r31.json`, `zjul5_deployed_blendnet_r31.json`, `zjul5_vits_r31.json`, `zjul5_ridge_train_r31.json` | the 8×8 footprint as % of valid pixels; the mask-difference list explaining the 6 % vs 15 % gap |
| T2 core timing | ⚠️ partial | `timing_pipeline_r31.json` — all six config × resolution cells | the tensor-size chain (item 16) and the in-node/offline ratio |
| T4 latency & data age | ⚠️ thinner than it looks | `age_budget_r31.json`, `profile_node_r31.json` | that run was **117 s at 8.455 Hz**, not 10 min stationary + 10 min driving; no map-ID, map-reuse fraction or queue-drop columns; its `engines` block is empty |
| T1 DEPTHOR matched | ❌ not started | — | see §0 — this is the priority |
| T3 tape reference | ❌ blocked | `tape_r31/tape_gt.json` is an empty template (`"points": []`) | **markers are not currently placed in the environment**; items 27–30 unresolved |
| T6 fallback frequency | ⚠️ needs respecifying | — | see §2 — the triggers do not exist |
| T9 refiner seeds | ❌ not started | — | needs a `--seed` flag; see §6 |

The offline work outside those gaps is complete: the full 1234-frame ablation, the 61-frame
val-split ablation, the tape re-analysis, the ZJU-L5 re-runs and the DEPTHOR re-timing have all
landed.

---

## 7. Two scheduling constraints the test plan should account for

**1. There is one GPU, and it is the Jetson.** The refiner trains with `device cuda` on the same
AGX Orin that runs the bench timing. T9 cannot "run in the background" while T1 and T2 are
measured — those require the GPU otherwise idle, which every r31 timing JSON asserts explicitly.
T9 and T1/T2 must be serialised. Given T1 gates the abstract, the natural order is T1 + T2
first (they are minutes of compute once the engines are built), then T9 overnight.

Measured refiner wall times on this hardware, from the r31 run directories:

| Run | Epochs | Wall time |
|---|---|---|
| `residual_v4_island` | 40 | 46 min |
| `residual_v5_mixed` | 40 | 40 min |
| `residual_v6_fov73` | 40 | 40 min |
| `residual_v7_long` (shipped) | 111 | 114 min |

≈ **1 min/epoch**, stable across configs. So T9 at 3 seeds:

- 2 configs (scattered + island) on the shipped 111-epoch schedule ≈ **11–12 h**
- 3 configs (adding mixed) ≈ **17 h**
- on the shorter 40-epoch schedule, ≈ 4.5 h and 7 h respectively

The shipped engine used the long schedule, so seeds should match it for the island-gain
comparison to be meaningful.

**2. `train_residual.py` has no seed argument.** The only seeding is a hardcoded
`torch.Generator().manual_seed(0)` on the train/val split (line 236); there is no global
`torch.manual_seed`, so weight init and batch order are not currently controllable. T9 needs a
`--seed` flag added first.

A design point for that flag: T9 specifies "the same data and hyperparameters", so `--seed`
should vary **weight init and batch ordering only** and leave the split generator pinned at 0.
Re-seeding the split as well would confound initialisation variance with split variance, and the
reported ± std would no longer mean what Table II needs it to mean.

---

## 8. Summary of what is asked of the author

1. **Decide the §IV-C fallback question** — rewrite to describe frame-dropping, or implement the
   guards and re-run dependents. Blocks items 3/4/5 and the specification of T6.
2. **Accept the §IV-F wording replacement** for item 14, and the ROI-mask correction for item 13.
3. **Withdraw the coverage-improvement claim** in Table IV / §V-E and replace it with the
   rank-correlation claim, which the data supports.
4. **Re-source §V-E** to the four held-out points above.
5. **Rewrite Eq. 8** to match the code — the `wgt` factor, the clipped angle term, and whether
   the printed constants are pre- or post-square (item 15, §2).
6. **Decide how §V-A describes timing provenance**, given there is no ESP32 clock mapping and
   `sensor_to_arrival_ms` is a DDS hop rather than transport time (item 22, §1.2). This also
   determines whether T4 step 2 is runnable as written.
7. **Re-measure or re-caption ρ = 0.996**, which came from the 2000-image pilot student, not the
   deployed `student_v4_heldout` backbone (item 26, §1.4).
8. **Produce items 27–30** from CAD or fresh measurement — the hard prerequisite for T3.
9. **Schedule marker placement** — the environment has none at present, so T3 cannot be booked
   until that physical work is done.
10. **Confirm the T9 schedule** (2 or 3 configs, 40 or 111 epochs) so the overnight run can start.

Items that **no longer need a test run**: T5's robust-pass fraction (§5.6), T5's weighting and
scattered rows (§5.5), T7 entirely (§5.1–5.2), and T8's footprint percentage (§5.7).
