# RingFusion r31 — revision working log

> **The numbers now live in [`PAPER-REVISION-RESULTS.md`](../PAPER-REVISION-RESULTS.md)** at
> the repo root — organised by paper location, written for the consultant, and the file to
> send when the tests are done. **New measurements go there.** This file stays as the method
> and process record: how each tool works, why each protocol is shaped the way it is, the
> pre-flight checklists, and the phase-by-phase history.

**Purpose:** running record of what the reviewer asked for, what we found, what we measured, and where every number came from. Companion to `ringfusion-data-collection-brief.md`, which it supersedes where the two disagree.

**Build under test:** `aba5edd` (2026-09-12), deployed config `student_v4_heldout_fp16` + `residual_v7_fov73_fp16`, blend + ROI on.

**Status:** Phase 0 complete. Phase 1 running (1.1, 1.2, 1.3a done). **No robot session has been run and none will start without explicit go-ahead** — see §6 for the pre-flight checklist.

---

## 1. The reviewer's five comments

| # | Comment | Phases |
|---|---|---|
| R1 | Ablation separating analytic anchoring, robust weighting, residual refiner, arbitration, **and uncertainty estimation**. Accuracy inside and outside the dToF coverage area. | 0.4, 0.9, 1.1, 1.2, 1.3, 3.1–3.4, 4.1 |
| R2 | Per-run variation or confidence intervals. Tape experiment must distinguish calibration from held-out points. | 0.1, 0.2, 0.8, 1.1, 3.3, 4.1 |
| R3 | Uncertainty-calibration procedure and constants fully documented; related claims **moderated** because the independent validation sample is limited. | 0.1, 0.3, 0.6, 3.3, 3.4 |
| R4 | Runtime: resolution, included stages, precision, hardware settings, measurement procedure. Relationship among core latency, publish rate, data age. | 0.4, 0.7, 1.5, 2.1–2.5 |
| R5 | Abstract/ridge on train split; define "real time"; qualify camera-generalization; document fallback; provide code, calibration, splits, runtime settings. | 0.5, 0.7, 1.3, 1.4, 4.2, Phase 5 |

---

## 2. Environment manifest (R4) — captured 2026-09-12

| Item | Value | Source |
|---|---|---|
| L4T | **R36.5.0**, GCID 43688277, aarch64, 2026-01-16 | `/etc/nv_tegra_release` |
| JetPack | metapackage not installed — quote L4T R36.5.0 instead | `dpkg -l nvidia-jetpack` → empty |
| TensorRT | **10.3.0.30-1+cuda12.5** | `dpkg -l \| grep tensorrt` |
| CUDA | **12.6.68** (SDK 12.6.11) | `nvcc --version` |
| cuDNN | **9.3.0.75** | `dpkg -l \| grep cudnn` |
| Power mode | **MAXN** ✅ (persists across reboots) | `nvpmodel -q` |
| `jetson_clocks` | ❌ **NOT applied** — see below | sysfs, 2026-09-12 19:0x, uptime 1h47m |
| Engine precision | **FP16** both networks | `build_engine.py --precision fp16` |
| Backbone | `student_v4_heldout_fp16.engine` (3.66M) | README deployed config |
| Residual | `residual_v7_fov73_fp16.engine` (0.46M) | README deployed config |

⚠️ `student_int8.engine` exists in the repo (Jul 22, superseded). The launch file has **no engine defaults**, so the deployed pair must be passed explicitly and recorded per run.

### ⚠️ `jetson_clocks` is not applied, and this affects every timing number (R4)

`nvpmodel` persists across reboots; **`jetson_clocks` does not** — it must be re-run after every boot. Read from sysfs (no root needed) on 2026-09-12 with the board up 1h47m:

| | Value | Locked? |
|---|---|---|
| CPU governor | `schedutil`, 729 600–2 201 600 kHz | ❌ min ≠ max |
| GPU governor (`17000000.gpu`) | `nvhost_podgov`, 306–1300.5 MHz, **idling at 306 MHz** | ❌ min ≠ max |
| Power mode | MAXN | ✅ |

So the board is in MAXN with clocks **free-running under DVFS**. Section V-A's "with the Jetson in maximum performance mode" is currently true only of `nvpmodel`, and Table III's note has a pending slot that would claim clocks were enabled.

**Consequences:**

1. Accuracy runs (1.1–1.4) are unaffected — clock state changes speed, not results.
2. The 1.5 DEPTHOR timings will record `jetson_clocks_applied: false` and must be re-run after `sudo jetson_clocks` so both arms exist.
3. **The Phase 2 robot session must apply `sudo jetson_clocks` and record the state**, or its numbers will not be comparable to 1.5's.
4. We do not know what state the historical numbers (79.4 ms, 25.8 ms, every `rate_live_*` capture) were taken in. Another reason not to mix old and new timing figures in one table.

`time_net.py` now records the full clock state alongside every timing result, so this can never again be a reconstruction problem.

---

## 3. Phase 0 — desk work

### 0.1–0.3 Tape re-analysis ✅

Tool: `tools/diagnostics/tape_stats_r31.py` → `docs/demo/benchmarks/tape_stats_r31.json`

#### Calibration vs held-out (R2, R3)

The 15-point set is **not** 15 independent points. Matching on pixel coordinates (≤45 px, ground truth agreeing within 10 cm ⇒ same marker re-measured):

| Role | n | Point IDs |
|---|---|---|
| Calibration marker re-measured | **8** | 3, 5, 6, 7, 8, 9, 10, 12 |
| Ambiguous (same pixel region, gt differs ≥0.1 m ⇒ separate target across a depth edge) | 2 | 1, 11 |
| **Strictly held out** | **5** | 2, 4, 13, 14, 15 |

**The independent sample is 5 points, at most 7.** The paper's "tuned on eleven of the points and scored here on all fifteen" reads as fifteen independent points and must be corrected. This is R2's second sentence and R3's "limited validation sample" in one.

#### Coverage, with binomial intervals (R2, R3)

The n=15 file's existing `lo`/`hi` are the spread across **5 repeat captures of the same points** — pipeline noise. They are *not* sampling uncertainty. A claim about where coverage sits relative to its target needs the binomial (Wilson) interval:

| Set | @1σ | 95% Wilson | @2σ | 95% Wilson |
|---|---|---|---|---|
| n=11, before sigma terms | 9/11 = 0.818 | [0.523, 0.949] | 9/11 = 0.818 | [0.523, 0.949] |
| n=11, after sigma terms | 10/11 = 0.909 | [0.623, 0.984] | 10/11 = 0.909 | [0.623, 0.984] |
| n=15, all points | 12/15 = 0.800 | [0.548, 0.930] | 15/15 = 1.000 | [0.796, 1.000] |
| n=15, held out only (n=7) | 4/7 = 0.571 | [0.250, 0.842] | 7/7 = 1.000 | [0.646, 1.000] |
| *targets* | *0.683* | | *0.954* | |

**Finding: every coverage interval contains the target, and every pair of intervals overlaps heavily.** The paper's claims that coverage "moves toward its target rather than away" (0.909 → 0.760) and that the sigma terms improved coverage (0.818 → 0.909) are **not supported at this sample size**. This is precisely what R3's "moderate the claims" asks for, and it is the highest-value correction available at zero cost.

#### Depth error against tape, bootstrapped over points (B=10000)

| Set | n | median | 95% CI | MAE | 95% CI |
|---|---|---|---|---|---|
| All 15 points | 15 | 0.051 | [0.009, 0.531] | 0.345 | [0.112, 0.656] |
| **Inside the cone** | 8 | **0.029** | **[0.012, 0.101]** | 0.152 | [0.020, 0.394] |
| On the cone edge | 3 | 0.002 | [0.000, 0.559] | 0.187 | [0.000, 0.559] |
| **Outside the cone** | 4 | **0.528** | **[0.265, 2.071]** | 0.848 | [0.331, 1.685] |
| Strictly held out | 5 | 0.526 | [0.002, 0.978] | 0.408 | [0.109, 0.708] |

**Two findings:**

1. ✅ **In-cone vs out-of-cone separation is real and survives the interval** — 0.029 [0.012, 0.101] against 0.528 [0.265, 2.071], non-overlapping. This is the one strong claim the tape data supports, and it is a direct answer to R1's "accuracy inside and outside the dToF coverage area."
2. ⚠️ **In-cone tape (0.029 m) independently confirms Table III center (0.032 m) to 3 mm.** The paper does not currently make this comparison. It is the best available answer to "your evaluation is circular" and should be stated explicitly.
3. ❌ The all-points median has a 30× interval. **The tape reference cannot carry an accuracy claim or rank methods** — only validation. See §5.

#### Out-of-cone claim re-sourced (R3)

Section V-E quotes "error reaches 1.07 and 1.52 m while it reports σ of 2.10 and 1.84". Located: `live4_sigma_2026-08-04.json` points 7 and 8 (gt 3.178 → pred 2.107, σ 2.103; gt 3.3635 → pred 1.845, σ 1.844). **These are calibration-set points** — the constants were fitted on them. The paper's `VERIFY, BLOCKING` note is resolved: the run exists.

Held-out out-of-cone points available as a **non-circular** replacement:

| id | Role | gt | error | σ | nσ |
|---|---|---|---|---|---|
| 1 | ambiguous | 1.70 | −0.265 | 1.932 | 0.14 |
| 11 | ambiguous | 4.03 | −2.071 | 1.962 | 1.05 |
| 13 | held out | 0.82 | +0.531 | 3.274 | 0.16 |
| 15 | held out | 1.86 | −0.526 | 0.468 | 1.12 |

Recommend re-sourcing Section V-E to these. Same conclusion (σ declares the extrapolation), independent data.

#### Pointing sensitivity ⚠️

`best_within_45px` on point 11: the reported error is 2.071 m but a pixel within 45 px errs by only 1.306 m. **~0.77 m of the largest out-of-cone error is where the click landed**, not what the pipeline predicted. Human pointing is a first-order error term in this experiment and belongs in the caveats.

### 0.4 Table III header is wrong (R1, R4) ✅

Table III's column group reads *"center, by angle from optical axis"*. [`baselines.py:218`](../tools/diagnostics/baselines.py#L218) computes `ang = dmat.min(1)` — angular distance from each held-out zone to the **nearest surviving anchor** — and the script prints "median AE (m) by angular distance from nearest anchor".

Consequence: the bands are a genuine extrapolation axis, so V-B's crossover argument is correct in substance. **Only the header is wrong.** Fix the header; the numbers stand.

### 0.5 The documented fallback does not exist (R5) ⚠️ BLOCKING

Section IV-C: *"the system falls back to a scale-only estimate"*. Discussion limitation 4: *"falls back to a scale-only estimate without signalling it"*. The paper also has pending slots for `N_min`, `v_min`, `κ_max` and a "held at its previous value" for `b̂`.

Actual code, [`pipeline.py:171-173`](src/ringfusion_perception/ringfusion_perception/pipeline.py#L171):

```python
fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
if fit is None:
    return {'ok': False, 'n_anchors': int(inb.sum())}
```

**The frame is dropped.** There is no scale-only fallback, no `b̂` hold, and no threshold constants. The only guards are inside `anchoring.solve_scale_shift`: `w.size < 2`, `w.sum() <= eps`, `|den| < eps`. `--max-cond 1e8` is a `baselines.py` flag the node never sees.

**Not measurable — needs an author decision:** rewrite both passages to describe frame-dropping, or implement the guards (and re-run every dependent number).

### 0.6 Uncertainty constants (R3) ✅

Verified against HEAD:

| Paper symbol | Value | Code | Status |
|---|---|---|---|
| `c_a` | 1.00 | `DISAGREE_K = 1.0`, squared | ✅ |
| `c_ν` | 1.00 | `SPREAD_K = 1.0`, squared | ✅ |
| `c_α` | 0.1225 | `SUPPORT_FRAC = 0.35`, squared | ✅ |
| `α_0` | 5.0° | `FAR_DEG = 5.0` | ✅ |
| smoothstep range | 2.0°–5.0° | `NEAR_DEG = 2.0`, `FAR_DEG = 5.0` | ✅ |
| `α_max` | — | ❌ **mis-mapped** | see below |

`var_r = disagree² + support² + spread²` in `blend.sigma_support_var` — matches Equation (11)'s form.

⚠️ **`α_max` does not exist as an angular threshold.** The "100% floor" is `ROI_OUTSIDE_SIGMA_FRAC = 1.0` at [`pipeline.py:31`](src/ringfusion_perception/ringfusion_perception/pipeline.py#L31), applied as a separate clamp keyed to the **geometric ROI mask** (reach ≤ 3.0 m, height ≤ 0.6 m), not to angular support. Visible in the data: n=15 points 8 and 9 have σ exactly equal to pred. The Method text needs rewriting, not a number.

**Still open:** the paper says constants were chosen "by minimizing the Gaussian negative log likelihood of the tape residuals". `live4_sigma_after` describes a hand sweep against coverage. Needs the author to confirm which, and an honest "hand-tuned against 11 points" is publishable.

### 0.7 Build/split manifest (R4, R5) — partial

Splits to release: `heldout_stems.txt` (200), `val_stems.txt` (61, **entirely a subset of the 200**), ZJU-L5 train/test, tape calibration/held-out assignment. `data/` is gitignored so no stem→frame mapping is currently public.

**Data locations** (the two `data/` trees are different datasets and are easy to confuse):

| Path | Contents | n |
|---|---|---|
| `ros2_ws/data/real/{rgb,tof}` | **the logged camera–dToF pairs** — `.png` + `.npz(dist_m, confidence, mirrored)` | **1234** ✅ |
| `data/rect/`, `data/teacher/` | distillation set: rectified frames + cached teacher targets | 2000 / 1228 |
| `data/zjul5/ZJUL5/` | ZJU-L5, 16 scenes, `data.json` | 527 test |

✅ **The paper's "1234 logged pairs" is correct.** (An earlier note in this log flagged 1228 — that was `data/rect/paired`, the distillation set, not the paired logs.)

Both Depth Anything V2 teachers (`Small-hf`, `Large-hf`) are in the local HF cache, so every Table VI row is reproducible offline.

### 0.8 Frame-level bootstrap utility ✅

`tools/diagnostics/bootstrap.py` — `wilson()`, `frame_bootstrap()`, `paired_diff()`, `summarise()`.

Resamples **frames, not pixels**: errors within a frame are strongly correlated, so pixel-level resampling gives intervals far too tight. `paired_diff()` shares the frame draw across both arms, which is what the arbitration A/B claim needs.

### 0.9 Extend `baselines.py` ✅

**Five weighting arms**, all on the same anchors and the same held-out zones so the column is a clean ablation:

| Method | Weighting | Answers |
|---|---|---|
| `W0_uniform_norobust` | w = 1, no Huber pass | Table V reference row |
| `W0_uniform` | w = 1, one Huber pass | isolates the robust pass |
| `W1_rangep1` | w = z | what Section IV-C **claims** ships |
| `W2_rangep2` | w = z² | what error propagation predicts |
| `W3_roi` | geometric ROI gate | what **actually** ships |

`W0_uniform` reaches the same fit as `B4c_affine_cl` by a different code path, so the report prints the gap as a self-check. **Measured 0.0000 m on both protocols** — the arms are wired correctly.

**Robust-pass instrumentation.** `anchoring.solve_robust` takes an optional `info` dict (no behaviour change when unused). ⚠️ **The paper's wording is wrong**: Huber *downweights*, it does not reject, so "removes X.X% of anchors" names neither quantity. Both are now reported. On a 20-frame smoke test: **30–34% of anchors downweighted, removing 9–12% of total weight mass.**

**New `edge` protocol** (task 1.2). Holds out zones whose depth differs from the median of their valid 8-neighbours by more than `--edge-thresh` (default 0.15 m, >10× the sensor's own 0.01 m accuracy), anchoring on the smooth remainder. A frame with no real discontinuity is **skipped** rather than contributing a relabelled smooth quartile. This attacks limitation 2 of the withheld-zone protocol — the one that is fixable offline.

Smoke test (20 frames) shows it bites hard: `B1_nearest` medAE **0.193 m** on edge zones against ~0.03 m on the center protocol, and the analytic arms land at 0.60–0.67 m. This is the failure mode the paper currently describes only anecdotally (the thin-pole marker).

**Frame-level bootstrap wired in** via `--bootstrap` (default 1000). Vectorised resampling: naive per-draw concatenation over 1228 frames would put the ablation into hours, so groups are flattened once and each draw is gathered arithmetically — verified identical to the naive result, 24 s per statistic at B=1000.

⚠️ **In/out-footprint split deliberately NOT added.** Every evaluation target is a dToF zone and zones exist only inside the cone, so the split is degenerate on own-hardware data. Fabricating one would misrepresent what was measured. R1's inside/outside requirement is carried by the tape reference, ZJU-L5 regions, and Phase 3's plane captures.

---

## 4. Phase 1 — offline compute

| Task | Status |
|---|---|
| 1.1 Full 8-row ablation with intervals | ✅ `baselines_r31.json` |
| 1.2 Discontinuity-targeted hold-out | ✅ same file, `edge` protocol |
| 1.3 ZJU-L5 re-run by region | running |
| 1.4 ZJU-L5 ridge sweep on train split | queued |
| 1.5 DEPTHOR-Small/Large re-timing | queued — see clock-state note below |

Run: 1234 frames, 3 protocols, `student_v4_heldout_fp16` + `residual_v7_fov73_fp16`, B=1000 frame-level bootstrap, 677 ms/frame, 14 min.

### 1.1 ✅ Table III reproduces exactly on the current build

The paper's existing numbers are sound. Angular bands, center protocol, medAE in m:

| Row | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| nearest-zone dToF — paper | 0.027 | 0.063 | 0.104 | 0.163 | 0.204 |
| nearest-zone dToF — **r31** | **0.027** | **0.063** | **0.104** | **0.163** | **0.204** |
| refiner island — paper | 0.033 | 0.028 | 0.031 | 0.032 | 0.036 |
| refiner island — **r31** | **0.033** | **0.028** | **0.031** | **0.032** | **0.036** |
| arbitration — paper | 0.026 | 0.027 | 0.031 | 0.032 | 0.036 |
| arbitration — **r31** | **0.026** | **0.027** | **0.031** | **0.032** | **0.036** |

The analytic reference also reproduces: **0.055 m on random, 0.054 m on center** against the paper's "scoring 0.055 m on both".

### 1.1 ✅ The weighting ablation refutes Section IV-C (R1, P0-2)

medAE with 95% frame-level bootstrap intervals:

| Row | random | center | edge |
|---|---|---|---|
| analytic, uniform, no robust pass | 0.0589 [0.0567, 0.0610] | 0.0552 [0.0533, 0.0570] | 0.4470 [0.4344, 0.4597] |
| **w ∝ z** *(what IV-C claims)* | **0.0716** [0.0682, 0.0752] | 0.0564 [0.0546, 0.0581] | 0.4754 [0.4642, 0.4866] |
| w ∝ z² | 0.1013 [0.0960, 0.1069] | 0.0658 [0.0636, 0.0679] | 0.5082 [0.4959, 0.5223] |
| **ROI gate** *(what ships)* | **0.0502** [0.0483, 0.0520] | 0.0545 [0.0527, 0.0564] | 0.4360 [0.4208, 0.4514] |
| uniform + robust pass | 0.0546 [0.0525, 0.0568] | 0.0540 [0.0522, 0.0558] | 0.4368 [0.4254, 0.4488] |

**On the random protocol `w ∝ z` is 43% worse than the deployed ROI gate, with non-overlapping intervals.** Section IV-C's "We use $w_i \propto z_i$" is refuted on our own data, not merely out of date.

Nuance worth stating rather than hiding: the ROI gate's advantage is real on `random` (0.0502 vs 0.0546, non-overlapping) but **indistinguishable on `center` and `edge`** (intervals overlap). Likewise the robust pass is a real improvement only on `random` (0.0589 → 0.0546, non-overlapping); on the other two protocols it is within noise.

**Robust pass rejection rate** (fills the pending slot): **22.7–22.9% of anchors downweighted, removing 6.8–6.9% of total weight mass**, stable across all three protocols. Use the second number if the sentence says "removes"; use the first if it says "downweights".

### 1.2 ✅ The discontinuity protocol — a new result for R1

New `edge` protocol: hold out zones whose depth differs from their 8-neighbour median by >0.15 m. 1146 of 1234 frames contained a real discontinuity; **88 were skipped and are reported, not silently backfilled**.

medAE, m:

| Method | random | center | **edge** | edge vs center |
|---|---|---|---|---|
| nearest-zone dToF | 0.018 | 0.117 | **0.228** | 1.9× |
| analytic + robust | 0.055 | 0.054 | **0.437** | **8.1×** |
| + residual refiner | 0.046 | 0.033 | **0.350** | **10.7×** |
| **+ source arbitration** | 0.014 | 0.032 | **0.133** | 4.2× |

Two findings:

1. **Arbitration is what rescues the discontinuity case.** On `center` the refiner and arbitration are indistinguishable (0.0326 vs 0.0319). On `edge` arbitration beats the refiner by **2.6×** (0.350 → 0.133). The blend was designed to take the dToF where a measurement is near; edge zones are surrounded by anchors, so it does exactly that, and the camera path is the one that gets the step wrong.
2. **Nearest-zone dToF beats the whole camera path on edge zones** (0.228 against 0.350–0.508). Only arbitration beats it. This is a clean, quantitative answer to the caveat in V-C — "error across a discontinuity falling between zones cannot appear" — which the paper currently supports only with the single thin-pole marker.

⚠️ Scope: `edge` holds out ~25% of zones at discontinuities while anchoring on the smooth remainder, so held-out zones sit mostly 0–3° from an anchor. It isolates the **discontinuity** effect, not extrapolation. State it that way.

### 1.3a ✅ Table VI's "as deployed" row is confirmed mislabelled — and the fix is *good* news

Re-run on the deployed pair (`zjul5_deployed_r31.json`, 527 test samples, ρ 0.466):

| Region | Configuration | AbsRel↓ | RMSE↓ | δ1↑ |
|---|---|---|---|---|
| all | analytic only (`B4c_affine_cl`) — **what Table VI prints** | 0.1808 | 1.1882 | 0.7272 |
| all | + residual refiner (`B5_ringfusion`) | **0.6754** | 3.0899 | **0.1429** |
| all | + arbitration over D0 (`B6_blend_over_D0`) | **0.1555** | 1.1768 | **0.7704** |
| inside footprint | analytic only | 0.1209 | 0.5081 | 0.8517 |
| inside footprint | + arbitration over D0 | **0.0756** | **0.4588** | **0.9286** |
| outside footprint | analytic only | 0.2553 | 1.6874 | 0.5722 |
| outside footprint | + arbitration over D0 | 0.2547 | 1.6870 | 0.5735 |

**Confirmed:** Table VI's printed 0.185 / 1.174 / 0.716 matches the *old* `B4c_affine_cl` (0.1847 / 1.1740 / 0.7160) to three decimals. The row is the **analytic stage with the refiner off**, while the caption calls it "the Complete 4.1M Pipeline".

**The refiner does not transfer, and v7 transfers worse than v4** — δ1 0.143 against the older engine's 0.292 on the same data. It improved on our own hardware (Table III reproduces exactly) and degraded off-domain. That is domain specialisation, and it strengthens the paper's own limitation 3 rather than undermining it.

**But arbitration more than rescues it.** Blending over the closed-form depth reaches δ1 0.7704 overall and **0.9286 inside the footprint**, beating the analytic row currently printed. So an honest "as deployed" row can be *better* than what the paper claims, not worse — provided it is the arbitration configuration and is labelled as such.

⚠️ **One variant still missing.** On the robot the blend sits **over the refiner output** (`pipeline.py`: `D_net` = refined depth, then blend), whereas `B6_blend_over_D0` skips the refiner. `zjul5_eval.py` defaults to `--blend-over d0` deliberately, because the refiner is out of domain on 8×8. The true deployed topology is `--blend-over net`, queued as **1.3c**. Until it lands, do not describe `B6_blend_over_D0` as "as deployed" — it is *arbitration without the refiner*.

### 1.3b ✅ The ViT-S row reproduces exactly

`zjul5_vits_r31.json`, 527 test samples, ρ 0.873:

| Row | AbsRel | RMSE | δ1 |
|---|---|---|---|
| Paper Table VI, "Ours, affine / ViT-S" | 0.094 | 1.056 | 0.902 |
| **r31 re-run** (`B4c_affine_cl`) | **0.0941** | **1.0564** | **0.9017** |

⚠️ One sub-claim does not reproduce. The Table VI note says RMSE is "0.418 inside the dToF footprint against 1.398 outside on the ViT-S row". Re-measured: **0.421 inside** (matches) but **1.511 outside**, not 1.398. The inside figure and the argument both survive; the outside number needs updating.

### 1.4 ✅ Ridge selection holds, but the quoted improvements do not

ZJU-L5 **train** split, n=483, all pixels:

| ridge | r31 RMSE | r31 MAE | r31 AbsRel | r31 δ1 | r31 bias | paper RMSE | paper AbsRel | paper δ1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.922 | 0.238 | 0.102 | 0.887 | −0.134 | 1.012 | 0.105 | 0.886 |
| **0.003** | 0.902 | 0.229 | 0.098 | 0.894 | −0.107 | 0.962 | 0.097 | 0.890 |
| 0.01 | **0.901** | **0.226** | 0.098 | **0.903** | −0.065 | 1.085 | 0.103 | 0.896 |
| 0.03 | 0.934 | 0.251 | 0.113 | 0.892 | +0.018 | 1.302 | 0.139 | 0.858 |
| ∞ | 2.369 | 0.903 | 0.438 | 0.606 | +0.760 | 3.006 | 0.529 | 0.569 |

δ1 and bias reproduce closely (0.887 vs 0.886; −0.134 vs −0.135). **RMSE and MAE are systematically lower** in the re-run, and at ridge 0.01 the paper shows a clear degradation (RMSE 1.085) that the re-run does not (0.901, the best row).

**The selection of 0.003 still holds — but on the near-field criterion, not the one the sentence leads with.** Inside the footprint, medAE is 0.0337 at ridge 0.003 against 0.0345 at 0 and **0.0370 at 0.01**. So 0.003 is the largest ridge with *no near-field cost*, which is exactly the paper's stated rule. The magnitudes in "AbsRel improves by 8% and RMSE by 5%" do not reproduce — re-measured they are **3.9% and 2.2%**.

⚠️ **Author question:** Table IV does not state its backbone. This re-run used the DAv2 ViT-S teacher. If Table IV was produced with a different backbone that would explain the RMSE offset, and the table should say which.

### 1.5 ✅ Table II's DEPTHOR numbers reproduce — the denominator is sound

`sudo jetson_clocks` was applied before these ran; `jetson_clocks_applied: true` is recorded in both files. Procedure: **100 warm-up + 500 timed iterations**, CUDA-synchronised each side, batch 1 at 480×640, dataloading excluded, GPU otherwise idle.

| Model | r31 median | p90 | sd | Hz | params | Paper Table II |
|---|---|---|---|---|---|---|
| DEPTHOR-Small | **80.0 ms** | 80.4 | 0.23 | 12.5 | 30.2M | 79.4 ms / 12.6 Hz |
| DEPTHOR-Large | **186.2 ms** | 188.6 | 2.20 | 5.4 | 36.9M | 183.8 ms / 5.4 Hz |

✅ **Correction to an earlier note in this log: the 79.4 ms *is* traceable and reproducible** — it comes from `time_net.py` in the DEPTHOR checkout and reproduces to 0.6 ms (0.8%). The 128 ms in `depthor_small_zjul5.json` is the dataloading-inclusive figure and is correctly caveated there. **The 3.1× denominator is sound.**

The remaining Table II problem is unchanged and is entirely in the **numerator**: the "Ours, deployed configuration" row (25.8 ms) excludes the uncertainty terms the deployed system runs. Fixing that is Phase 2 task 2.4.

⚠️ **Precision asymmetry, which R4 asks to be identified.** DEPTHOR runs `torch.float32`; our engines are FP16 TensorRT. "3.1× faster" is therefore partly a precision comparison and the table note must say so. The sd of 0.23 ms also shows DVFS was already reaching max clocks under sustained load, so the clock state matters less here than feared — but it is now recorded either way.

### Known artefacts reproduced (both already documented in the code)

- `B2_bilinear` coverage **0.00** on `center` — cannot extrapolate outside its convex hull. Reported as collapsed coverage rather than papered over.
- `B4_affine` MAE **11.157 m** on `center` against `B4c_affine_cl` 0.314 m — the far-field clamp-policy artefact. The gap is the size of the artefact, which is why the paper should quote B4c.

---

## 5. Standing findings

### The tape reference cannot be the accuracy metric

Bootstrapped, the 15-point median is 0.051 m with a 95% interval of [0.009, 0.531] — a 30× range. It cannot resolve a 1 cm difference, rank two methods, or support an ablation. Against that the withheld-zone protocol scores ~10⁵ points and lands at 0.014 m.

Three caps, only one fixable by adding points: pointing error is first-order (§0.1), it is one scene in one session, and it is point-wise so it cannot produce frame-level RMSE/δ1.

**The fix is dense independent ground truth, not more markers:** floor-plane residuals with the plane anchored to tape (floor geometry already validated to **2.3 mm** on this rig — `fov_v_session_2026-08-04.json`: implied lens height 15.97 cm vs 16.2 cm tape), plus a flat board at tape-measured ranges inside, at the edge of, and outside the cone. That yields ~10⁵ px/frame of dToF-independent ground truth covering the periphery, in the same metric form as Tables III and VI. **Condition: the plane must come from tape, not from dToF anchors, or the circularity returns.**

### The withheld-zone protocol: what it can and cannot say

It works and it is the backbone of Table III. Four limits:

1. **Every target is a dToF zone, and zones exist only inside the cone.** No cell in Table III is out-of-coverage — not by band choice, but because no ground truth exists out there.
2. **Targets are zone centres**, where nearest-neighbour looks best. Errors across a discontinuity between zones cannot appear. *Fixable offline* — task 1.2.
3. **Common-mode error cancels** — dToF bias is in both anchors and targets.
4. **Noise floor.** `calibration.yaml` records the dToF at MAE 0.010 m vs tape over 7 markers. The random protocol reports 0.014 m — within 4 mm of the reference's own accuracy. That column is partly measuring the sensor.

### Numbers ready to use — provenance table

Every figure below is measured on build `aba5edd` with `student_v4_heldout_fp16` + `residual_v7_fov73_fp16`. Quote the source file alongside, so a reviewer question has an answer.

| Paper location | Number | Source |
|---|---|---|
| IV-C weighting, `w ∝ z` | 0.0716 m [0.0682, 0.0752] random | `baselines_r31.json` → `random.W1_rangep1` |
| IV-C weighting, `w ∝ z²` | 0.1013 m [0.0960, 0.1069] random | `random.W2_rangep2` |
| IV-C weighting, ROI gate (deployed) | 0.0502 m [0.0483, 0.0520] random | `random.W3_roi` |
| IV-C robust pass rejection rate | **22.8% downweighted / 6.8% of weight mass** | `*.robust_pass` |
| IV-F `c_a`, `c_ν`, `c_α`, `α_0` | 1.00, 1.00, 0.1225, 5.0° | `blend.py` HEAD |
| IV-F `α_max` | ❌ not an angular threshold — `ROI_OUTSIDE_SIGMA_FRAC = 1.0` on the ROI mask | `pipeline.py:31` |
| V preamble bootstrap count | **B = 1000**, resampled over **frames** | `baselines_r31.json` → `*.ci.*.B` |
| V-B analytic reference | 0.055 random / 0.054 center | `random.W0_uniform`, `center.W0_uniform` |
| Table V, all 8 rows + angular bands | see §1.1, §1.2 | `baselines_r31.json` |
| Table V note, worst-case interval | **±0.0056 m** if the table is random+center only; **±0.0154 m** if `edge` rows are included (widest: `edge`/`W3_roi`). Per protocol: random ±0.0056 (`W2_rangep2`), center ±0.0037 (`B1_nearest`), edge ±0.0154 | `baselines_r31.json` |
| V-C 15-point median | 0.051 m [0.009, 0.531] | `tape_stats_r31.json` → `accuracy.all_15` |
| V-C in-cone median | 0.029 m [0.012, 0.101], n=8 | `accuracy.in_cone` |
| V-C out-of-cone median | 0.528 m [0.265, 2.071], n=4 | `accuracy.out_of_cone` |
| V-C calibration vs held-out | **8 re-measured, 2 ambiguous, 5 strictly held out** | `provenance.rows` |
| Table VII coverage @1σ, n=11 | 0.909, **95% [0.623, 0.984]** | `coverage.n11_after_sigma_terms` |
| Table VII coverage @1σ, n=15 | 0.800, **95% [0.548, 0.930]** | `coverage.n15_all` |
| Table VII coverage @2σ, n=15 | 1.000, **95% [0.796, 1.000]** | `coverage.n15_all` |
| V-E out-of-cone error/σ pair | 1.07 & 1.52 m vs σ 2.10 & 1.84 — **calibration set** | `live4_sigma_2026-08-04.json` pts 7, 8 |
| V-E held-out replacement | 4 out-of-cone held-out points, errors −0.27/−2.07/+0.53/−0.53 | `out_of_cone_claim.heldout_replacement` |
| V-A deployed rate | 9.38 Hz | `rate_live_v7_lights_2026-08-04.json` — **re-measure in Phase 2** |
| V-A map age | 143.6 ms | same file — **re-measure in Phase 2** |
| V-E sigma term cost | 10.643 ms median (bypass A/B) | `sigma_cost_corrected_2026-08-04.json` |
| Table III note, TensorRT / L4T / CUDA / cuDNN | 10.3.0.30 / R36.5.0 / 12.6.68 / 9.3.0.75 | §2 |
| Table III note, `jetson_clocks` | ❌ **not applied** at time of writing | §2 |

**New, not previously in the paper** (all from `baselines_r31.json`, `edge` protocol):

| Claim | Number |
|---|---|
| Discontinuity-zone error, nearest-zone dToF | 0.228 m [0.223, 0.232] |
| Discontinuity-zone error, analytic + robust | 0.437 m [0.425, 0.449] |
| Discontinuity-zone error, + refiner | 0.350 m [0.340, 0.359] |
| Discontinuity-zone error, + arbitration | **0.133 m** [0.124, 0.141] |
| Frames containing a real discontinuity | 1146 of 1234 (88 skipped, reported) |

### Numbers at risk

| Claim | Where | Risk |
|---|---|---|
| **3.1× faster than DEPTHOR-Small** | abstract, V-A, conclusion | Divides by Table II's 79.4 ms, which has no traceable source; our own file says 128 ms/frame w/ dataloading. Numerator 25.8 ms excludes the sigma terms. |
| **4.0× with optional stages off** | V-A | Same denominator. |
| **"roughly twice as fast" offline vs deployed** | V-A | Offline is 70.7–80.7 ms (12.4–14.1 Hz) vs deployed 9.38 Hz ⇒ **1.3–1.5×**. The 2× matches 14.1/7.16 — the pre-queue-fix rate. |
| **Table VI "Ours, as deployed"** 0.185/1.174/0.716 | Table VI | Matches `zjul5_student.json` **`B4c_affine_cl`** — analytic stage, **refiner off**. Full pipeline `B5_ringfusion` is 0.613/3.051/0.292. Caption says "Complete 4.1M Pipeline"; table body says "distilled 3.66M". |
| **9.4 Hz / 128 ms** | abstract, V-A, Table III, conclusion | Two different builds. 9.4 Hz = `rate_live_v7_lights` (current). 128 ms = `rate_live_q1` (Jul 30, pre-v7, pre-sigma). Current-build age is **143.6 ms**. |
| **Coverage 0.909 → 0.760 "better calibrated"** | V-E, Table VII | Not supported — see §0.1. |
| **MAE 0.264→0.247, frame-to-frame 0.0106→0.0100** | V-B | No interval. Needs paired bootstrap (4.1). |

### Provenance of the rate/age numbers

| File | Date | Rate | Age | Build |
|---|---|---|---|---|
| `rate_live_on_verified.json` | Jul 30 16:46 | 7.16 Hz | 426.5 ms | **pre-queue-fix** — not the current build |
| `rate_live_q1.json` | Jul 30 17:57 | 10.37 Hz | **128.5 ms** | queue-depth-1, pre-v7, pre-sigma |
| `rate_live_v7.json` | Aug 4 03:36 | 10.29 Hz | 137.3 ms | v7, pre-sigma |
| **`rate_live_v7_lights_2026-08-04.json`** | Aug 4 22:40 | **9.38 Hz** | **143.6 ms** | **v7 + sigma — current** |

`sigma_cost_corrected_2026-08-04.json` closes the loop: predicted 10.29 → 9.27 Hz when the sigma terms went in; 9.38 measured. It also answers the paper's 10.6 ms question — that is the Stage 7c sigma block measured by **bypass** (10.643 ms median), not `6_variance` (6.40 ms).

The brief's premise that the honest deployed rate is "nearer 7 Hz" is **wrong** — 7.16 Hz predates the queue fix by 70 minutes and the v7 build by five days.

---

## 6. Phase 2 pre-flight — NOT STARTED, awaiting go-ahead

⛔ **No robot session has been run. Phases 0 and 1 touched no hardware** — they used the 1234 logged pairs, ZJU-L5, and the saved tape captures only.

Before the bench session, in order:

1. `sudo jetson_clocks` — **the one that resets every boot.** `nvpmodel` is already MAXN and persists.
2. Verify with `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` and `cat /sys/class/devfreq/17000000.gpu/{min_freq,max_freq}` — locked means min == max on both.
3. Stop everything else on the Jetson. The `rate_live_on_verified` capture carries the annotation "leroi stopped" and differs from its neighbours because of it.
4. Confirm the launch passes `student_v4_heldout_fp16.engine` + `residual_v7_fov73_fp16.engine` explicitly — the launch file has no defaults.
5. Confirm `blend=true`, `roi_enable=true` against the running node's parameters, not the launch command. `rate_live_on_final.json` was mislabelled at capture for exactly this reason.
6. **Let the board run 5 minutes under load before the timed capture**, so the number is thermally settled rather than cold-boot.
7. Record the git commit.

Expected results, so a bad run is recognisable at the time:

| Quantity | Expect | If you see |
|---|---|---|
| publish rate | **9.3–9.5 Hz** | ~7 Hz → misconfigured launch, **stop and check params** |
| map age | **~140 ms** | ~400 ms → queue depth not 1 |
| `frame_total_ms` | ~90–120 ms | — |

Also required in Phase 2 and not measured anywhere yet: **the data-age decomposition** (exposure → USB transport → queue wait → pipeline → publish). `profile_node` covers the pipeline, `rate_live` covers end-to-end age, and nothing measures the gap. R4 asks for that relationship explicitly.

---

## 4b. Phase 2 — bench session ✅ COMPLETE (2026-09-12/13)

Session conditions, all recorded in each JSON's embedded `env` block: `jetson_clocks` applied, MAXN, **leroi stack stopped** (`systemctl stop leroi-robot.service`), engines `student_v4_heldout_fp16` + `residual_v7_fov73_fp16` verified by SHA, node parameters read back from the **running node**: `blend=True`, `roi_enable=True`, `min_confidence=-1`, `plane_refit_every=1`.

### 2.2 Publish rate and map age — two 900 s runs

| | Run 1 | Run 2 | Paper |
|---|---|---|---|
| Rate | **9.40 Hz** | **9.49 Hz** | 9.4 Hz ✅ |
| p05–p95 rate | not stored | **8.52 – 10.83 Hz** | fills the V-A pending |
| Age median | **146.9 ms** | 135.7 ms | 128 ms ❌ |
| Age p90 / max | 172.6 / 230.3 ms | 160.9 / **572.6** ms | — |
| Thermals | 52.2 → 55.1 °C | — | no throttling, clocks stayed locked |

✅ **The 9.4 Hz claim reproduces.** ❌ **The 128 ms map age does not** — it came from `rate_live_q1` (Jul 30, pre-v7, pre-sigma). Four independent measurements now agree on ~135–147 ms.

Run-to-run: rate stable to ~1%, age varies ~8%. Run 2 caught a single 554 ms gap that run 1 did not — keep it, since that tail is what a controller feels.

### 2.3 Data-age budget (R4's core-latency / rate / age relationship) — NEW

| Component | median | p05 | p95 |
|---|---|---|---|
| sensor → arrival | **1.4 ms** | 1.1 | 12.9 |
| node latency | **143.6 ms** | 103.7 | 175.8 |
| total age | 145.5 ms | 115.5 | 177.1 |
| **image staleness** | **10.9 ms** | 2.6 | 45.6 |

1. **ToF transport is essentially free (1.4 ms)** — the age is a compute-and-queue story, not a sensor one.
2. **A map is 1.38 publish periods old when it lands** (145.5 ms against a 105.7 ms period).
3. ⚠️ **Image staleness 10.9 ms (p95 45.6 ms)**: `on_tof` consumes whatever `on_image` last cached, so the camera half of the fusion is older than the published stamp claims. Not mentioned anywhere in the paper.

⚠️ `age_budget`'s own rate reading (8.46 Hz) is **not** quotable — it subscribes to three topics including 30 Hz `/image` and drops some `/depth`. Quote `rate_live`'s 9.40 Hz. The budget components are per-matched-pair and unaffected.

### 2.1 Per-stage frame budget, 500 frames

| Stage | median ms | % frame |
|---|---|---|
| `1_rectify` | 4.10 | 4.2 |
| `2_backbone` | 10.17 | 10.5 |
| `3_4_project_pair` | 0.88 | 0.9 |
| `4b_roi_plane` | 0.10 | 0.1 |
| `5_fit_metric` | 3.64 | 3.8 |
| `6_variance` | 6.07 | 6.2 |
| `7_residual` | 27.59 | 28.4 |
| `7b_clamp` | 1.05 | 1.1 |
| `7c_blend` | 24.50 | 25.2 |
| `7d_roi_sigma` | 6.11 | 6.3 |
| `8_cloud` | 2.62 | 2.7 |
| `9_publish` | 8.09 | 8.3 |
| **`pipeline.run` total** | **84.72** | 87.2 |
| **frame total** | **97.17** | 100 |

**Mapped onto Table III's six published rows:**

| Paper row | ms | Stages summed |
|---|---|---|
| camera branch: capture, rectify, backbone | **14.27** | `1_rectify` + `2_backbone` |
| dToF branch: projection and validity | **0.98** | `3_4_project_pair` + `4b_roi_plane` |
| fit, covariance and refiner | **31.24** | `5_fit_metric` + `7_residual` |
| source arbitration | **25.54** | `7c_blend` + `7b_clamp` |
| uncertainty terms | **12.18** | `6_variance` + `7d_roi_sigma` |
| unprojection, assembly and publication | **10.70** | `8_cloud` + `9_publish` |
| *sum of rows* | *94.92* | vs frame total 97.17 — **2.25 ms is untimed glue**, state it |

⚠️ **The "uncertainty terms cost 10.6 ms" figure is a different decomposition.** 10.6 ms is `sigma_cost_corrected`'s bypass A/B of the Stage 7c σ block, which lives *inside* `7c_blend`. The 12.18 ms row above is `6_variance` (analytic delta-method) + `7d_roi_sigma` (the ROI floor). They are not the same quantity and the paper must not present one as the other.

### 2.4 Table II deployed row — ⚠️ 3.1× becomes 2.8×

`time_pipeline`, offline, GPU otherwise idle:

| Config @ 480×640 | r31 | Paper |
|---|---|---|
| stages off | **20.2 ms** | 20.0 ms ✅ |
| blend only | 25.7 ms | — |
| **deployed (blend + ROI + σ)** | **28.3 ms** | 25.8 ms ❌ |

At 1640×1232: 50.2 / 73.4 / **79.3 ms** (12.6 Hz) for the same three configs.

The paper's 25.8 ms matches the **pre-σ** `timing_after_gpufixes` figure exactly. Recomputing against the re-measured DEPTHOR-Small (80.0 ms):

| Claim | Paper | r31 |
|---|---|---|
| vs DEPTHOR-Small, deployed | 3.1× | **2.83×** ❌ |
| vs DEPTHOR-Small, stages off | 4.0× | **3.96×** ✅ |
| offline vs deployed core | "roughly twice" | **1.33×** ❌ (79.3 ms = 12.6 Hz vs 9.45 Hz) |

### Age attribution, closed

`node_latency` 143.6 ms − replica frame total 97.2 ms = **46.5 ms of queue and transport** the pipeline itself does not account for. Deployed period is 105.7 ms, so the node is working on frame *N* while *N+1* waits — which is what makes the age 1.38 periods rather than 1.0.

⚠️ One caveat on the replica: `profile_node` implies 10.29 Hz against the deployed 9.40 Hz, because it publishes to `/prof_*` with no downstream subscribers and skips serialisation the real node pays on three 8 MB topics. Use it for the **stage split**, not for the rate.

---

## 6b. Implementation needed for Phases 3–5 — ANALYSIS ONLY, nothing built

### Phase 2 tooling ✅ built (no data collected)

| Tool | Status |
|---|---|
| `tools/diagnostics/envinfo.py` | **new** — captures L4T/TensorRT/CUDA/cuDNN, clock lock state, thermals, engine SHAs, git commit. `warn_if_unlocked()` shouts if `jetson_clocks` is missing |
| `tools/diagnostics/age_budget.py` | **new** — the data-age decomposition R4 asks for (2.3) |
| `rate_live.py`, `profile_node.py` | patched to embed `envinfo.capture()` in their JSON — a capture can no longer drift from its label |
| 2.4 Table II row with sigma terms | ✅ **no code needed** — `sigma_support_var` is called inside `pipeline.run` at stage 7c, so `time_pipeline.py` re-run today already includes it. `timing_after_gpufixes.json` is stale, not wrong |

### Phase 3 — three gaps, one substantial

| Item | Exists | Needed |
|---|---|---|
| 3.1/3.2 **plane-based dense validation** | ❌ nothing | **NEW TOOL, the biggest build in the revision.** Score dense published depth against a *tape-anchored* plane, split in/out of footprint. `tape_eval.py` is point-wise only; `roi.fit_ground_plane` fits from dToF anchors and is therefore circular for validation. This is what turns R1's out-of-coverage accuracy from 4 points into ~10⁵ px |
| 3.3 pre-registered split | ❌ no `role` field | Small — write `role: calibration\|heldout` into `tape_gt.json` at capture time. `instrument` and `origin_offset_m` are already recorded ✅ |
| 3.3 repeats aggregation | ❌ | Medium — `tape_eval.py` scores one directory; the n=15 mean/lo/hi was assembled by hand. Can reuse `bootstrap.py` |
| 3.3 in/out split, σ from `/depth_var`, <20-point warning | ✅ | none |
| 3.4 **per-term uncertainty ablation** | ❌ | See below — a design problem, not just a script |

#### 3.4 resolved ✅ — offline replay works, no five live runs needed

**Checked `tape_capture.py`. It already saves the raw inputs**, per frozen frame (deduplicated across points that share a frame — otherwise ~16 MB of identical float32 per point):

| File | Contents | Replay-ready? |
|---|---|---|
| `<stem>_rgb.png` | camera frame, BGR-converted before `imwrite` so `cv2.imread` round-trips | ✅ raw fisheye — rectify offline |
| `<stem>_tof.npy` | `dist_m` (rows, cols) float32 | ✅ `valid` derives as `np.isfinite` |
| `<stem>_depth.npy`, `<stem>_var.npy` | the published depth and variance | ✅ the replay target to validate against |

**`confidence` is discarded by `on_tof` — and it does not matter.** `perception_node` declares `min_confidence` with a default of **−1**, and `pipeline.run` only builds `conf_flat` when `min_confidence >= 0`. At the default the branch never runs and `weights = np.ones_like(inv_depth)`.

✅ **This also confirms the Table V reference row.** The deployed anchor weighting is **uniform × the geometric ROI gate** — there is no ToF-confidence weighting in the default configuration. P0-2's original reading was correct.

**Three conditions for the replay to be trustworthy:**

1. ⚠️ **Pin `min_confidence = -1`** (the default) for the session and record it. If anyone sets it ≥ 0, the deployed fit gains confidence weighting that the replay cannot reproduce, and it would fail silently. `envinfo.capture()` does not currently read node parameters — worth adding before the session.
2. ⚠️ `rgb`, `tof`, `depth` and `var` arrive on **four independent subscriptions** and are frozen together, so the saved `rgb` may be a slightly newer frame than the one that produced the saved `depth`. Harmless on a static scene — which this protocol requires anyway — but replay will not be bit-identical.
3. ⚠️ `PlaneTracker` is **stateful** (EMA `alpha=0.2` plus jump rejection) even at the default `refit_every=1`, so a replay starting fresh converges to the plane rather than matching it exactly.

Because of (2) and (3), the replay tool must carry a **validation step**: re-run with the deployed constants and check it reproduces `_var.npy` closely before trusting any ablated arm. Cheap, and it converts two unquantified caveats into a measured agreement.

**Remaining 3.4 work is then small:** parameterise `sigma_support_var` to accept `DISAGREE_K`/`SUPPORT_FRAC`/`SPREAD_K` (currently module globals; `near_deg`/`far_deg` are already parameters), and write the replay-and-score tool.

### Phase 4 — ⚠️ the brief names the wrong tool

The brief's Task D says `moving_ab.py --engine-v1 <arbitration off> --engine-v2 <arbitration on>`. **`moving_ab.py` compares two residual engines against an analytic baseline — it has no arbitration mode.** The arbitration A/B is `blend_ab_live.py`, which is where `blend_ab_live_v7_2026-08-04.json` (MAE 0.264 → 0.247) came from.

| Item | Status |
|---|---|
| Paired design | ✅ **already ideal** — both arms computed from ONE backbone+residual pass on the same frame, so every difference is the blend. Its docstring argues explicitly against two drives, and the paper already uses this design |
| Per-frame structure for bootstrapping | ✅ preserved in memory (`rows[name]['p'].append(...)` per frame), flattened only for the metrics — so `bootstrap.paired_diff` is a small addition |
| Per-run variation (R2) | ❌ needs two drives reported separately, plus a small aggregator |
| 4.2 platform top speed | ❌ no odometry or IMU anywhere in the stack — manual measurement (tape a distance, time it). No code |

Correction to the earlier Phase 4 plan in this log: because `blend_ab_live.py` is inherently paired, **taping the route is not critical for the A/B itself**. Two drives are needed only for the between-run variation the reviewer asked about.

### Phase 5 — one potentially very large change

| Decision | Code impact |
|---|---|
| Fallback: rewrite the text to match the code | none |
| Fallback: **implement the guards** | ⚠️ **Largest change in the revision.** `N_min`/`v_min`/`κ_max` + holding `b̂` in `anchoring.py`/`pipeline.py`, plus a health signal in the output — and `/depth`, `/depth_var` are plain `sensor_msgs/Image`, so exposing a flag needs a new message type or a separate diagnostic topic. **It would invalidate every number measured so far and require re-running Phases 1–4.** Recommendation: rewrite the text, defer the feature to future work |
| Real-time definition | none — needs the planner rate and 4.2's top speed |
| Table VI restructure | none — the data is already in hand |
| Artifact release | packaging only: splits, calibration, runtime settings |
| 1234 vs 1228 pairs | ✅ resolved, no change |

---

## 6c. Phase 3/4 tooling ✅ BUILT AND VALIDATED (offline, 2026-09-13)

No robot involved. Every tool was validated against either existing captured data or a synthetic case with a known answer.

### Deployed-code changes — all behaviour-preserving

| File | Change | Why safe |
|---|---|---|
| `anchoring.solve_robust` | optional `info` dict out-param | populated only when passed; no behaviour change |
| `blend.sigma_support_var` | `disagree_k` / `support_frac` / `spread_k` as arguments | default to the module constants |
| `pipeline.run` | `sigma_terms=None`, `learned_var=True` | defaults reproduce the deployed path exactly |

**Regression test on a real logged frame (821 anchors):** default call vs explicit-`None` vs explicit deployed constants all produce **bit-identical** depth and variance. Zeroing a σ term changes the **variance only** — depth stays bit-identical, which is the property that makes the ablation valid. Setting `residual=None` would have changed both and made the comparison meaningless.

### New tools

| Tool | Purpose | Validation |
|---|---|---|
| `sigma_ablation.py` | per-term uncertainty ablation (R1's fifth component), replayed offline | its scoring reproduces the recorded n=15 result exactly — `rank_corr 0.6250` against the n14 file's stored 0.6250; its Spearman matches scipy to 1e-9 including ties |
| `plane_eval.py` | dense accuracy against a **tape-anchored** plane, split in/out of cone | synthetic: exact plane → medAE **0.0000**; 5% scale error → AbsRel **0.0500** exactly; a 12 cm corrupted corner → planarity 28 mm, **rejected** |
| `tape_repeats.py` | capture N repeats of a static scene and aggregate the spread | synthetic 5×15: coverage 1.000 every repeat as designed, rank corr swings 0.207–0.489 — the repeat-noise effect the ±0.13 figure describes |
| `envinfo.py`, `age_budget.py`, `marker_view.py` | see §4b, §6b | in service during Phase 2 |

### Modified tools

- `blend_ab_live.py` — **paired bootstrap** on the arbitration difference. Validated on synthetic data: resolves a real 6% effect (p=0.0000) and correctly declines a null one (p=0.394). Both arms score the same frames, so the frame draw is shared; bootstrapping the arms separately would discard the pairing and call a real effect insignificant.
- `tape_capture.py` — records `cone` (in/edge/out from the calibrated FOV) and `role` per point, prints a running composition tally. Cone classification agrees with the hand-labelled n=15 set on **14/15**; the one disagreement sits 1.0° inside the band boundary, a threshold convention rather than a geometry error. `tape_eval` still recomputes it independently for scoring.
- `time_pipeline.py` — embeds `envinfo`. Re-run reproduced to within 0.3 ms.

### How `plane_eval` stays non-circular

Two rules, and the result is worthless without either:

1. **The plane comes from tape, never from a dToF fit.** `roi.fit_ground_plane` would be easier but fits through the same measurements the pipeline anchors to — scoring against it measures self-consistency, the exact circularity this workstream exists to escape. Here the plane is fitted to backprojected tape ranges and the dToF is never consulted.
2. **The region is declared, not inferred.** Deciding which pixels are floor by asking whether their predicted depth looks floor-like grades the prediction against itself. The operator measures the corners; only pixels inside that hull are scored, eroded by 6 px so marker tape does not contribute.

**Capture protocol:** in `tape_capture`, click and measure ≥3 points on each planar target and label them all `plane:<name>` (e.g. `plane:floor_left`, `plane:board_2m`). Four or more gives a planarity residual, which is the only check that the surface was flat and the ranges right. Pooled intervals resample over **plane regions**, not pixels — pixels on one plane are not independent observations.

---

## 6d. Consultant cross-check + item 2 closed ✅ (offline, 2026-09-13)

### The ten items against our list

| # | Item | Status |
|---|---|---|
| 3 | JetPack/TensorRT + `jetson_clocks` beside the timing | ✅ already embedded in every r31 JSON |
| 4 | Deployed row split by footprint, six cells | ✅ 0.2701/1.4984/0.6179 in, 0.7200/3.3838/0.1081 out |
| 5 | AbsRel + δ1 outside the footprint, both baselines | ✅ ViT-S 0.1417/0.8212, DEPTHOR-Small 0.1186/0.8566 |
| 6 | Footprint share on ZJU-L5 | ✅ **54.43 %** (85,375,741 / 156,851,220), vs 12.4 % on our own sensor |
| 7 | Rank correlation + worst residual, held-out column | ✅ §6d.2 |
| 8 | Five points or seven | ✅ **seven** — §6d.3 |
| 9 | How the σ constants were chosen | ✅ the paper's sentence is false — §6d.4 |
| 10 | Planner rate + platform top speed | ◐ planner **3.0 Hz**; top speed still needs the robot |
| 1 | Per-session driving MAE, paired bootstrap | ⛔ Phase 4, tool built, needs two drives |
| 2 | Scattered hold-out refiner row, one split | ✅ **closed below** — this was a genuine gap in our list |

Item 2 was in the original brief (Task C, "one baseline, one split") and had been dropped
when the brief was turned into phases. It is the only item the consultant's list added.

### 6d.1 Item 2 — Table V now comes from one run on one split

**The defect.** The paper reports the analytic output as **0.055 m** in Table V and as
**0.064 m** in the scattered-hold-out paragraph. Those came from different runs on
different splits, so no row of that table could be compared with any other row.

**Code change.** `baselines.py` now takes `--residual-engine NAME=PATH` repeatably and
holds every engine resident at once, so all refiner rows are scored inside the same frame
loop, against the same per-frame anchor/hold-out split, off one shared backbone pass.
`--deployed-engine` keeps the shipped engine on the canonical `B5_ringfusion`/`B6_blend`
row names so numbers cited elsewhere stay findable. Two further additions:

- **`B6_analytic`** — arbitration with *no refiner in the path*. This is the topology
  Table VI actually printed while labelling it "as deployed"; it is now its own row and
  can no longer be confused with the deployed one.
- **`--paired A:B`** — two rows scored on the same frames are paired data, and overlapping
  marginal CIs are the wrong test for them. Each pair is re-tested with one shared frame
  draw applied to both arms (`bootstrap.paired_diff`).

The run also now embeds `envinfo.capture()`, every engine SHA, and the full argument set.
The previous `baselines_r31.json` recorded **none** of that — a provenance hole in a file
headed for the paper.

**The split.** `B5` is a trained net and the 1234 logged pairs are its training set, so the
refiner rows are scored on the 61-frame `random_split(..., manual_seed(0))` validation
split (`docs/demo/benchmarks/val_stems.txt`) that no version trained on.
→ `docs/demo/benchmarks/baselines_valsplit_r31.json`.

**medAE (m), 61 val frames, one run, 95 % frame-level bootstrap CI:**

| row | `center` (extrapolation) | `random` (interpolation) | `edge` (discontinuity) |
|---|---|---|---|
| B1 nearest zone | 0.1205 [0.1026, 0.1387] | 0.0177 [0.0147, 0.0206] | 0.2257 [0.2058, 0.2497] |
| B4c analytic **(reference)** | **0.0519 [0.0426, 0.0608]** | 0.0512 [0.0414, 0.0600] | 0.4611 [0.4122, 0.5080] |
| + refiner, **scattered** (v3) | 0.0533 [0.0453, 0.0624] | 0.0305 [0.0278, 0.0336] | 0.2988 [0.2585, 0.3299] |
| + refiner, **island** (v4) | 0.0480 [0.0435, 0.0529] | 0.0959 [0.0874, 0.1080] | 0.4276 [0.3876, 0.4775] |
| + refiner, **deployed** (v7) | 0.0320 [0.0281, 0.0362] | 0.0451 [0.0389, 0.0520] | 0.3345 [0.3053, 0.3807] |
| + arbitration over analytic | 0.0491 [0.0406, 0.0566] | 0.0155 [0.0127, 0.0182] | 0.1493 [0.1148, 0.1740] |
| + arbitration, **as deployed** | **0.0317 [0.0280, 0.0354]** | 0.0142 [0.0117, 0.0168] | 0.1384 [0.0997, 0.1684] |

**The reference row is 0.0519 m [0.0426, 0.0608], not 0.055 and not 0.064.** Both published
figures sit inside that interval, so neither was wrong — they were just never the same
measurement. The surrounding paragraph has to be rewritten around this number.

### 6d.2 ⚠️ The supervision-geometry claim survives — but in MAE, not the median

Paired differences, shared frame draw, B = 1000. **The two statistics disagree, and that
disagreement is the finding.**

`center` (extrapolation) — the deployment-relevant protocol:

| comparison | medAE | MAE |
|---|---|---|
| scattered − analytic | +0.0014, p = 0.128 | −0.0143 [−0.0193, −0.0097], p = 0.000 |
| island − analytic | −0.0040, p = 0.270 | −0.0724 [−0.0911, −0.0537], p = 0.000 |
| **island − scattered** | −0.0053 [−0.0118, +0.0010], **p = 0.116** | **−0.0581 [−0.0747, −0.0412], p = 0.000** |
| deployed v7 − analytic | −0.0199 [−0.0267, −0.0136], p = 0.000 | −0.1214 [−0.1399, −0.1021], p = 0.000 |
| deployed − analytic-arbitration | −0.0174 [−0.0231, −0.0117], p = 0.000 | −0.1102 [−0.1270, −0.0930], p = 0.000 |

`random` (interpolation) and `edge` (discontinuity):

| comparison | `random` medAE | `random` MAE | `edge` medAE | `edge` MAE |
|---|---|---|---|---|
| island − scattered | +0.0655, p = 0.000 | +0.0891, p = 0.000 | +0.1288, p = 0.000 | +0.0786, p = 0.000 |
| deployed v7 − analytic | −0.0060, p = 0.014 | −0.0779, p = 0.000 | −0.1266, p = 0.000 | −0.1421, p = 0.000 |

**What this means.** Island supervision cuts extrapolation MAE by **0.058 m, a 22 % reduction
(0.2676 → 0.2094), p = 0.000** — while leaving the median untouched (p = 0.116). It does not
make the typical pixel better; it stops the far-field pixels from blowing up. That is exactly
what supervising beyond the anchor island should do, and MAE is the statistic that sees it.

So the paper's claim is **directionally right and materially understated in kind**: it was
argued on the median, where it does not hold, when the evidence for it is in the tail, where
it is overwhelming. The table should carry **both statistics**, as the README's own
`medAE 0.066 / MAE 18.092` warning already argues.

**The cost is real and symmetric.** On interpolation island is worse on both statistics
(medAE +0.0655, MAE +0.0891, both p = 0.000), and on discontinuities scattered wins on both.
The honest framing is a **trade**, not an upgrade: island buys far-field tail behaviour and
sells near-field accuracy and edge behaviour. Stage 7c's blend exists to buy the near field
back, and §6d.1 shows it does — `B6_blend` beats every single-source row on all three
protocols.

**What is not separable.** The shipped v7 beats the analytic baseline outright on both
statistics and all three protocols, so the refiner earns its place. But v7 differs from v4 by
the FOV fix and a 5× epoch budget as well as supervision geometry, so its margin cannot be
attributed to supervision alone.

**Caveat, stated because it cuts both ways.** v3 and v4 are both pre-FOV-fix (Jul 28/29;
`fov_h` corrected 45° → 73.5° on Aug 3–4), so both are evaluated under a calibration neither
trained against. That is a fair A/B — one changed argument, both equally mismatched — but it
is not the regime either was trained for. Settling it properly needs a v3/v4 retrain under
the corrected calibration, which is not authorised.

**Two secondary results from the same run, both needing the 1234-frame numbers before they
are quoted** (the weighting and robust rows are untrained, so the larger split is the better
estimate):

- **The robust pass** helps the median on `center` (−0.0016, p = 0.036) but does *nothing* to
  MAE there (−0.0003, p = 0.826). Its value is concentrated on `random` and `edge`.
- **`w ∝ z` is mixed, not simply refuted.** On `center` it is worse on the median
  (p = 0.118, i.e. no difference) but *better* on MAE (−0.0135, p = 0.000); on `random` worse
  on the median (+0.0147, p = 0.000); on `edge` worse on the median (+0.0640, p = 0.006) yet
  better on MAE (−0.0696, p = 0.010). The Phase 1 statement that `w ∝ z` is refuted was based
  on a single statistic and **must be re-checked against the full-split numbers before it
  goes in the paper.**

### 6d.3 Item 8 — the held-out set is **seven**

The two ambiguous points sit 12 px and 26 px from an earlier marker but at depths differing
by **0.67 m and 0.96 m** — far beyond any marker's physical extent, so they cannot be the
same surface. They are different targets, therefore held out. The "5, at most 7" hedge was
mine and should read **seven** throughout, with that one sentence as justification. The
fresh 20-point session retires the question.

### 6d.4 Item 9 — the σ-constant provenance sentence is false

The paper says the constants were chosen "by minimizing the Gaussian negative log
likelihood of the tape residuals". **There is no NLL minimisation anywhere in the record.**
From `blend.py`'s comments and `live4_sigma_after_2026-08-04.json`:

- terms were **designed against two identified LIVE-4 failures** (a mixed-return marker at
  3.9σ, an angle asymmetry), not fitted;
- `DISAGREE_K` and `SPREAD_K` are both **1.0** — a natural unit scale, not a fitted value;
- `SUPPORT_FRAC = 0.35` is the only non-trivial constant;
- `SPREAD_WIN = 11` is fixed by **geometry** (~17.7 px/zone → 11 cells ≈ 2.5 zones); the
  first attempt at 3 cells was smaller than one zone and did nothing;
- a single global multiplier rescale **was swept and rejected** — "no single scale serves
  both ends";
- scored against 11 tape points, with the recorded caveat "overfitting risk is real".

Honest replacement: *designed against two identified failure modes, two constants at unity,
one at 0.35, window set by zone geometry, scored against eleven tape points; a global
rescale was swept and rejected.* This is a better story than NLL fitting, and it is true.

### 6d.5 Item 10 — the planner rate makes the real-time claim comfortable

`route_rate_hz = 3.0` (`src/leroi_bringup/launch/serial_messenger.launch.py`, and declared
at `route_planner_node.py:95`). Period **333 ms**. We publish at 9.4 Hz (105.7 ms) and maps
arrive **146.9 ms** old — **0.44 planner periods**. Every map the planner consumes is less
than half a cycle old, and the pipeline runs 3.1× faster than the planner replans and above
the sensor's 8.3 Hz assembly rate. Both halves of the real-time definition hold with margin
**even at the corrected 147 ms**, so the age correction does not weaken V-A.

Still needs the robot: platform top speed.

## 7. Author decisions outstanding

1. **Fallback** (§0.5) — rewrite to match the code, or implement the guards.
2. **Real-time definition** — planner rate **found: 3.0 Hz** (§6d.5); only platform top speed is still open.
3. **Table VI** — report both rows (analytic vs full pipeline) or scope the deployed row to the analytic stage. §6d.1 now supplies both as `B6_analytic` and `B6_blend` from one run.
4. **Uncertainty fitting objective** (§0.6) — *settled by §6d.4: the NLL sentence is false.* The remaining choice is wording, not substance.
5. **Artifact release scope** — R5 asks for code, calibration, splits, runtime settings.
6. **1234 vs 1228** logged pairs.
7. ⚠️ **The supervision-geometry claim** (§6d.2) — it holds, but on **MAE** (−0.058 m, −22 %, p = 0.000), not the median (p = 0.116), and it comes with a significant interpolation cost. The paper argues it on the median, where it does not hold. Recommend: **re-argue it on MAE, report both statistics, and frame it as a trade the blend then recovers.** No new runs needed. (Retraining v3/v4 under the corrected calibration would settle it outright, but needs training authorisation.)
8. ⚠️ **`w ∝ z`** (§6d.2) — the Phase 1 "refuted" finding rests on one statistic and flips sign between medAE and MAE on two of three protocols. Must be re-checked on the full split before any version of it goes in the paper.

---

## 8. Files produced

| File | Contents |
|---|---|
| `tools/diagnostics/bootstrap.py` | Frame-level resampling, Wilson intervals, paired difference test |
| `tools/diagnostics/tape_stats_r31.py` | Tape re-analysis: provenance, coverage intervals, error intervals |
| `docs/demo/benchmarks/tape_stats_r31.json` | Output of the above |
| `tools/diagnostics/baselines.py` *(modified)* | Multi-engine `NAME=PATH` rows on one split, `B6_analytic`, `--paired`, embedded env + engine SHAs |
| `docs/demo/benchmarks/baselines_valsplit_r31.json` | **Table V** — every row, 61 val frames, one run, CIs + paired tests |
| `docs/demo/benchmarks/baselines_r31.json` | Same on all 1234 frames — larger *n* for the untrained rows and the angular bands; refiner rows there are contaminated by construction |
