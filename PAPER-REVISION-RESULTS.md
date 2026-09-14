# RingFusion — measurement record for the r31 revision

**What this is.** Every number measured for the revision, organised by where it belongs in
the paper. For each claim: what the submitted paper says, what we measure now, the verdict,
and the file the number comes from. Intended to be read alongside the submitted PDF and the
reviewer comments.

**Build under test.** Commit `aba5edd` (2026-09-12). Deployed configuration
`student_v4_heldout_fp16.engine` (backbone, 3.66 M) + `residual_v7_fov73_fp16.engine`
(refiner, 0.46 M), blend and ROI gate on, FP16 TensorRT throughout. Every JSON referenced
below embeds its own environment block: L4T/TensorRT/CUDA/cuDNN versions, clock lock state,
thermals, engine SHA-256, and the git commit. A number without that block is an old number
and is labelled as such.

**Status legend**

| | Meaning |
|---|---|
| ✅ | Reproduces. The published number stands. |
| ❌ | Changed. The published number is wrong and must be replaced. |
| ⚠️ | Unsupported. The claim outruns the evidence; soften or remove. |
| 🆕 | New result, not currently in the paper. |
| ⏳ | Pending. Needs data not yet collected. |

**Working notes**, including how each tool works, the pre-flight checklists and the
phase-by-phase history, live in [`ros2_ws/ringfusion-r31-revision-log.md`](ros2_ws/ringfusion-r31-revision-log.md).
This file is the numbers; that file is the method.

---

## 1. Summary — what has to change

Ordered by how much of the paper each one touches.

| # | Change | Where it lands | Status |
|---|---|---|---|
| 1 | **"3.1× faster than DEPTHOR-Small" → 2.83×.** The deployed row's 25.8 ms excludes the uncertainty terms the deployed system runs; re-measured it is 28.3 ms. The *denominator* (79.4 ms) is sound and reproduces at 80.0 ms. | abstract, Table II, §V-A, conclusion | ❌ |
| 2 | **Map age 128 ms → 146.9 ms.** The 128 ms came from a Jul 30 capture predating both the v7 engine and the uncertainty terms. Four independent current-build measurements agree on 135–147 ms. | abstract, Table III note, §V-A, conclusion | ❌ |
| 3 | **Table VI's "as deployed" row is the analytic stage with the refiner off.** Its printed 0.185/1.174/0.716 matches `B4c_affine_cl` to three decimals, while the caption says "Complete 4.1M Pipeline". The correctly-labelled deployed row is *better*, not worse. | Table VI + caption | ❌ |
| 4 | **Coverage claims are not supported at n = 11/15.** Every Wilson interval contains its target and every pair overlaps heavily. This is exactly R3's "moderate the claims". | §V-E, Table VII | ⚠️ |
| 5 | **Table V's rows came from different runs on different splits.** The analytic output appears as 0.055 m in the table and 0.064 m in the scattered-hold-out paragraph. Now re-run as one run on one split: the reference is **0.0519 m [0.0426, 0.0608]**. | Table V, §V-B | ❌ |
| 6 | **The documented fallback does not exist in the code.** There is no scale-only fallback and no held `b̂`; the frame is dropped. `N_min`, `v_min`, `κ_max` have no counterparts. | §IV-C, Discussion limitation 4 | ❌ |
| 7 | **`α_max` is not an angular threshold.** The "100 % floor" is `ROI_OUTSIDE_SIGMA_FRAC = 1.0` keyed to the geometric ROI mask. | §IV-F | ❌ |
| 8 | **The σ constants were not chosen by NLL minimisation.** No NLL fit exists anywhere in the record. | §IV-F | ❌ |
| 9 | **§IV-C says the deployed system weights anchors `w ∝ z`. It does not** — `RANGE_WEIGHT_P = 0.0` and a geometric ROI gate ships instead. The ablation shows the deployed choice is better on the median everywhere but gives up a real MAE advantage in extrapolation. | §IV-C, §V-B | ❌ |
| 10 | **The tape experiment has seven independent points, not fifteen.** Eight of the fifteen are calibration markers re-measured. | §V-C, R2 | ❌ |
| 11 | **Table III's column header is wrong** — the bands are angle from the *nearest anchor*, not from the optical axis. The numbers are correct and the argument survives. | Table III header | ❌ |
| 12 | **The supervision-geometry claim is argued on the wrong statistic.** It holds on MAE (−22 %, p = 0.000) and not on the median (p = 0.116). | §V-B | ❌ |
| 13 | **Discontinuity behaviour** — a new protocol quantifying what the paper currently supports with one anecdote. | §V-C, Table V | 🆕 |
| 14 | **Arbitration's cost/benefit should be stated explicitly.** It consumes ~25 % of the frame budget (25.54 of 97.17 ms) and returns 2.6–3.2× at discontinuities and near anchors, but only ~7 % on open-field driving average. The paper should say where it earns its budget. | §V-A, §V-B, limitations | 🆕 |
| 15 | **Image staleness (10.9 ms, p95 45.6 ms)** — the camera half of each fusion is older than the published stamp implies. Not mentioned anywhere. | §V-A, limitations | 🆕 |

---

## 2. Environment and build (R4)

R4 asks for resolution, included stages, precision, hardware settings and measurement
procedure. All of it is below and all of it is embedded per-run in the JSONs.

| Item | Value | Source |
|---|---|---|
| L4T | **R36.5.0**, GCID 43688277, aarch64 | `/etc/nv_tegra_release` |
| JetPack | metapackage not installed — **quote L4T R36.5.0** | `dpkg -l nvidia-jetpack` empty |
| TensorRT | **10.3.0.30-1+cuda12.5** | `dpkg -l` |
| CUDA | **12.6.68** (SDK 12.6.11) | `nvcc --version` |
| cuDNN | **9.3.0.75** | `dpkg -l` |
| Power mode | **MAXN** (persists across reboots) | `nvpmodel -q` |
| `jetson_clocks` | **applied** for every r31 timing run | sysfs, `min_freq == max_freq` on CPU and GPU |
| Engine precision | **FP16**, both networks | `build_engine.py --precision fp16` |
| Deployed resolution | **1640 × 1232** on-robot; 480 × 640 for the Table II comparison | — |
| Thermals during capture | 52.2 → 55.1 °C, no throttling, clocks stayed locked | `rate_live_r31.json` |

**Two procedural points R4 asks for explicitly:**

- **`jetson_clocks` does not persist across reboots** (`nvpmodel` does). It must be re-applied
  every boot. Detection is `min_freq == max_freq` on both CPU and GPU — the governor *name*
  does not tell you. Every r31 JSON records the state, so this can never again be a
  reconstruction problem. The historical numbers (79.4 ms, 25.8 ms, the older `rate_live_*`
  captures) were taken in an unknown clock state, which is a second reason not to mix them
  with current ones in one table.
- **Precision asymmetry.** DEPTHOR runs `torch.float32`; our engines are FP16 TensorRT. The
  speed comparison is therefore partly a precision comparison and the Table II note must say
  so.

---

## 3. Results by paper location

### 3.1 Abstract

| Claim | Verdict | Replace with |
|---|---|---|
| 3.1× faster than DEPTHOR-Small | ❌ | **2.83×** — see §3.2 |
| 9.4 Hz | ✅ | reproduces: 9.40 and 9.49 Hz on two 900 s runs |
| 128 ms map age | ❌ | **146.9 ms** — see §3.10 |

### 3.2 Table II — runtime comparison

**The denominator is sound.** 100 warm-up + 500 timed iterations, CUDA-synchronised each
side, batch 1 at 480×640, dataloading excluded, GPU otherwise idle, `jetson_clocks` applied.

| Model | r31 median | p90 | sd | Hz | params | Paper |
|---|---|---|---|---|---|---|
| DEPTHOR-Small | **80.0 ms** | 80.4 | 0.23 | 12.5 | 30.2 M | 79.4 ms / 12.6 Hz ✅ |
| DEPTHOR-Large | **186.2 ms** | 188.6 | 2.20 | 5.4 | 36.9 M | 183.8 ms / 5.4 Hz ✅ |

Both reproduce to under 1.5 %. → `depthor_small_timing_r31.json`, `depthor_large_timing_r31.json`

**The numerator is not.** The paper's 25.8 ms matches the *pre-σ* `timing_after_gpufixes`
figure exactly — it excludes the uncertainty terms the deployed system actually runs.

| Config @ 480×640 | r31 | Paper |
|---|---|---|
| optional stages off | **20.2 ms** | 20.0 ms ✅ |
| blend only | 25.7 ms | — |
| **deployed (blend + ROI + σ)** | **28.3 ms** | 25.8 ms ❌ |

At the deployed 1640×1232: 50.2 / 73.4 / **79.3 ms** (12.6 Hz) for the same three configs.

| Claim | Paper | r31 |
|---|---|---|
| vs DEPTHOR-Small, deployed | 3.1× | **2.83×** ❌ |
| vs DEPTHOR-Small, stages off | 4.0× | **3.96×** ✅ |
| "roughly twice as fast" offline vs deployed | 2× | **1.33×** ❌ (79.3 ms = 12.6 Hz vs 9.45 Hz) |

→ `timing_pipeline_r31.json`

### 3.3 Table III — angular bands

**✅ Every published cell reproduces exactly on the current build.** medAE in m, `center`
protocol:

| Row | 0–3° | 3–6° | 6–10° | 10–15° | 15–30° |
|---|---|---|---|---|---|
| nearest-zone dToF — paper | 0.027 | 0.063 | 0.104 | 0.163 | 0.204 |
| nearest-zone dToF — **r31** | **0.027** | **0.063** | **0.104** | **0.163** | **0.204** |
| refiner island — paper | 0.033 | 0.028 | 0.031 | 0.032 | 0.036 |
| refiner island — **r31** | **0.033** | **0.028** | **0.031** | **0.032** | **0.036** |
| arbitration — paper | 0.026 | 0.027 | 0.031 | 0.032 | 0.036 |
| arbitration — **r31** | **0.026** | **0.027** | **0.031** | **0.032** | **0.036** |

❌ **The column header is wrong.** It reads "center, by angle from optical axis". The code
computes angular distance from each held-out zone to the **nearest surviving anchor**
(`baselines.py`, `ang = dmat.min(1)`), and the script's own output says so. This *helps* the
paper — the bands are a genuine extrapolation axis, so §V-B's crossover argument is correct
in substance. Only the header needs fixing.

→ `baselines_r31.json`

### 3.4 Table IV — ridge selection

ZJU-L5 **train** split, n = 483, all pixels:

| ridge | r31 RMSE | r31 AbsRel | r31 δ1 | r31 bias | paper RMSE | paper AbsRel | paper δ1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.922 | 0.102 | 0.887 | −0.134 | 1.012 | 0.105 | 0.886 |
| **0.003** | 0.902 | 0.098 | 0.894 | −0.107 | 0.962 | 0.097 | 0.890 |
| 0.01 | **0.901** | 0.098 | **0.903** | −0.065 | 1.085 | 0.103 | 0.896 |
| 0.03 | 0.934 | 0.113 | 0.892 | +0.018 | 1.302 | 0.139 | 0.858 |
| ∞ | 2.369 | 0.438 | 0.606 | +0.760 | 3.006 | 0.529 | 0.569 |

✅ **The selection of 0.003 holds** — but on the near-field criterion, not the one the
sentence leads with. Inside the footprint, medAE is 0.0337 at ridge 0.003 against 0.0345 at
0 and **0.0370 at 0.01**, so 0.003 is the largest ridge with no near-field cost, which is
exactly the paper's stated rule.

❌ **The quoted magnitudes do not reproduce.** "AbsRel improves by 8 % and RMSE by 5 %" is
re-measured at **3.9 % and 2.2 %**. δ1 and bias reproduce closely (0.887 vs 0.886; −0.134 vs
−0.135), but RMSE and MAE are systematically lower in the re-run and the paper's clear
degradation at ridge 0.01 does not appear.

⚠️ **Author question:** Table IV does not state its backbone. This re-run used the DAv2
ViT-S teacher. A different backbone would explain the RMSE offset, and the table should say
which.

→ `zjul5_ridge_train_r31.json`

### 3.5 Table V — component ablation

❌ **The defect R1 and the consultant both flagged:** the paper reports the analytic output
as **0.055 m** in Table V and **0.064 m** in the scattered-hold-out paragraph. Those came
from different runs on different splits, so no row of the table could be compared with any
other row.

**Fixed.** `baselines.py` now holds every refiner engine resident at once and scores them
inside one frame loop, against the same per-frame anchor/hold-out split, off one shared
backbone pass. Comparability is by construction rather than by hoping two runs matched.

`B5` is a trained network and the 1234 logged pairs are its training set, so the refiner
rows are scored on the **61-frame `random_split(..., manual_seed(0))` validation split** that
no version trained on (`val_stems.txt`).

**medAE (m), 61 val frames, one run, 95 % frame-level bootstrap CI:**

| Row | `center` (extrapolation) | `random` (interpolation) | `edge` (discontinuity) |
|---|---|---|---|
| nearest-zone dToF | 0.1205 [0.1026, 0.1387] | 0.0177 [0.0147, 0.0206] | 0.2257 [0.2058, 0.2497] |
| **analytic fit (reference)** | **0.0519 [0.0426, 0.0608]** | 0.0512 [0.0414, 0.0600] | 0.4611 [0.4122, 0.5080] |
| + refiner, **scattered** hold-out | 0.0533 [0.0453, 0.0624] | 0.0305 [0.0278, 0.0336] | 0.2988 [0.2585, 0.3299] |
| + refiner, **island** hold-out | 0.0480 [0.0435, 0.0529] | 0.0959 [0.0874, 0.1080] | 0.4276 [0.3876, 0.4775] |
| + refiner, **deployed** (v7) | 0.0320 [0.0281, 0.0362] | 0.0451 [0.0389, 0.0520] | 0.3345 [0.3053, 0.3807] |
| + arbitration over analytic | 0.0491 [0.0406, 0.0566] | 0.0155 [0.0127, 0.0182] | 0.1493 [0.1148, 0.1740] |
| **+ arbitration, as deployed** | **0.0317 [0.0280, 0.0354]** | 0.0142 [0.0117, 0.0168] | 0.1384 [0.0997, 0.1684] |

**The reference row is 0.0519 m [0.0426, 0.0608] — not 0.055 and not 0.064.** Both published
figures fall inside that interval, so neither was wrong; they were simply never the same
measurement. The surrounding paragraph must be rewritten around this number.

→ `baselines_valsplit_r31.json`

**A note on which row is "as deployed".** On the robot the blend sits *over the refiner
output*. "Arbitration over analytic" is the same arbitration with the refiner skipped — it is
the topology Table VI actually printed (see §3.6), and it is now reported as its own row so
the two can no longer be confused.

### 3.5b The same ablation on the full 1234-frame split, both statistics

The val split above is the **quotable** one for the refiner rows, because the refiner trained
on the other 1173 frames. This table has 20× the *n* and is the better estimate for every row
with **no learned parameters** — the analytic fit, the weighting arms, nearest-zone dToF, and
arbitration over the analytic map. ⚠️ The three rows marked *(trained)* are contaminated here
by construction and must be quoted from the val-split table instead.

**`center` — extrapolation, the deployment-relevant protocol:**

| Row | medAE (m) | MAE (m) |
|---|---|---|
| nearest-zone dToF | 0.1170 [0.1135, 0.1207] | 0.2322 [0.2267, 0.2376] |
| analytic, uniform, no robust pass | 0.0552 [0.0533, 0.0570] | 0.3158 [0.3023, 0.3332] |
| **analytic fit (reference)** | 0.0540 [0.0522, 0.0558] | 0.3142 [0.2989, 0.3337] |
| + refiner, scattered *(trained)* | 0.0565 [0.0547, 0.0582] | 0.2925 [0.2791, 0.3099] |
| + refiner, island *(trained)* | 0.0531 [0.0518, 0.0546] | 0.2163 [0.2107, 0.2221] |
| + refiner, deployed v7 *(trained)* | 0.0326 [0.0318, 0.0336] | 0.1551 [0.1502, 0.1602] |
| **+ arbitration over analytic** | **0.0504 [0.0491, 0.0518]** | **0.2953 [0.2803, 0.3144]** |
| + arbitration, as deployed *(trained)* | 0.0319 [0.0311, 0.0327] | 0.1491 [0.1444, 0.1539] |

**`random` — interpolation:**

| Row | medAE (m) | MAE (m) |
|---|---|---|
| nearest-zone dToF | 0.0178 [0.0173, 0.0184] | 0.0734 [0.0718, 0.0749] |
| analytic, uniform, no robust pass | 0.0589 [0.0567, 0.0610] | 0.2726 [0.2658, 0.2789] |
| **analytic fit (reference)** | 0.0546 [0.0525, 0.0568] | 0.2616 [0.2554, 0.2671] |
| + refiner, scattered *(trained)* | 0.0315 [0.0308, 0.0323] | 0.1488 [0.1441, 0.1535] |
| + refiner, island *(trained)* | 0.0999 [0.0970, 0.1030] | 0.2693 [0.2610, 0.2779] |
| + refiner, deployed v7 *(trained)* | 0.0462 [0.0446, 0.0476] | 0.1754 [0.1704, 0.1805] |
| **+ arbitration over analytic** | 0.0157 [0.0152, 0.0162] | 0.0648 [0.0632, 0.0664] |
| + arbitration, as deployed *(trained)* | 0.0144 [0.0140, 0.0149] | 0.0580 [0.0566, 0.0595] |

**`edge` — discontinuity:**

| Row | medAE (m) | MAE (m) |
|---|---|---|
| nearest-zone dToF | 0.2275 [0.2226, 0.2324] | 0.5116 [0.5007, 0.5232] |
| analytic, uniform, no robust pass | 0.4470 [0.4344, 0.4597] | 0.7289 [0.7111, 0.7476] |
| **analytic fit (reference)** | 0.4368 [0.4254, 0.4488] | 0.7004 [0.6841, 0.7172] |
| + refiner, scattered *(trained)* | 0.2677 [0.2585, 0.2766] | 0.5353 [0.5175, 0.5527] |
| + refiner, island *(trained)* | 0.4400 [0.4302, 0.4491] | 0.6544 [0.6390, 0.6696] |
| + refiner, deployed v7 *(trained)* | 0.3495 [0.3399, 0.3590] | 0.5895 [0.5735, 0.6059] |
| **+ arbitration over analytic** | 0.1393 [0.1309, 0.1468] | 0.3942 [0.3828, 0.4054] |
| + arbitration, as deployed *(trained)* | 0.1331 [0.1236, 0.1407] | 0.3764 [0.3656, 0.3873] |

⚠️ **A correction recorded here because it would otherwise be invisible.** The
"arbitration over analytic" row initially blended *unclamped* analytic depth and read
**medAE 0.0504 / MAE 10.9487 m** on `center`. Stage `7b_clamp` runs before `7c_blend` in the
deployed pipeline, so every arbitration row must blend clamped depth. Corrected, the row is
**medAE 0.0504 / MAE 0.2953 m** — the median never moved, because the artefact was a handful
of extreme pixels. That medAE/MAE divergence is the same signature that originally exposed
the missing far-field clamp, and it is why both statistics are reported throughout.

→ `baselines_r31.json`

### 3.6 Table V — the supervision-geometry claim is argued on the wrong statistic

The paper claims island supervision beats scattered by ~30 % on extrapolation (0.045 vs
0.066). Paired bootstrap, shared frame draw across both arms, B = 1000 — **the two statistics
disagree, and that disagreement is the finding.**

`center` (extrapolation), the deployment-relevant protocol:

| comparison | medAE | MAE |
|---|---|---|
| scattered − analytic | +0.0014, p = 0.128 | −0.0143 [−0.0193, −0.0097], p = 0.000 |
| island − analytic | −0.0040, p = 0.270 | −0.0724 [−0.0911, −0.0537], p = 0.000 |
| **island − scattered** | −0.0053 [−0.0118, +0.0010], **p = 0.116** | **−0.0581 [−0.0747, −0.0412], p = 0.000** |
| deployed v7 − analytic | −0.0199 [−0.0267, −0.0136], p = 0.000 | −0.1214 [−0.1399, −0.1021], p = 0.000 |
| deployed − analytic-arbitration | −0.0174 [−0.0231, −0.0117], p = 0.000 | −0.1102 [−0.1270, −0.0930], p = 0.000 |

`random` and `edge`:

| comparison | `random` medAE | `random` MAE | `edge` medAE | `edge` MAE |
|---|---|---|---|---|
| island − scattered | +0.0655, p = 0.000 | +0.0891, p = 0.000 | +0.1288, p = 0.000 | +0.0786, p = 0.000 |
| deployed v7 − analytic | −0.0060, p = 0.014 | −0.0779, p = 0.000 | −0.1266, p = 0.000 | −0.1421, p = 0.000 |

**Island supervision cuts extrapolation MAE by 0.058 m — a 22 % reduction (0.2676 → 0.2094),
p = 0.000 — while leaving the median untouched (p = 0.116).** It does not make the typical
pixel better; it stops far-field pixels blowing up. That is precisely what supervising beyond
the anchor island should do, and MAE is the statistic that sees it.

So the claim is **directionally right but defended on the median, where it does not hold**,
when the evidence for it is in the tail, where it is overwhelming. Report both statistics.

**The cost is real and symmetric.** On interpolation island is worse on both statistics
(medAE +0.0655, MAE +0.0891, both p = 0.000); on discontinuities scattered wins on both. The
honest framing is a **trade**, not an upgrade: island buys far-field tail behaviour and sells
near-field accuracy and edge behaviour. Stage 7c's arbitration exists to buy the near field
back, and §3.5 shows it does — the deployed arbitration row beats every single-source row on
all three protocols.

**What is not separable.** The shipped v7 beats the analytic baseline outright on both
statistics and all three protocols, so the refiner earns its place. But v7 differs from the
island engine by the FOV fix and a 5× epoch budget as well as supervision geometry, so its
margin cannot be attributed to supervision alone.

**Caveat, stated because it cuts both ways.** The scattered and island engines are both
pre-FOV-fix (trained Jul 28/29; `fov_h` corrected 45° → 73.5° on Aug 3–4), so both are
evaluated under a calibration neither trained against. That is a fair A/B — one changed
argument, both equally mismatched — but it is not the regime either was trained for. Settling
it outright needs a retrain of both under the corrected calibration.

### 3.7 Table VI — ZJU-L5

❌ **The "as deployed" row is the analytic stage with the refiner off.** Its printed
0.185 / 1.174 / 0.716 matches `B4c_affine_cl` (0.1847 / 1.1740 / 0.7160) to three decimals,
while the caption calls it "the Complete 4.1M Pipeline".

Re-run on the deployed pair, 527 test samples:

| Region | Configuration | AbsRel ↓ | RMSE ↓ | δ1 ↑ |
|---|---|---|---|---|
| all | analytic only — **what Table VI prints** | 0.1808 | 1.1882 | 0.7272 |
| all | + residual refiner | 0.6754 | 3.0899 | 0.1429 |
| all | + arbitration over analytic | **0.1555** | **1.1768** | **0.7704** |
| inside footprint | analytic only | 0.1209 | 0.5081 | 0.8517 |
| inside footprint | + arbitration | **0.0756** | **0.4588** | **0.9286** |
| outside footprint | analytic only | 0.2553 | 1.6874 | 0.5722 |
| outside footprint | + arbitration | 0.2547 | 1.6870 | 0.5735 |

**The correctly-labelled row is better than what the paper claims, not worse** — δ1 0.7704
overall and 0.9286 inside the footprint, against the printed 0.716.

**The refiner does not transfer off-domain**, and v7 transfers worse than the older engine
(δ1 0.143 against 0.292 on the same data) while improving on our own hardware, where Table III
reproduces exactly. That is domain specialisation, and it **strengthens the paper's own
limitation 3** rather than undermining it.

**Deployed row split by footprint** (R1's inside/outside requirement):

| | AbsRel | RMSE | δ1 |
|---|---|---|---|
| deployed, inside footprint | 0.2701 | 1.4984 | 0.6179 |
| deployed, outside footprint | 0.7200 | 3.3838 | 0.1081 |
| ViT-S, outside | 0.1417 | 1.5113 | 0.8212 |
| DEPTHOR-Small, outside | 0.1186 | 1.0269 | 0.8566 |

**Footprint share on ZJU-L5: 54.43 %** (85,375,741 of 156,851,220 evaluated pixels) — against
**12.4 %** on our own sensor. The two "inside the footprint" numbers are therefore not
comparable quantities, and the paper should say so.

**✅ The ViT-S row reproduces exactly:** 0.0941 / 1.0564 / 0.9017 against the published
0.094 / 1.056 / 0.902.

❌ **One sub-claim does not.** The Table VI note says RMSE is "0.418 inside the dToF footprint
against 1.398 outside" on the ViT-S row. Re-measured: **0.421 inside** (matches) but
**1.511 outside**, not 1.398. The inside figure and the argument both survive; the outside
number needs updating.

→ `zjul5_deployed_r31.json`, `zjul5_vits_r31.json`, `zjul5_deployed_blendnet_r31.json`

### 3.8 Table VII and §V-E — uncertainty coverage

⚠️ **This is the highest-value correction available at zero cost, and it is exactly what R3
asks for.**

The existing `lo`/`hi` in the tape files are the spread across **5 repeat captures of the same
points** — pipeline noise. They are *not* sampling uncertainty. A claim about where coverage
sits relative to its target needs the binomial (Wilson) interval:

| Set | @1σ | 95 % Wilson | @2σ | 95 % Wilson |
|---|---|---|---|---|
| n = 11, before σ terms | 9/11 = 0.818 | [0.523, 0.949] | 0.818 | [0.523, 0.949] |
| n = 11, after σ terms | 10/11 = 0.909 | [0.623, 0.984] | 0.909 | [0.623, 0.984] |
| n = 15, all points | 12/15 = 0.800 | [0.548, 0.930] | 15/15 = 1.000 | [0.796, 1.000] |
| n = 15, held out only (n = 7) | 4/7 = 0.571 | [0.250, 0.842] | 7/7 = 1.000 | [0.646, 1.000] |
| *targets* | *0.683* | | *0.954* | |

**Every interval contains its target, and every pair of intervals overlaps heavily.** The
claims that coverage "moves toward its target rather than away" (0.909 → 0.760) and that the
σ terms improved coverage (0.818 → 0.909) are **not supported at this sample size**.

🆕 **What the data does support — σ ranks errors correctly even where it was not tuned:**

| Column | n | rank corr | cov@1σ | worst residual |
|---|---|---|---|---|
| all fifteen | 15 | +0.625 | 0.800 [0.55, 0.93] | 1.724 |
| calibration re-measured | 8 | +0.762 | 1.000 [0.68, 1.00] | 0.215 |
| **held out** | 7 | **+0.679** | 0.571 [0.25, 0.84] | 1.724 |
| strictly held out | 5 | +0.800 | 0.600 [0.23, 0.88] | 1.724 |

σ **orders** errors as well on held-out points as on the ones it was tuned against
(+0.679–0.800 vs +0.762), but is **under-sized in magnitude** there, and the worst residual
comes entirely from the held-out side. That is a nuanced, defensible claim and a better one
than the coverage claim it replaces.

→ `tape_stats_r31.json`

### 3.9 §IV-C and §IV-F — method text that does not match the code

❌ **The documented fallback does not exist.** §IV-C says "the system falls back to a
scale-only estimate"; limitation 4 repeats it. The code (`pipeline.py:171-173`) does:

```python
fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
if fit is None:
    return {'ok': False, 'n_anchors': int(inb.sum())}
```

**The frame is dropped.** There is no scale-only fallback, no held `b̂`, and no threshold
constants — `N_min`, `v_min` and `κ_max` have no counterparts in the code. The only guards are
inside `solve_scale_shift`: `w.size < 2`, `w.sum() <= eps`, `|den| < eps`. Either rewrite both
passages to describe frame-dropping, or implement the guards and re-run every dependent number.

✅ **The σ constants verify against HEAD:**

| Paper symbol | Value | Code | Status |
|---|---|---|---|
| `c_a` | 1.00 | `DISAGREE_K = 1.0`, squared | ✅ |
| `c_ν` | 1.00 | `SPREAD_K = 1.0`, squared | ✅ |
| `c_α` | 0.1225 | `SUPPORT_FRAC = 0.35`, squared | ✅ |
| `α_0` | 5.0° | `FAR_DEG = 5.0` | ✅ |
| smoothstep range | 2.0–5.0° | `NEAR_DEG = 2.0`, `FAR_DEG = 5.0` | ✅ |
| `α_max` | — | ❌ **mis-mapped** | below |

❌ **`α_max` is not an angular threshold.** The "100 % floor" is `ROI_OUTSIDE_SIGMA_FRAC = 1.0`
(`pipeline.py:31`), applied as a separate clamp keyed to the **geometric ROI mask** (reach
≤ 3.0 m, height ≤ 0.6 m), not to angular support. It is visible in the data: n = 15 points 8
and 9 have σ exactly equal to the prediction. This needs rewritten method text, not a
different number.

❌ **The constants were not chosen by NLL minimisation.** The paper says they were chosen "by
minimizing the Gaussian negative log likelihood of the tape residuals". **No NLL fit exists
anywhere in the record.** What actually happened, from `blend.py`'s comments and
`live4_sigma_after_2026-08-04.json`:

- terms were **designed against two identified failure modes** observed in a live session (a
  mixed-return marker at 3.9σ, an angle asymmetry), not fitted;
- `DISAGREE_K` and `SPREAD_K` are both **1.0** — a natural unit scale, not a fitted value;
- `SUPPORT_FRAC = 0.35` is the only non-trivial constant;
- `SPREAD_WIN = 11` is fixed by **geometry** (~17.7 px per zone → 11 cells ≈ 2.5 zones); a
  first attempt at 3 cells was smaller than one zone and did nothing;
- a single global multiplier rescale **was swept and rejected** — "no single scale serves both
  ends";
- scored against 11 tape points, with the contemporaneous note "overfitting risk is real".

Honest replacement: *designed against two identified failure modes, two constants at unity,
one at 0.35, window set by zone geometry, scored against eleven tape points; a global rescale
was swept and rejected.* This is a better story than NLL fitting, and it has the advantage of
being true.

### 3.10 §V-A — rate, latency and "real time"

Two 900 s runs, `jetson_clocks` applied, leroi stack stopped, node parameters read back from
the **running node** (`blend=True`, `roi_enable=True`, `min_confidence=-1`,
`plane_refit_every=1`), engines verified by SHA.

| | Run 1 | Run 2 | Paper |
|---|---|---|---|
| Rate | **9.40 Hz** | **9.49 Hz** | 9.4 Hz ✅ |
| p05–p95 rate | — | **8.52 – 10.83 Hz** | fills the pending slot |
| Age median | **146.9 ms** | 135.7 ms | 128 ms ❌ |
| Age p90 / max | 172.6 / 230.3 ms | 160.9 / **572.6** ms | — |
| Thermals | 52.2 → 55.1 °C | — | no throttling |

✅ **9.4 Hz reproduces.** ❌ **128 ms does not** — it came from a Jul 30 capture predating both
the v7 engine and the σ terms. Four independent current-build measurements agree on 135–147 ms.
Run-to-run: rate stable to ~1 %, age varies ~8 %. Run 2 caught a single 554 ms gap run 1 did
not; keep it, because that tail is what a controller feels.

🆕 **Data-age budget** — R4 asks for the core-latency / publish-rate / data-age relationship
explicitly, and nothing measured it before:

| Component | median | p05 | p95 |
|---|---|---|---|
| sensor → arrival | **1.4 ms** | 1.1 | 12.9 |
| node latency | **143.6 ms** | 103.7 | 175.8 |
| total age | 145.5 ms | 115.5 | 177.1 |
| **image staleness** | **10.9 ms** | 2.6 | 45.6 |

1. **dToF transport is essentially free (1.4 ms)** — the age is a compute-and-queue story, not
   a sensor one.
2. **A map is 1.38 publish periods old when it lands** (145.5 ms against a 105.7 ms period).
3. ⚠️ **Image staleness 10.9 ms, p95 45.6 ms.** `on_tof` consumes whatever `on_image` last
   cached, so the camera half of each fusion is older than the published stamp implies. Not
   mentioned anywhere in the paper; it belongs in the limitations.

**Per-stage frame budget**, 500 frames, mapped onto Table III's six published rows:

| Paper row | ms | Stages summed |
|---|---|---|
| camera branch: capture, rectify, backbone | **14.27** | `1_rectify` + `2_backbone` |
| dToF branch: projection and validity | **0.98** | `3_4_project_pair` + `4b_roi_plane` |
| fit, covariance and refiner | **31.24** | `5_fit_metric` + `7_residual` |
| source arbitration | **25.54** | `7c_blend` + `7b_clamp` |
| uncertainty terms | **12.18** | `6_variance` + `7d_roi_sigma` |
| unprojection, assembly and publication | **10.70** | `8_cloud` + `9_publish` |
| *sum of rows* | *94.92* | vs **frame total 97.17** — 2.25 ms is untimed glue; state it |

⚠️ **"The uncertainty terms cost 10.6 ms" is a different decomposition.** 10.6 ms is a bypass
A/B of the Stage 7c σ block, which lives *inside* `7c_blend`. The 12.18 ms row above is
`6_variance` (analytic delta-method) + `7d_roi_sigma` (the ROI floor). They are not the same
quantity and the paper must not present one as the other.

**Age attribution closes:** node latency 143.6 ms − replica frame total 97.2 ms = **46.5 ms of
queue and transport** the pipeline does not itself account for. The deployed period is
105.7 ms, so the node works on frame *N* while *N+1* waits — which is what makes the age 1.38
periods rather than 1.0.

✅ **Defining "real time" (R5).** The consumer is the route planner at
**`route_rate_hz = 3.0`** (`serial_messenger.launch.py`; declared at `route_planner_node.py:95`)
— a 333 ms period. We publish every 105.7 ms and maps land **146.9 ms** old, i.e. **0.44
planner periods**. Every map the planner consumes is less than half a cycle old, and the
pipeline runs 3.1× faster than the planner replans and above the sensor's 8.3 Hz assembly
rate. **Both halves of the definition hold with margin even at the corrected 146.9 ms**, so
the age correction does not weaken §V-A.

⏳ **Platform top speed — optional, and not part of the definition.** Both halves of §V-A's
stated definition (publish at least as fast as the sensor produces data, and at least as fast
as the planner replans) are rate-versus-rate and are **already satisfied above without it**.
Top speed supports a *separate* argument the brief lists under §V-A as "travel distance", and
which the brief itself flags as needing an author decision.

Its value is specific: the map age is being **revised upward from 128 ms to 146.9 ms**, and a
reviewer can accept both rate comparisons and still ask whether 147 ms of staleness is
acceptable. Rate ratios cannot answer that — they establish that we publish often, not that
the data is fresh when it is used. Top speed converts the age into a bound: *at top speed the
robot travels v × 0.147 m between a map being true and being acted on* (~15 cm at 1 m/s). That
turns the age correction from a disclosure into a bounded quantity.

**Which speed:** the maximum the platform actually reaches under its own control program on
its deployment surface — not the speed during any particular recording, and not the drivetrain
free-speed. Motion control runs on the V5 brain (the Jetson sends route points, never
velocities), so the commanded cap is the honest bound.

→ `rate_live_r31.json`, `rate_live_r31_run2.json`, `age_budget_r31.json`, `profile_node_r31.json`

### 3.11 §V-B — anchor weighting

❌ **Section IV-C says "we use $w_i \propto z_i$". The deployed code does not.**
`anchoring.py` has `RANGE_WEIGHT_P = 0.0`, and `pipeline.run` folds in the geometric ROI gate
from `roi.py` instead. **This is wrong as a statement of fact, independently of which
weighting is better.**

Five weighting arms were run on the **same anchors and the same held-out zones**, so the
column is a clean ablation rather than five separate configurations. A consistency self-check
is built in: `W0_uniform` and `B4c_affine_cl` reach the same fit by two different code paths,
and the report prints the gap. **Measured 0.0000 m on every protocol** — the arms are wired
correctly.

**medAE (m), full 1234-frame split, 95 % frame-level bootstrap CI:**

| Weighting | `random` | `center` | `edge` |
|---|---|---|---|
| uniform, no robust pass | 0.0589 [0.0567, 0.0610] | 0.0552 [0.0533, 0.0570] | 0.4470 [0.4344, 0.4597] |
| **w ∝ z** *(what §IV-C claims ships)* | 0.0716 [0.0682, 0.0752] | 0.0564 [0.0546, 0.0581] | 0.4754 [0.4642, 0.4866] |
| w ∝ z² | 0.1013 [0.0960, 0.1069] | 0.0658 [0.0636, 0.0679] | 0.5082 [0.4959, 0.5223] |
| **geometric ROI gate** *(what actually ships)* | **0.0502 [0.0483, 0.0520]** | 0.0545 [0.0527, 0.0564] | **0.4360 [0.4208, 0.4514]** |
| uniform + robust pass | 0.0546 [0.0525, 0.0568] | **0.0540 [0.0522, 0.0558]** | 0.4368 [0.4254, 0.4488] |

**`w ∝ z` trades median accuracy for tail accuracy, and only in the extrapolation regime.**
Paired against uniform + robust pass, shared frame draw, B = 1000 — at n = 1234 every
comparison reaches significance, so the *sign* is what matters, not the p-value:

| `w ∝ z` − uniform | medAE | MAE |
|---|---|---|
| `random` | **+0.0169** [+0.0147, +0.0194] ❌ worse | **+0.0062** [+0.0034, +0.0092] ❌ worse |
| `center` | **+0.0024** [+0.0014, +0.0033] ❌ worse | **−0.0248** [−0.0307, −0.0204] ✅ better |
| `edge` | **+0.0386** [+0.0281, +0.0483] ❌ worse | **−0.0572** [−0.0699, −0.0448] ✅ better |

Range weighting upweights distant anchors, which pulls the fit toward the far field. That
helps the **tail** when extrapolating or crossing a discontinuity, and hurts the **typical
pixel** everywhere — decisively so on interpolation, where it is worse on both statistics.

**So the deployed choice is defensible but not dominant**, and the honest sentence says so:
uniform-plus-ROI-gate is better on the median on all three protocols and better on MAE on
interpolation; `w ∝ z` has a real MAE advantage in extrapolation that the deployed
configuration gives up. An earlier single-statistic reading of this as "refuted" was too
strong.

**The ROI gate's own advantage is narrower than it looks.** It is real on `random`
(0.0502 vs 0.0546, non-overlapping) but **indistinguishable from uniform on `center` and
`edge`** (intervals overlap). Worth stating rather than hiding.

✅ **Robust-pass rate** (fills a pending slot): Huber **downweights, it does not reject**, so
"removes X.X % of anchors" names neither quantity. Both are now reported: **22.7–22.9 % of
anchors downweighted, removing 6.8–6.9 % of total weight mass**, stable across all three
protocols. Use the second number if the sentence says "removes"; the first if it says
"downweights".

Its benefit is small and protocol-dependent: medAE improves on all three (−0.0042 `random`,
−0.0012 `center`, −0.0102 `edge`, all p = 0.000), but MAE improves only on `random` and `edge`
(−0.0109 and −0.0286, p = 0.000) and is **indistinguishable on `center`** (−0.0016, p = 0.568).

→ `baselines_r31.json`

### 3.11b §V-B — the arbitration A/B under motion ✅ *(two drives, 2026-09-13)*

R2 asked for intervals on the §V-B arbitration claim, which the paper quotes with none.
Measured on **two live drives**, 600 frames each, `jetson_clocks` applied, MAXN, leroi stack
stopped, commit `aba5edd`, engines SHA-verified in run 2.

**This is a paired comparison, and the two drives are not the A/B.** Both arms come from one
backbone + refiner pass on the *same* frame — arm A is the refined map, arm B is that same map
with the blend applied. A central 16×16 dToF island anchors the fit and feeds the blend;
everything outside is held out and used only for scoring, so **the blend is never graded on a
zone it saw**. The second drive exists for *between-run* variation and a different scene
depth, not to form the comparison.

**Session conditions.** Both drives: 600 frames, coverage 1.00, `jetson_clocks` applied,
MAXN, leroi stack stopped, commit `aba5edd`, engines `student_v4_heldout_fp16` +
`residual_v7_fov73_fp16`. dToF range gate 0.15–6.5 m. Paired bootstrap B = 2000 over 600
frames.

**Run 1 — short-range scene** (415,368 held-out zones, median max depth **2.49 m**):

| metric | A: no blend | B: blend | paired Δ (B − A) | p |
|---|---|---|---|---|
| **MAE** | 0.1610 | **0.1526** | **−0.00839 [−0.00933, −0.00751]** | 0.000 |
| **medAE** | 0.0342 | **0.0331** | **−0.00110 [−0.00129, −0.00091]** | 0.000 |
| p95 | 0.8860 | **0.8491** | — | — |
| RMSE | 0.3560 | **0.3439** | — | — |
| AbsRel | 0.1809 | **0.1766** | — | — |
| bias | −0.0408 | **−0.0367** | — | — |
| δ1 ↑ | 0.7890 | **0.7999** | — | — |
| δ2 ↑ | 0.8912 | **0.8965** | — | — |
| δ3 ↑ | 0.9402 | **0.9428** | — | — |
| frame-to-frame \|Δdepth\| | 0.00816 | **0.00780** | — | — |

**Run 2 — long sightline** (406,223 held-out zones, median max depth **4.56 m**):

| metric | A: no blend | B: blend | paired Δ (B − A) | p |
|---|---|---|---|---|
| **MAE** | 0.1984 | **0.1850** | **−0.01340 [−0.01400, −0.01279]** | 0.000 |
| **medAE** | 0.0455 | **0.0443** | **−0.00121 [−0.00182, −0.00072]** | 0.000 |
| p95 | 0.8725 | **0.8420** | — | — |
| RMSE | 0.3619 | **0.3461** | — | — |
| AbsRel | 0.1439 | **0.1382** | — | — |
| bias | +0.0378 | **+0.0332** | — | — |
| δ1 ↑ | 0.8219 | **0.8362** | — | — |
| δ2 ↑ | 0.9385 | **0.9416** | — | — |
| δ3 ↑ | 0.9747 | **0.9751** | — | — |
| frame-to-frame \|Δdepth\| | 0.00493 | **0.00461** | — | — |

**Every metric improves in both drives** — including p95, which is the tail, and all three δ
thresholds. There is no metric on which the blend loses.

*Two scene differences worth noting when comparing the runs, neither of which affects the A/B:*
run 2 has **lower AbsRel** (0.144 vs 0.181) and **better δ2/δ3** despite its higher absolute
MAE, because relative error falls as scenes get deeper; and **bias flips sign** (−0.041 in run
1, +0.038 in run 2), i.e. the system slightly under-estimates depth in the near scene and
over-estimates in the far one. Both are properties of the scenes, not of the blend — the
paired differences are computed within-frame and are unaffected.

**The blend wins every metric in both drives, and every interval excludes zero.** That is the
interval §V-B was missing.

🆕 **The benefit scales with scene depth, and this is the payoff from running twice.** The MAE
gain is **60 % larger in the longer-range drive** (−0.0134 vs −0.0084), and the two intervals
**do not overlap** — so that is a real scene effect, not run-to-run noise. The median gain, by
contrast, is essentially unchanged (−0.0011 vs −0.0012). Read together: arbitration does not
change the typical pixel much in either scene, but the harder and further the scene, the more
it suppresses large errors. Between-run variation therefore lands on the *effect size*, while
the *direction and significance* are stable — which is a more useful statement for R2 than a
single number with an interval.

**medAE by angular distance from the dToF island** — the tool's built-in sanity check:

| arm | run 1: 10–15° | run 1: 15–30° | run 1: 30–90° | run 2: 15–30° | run 2: 30–90° |
|---|---|---|---|---|---|
| A: no blend | 0.0169 | 0.0369 | 0.0326 | 0.0735 | 0.0370 |
| B: blend | **0.0099** | **0.0342** | 0.0326 | **0.0554** | 0.0370 |

**The blend's benefit is concentrated near the island — −41 % at 10–15° (run 1) and −25 % at
15–30° (run 2) — and fades to nothing by 30–90°, where the two arms are numerically identical
in both drives.** That identity is by construction and is the desired result, not a shortfall:
Stage 7c is a smoothstep that pulls toward dToF where a measurement is near and hands back to
the network where none is, so far from the island B returns A's depth untouched. The check
this table exists to run is whether B *diverges* far out — that would mean the blend is
leaking near-field depth into the far field. It does not, in either drive.

⚠️ **One limit neither drive reached, with a concrete requirement for next time.**
`fraction beyond dToF range` is **0.00 % in both**: the dToF gate is **6.5 m** and the deepest
scene reached only a 4.56 m median maximum, so nothing was ever scored past the sensor's
reach. The far-field extrapolation regime is therefore still evidenced by Table III's angular
bands and the tape reference (§3.12), not by these drives.

> 📏 **Scene requirement for the next drive: a clear sightline of more than 6.5 m, and
> preferably 10 m or more.** 6.5 m is the bare threshold — the dToF gate itself — so a scene
> that merely touches it produces almost no pixels past the gate and the band stays empty in
> practice. A 10 m+ sightline puts real content beyond dToF reach and finally exercises the
> regime the camera exists for: depth where the sensor has nothing to say. The robot does not
> need to *travel* 6.5 m, it needs to *see* that far, so a long corridor, a hallway, or a
> gym/field diagonal all work. Keep some near structure in frame as well, so the 10–15 ° band
> still has content and the drive is not testing only one end.

**Relation to the published figures.** The paper's MAE 0.264 → 0.247 and frame-to-frame
0.0106 → 0.0100 come from `blend_ab_live_v7_2026-08-04.json`, a different session. Direction,
significance and the angular signature all reproduce here; the absolute magnitudes differ with
scene depth, as run 1 vs run 2 shows directly. Quote the r31 pair with its intervals and name
the scene, rather than carrying the older bare numbers.

*Minor provenance note:* run 1's `env` block records clock state, thermals and git commit but
not the engine SHAs, because `blend_ab_live` called `envinfo.capture()` without the engine
list. The engines are `student_v4_heldout_fp16` + `residual_v7_fov73_fp16`, named on the
command line and SHA-verified the same day under the same commit in the other r31 captures.
The tool was fixed between the two drives, so **run 2 carries the SHAs directly.**

**⚠️ How to read the size of this effect — the drive understates the component by
construction.** Written out because a reader coming to these numbers cold would reasonably
conclude arbitration is marginal, and that is not what the data says.

The protocol anchors a central 16×16 dToF island and scores only zones *outside* it. That
island spans roughly **0–10°**, so the three nearest angular bands are **empty in both drives**
— the zones where the blend does its heaviest lifting are precisely the zones being used as
anchors, and can never be scored. What the drive can see is the handoff band and beyond:
a large win at 10–15° / 15–30°, and correctly nothing past 30°. The headline **−7 % MAE is
those bands averaged together, diluted by the far ones where no effect is possible.**

Scored where arbitration *can* act, the same component is worth far more (§3.5b, full split):

| protocol | refiner alone | + arbitration | gain |
|---|---|---|---|
| discontinuities (`edge`) | 0.3495 m | **0.1331 m** | **2.6×** |
| interpolation (`random`) | 0.0462 m | **0.0144 m** | **3.2×** |

So the two measurements agree and answer different questions: **the drive establishes that
arbitration never hurts under motion and helps materially in the transition band; Table V
establishes how large the benefit is where the component is actually active.** Cite both, and
do not let the drive's −7 % stand alone as the effect size.

⚠️ **The cost/benefit tension should be stated rather than left for a reviewer to find.**
Arbitration costs **25.54 ms of the 97.17 ms frame budget — roughly a quarter of the compute**
(§3.10). For a 2.6× improvement at depth discontinuities that is clearly worth paying. For
7 % in open driving it is much less obviously so. The honest position is that the component
earns its budget on edges and near-field structure, which is where a ground robot's collision
risk lives, and not on open-field average error.

→ `blend_ab_live_r31.json`, `blend_ab_live_r31_run2.json`, with full console output preserved
at `docs/demo/benchmarks/logs/blend_ab_live_r31*.log`.

*Note on the far-field figures:* median max depth and `fraction beyond dToF range` were
printed by the tool but **not written into the JSON** for these two runs, so they are quoted
here from the preserved console logs. The tool has since been fixed to persist them as a
`far_field` block alongside the range gate, so later runs will self-certify.

### 3.12 §V-C — the tape experiment

❌ **The 15-point set is not 15 independent points.** Matching on pixel coordinates (≤ 45 px,
ground truth agreeing within 10 cm ⇒ the same marker re-measured):

| Role | n | Point IDs |
|---|---|---|
| Calibration marker re-measured | **8** | 3, 5, 6, 7, 8, 9, 10, 12 |
| Ambiguous — same pixel region, ground truth differs ≥ 0.1 m | 2 | 1, 11 |
| **Strictly held out** | **5** | 2, 4, 13, 14, 15 |

**The independent sample is seven points.** The two ambiguous points sit 12 px and 26 px from
an earlier marker but at depths differing by **0.67 m and 0.96 m** — far beyond any marker's
physical extent — so they cannot be the same surface. They are different targets and therefore
held out. The paper's "tuned on eleven of the points and scored here on all fifteen" reads as
fifteen independent points and must be corrected. This is R2's second sentence and R3's
"limited validation sample" in one.

**Depth error against tape, bootstrapped over points, B = 10000:**

| Set | n | median | 95 % CI | MAE | 95 % CI |
|---|---|---|---|---|---|
| All 15 points | 15 | 0.051 | [0.009, 0.531] | 0.345 | [0.112, 0.656] |
| **Inside the cone** | 8 | **0.029** | **[0.012, 0.101]** | 0.152 | [0.020, 0.394] |
| On the cone edge | 3 | 0.002 | [0.000, 0.559] | 0.187 | [0.000, 0.559] |
| **Outside the cone** | 4 | **0.528** | **[0.265, 2.071]** | 0.848 | [0.331, 1.685] |
| Strictly held out | 5 | 0.526 | [0.002, 0.978] | 0.408 | [0.109, 0.708] |

✅ **The in-cone vs out-of-cone separation is real and survives the interval** — 0.029
[0.012, 0.101] against 0.528 [0.265, 2.071], non-overlapping. This is the one strong claim the
tape data supports, and it directly answers R1's "accuracy inside and outside the dToF
coverage area".

🆕 **In-cone tape (0.029 m) independently confirms Table III's centre band (0.032 m) to 3 mm.**
The paper does not currently make this comparison. It is the best available answer to a
"your evaluation is circular" objection and should be stated explicitly.

⚠️ **The tape reference cannot carry an accuracy claim or rank methods.** The 15-point median
is 0.051 m with a 95 % interval of [0.009, 0.531] — a 30× range. It cannot resolve a 1 cm
difference or support an ablation. Against that, the withheld-zone protocol scores ~10⁵ points
and lands at 0.014 m. Three caps, only one fixable by adding markers: pointing error is
first-order, it is one scene in one session, and it is point-wise so it cannot produce
frame-level RMSE or δ1.

⚠️ **Pointing sensitivity.** On point 11 the reported error is 2.071 m, but a pixel within
45 px errs by only 1.306 m. **~0.77 m of the largest out-of-cone error is where the click
landed**, not what the pipeline predicted. Human pointing is a first-order error term in this
experiment and belongs in the caveats.

**§V-E's out-of-cone example is a calibration-set point.** The quoted "error reaches 1.07 and
1.52 m while σ is 2.10 and 1.84" is `live4_sigma_2026-08-04.json` points 7 and 8 — points the
constants were fitted on. Held-out replacements with the same conclusion:

| id | Role | gt | error | σ | nσ |
|---|---|---|---|---|---|
| 1 | ambiguous | 1.70 | −0.265 | 1.932 | 0.14 |
| 11 | ambiguous | 4.03 | −2.071 | 1.962 | 1.05 |
| 13 | held out | 0.82 | +0.531 | 3.274 | 0.16 |
| 15 | held out | 1.86 | −0.526 | 0.468 | 1.12 |

Recommend re-sourcing §V-E to these: same conclusion (σ declares the extrapolation),
independent data.

→ `tape_stats_r31.json`

### 3.13 The withheld-zone protocol — what it can and cannot say

It works, and it is the backbone of Table III. Four limits worth stating:

1. **Every target is a dToF zone, and zones exist only inside the cone.** No cell in Table III
   is out-of-coverage — not by band choice, but because no ground truth exists out there.
2. **Targets are zone centres**, where nearest-neighbour looks best. Error across a
   discontinuity falling *between* zones cannot appear. → addressed by the new `edge` protocol.
3. **Common-mode error cancels** — dToF bias is present in both anchors and targets.
4. **Noise floor.** `calibration.yaml` records the dToF at MAE 0.010 m against tape over 7
   markers. The `random` protocol reports 0.014 m — within 4 mm of the reference's own
   accuracy. That column is partly measuring the sensor.

⚠️ **An in/out-footprint split was deliberately NOT added to this protocol.** Every evaluation
target is a dToF zone and zones exist only inside the cone, so the split is degenerate on
own-hardware data. Fabricating one would misrepresent what was measured. R1's inside/outside
requirement is carried instead by the tape reference (§3.12), the ZJU-L5 regions (§3.7), and
the pending plane captures (§5).

---

## 4. New results, not currently in the paper

### 4.1 Discontinuity behaviour — the `edge` protocol

Holds out zones whose depth differs from their 8-neighbour median by more than 0.15 m (>10×
the sensor's own 0.010 m accuracy), anchoring on the smooth remainder. **1146 of 1234 frames
contained a real discontinuity; the 88 without one are skipped and reported, not silently
backfilled.**

medAE, m, full split:

| Method | random | center | **edge** | edge vs center |
|---|---|---|---|---|
| nearest-zone dToF | 0.0178 | 0.1170 | **0.2275** [0.2226, 0.2324] | 1.9× |
| analytic + robust | 0.0546 | 0.0540 | **0.4368** [0.4254, 0.4488] | **8.1×** |
| + residual refiner | 0.0462 | 0.0326 | **0.3495** [0.3399, 0.3590] | **10.7×** |
| + arbitration over analytic | 0.0157 | 0.0504 | **0.1393** [0.1309, 0.1468] | 2.8× |
| **+ arbitration, as deployed** | 0.0144 | 0.0319 | **0.1331** [0.1236, 0.1407] | 4.2× |

1. **Arbitration is what rescues the discontinuity case.** On `center` the refiner and
   arbitration are indistinguishable (0.0326 vs 0.0319); on `edge` arbitration beats the
   refiner by **2.6×** (0.350 → 0.133). The blend takes the dToF where a measurement is near,
   and edge zones are surrounded by anchors, so it does exactly that — while the camera path
   is the one that gets the step wrong.
2. **Nearest-zone dToF beats the entire camera path on edge zones** (0.228 against
   0.350–0.508). Only arbitration beats it.

This is a clean quantitative answer to the §V-C caveat — "error across a discontinuity falling
between zones cannot appear" — which the paper currently supports with a single thin-pole
marker.

⚠️ **Scope.** `edge` holds out ~25 % of zones at discontinuities while anchoring on the smooth
remainder, so held-out zones sit mostly 0–3° from an anchor. It isolates the **discontinuity**
effect, not extrapolation. State it that way.

### 4.2 Known artefacts, both already documented in the code

- `B2_bilinear` coverage **0.00** on `center` — it cannot extrapolate outside its convex hull.
  Reported as collapsed coverage rather than papered over with a nearest-neighbour fallback.
- `B4_affine` MAE **11.157 m** on `center` against `B4c_affine_cl` 0.314 m — the far-field
  clamp-policy artefact. The gap *is* the size of the artefact, which is why the paper should
  quote the clamped variant.

---

## 5. Still outstanding

### Needs the robot

> 📏 **Scene note carried forward:** any further arbitration drive needs a **clear sightline
> over 6.5 m, ideally 10 m+** — see §3.11b. Both drives so far topped out at 4.56 m, under the
> dToF's own 6.5 m gate, so the beyond-range regime has never been measured under motion.

| # | Item | What it fills | Blocked on |
|---|---|---|---|
| 1 | **Marker session** — 20+ points, composition 8+ in cone / 3 edge / 4+ out / 4+ discontinuities, 0.3–4.0 m span, static scene, 5 repeats | Replaces the seven-point independent sample with a properly composed one; retires the calibration/held-out ambiguity entirely, since σ constants are frozen and every new point is automatically held out | markers not yet placed |
| 2 | **Plane captures** — `plane:<name>` groups of ≥ 3 tape-anchored points | Dense dToF-independent ground truth, ~10⁵ px/frame, covering the periphery, in the same metric form as Tables III and VI. **The plane must come from tape, not from dToF anchors, or the circularity returns.** | same session |
| ~~3~~ | ~~**Driving session**~~ ✅ **done 2026-09-13**, two drives — see §3.11b | §V-B now has paired intervals: MAE −0.0084 [−0.0093, −0.0075] and −0.0134 [−0.0140, −0.0128], both p = 0.000 | — |
| 3b | **Optional third drive, long sightline** | The only untested regime: scene content **beyond the dToF's 6.5 m gate**, where the camera path is unsupported. Needs a **>6.5 m, ideally 10 m+** clear sightline; both drives so far reached only 4.56 m | a longer space |
| ~~4~~ | ~~**Platform top speed**~~ ⏸️ **deferred by author decision, 2026-09-14** | Optional throughout — §V-A's definition is satisfied on both rate halves without it. If it is never measured, §V-A rests on the two rate comparisons alone, which is sufficient; the travel-distance sentence is simply omitted. Revisit only if a reviewer challenges the 146.9 ms age | ~10 min if wanted later |

Floor geometry on this rig is already validated to **2.3 mm** (implied lens height 15.97 cm vs
16.2 cm tape), which is what makes the plane approach viable.

### Offline — complete

Nothing offline is outstanding. The full 1234-frame ablation, the 61-frame val-split
ablation, the tape re-analysis, the ZJU-L5 re-runs and the DEPTHOR re-timing have all
landed, each with an embedded environment and engine-SHA block.

---

## 6. Author decisions

These are judgement calls, not measurements.

0. ⏸️ **Platform top speed — deferred 2026-09-14.** Not required: §V-A's real-time definition
   is rate-versus-rate and already satisfied. Omitting it means §V-A carries no
   travel-distance sentence, which is a presentational loss, not an evidential one. Revisit
   only if a reviewer challenges whether 146.9 ms of staleness is acceptable.
1. **Fallback** (§3.9) — rewrite both passages to describe frame-dropping, or implement the
   guards and re-run every dependent number.
2. **Table VI** (§3.7) — report both rows (analytic and full pipeline), or scope the deployed
   row to the analytic stage and say so. Both are now available from one run.
3. **Supervision geometry** (§3.6) — re-argue on MAE and report both statistics
   *(recommended, no new runs)*; or drop the comparison and claim only that the shipped
   refiner beats the analytic baseline; or retrain both engines under the corrected
   calibration to settle it outright.
4. **Table IV backbone** (§3.4) — state which backbone produced it.
5. **Artifact release scope** (R5) — code, calibration, splits, runtime settings. Splits to
   release: `heldout_stems.txt` (200), `val_stems.txt` (61, entirely a subset of the 200),
   the ZJU-L5 train/test split, and the tape calibration/held-out assignment. `data/` is
   gitignored, so no stem→frame mapping is currently public.
6. **σ-constant provenance wording** (§3.9) — the substance is settled; only the phrasing is
   open.

---

## 7. Reproduction

Every run below is offline against logged data except where noted. All write their own
environment block into the output JSON.

```bash
# component ablation -- every row, one run, one split
python3 tools/diagnostics/baselines.py \
  --rgb-dir ros2_ws/data/real/rgb --tof-dir ros2_ws/data/real/tof \
  --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
  --backbone-engine student_v4_heldout_fp16.engine \
  --residual-engine v7=residual_v7_fov73_fp16.engine \
                    scattered=residual_v3_best_fp16.engine \
                    island=residual_v4_best_fp16.engine \
  --deployed-engine v7 \
  --stems-file docs/demo/benchmarks/val_stems.txt \
  --protocol random center edge --bootstrap 1000 \
  --paired B4c_affine_cl:B5_scattered B5_scattered:B5_island \
           B4c_affine_cl:B5_ringfusion B6_analytic:B6_blend \
  --out docs/demo/benchmarks/baselines_valsplit_r31.json

# tape re-analysis: provenance, Wilson coverage intervals, bootstrapped errors
python3 tools/diagnostics/tape_stats_r31.py --out docs/demo/benchmarks/tape_stats_r31.json

# on-robot: rate and map age (900 s), data-age budget, per-stage frame budget
sudo jetson_clocks          # DOES NOT PERSIST ACROSS REBOOTS -- re-run every boot
python3 tools/diagnostics/rate_live.py    --seconds 900 --out .../rate_live_r31.json
python3 tools/diagnostics/age_budget.py   --seconds 300 --out .../age_budget_r31.json
python3 tools/diagnostics/profile_node.py --frames 500  --out .../profile_node_r31.json
```

---

## 8. Source files

All under `docs/demo/benchmarks/`. Each embeds its own environment and engine SHA block.

| File | Contents |
|---|---|
| `baselines_valsplit_r31.json` | **Table V** — every row, 61 val frames, one run, CIs + paired tests |
| `baselines_r31.json` | Same on all 1234 frames — larger *n* for the untrained rows and the angular bands. Refiner rows there are contaminated by construction (the refiner trained on those frames) |
| `blend_ab_live_r31.json`, `blend_ab_live_r31_run2.json` | **§V-B arbitration A/B under motion** — two 600-frame drives, paired intervals, angular bands, temporal stability, two scene depths |
| `tape_stats_r31.json` | Tape provenance, Wilson coverage intervals, bootstrapped errors, out-of-cone re-sourcing |
| `rate_live_r31.json`, `rate_live_r31_run2.json` | Publish rate and map age, two 900 s runs |
| `age_budget_r31.json` | Sensor→arrival, node latency, total age, image staleness |
| `profile_node_r31.json` | Per-stage frame budget, 500 frames |
| `timing_pipeline_r31.json` | Table II deployed row, three configs, two resolutions |
| `depthor_small_timing_r31.json`, `depthor_large_timing_r31.json` | Table II denominators |
| `zjul5_deployed_r31.json`, `zjul5_deployed_blendnet_r31.json` | Table VI deployed rows by region |
| `zjul5_vits_r31.json` | Table VI ViT-S row |
| `zjul5_ridge_train_r31.json` | Table IV ridge sweep |
| `val_stems.txt`, `heldout_stems.txt` | The splits, for release under R5 |

**Tools**, all under `tools/diagnostics/`: `baselines.py`, `bootstrap.py`, `tape_stats_r31.py`,
`envinfo.py`, `age_budget.py`, `plane_eval.py`, `sigma_ablation.py`, `tape_repeats.py`,
`marker_view.py`, `rate_live.py`, `profile_node.py`, `time_pipeline.py`, `blend_ab_live.py`.

---

## 9. Document history

| Date | Change |
|---|---|
| 2026-09-13 | Created. Phases 0–2 complete (desk work, offline compute, bench session). Consultant's ten-item list cross-checked; item 2 (Table V on one split) closed. Outstanding: marker session, plane captures, driving session, platform top speed. |
| 2026-09-14 | Completed §3.11b with the **full metric set for both drives** (p95, RMSE, AbsRel, bias, δ1/δ2/δ3 alongside MAE/medAE) — every metric improves under the blend in both runs, including the p95 tail. Console logs preserved under `docs/demo/benchmarks/logs/`; `blend_ab_live` fixed to persist the far-field block it previously only printed. |
| 2026-09-14 | **Platform top speed deferred by author decision.** Not required — §V-A's definition holds on both rate halves without it. Consequence if never measured: the travel-distance sentence is omitted from §V-A and the real-time claim rests on the two rate comparisons, which is sufficient. |
| 2026-09-14 | Corrected the framing of platform top speed: it is **not** part of §V-A's real-time definition, both halves of which are rate-versus-rate and already satisfied (9.4 Hz publish vs 8.3 Hz sensor and 3.0 Hz planner). It supports the separate travel-distance argument that bounds the 146.9 ms age, and is optional pending an author decision. |
| 2026-09-13 | Recorded a scene requirement for any future arbitration drive: a clear sightline **over 6.5 m (the dToF gate), ideally 10 m+**. Both drives reached only 4.56 m, so the beyond-range regime remains unmeasured under motion; logged as optional item 3b. |
| 2026-09-13 | Added the effect-size interpretation to §3.11b: the drive's three nearest angular bands are empty by construction (the 16×16 anchor island spans ~0–10°), so its −7 % MAE understates the component; scored where arbitration can act it is worth 2.6–3.2×. Cost/benefit tension (~25 % of frame budget) recorded as summary item 14. |
| 2026-09-13 | **Second driving A/B** on a long sightline (median max depth 4.56 m vs 2.49 m). Blend again wins every metric, p = 0.000. The MAE benefit is **60 % larger in the deeper scene** (−0.0134 vs −0.0084) with non-overlapping intervals, while the median benefit is unchanged — between-run variation lands on effect size, not direction. The 30–90° convergence reproduces exactly. Engine SHAs recorded in run 2. |
| 2026-09-13 | **Driving A/B captured** (§3.11b) — 600 frames, 415,368 held-out zones. Blend beats no-blend on MAE (−0.0084 [−0.0093, −0.0075], p = 0.000) and medAE (−0.0011, p = 0.000), arms converge by 30–90° as they should. Scene was short-range (median max depth 2.49 m), so magnitudes are below the published longer-range figures while direction and significance reproduce. `blend_ab_live` fixed to record engine SHAs. |
| 2026-09-13 | Added §3.5b — the full 1234-frame ablation with both statistics and CIs for all three protocols, rows marked where contaminated. |
| 2026-09-13 | Full 1234-frame ablation re-run after a clamp-ordering fix in the arbitration rows (stage `7b_clamp` precedes `7c_blend`, so every arbitration row must blend clamped depth; the unclamped version showed medAE 0.0504 with MAE 10.95 m on `center`; corrected to **medAE 0.0504 / MAE 0.2953 m**, recorded in §3.5b). §3.11 (anchor weighting) completed on the full split — `w ∝ z` trades median accuracy for tail accuracy and only in the extrapolation regime, so the earlier single-statistic "refuted" reading is withdrawn. Table III bands, the `B4_affine` clamp artefact (MAE 11.157 m) and `B2_bilinear` coverage 0.00 all reproduce. Held-out point count made consistent at **seven** throughout. |
