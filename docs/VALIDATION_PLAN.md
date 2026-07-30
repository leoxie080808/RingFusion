# Validation plan — breaking the closed loop

**Status: workstreams 1–4 EXECUTED 2026-07-28/30. Created 2026-07-28.**

> This document is the plan as written *before* execution, kept for the reasoning. Several of
> its premises were overturned by carrying it out, and it should be read against the results
> rather than as current fact:
> * `0.199 m` is used throughout as the reference number. It was later found to be
>   **in-sample** — the on-robot harnesses fit and score on the same ToF zones.
> * Baselines, ZJU-L5 and DEPTHOR are **done**; the tape-measure session is **not**.
> * The checkerboard proposal in workstream 2 was **rejected** (too small at range).
> * `B5` here refers to `residual_v3_last`; the recommended engine is now `residual_v4_last`.
>
> Current results live in
> [`ros2_ws/README.md`](../ros2_ws/README.md#benchmarks-vs-trivial-baselines).

Every depth number RingFusion currently reports is **scored against held-out ToF zones —
the same sensor the pipeline anchors to**. That is a closed loop. It cannot detect a
systematic ToF bias, it cannot say anything about the 92.5 % of the frame the ToF never
sees, and `0.199 m` has no reference point, so no reader can tell whether it is excellent
or poor.

Four workstreams fix that. They are ordered below by *value per hour*, not by ambition.

| # | Workstream | Breaks the loop? | New hardware/data | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **Trivial baselines** | no | none | ~1 day | very low |
| 2 | **Independent GT (tape / board)** | **yes** | none, but robot + room time | ~1 day | low |
| 3 | **DEPTHOR-Small on the Orin** | no | model weights | ~2 days | **high** |
| 4 | **ZJU-L5 public benchmark** | partly | ~large download | ~2–3 days | medium |

---

## Metric contract — do this before anything else

Every harness must emit the **same** metric block, or none of the four workstreams are
comparable to each other and workstreams 3–4 are not comparable to the literature.

| Metric | Definition | Why |
|---|---|---|
| `MAE` | `mean(\|d − gt\|)`, metres | What we report today. Intuitive, scale-dependent. |
| `RMSE` | `sqrt(mean((d − gt)²))` | Punishes the far-field blow-ups we specifically fixed. |
| `AbsRel` | `mean(\|d − gt\| / gt)` | **The literature standard.** Without it, workstreams 3–4 produce numbers no paper can be compared against. |
| `δ<1.25`, `δ<1.25²` | `mean(max(d/gt, gt/d) < τ)` | The other literature standard. |
| `coverage` | fraction of eval points with a finite prediction | A method that predicts nothing everywhere would otherwise score perfectly. |
| `corr(σ, \|e\|)` | as in `sigma_cal.py` | Ours; only meaningful for methods with a variance head. |

`AbsRel` and `δ` already exist in [`training/eval_student.py:82-84`](../training/eval_student.py#L82-L84)
but only for student-vs-teacher. **Lift them into a shared `metrics.py` and call it from
every harness.** This is a ~40-line change and it gates everything else.

> **Report `coverage` next to every row.** Baselines 1–2 below only produce depth *inside
> the ToF footprint*; scoring them on a footprint-only eval set and reporting a bare MAE
> would silently flatter them.

---

## Workstream 1 — Trivial baselines *(do this first)*

**Question it answers:** is the neural stack buying meaningful accuracy over naive
interpolation of the sensor we already have?

This is a genuine risk, not a formality. If bilinear upsampling scores `0.21 m`, the whole
architecture is buying 5 % and the project's framing has to change. It costs one day and
needs no new data.

### Methods

All run offline on the **1,234 existing paired logs** (`ros2_ws/data/real/{rgb,tof}`, already
mirror-migrated — `mirrored=True` in every `.npz`). All use the **identical anchor/hold-out
split** as training (`anchoring_bridge.build_real_supervision`, fixed seed) so every method
sees the same anchors and is scored at the same held-out zones.

| ID | Method | Uses camera? | Params |
|---|---|---|---|
| `B0` | **Global constant** — predict `median(anchor depths)` everywhere | no | 1 |
| `B1` | **Nearest-zone ToF** — each held-out zone takes its nearest *anchor* zone's depth | no | 0 |
| `B2` | **Bilinear ToF upsample** — linear interp over anchor zones in grid space | no | 0 |
| `B3` | **Mono + median scale** — `1/z = s·disp`, `s = median(inv_z / disp)` over anchors | yes | 1 |
| `B4` | **Closed-form affine** *(current, Network B off)* — `solve_robust`, 2-param | yes | 2 |
| `B5` | **Full RingFusion** *(`residual_v3_last`)* | yes | ~0.46 M |

`B0` is not in the peer's list; it is added because it is free and it is the true
zero-information floor. If `B1` does not comfortably beat `B0`, the eval set is degenerate.

### Two hold-out protocols — this is the important part

Random 25 % hold-out leaves an anchor roughly **one zone away** from every evaluation point.
That is trivially interpolable, and `B2` will look strong for reasons that have nothing to
do with the real deployment problem.

So run **both**:

| Protocol | Split | What it tests |
|---|---|---|
| `random` | random 25 % of valid zones (current training split) | Interpolation. `B2` should do well here. |
| `block` | hold out a contiguous block — the outer ring, and separately an 8×8 corner | **Extrapolation.** Anchors are far away. `B1`/`B2` should collapse; if `B4`/`B5` hold up, that is the argument for the whole architecture. |

The `block` protocol is the cheapest available proxy for "what happens outside the ToF
box", and it is the single highest-value item in this plan. It does not replace
workstream 2 — it is still ToF-scored — but it can be run today.

### Deliverable

`tools/diagnostics/baselines.py` → a 6×2 table (methods × protocols) with the full metric
block, plus `docs/demo/baselines.json`.

### Honest caveat to publish alongside it

`B1`/`B2` are **structurally incapable of predicting outside the ToF footprint**, and the
eval set is entirely inside it. Their numbers therefore describe 7.5 % of the frame. State
this in the table caption rather than letting a reader infer a like-for-like comparison.

---

## Workstream 2 — Independent ground truth *(do this second; schedule the room now)*

**Question it answers:** is the pipeline actually right, or only self-consistent with a ToF
that may itself be biased?

This is the only workstream that breaks the closed loop, and it is the long-lead item
because it needs physical access to the robot and a static room. **Compute-bound
workstream 1 and room-bound workstream 2 do not compete for the same resource — start
workstream 1 today and book the room in parallel.**

### Method: ~20 marked points on real scene geometry

**A checkerboard was considered and rejected.** The printed target is 9×6 at 25 mm ≈
22×15 cm, which at 4–5 m spans only ~20–30 px — it only yields dense ground truth at close
range, which is the range that needs it least. It also introduces a flat, high-contrast,
cooperative surface that is not representative of what the backbone sees in deployment.
`checkerboard_9x6_25mm.pdf` stays what it is: camera-intrinsics calibration only.

The problem the board was meant to solve is **correspondence** — after measuring "3.2 m to
that wall", you still have to know *which pixel* the measurement belongs to, and being 20 px
off onto another surface corrupts the ground truth silently.

Solve it directly instead: stick a **small visual marker** (printed tag or a bright card
square, a few cm) at each measured point on **real scene geometry** — wall, door frame, box
edge, chair back. That fixes correspondence, keeps the target representative, and scales to
any distance.

- **~20 points**, spread across 0.3–6 m.
- Deliberately place **at least half outside the ToF footprint**, including near the frame
  edges — that region is the whole reason for the exercise.
- Record the marker's pixel coordinate at capture time, not from memory afterwards.

### Gotchas that will silently ruin the session

1. **Tape gives slant range `r`; the pipeline predicts optical-axis depth `z`.**
   `z = r · cosθ`, where `θ` comes from the pixel coordinate and `K_rect`. Forgetting this
   produces a fake radial bias that grows toward the frame edge — exactly the shape of a
   real error. **Convert every measurement.**
2. **The tape origin is the camera optical centre**, which is inside the lens barrel, not
   the front face. Measure the housing-to-centre offset once and apply it to every point.
3. **A laser distance meter (±1–3 mm) beats a tape (±5 mm) and is far faster.** Use one if
   available.
4. **Freeze the scene and the robot.** Anything that moves between the measurement and the
   capture is unrecoverable error.
5. **Record the raw `/depth` float32**, not a colourised frame.

### Uncertainty budget

Tape ±5 mm ⊕ optical-centre offset ±10 mm ⊕ board pose ±5 mm ≈ **±12 mm**. Against a
`0.199 m` MAE the ground truth is ~16× tighter than the signal being measured — comfortably
sufficient. Publish this budget; a reviewer will ask.

### Deliverable

`tools/diagnostics/tape_eval.py` + `docs/demo/tape_gt.json`, reporting the metric block
**split inside vs outside the ToF footprint**. That split is the headline result of the
entire plan.

### What would count as a bad outcome

If in-footprint error matches the ToF-scored `0.199 m` but out-of-footprint error is
several times worse, that is a **real finding**, not a failure — it quantifies the limit
the READMEs currently describe only qualitatively. Plan to publish it either way.

---

## Workstream 3 — DEPTHOR-Small on the Orin

**Question it answers:** how does RingFusion compare to a current published method?

**Gate this before committing two days.** Spend 30 minutes *at the start of workstream 1*
confirming: (a) public weights exist, (b) the licence permits use, (c) the sparse-input
format can be built from a 32×32 ToF box. If any fails, drop to reporting the architectural
comparison qualitatively and move on.

### The domain-gap problem, stated honestly

DEPTHOR-family models expect **LiDAR-like scattered sparse depth**. Ours is a **dense
contiguous 32×32 block covering 7.5 % of the frame**. Feeding it our anchor set is
out-of-distribution for it in exactly the way ZJU-L5 is out-of-distribution for Network B.

That cuts both ways, so run it in both directions and say so:

| Run | Purpose |
|---|---|
| DEPTHOR on our data | Direct comparison — but flag the OOD input pattern. |
| DEPTHOR on ZJU-L5 | **Harness validation.** If we cannot reproduce its published number, our comparison is measuring our own bugs. |

That second row is the real reason workstream 4 exists.

### Also report

Latency and memory on the Orin, not just accuracy. A method that is 3× more accurate at
2 Hz is not competing with RingFusion at 13.7 Hz, and that is a legitimate part of the
comparison.

---

## Workstream 4 — ZJU-L5

**Question it answers:** what does RingFusion score on a public benchmark other people
have numbers for?

**Run the closed-form path only.** Network B was trained on 32×32 anchor geometry; ZJU-L5
uses an 8×8 VL53L5CX. The peer's own analysis concedes this, and it is a strength of the
architecture, not a dodge: **the closed-form path has no learned parameters, so it
transfers by construction.** That is a genuinely publishable framing.

### Work required

1. Download + adapter mapping their intrinsics/extrinsics and 8×8 zone grid into
   `geo.project_zone_to_pixel`. Reuse `anchoring_bridge.calib_from_yaml`'s structure.
2. Verify projection with `orient_full.py` **before scoring anything** — the mirror bug
   cost this project a week, and a new dataset is precisely where it recurs.
3. Score `B1`–`B4` from workstream 1 on it too, so the baselines travel with the method.
4. *Optional, separate experiment:* retrain Network B on the ZJU-L5 train split. Report it
   as a distinct row; do not blend it with the zero-shot analytic number.

### Expected outcome

The analytic path on 8×8 (64 zones vs our 1024) will be **worse in absolute terms**, and
that is fine and expected. The publishable claim is *transfer without retraining*, not a
leaderboard position.

---

## Recommended order, with rationale

1. **Metric contract** (~1 h) — gates everything.
2. **Workstream 1, baselines** (~1 day) — no hardware, and it can change the framing of
   everything downstream. Includes the `block` protocol.
3. **Workstream 3 feasibility gate** (~30 min, run in parallel with 2) — cheap, and its
   answer determines whether to budget two days later.
4. **Workstream 2, independent GT** (~1 day) — book the room during step 2. The only
   open-loop evidence in the plan; highest scientific value.
5. **Workstream 3, DEPTHOR** (~2 days) — only if the gate passed.
6. **Workstream 4, ZJU-L5** (~2–3 days) — last, and partly in service of validating 3.

**Steps 1–4 are the ones that must happen.** They cost about two days, need no new
hardware, and together they turn `0.199 m` from a free-floating number into a defended one.
Steps 5–6 are comparative polish and carry real risk of consuming a week on integration
rather than on findings.

## Where results go

- Numbers → new `## Benchmarks` section in [`ros2_ws/README.md`](../ros2_ws/README.md),
  with the closed-loop caveat retired once workstream 2 lands.
- Headline table → root [`README.md`](../README.md) status block.
- Raw artefacts → `docs/demo/*.json`.
- Harnesses → `tools/diagnostics/`, listed in its
  [`README`](../tools/diagnostics/README.md).
