# RingFusion revision: data collection brief

**For:** whoever is running the Jetson and the robot
**Paper file:** `ringfusion_r31.tex` (revision 31, nine pages)
**Code:** https://github.com/leoxie080808/RingFusion
**Status:** accepted with minor required changes. Nothing here changes a claim. It fills in numbers the paper currently marks as missing, and resolves three places where the paper and the code disagree.

---

## 1. Background: what changed and why we need this

The reviewer accepted the paper and asked for five things. Four of them need measurements:

1. A component ablation separating analytic anchoring, robust weighting, the residual refiner, arbitration, and uncertainty estimation, with accuracy reported inside and outside the dToF coverage area.
2. Confidence intervals or per-run variation on the statistics, and a clean split between calibration points and held-out points in the tape experiment.
3. Full documentation of the uncertainty calibration procedure and its constants.
4. Runtime comparisons that state resolution, included stages, precision, hardware settings, and the measurement procedure, and that explain the relationship between core latency, publish rate, and data age.

The paper has been rewritten to answer all of these. It now contains new tables and new paragraphs with the numbers left blank. Those blanks are the job.

### How the paper marks things

Compile with `\draftmodetrue` (the default in the file). Then:

| Macro | Colour | Meaning |
|---|---|---|
| `\pending{...}` | red | a number we do not have yet. **112 of these.** |
| `\revised{...}` | green | text changed in this revision round, for review |
| `\tentative{...}` | blue | measured but likely to move |

There are also 10 `% VERIFY` comments in the source marking open questions. Three are marked `VERIFY, BLOCKING`.

When a number comes in, replace the whole `\pending{$0.0XX$}` with `$0.055$` (keep the maths delimiters, drop the macro). When everything is filled, flip to `\draftmodefalse` and all colour disappears with no text change.

---

## 2. Priority 0: three things the code and the paper disagree about

Do these first. They are cheap, and two of them affect numbers that appear in the abstract.

### P0-1. The deployed publish rate and map age do not match the paper

The paper claims **9.4 Hz** and a **median map age of 128 ms**, in the abstract, the introduction, Section V-A, Table III and the conclusion.

`docs/demo/benchmarks/rate_live_on_verified.json` (labelled "blend+ROI ON, params verified") reports:

| Quantity | Verified capture | Paper claims |
|---|---|---|
| depth topic rate | 7.16 Hz | 9.4 Hz |
| depth inter-message median | 138.8 ms | 106 ms implied |
| `depth_latency.median_ms` | 426.5 ms | 128 ms |

Also note `rate_live_on_final.json` carries a label saying it was **mislabelled at capture** and actually measured a stale blend-and-ROI-**off** launch, and that it is superseded by the verified file. Do not use it.

**What to do:** find which capture the 9.4 Hz and 128 ms figures came from, and determine which configuration is the one we want to publish. Then re-run a clean rate capture on the current build and use that. If the honest deployed number is nearer 7 Hz with a several hundred millisecond age, the paper has to say so, and Section V-A's real-time definition has to be written around the real figure rather than the other way round. That is fine. What is not fine is shipping a number we cannot reproduce.

Note that `expect_hz: 9.9` in those JSON files is the script's default argument, not a measurement.

### P0-2. The paper says we use range weighting. The code says it is off.

Section IV-C of the paper currently states: *"We use $w_i \propto z_i$ and record the disagreement."*

`ros2_ws/src/ringfusion_perception/ringfusion_perception/anchoring.py` line 11 sets `RANGE_WEIGHT_P = 0.0`, and the docstring says p=1 was chosen at one point and then disabled, because re-scored on anchors within 2.5 m it was the worst option (9.7% and 10.4% against 7.5% and 7.8% for uniform), and that the geometric ROI weighting in `roi.py` replaced it and reached 6.0% and 6.1%.

So the deployed system uses **uniform anchor weights plus a geometric ROI weight**, not `w ∝ z`.

**What to do:** confirm that reading against the running build, then tell the author. The Method text and the ablation table both have to change to describe the ROI weighting rather than the range exponent. The existing sweep in the docstring is useful background but it is in relative-error and near/far-ratio terms on two scenes, not median absolute error under our two protocols, so the table rows still need a run.

### P0-3. Table VII's two columns are not comparable

The paper's uncertainty table shows n=11 and n=15 columns where the worst residual falls from 2.30σ to 1.71σ and 2σ coverage rises from 0.909 to 1.000. Neither is possible if the eleven are a subset of the fifteen, and 0.760 is not a reachable coverage fraction with 15 points.

`docs/demo/benchmarks/live4_sigma_n15_2026-08-05.json` explains it. The n=15 figures are **means over 5 repeats** with lo and hi bounds (`corr` mean 0.613, lo 0.586, hi 0.654; `cov1` mean 0.760, lo 0.733, hi 0.800; `worst` mean 1.705, lo 1.637, hi 1.753), while the n=11 figures sit in a `reference_n11` block as single-run point estimates from an earlier session.

**What to do:** re-report both columns the same way, as a mean over repeats with an interval, and say so in the caption. The file also contains an `excl_disputed_n13` block (corr 0.723), so find out what "disputed" means and whether those two points should be in or out. Whatever the answer, state it.

---

## 3. Do an inventory pass before running anything

A large fraction of the 112 placeholders may already be answered by files in `docs/demo/benchmarks/`. Go through these first and mark which placeholders they close, which are stale, and which need a re-run on the current build.

| File | Likely covers | Watch out for |
|---|---|---|
| `profile_node_on.json`, `profile_node_off.json`, `profile_node_on_gpuup.json`, `profile_node_on_roigpu.json` | Table III lower panel, per-stage frame budget | `frame_total_ms` is 140.1 and `pipeline_total_ms` is 120.5, neither of which is the 106 ms the paper's 9.4 Hz implies. See P0-1. |
| `rate_live_on_verified.json` | publish rate, percentiles, map age | See P0-1. Prefer this over `rate_live_on_final.json`, which is mislabelled. |
| `rate_live_v7.json`, `rate_live_fov73_v6.json`, `rate_live_stagesoff.json`, `rate_live_blendroi_clean.json` | earlier rate captures | identify which build each came from |
| `baselines*.json` (9 files) | Table V ablation, both protocols, `by_angle` bins | many variants (`pre_fov_fix`, `v7`, `valsplit`, `fov73_residualv4`). Only the current build counts. |
| `heldout_v4.json`, `heldout_v6.json`, `heldout_v7.json`, `heldout_stems.txt` | the 200 held-out distillation frames and hold-out scoring | confirm which stems list matches the re-distilled backbone |
| `ceiling_ours.json` | the 1.43 m ceiling, ridge sweep on our logs | |
| `depthor_small_regions.json`, `depthor_large_regions.json` | **the whole DEPTHOR-Small row of Table IX** | see section 7 below, the numbers are already there |
| `depthor_small_zjul5.json` | DEPTHOR-Small overall on ZJU-L5 | |
| `score_v*.json`, `blend_ab_live*.json`, `moving_ab` outputs | driving-frames arbitration comparison | |
| `live4_sigma_*.json` (5 files) | Table VII | see P0-3 |
| `fov_v_session_2026-08-04.json`, `blend_depth_equality_2026-08-04.json` | supporting checks | |

**Deliverable for this step:** a short table, one row per placeholder, saying `already have / stale, re-run / never measured`. That tells us how much robot time is actually needed.

---

## 4. Environment capture (do this once, takes five minutes)

Table III's note needs these exactly. Record them from the Jetson that produces the timing numbers:

- JetPack version (`cat /etc/nv_tegra_release` or `dpkg -l nvidia-jetpack`)
- TensorRT version used to build the engines (`dpkg -l | grep -i tensorrt`), and the `tools/build_engine.py` flags used
- CUDA and cuDNN versions
- Power mode (`nvpmodel -q`), confirm MAXN
- Whether `jetson_clocks` was enabled during the timed runs, yes or no
- Fan profile and ambient conditions if the board thermally throttles over a long run
- Git commit hash of the build under test

Fills: `\pending{X.Y}` for TensorRT and JetPack, and `\pending{enabled}` for `jetson_clocks`, in the Table III note.

---

## 5. On-robot tasks

### Task A. Per-stage frame budget and rate (Table III lower panel, Section V-A)

**Runs on:** the robot, deployed configuration, full resolution.

```
python3 tools/diagnostics/profile_node.py \
  --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
  --backbone-engine <path> --residual-engine <path> \
  --blend true --roi-enable true \
  --frames 500 --out docs/demo/benchmarks/profile_node_r31.json
```

The existing `profile_node_on.json` uses `n=50`. Raise it so the medians are stable, and capture with the GPU in the deployed state rather than idle.

The paper's table groups stages, so map the script's stage keys onto the five published rows:

| Paper row | Script stages to sum |
|---|---|
| camera branch: capture, rectify, backbone | `1_rectify`, `2_backbone`, plus capture and USB transport if the script does not already include them |
| dToF branch: projection and validity | `3_4_project_pair`, `4b_roi_plane` |
| fit, covariance and refiner | `5_fit_metric`, `7_residual` |
| source arbitration | `7c_blend`, `7b_clamp` |
| uncertainty terms | `6_variance` and any sigma work inside blend |
| unprojection, assembly and publication | remaining stages plus publish |

**Open question to resolve while doing this:** the paper says the uncertainty terms cost **10.6 ms**. `profile_node_on.json` reports `6_variance` at 6.40 ms. Find out whether 10.6 ms was `6_variance` plus part of the blend sigma work, or a different run. The 10.6 ms figure appears twice in the paper and both instances need to be right.

**Also needed:** total per published frame, and median age of a published map. These are the two bottom rows of the table and they are the same quantities as P0-1, so settle that first.

**Fills:** six stage rows, the total row, the rate percentiles in Section V-A, and the offline-versus-deployed speed ratio.

### Task B. Publish rate distribution (Section V-A)

```
python3 tools/diagnostics/rate_live.py --secs 900 \
  --label "r31 deployed, blend+ROI on" \
  --out docs/demo/benchmarks/rate_live_r31.json
```

Run it long. The paper wants a 5th to 95th percentile over a stated number of minutes of continuous operation, so give it at least fifteen minutes and record the actual duration. Verify the node parameters really are the deployed ones before starting, given what happened with `rate_live_on_final.json`.

**Fills:** `\pending{X.X}` to `\pending{X.X}` Hz, `\pending{XX}` minutes.

### Task C. Component ablation (Table V)

**Runs on:** the 1234 logged camera and dToF pairs, on the Jetson or a workstation, same code path.

```
python3 tools/diagnostics/baselines.py \
  --rgb-dir <logs>/rgb --tof-dir <logs>/tof \
  --calib <calib.yaml> \
  --backbone-engine <path> --residual-engine <path> \
  --protocol random center --island 16 \
  --out docs/demo/benchmarks/baselines_r31.json
```

The output already has the right shape: `protocols → methods → overall` plus `by_angle` bins, which maps directly onto the table's two protocol columns and five angular bands.

The table needs **eight rows**, and the weighting rows are the ones that do not exist yet:

| Row | Status |
|---|---|
| nearest-zone dToF, no camera | have |
| analytic fit, uniform weights | need, and it is the table's reference row |
| range weighting variant 1 | need, see P0-2, this row's identity depends on that answer |
| range weighting variant 2 | need |
| plus one robust reweighting pass | partly have (0.055 on both protocols), angular bands missing |
| plus refiner, scattered hold-out | need, **on this split** |
| plus refiner, island hold-out | have |
| plus source arbitration, deployed | have |

Two things to get right:

- **One baseline, one split.** The paper currently reports the analytic output as 0.055 m in the table and as 0.064 m in the scattered hold-out comparison, on different splits. Every row of this table has to come from one run on one split. If the 0.064 and 0.066 pair cannot be reproduced on that split, re-score it, and the surrounding paragraph gets rewritten.
- **Robust pass rejection rate.** The paper says the robust pass "removes `\pending{X.X}%` of anchors on a typical frame". Instrument `solve_robust` to report it.

**Fills:** roughly 40 cells in Table V, plus three numbers in the weighting paragraph in Section V-B.

### Task D. Driving-frames arbitration comparison, per session (Section V-B)

```
python3 tools/diagnostics/moving_ab.py \
  --calib <calib.yaml> --backbone-engine <path> \
  --engine-v1 <arbitration off> --engine-v2 <arbitration on> \
  --secs 120 --hz 6 --out docs/demo/benchmarks/moving_ab_r31.json
```

The paper currently pools two sessions and caveats that route and speed differ between them. The reviewer asked for per-run variation, so report MAE for each session separately, both variants.

**Fills:** four numbers, `\pending{$0.XXX$}` against `\pending{$0.XXX$}` m on the first session and the same on the second.

### Task E. Fallback trigger documentation (Section IV-C)

**No measurement, mostly a code read, plus one log pass.**

The paper now documents the fallback explicitly and needs four values:

| Symbol | What it is | Where to look |
|---|---|---|
| `N_min` | minimum surviving anchors before falling back | `pipeline.py` and `anchoring.py` |
| `v_min` | minimum weighted variance of `disp` | same |
| `κ_max` | condition number bound | `baselines.py` uses `--max-cond` default `1e8`, confirm the node uses the same |
| fire rate | percent of frames where the fallback triggers | count over the 1234 logged pairs and the driving logs |

Also confirm the sentence "`b̂` is held at `\pending{its previous value}`" is what the code actually does, rather than resetting to zero.

While here, check whether the published message exposes a fit-conditioning or fallback flag. The fourth limitation in the Discussion says it does not. If it now does, that limitation gets rewritten as a feature, which is a free win.

### Task F. Tape reference, split by role (Section V-C and Table VII)

**Runs on:** the robot, with the existing tape captures.

```
python3 tools/diagnostics/tape_eval.py --dir <tape capture dir> \
  --calib <calib.yaml> --out docs/demo/benchmarks/tape_r31.json
```

What the paper needs, and the reason each is needed:

1. **Which eleven points are the calibration set** and which four are held out. `blend.py`'s comment header says the sigma constants were tuned against 11 tape points in LIVE-4, so the provenance exists. Write the list down.
2. **Whether the four out-of-cone points are the same four as the held-out set.** The paper has a `\pending{are / are not}` waiting on this. If they are the same four, that is a confound worth stating plainly, because the held-out column would then be measuring extrapolation and calibration transfer at once.
3. **Depth error over all fifteen points**, median and MAE, and split by coverage: eleven inside the cone against four outside. The outside figure is the only direct out-of-cone accuracy measurement we have on our own hardware, so it carries weight in Section V-B.
4. **A 95% bootstrap interval on the seven-marker median** currently quoted as 0.044 m.
5. **The out-of-cone error and sigma pair.** The paper quotes errors of 1.07 and 1.52 m against sigma of 2.10 and 1.84. There is a `VERIFY, BLOCKING` note saying this pair could not be reproduced from a saved capture. Either find the run it came from or replace it with a current out-of-cone pair.
6. **Table VII's three columns** recomputed consistently, see P0-3. All three columns as means over repeats with intervals, one set of constants, one stated denominator.

### Task G. Bootstrap intervals (Section V preamble)

The paper now promises: 95% confidence intervals from `\pending{1000}` bootstrap resamples, **resampled over frames rather than pixels**, and timing as the median of `\pending{500}` iterations after `\pending{100}` warm-up.

Confirm or change those three counts to whatever is actually run, then produce intervals for:

- the headline ablation numbers, the table note promises a worst-case interval of at most `\pending{±0.00X}` m
- the ZJU-L5 AbsRel and delta-1 figures
- the tape median

Frame-level resampling matters. Pixel-level bootstrap on depth maps will give intervals that are far too tight because errors within a frame are strongly correlated.

---

## 6. Uncertainty constants: mostly already answered

The paper's Equation (11) and the sentence after it need five constants. Reading `blend.py` at HEAD, they appear to be:

```
DISAGREE_K   = 1.0     # sigma in metres per metre the blend moves the depth
SUPPORT_FRAC = 0.35    # sigma as a fraction of D per FAR_DEG of missing angular support
SPREAD_K     = 1.0     # sigma in metres per metre of disagreement between nearby zones
NEAR_DEG     = 2.0
FAR_DEG      = 5.0
```

and `sigma_support_var` combines them as `var_r = disagree**2 + support**2 + spread**2`, added to the analytic and learned variance. That is exactly the form Equation (11) assumes, so the mapping is:

| Paper symbol | Value | Note |
|---|---|---|
| `c_a` | 1.00 | `DISAGREE_K` squared, dimensionless |
| `c_ν` | 1.00 | `SPREAD_K` squared, dimensionless |
| `c_α` | 0.1225 | `SUPPORT_FRAC` squared, dimensionless |
| `α_0` | 5.0 deg | `FAR_DEG` |
| `α_max` | check | the paper calls this "where the 100% floor takes over". With `SUPPORT_FRAC = 0.35` the support term reaches `D` at about 14.3 degrees of missing support, so confirm whether the floor is a separate clamp or emerges from this term |

**Please verify these against the running build rather than trusting my read of HEAD**, then confirm the fitting objective. The paper currently says the constants were chosen "by `\pending{minimizing the Gaussian negative log likelihood of the tape residuals}`". If they were hand-tuned against coverage instead, say that. It is a documentation requirement, not a quality judgement, and an honest "tuned by hand against 11 points to keep coverage conservative" is perfectly publishable.

Also confirm `NEAR_DEG = 2.0` and `FAR_DEG = 5.0` still match the smoothstep range quoted in Section IV-E.

One more to check: `SCENE_CAP_K = 2.0` and `SCENE_CAP_FLOOR_M = 1.0` implement a cap at a multiple of the furthest anchor. Section V-D says that remedy is "monotonically worse than the fixed ceiling". The code comment says the cap is fed to the variance channel as a bound rather than used as an estimate. Make sure the paper's sentence is not read as saying we do not do it at all.

---

## 7. Offline ZJU-L5 work (can run anywhere, but the DEPTHOR timings must be on our Orin)

### Table IX, the footprint split

`docs/demo/benchmarks/depthor_small_regions.json` already contains the DEPTHOR-Small row:

| Region | AbsRel | RMSE | delta-1 |
|---|---|---|---|
| all | 0.0813 | 0.7341 | 0.9201 |
| inside | 0.0501 | 0.3274 | 0.9733 |
| outside | (in file) | 1.0269 | (in file) |

Pull the remaining outside-region fields and the DEPTHOR-Large equivalents from `depthor_large_regions.json`.

**A discrepancy to resolve while you are in there.** Section V-F says re-scoring DEPTHOR through our masks "reproduces their published figures to within 0.002 to 0.004". That holds for AbsRel (0.0813 against a published 0.079) and delta-1 (0.9201 against 0.923), but the re-scored RMSE is 0.734 against a published 0.371. The sentence needs to be scoped to the metrics it is true of, or the RMSE difference needs explaining.

Then produce our own rows:

```
python3 tools/diagnostics/zjul5_eval.py --root <zjul5 root> \
  --backbone-engine <ViT-S path> --split test \
  --out docs/demo/benchmarks/zjul5_vits_r31.json

python3 tools/diagnostics/zjul5_eval.py --root <zjul5 root> \
  --backbone-engine <deployed distilled engine> --residual-engine <path> \
  --split test --out docs/demo/benchmarks/zjul5_deployed_r31.json
```

**Fills:** the DEPTHOR-Small row of Table IX (six cells), the two missing cells on the ViT-S row, the whole deployed row (six cells), and the footprint percentage in the note.

### Ridge selection

Section V-D says the ridge strength of 0.003 was selected on the ZJU-L5 **train** split and applied unchanged at test. `zjul5_eval.py` takes `--split train` and `--sweep-bprior`, so reproduce that sweep and keep the output, since the abstract now makes this claim explicitly and a reviewer may ask.

---

## 8. Complete placeholder checklist

Grouped by where it lives in `ringfusion_r31.tex`. Line numbers are approximate and will drift as numbers go in.

| Section | Placeholders | Task |
|---|---|---|
| IV-C fallback | `N_min`, `v_min`, `κ_max`, held-at value, fire rate | E |
| IV-F uncertainty | `c_a`, `c_α`, `c_ν`, `α_0`, `α_max`, fitting objective | 6 |
| V preamble | bootstrap count, timed iterations, warm-up count | G |
| V-A real time | planner rate, platform top speed, travel distance | E, plus a decision from the author |
| V-A rate | 5th and 95th percentile, duration, offline speed ratio | B |
| Table III note | TensorRT version, JetPack version, clocks state, iteration counts | 4 |
| Table III lower panel | six stage rows, total per frame | A |
| Table V | roughly 40 cells across eight rows | C |
| Table V note | worst-case bootstrap interval | G |
| V-B weighting | three medians, robust pass rejection rate | C, P0-2 |
| V-B driving | four per-session MAE values | D |
| V-C tape | are/are not, bootstrap interval, 15-point median and MAE, in-cone and out-of-cone medians | F |
| Table VII | held-out column (4 cells), all-points coverage (2 cells), worst residuals (2 cells) | F, P0-3 |
| Table VII note | coverage denominator and unit | P0-3 |
| V-E | out-of-cone error and sigma pair (4 values), worst residual | F |
| Table IX | DEPTHOR-Small row (6), ViT-S outside (2), deployed row (6) | 7 |
| Table IX note | footprint percentage on ZJU-L5 | 7 |

---

## 9. Output format

For each task, please produce:

1. The raw JSON in `docs/demo/benchmarks/` with an `r31` suffix and a `label` field saying what build and configuration it came from. The mislabelled `rate_live_on_final.json` is the reason this matters.
2. A one-line note per number saying which file it came from, so the author can fill the `\pending{}` slots without guessing and so we can answer a reviewer who asks.
3. For anything that could not be reproduced, say so plainly rather than substituting a nearby number. Three of the paper's current problems are exactly that.

---

## 10. What not to do

- Do not re-run anything the inventory pass in section 3 shows we already have on the current build. Robot time is the scarce thing here.
- Do not fill a placeholder from a run on a different build than the one the neighbouring numbers came from. A mixed table is worse than a table with a gap.
- Do not tune anything to make a number look better. Every figure in this paper is either measured or marked as pending, and the reviewer accepted it on that basis.
- Do not change `blend.py` or `anchoring.py` constants while collecting. If a constant needs changing, that is a separate decision and every dependent number has to be re-run.

---

## 11. Two things only the author can decide

Flag these back rather than guessing:

1. **The real-time definition.** Section V-A needs the planner rate and the platform's top speed. It defines real time as publishing at least as fast as the sensor produces data and as the planner replans. If P0-1 lands well below 9.4 Hz, this definition may need rewriting.
2. **Whether we publish a repository URL.** The paper currently carries a first-page footnote pointing at the GitHub repo and a paragraph listing ten artifact categories. If the release will not contain all ten by camera-ready, the footnote should be shortened or dropped. A link to a partial repo is worse than no link.
