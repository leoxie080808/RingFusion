# RingFusion Revision: Open Items and Tests

Sep 27, 2026 · @Leo

Paper r33 answers most ICRAE reviewer comments in its text, but its red placeholders still need values from the code, the logs and nine tests. Please fill the Value and Answer columns below directly. Run T1 first, because it can change the 3.1× speedup claimed in the abstract.

## 1. Values to pull from the code and config

These 30 items need no new experiment, only the deployed config, the source code or the training scripts. They are listed in the order they appear in the paper.

| # | Item | Symbol | What we need | Paper location | Value |
| --- | --- | --- | --- | --- | --- |
| 1 | Validity filter | none | Rule that rejects a zone: confidence threshold, range limits, other flags | IV-B |  |
| 2 | Robust reweighting | none | Weight function (Huber, Tukey, other), its threshold, and whether it down-weights or fully rejects anchors | IV-C, V-B |  |
| 3 | Fallback: minimum anchors | `N_min` | Minimum surviving anchors before the shift is held | IV-C |  |
| 4 | Fallback: variance test | `v_min` or `kappa_max` | Which test the code runs (weighted variance of the predicted disparity, or condition number of the normal matrix) and its threshold | IV-C |  |
| 5 | Fallback: edge cases | none | Value the shift is held at; behavior at startup (no previous shift), with zero anchors, with a stale dToF map, and when the fitted scale is zero or negative | IV-C |  |
| 6 | Arbitration limits | 2 and 5 degrees | Confirm both angles and the smoothstep profile | IV-E |  |
| 7 | Residual variance | `sigma_hat^2` | Weighted residual variance or unweighted; degrees of freedom (n minus 2?); computed before or after the robust pass | IV-F |  |
| 8 | Neighbor spread | `nu(p)` | Standard deviation or range; 4 or 8 neighboring zones; around which anchor | IV-F |  |
| 9 | Disagreement weight | `c_a` | Constant in Eq. 8 | IV-F |  |
| 10 | Angle weight | `c_alpha` | Constant in Eq. 8 | IV-F |  |
| 11 | Spread weight | `c_nu` | Constant in Eq. 8 | IV-F |  |
| 12 | Angle normalizer | `alpha_0` | Degrees | IV-F |  |
| 13 | Floor cutoff | `alpha_max` | Degrees from the nearest anchor beyond which variance is floored at depth squared | IV-F |  |
| 14 | How items 9 to 13 were fitted | none | Objective (Gaussian negative log-likelihood?), optimizer, which 11 tape points, which captures | IV-F |  |
| 15 | Eq. 8 as coded | `V_t` | Confirm the code computes V\_fit + tau^2 + c\_a Delta^2 + c\_alpha (alpha/alpha\_0)^2 D^2 + c\_nu nu^2, floored at D^2. If not, paste the code line | IV-F |  |
| 16 | Resolution chain | none | Pixel size at capture, after rectification, backbone input, backbone output, refiner, and the unprojection stride. Which one is "full resolution" | Table I, V-A |  |
| 17 | Software versions | none | JetPack, CUDA, cuDNN, TensorRT | Table I note |  |
| 18 | Orin configuration | none | 32 GB or 64 GB module; nvpmodel mode; jetson\_clocks on or off | Table I note |  |
| 19 | DEPTHOR timing setup | none | Runtime (PyTorch or TensorRT) and precision (FP32 or FP16) behind the 79.4 ms and 183.8 ms rows | Table I |  |
| 20 | Input queue | none | Confirm the queue is bounded to one frame | V-A |  |
| 21 | Data-age timestamp | none | Which stamp data age starts from (camera driver capture time?) | V-A |  |
| 22 | Clock alignment | none | How ESP32 hardware time is mapped to Jetson time (offset estimate, drift correction) | V-A |  |
| 23 | Frame pairing | none | How a camera frame picks its dToF map (latest complete map, nearest stamp?); whether 16 Hz partial subframes are used | V-A |  |
| 24 | Planner rate and top speed | none | Planner rate in Hz and platform top speed in m/s | V-A |  |
| 25 | Refiner training data | none | Logs used, frame count, hold-out fraction, epochs, optimizer, learning rate; confirm no overlap with any evaluation set | V intro |  |
| 26 | Distillation check | rho = 0.996 | Which frames and which model this correlation was measured on | V intro |  |
| 27 | Camera offset | x\_c, y\_c, z\_c | Position of the camera optical center relative to the robot center: forward, sideways and height, in mm (CAD) | V-C, T3 |  |
| 28 | Optical center in the lens | none | Distance from the lens front glass back to its optical center, in mm (lens datasheet or calibration) | V-C, T3 |  |
| 29 | Camera pitch and yaw | theta, psi | Camera orientation relative to the robot body, in degrees, from the calibrated extrinsic or CAD | V-C, T3 |  |
| 30 | dToF center offset | none | Position of the dToF sensor center relative to the robot center, in mm, for the raw dToF check in T3 | V-C, T3 |  |

## 2. Facts to confirm

Each question checks how an existing result was measured. The paper currently states the assumption in the Paper location column.

| # | Question | Paper location | Answer |
| --- | --- | --- | --- |
| 1 | Are the four held-out tape points the same four that lie outside the dToF cone? | V-C |  |
| 2 | What counts as one tape capture: a single frame, or the median over a burst of N frames? | V-C |  |
| 3 | Was tape depth measured as axial depth or as slant range, and from which reference point on the module? If slant range was compared against the pipeline's axial depth, off-axis errors are inflated: at 38.9 degrees, r is about 1.29 times Z | V-C |  |
| 4 | Were the 8% AbsRel and 5% RMSE ridge gains measured on the training split? At test, AbsRel only moves 0.094 to 0.091 | V-D |  |
| 5 | Is the 12.1 m underestimate beyond 10 m on ZJU-L5 the mean signed error of those pixels? | V-D |  |
| 6 | Were the CFPNet and PENet rows in Table V re-scored through our masks, or copied from their papers? | Table V note |  |
| 7 | In the Table I note, does "the last row includes capture latency" refer to the 128 ms data-age row? | Table I note |  |
| 8 | Is the 25 degree extent of the center protocol measured from the optical axis or from the island edge? | V intro, V-B |  |
| 9 | What is the largest difference between our re-score of DEPTHOR-Small and its published numbers: 0.004 or 0.008? | V-F |  |
| 10 | What did "world environment path" mean in the old distillation description? | V intro |  |
| 11 | Which rows of Table II used the 200 held-out distillation frames? | Table II caption |  |

## 3. Blocking issues

Two uncertainty results cannot ship as they stand. Reprocessing the saved tape captures in test T3 fixes both.

1. **Table IV is internally inconsistent.** The eleven calibration points are a subset of the fifteen, so the worst residual over fifteen cannot be smaller than over eleven, and 2-sigma coverage cannot rise from 0.909 to 1.000. The old coverage value 0.760 is also impossible with 15 points. Recompute all three columns (calibration, held out, all) under one set of constants and state whether coverage counts points or point captures.
2. **The out-of-cone example cannot be reproduced.** V-E reports errors of 1.07 and 1.52 m with sigma of 2.10 and 1.84, but no saved capture gives these numbers. Re-derive them from the original run, or replace them with fresh values for all four out-of-cone points. First check whether they compared tape slant range with pipeline axial depth, since at these angles that mismatch alone adds tens of centimeters.

## 4. Tests to run

Nine tests remain: two on the Jetson bench, two on the robot, four offline on existing logs, and one training run. Start T9 first because it runs in the background, and finish T1 before anything else that cites the speedup.

| Test | Where | Reviewer comment | Fills in the paper | When |
| --- | --- | --- | --- | --- |
| T1. DEPTHOR at matched precision | Jetson bench | 5 | Table I, abstract, V-A | First |
| T2. Offline core timing and resolution | Jetson bench | 5 | Table I, V-A | With T1 |
| T3. Tape reference | Robot | 3, 4 | V-C, V-E, Table IV, new tape figure | After constants are frozen |
| T4. In-node latency and data age | Robot | 5, 6 | Table I lower panel, V-A | With T3 |
| T5. Ablation rows | Offline, 1234 pairs | 2, 3 | Table II, V-B | Anytime |
| T6. Fallback frequency | Offline, all logs | 6 | IV-C | Anytime |
| T7. Driving per session | Offline, 600 frames | 3 | V-B | Anytime |
| T8. ZJU-L5 footprint and intervals | Offline | 2, 3 | Tables III and V, V-F | Anytime |
| T9. Refiner seeds | Training GPU | 3 | Table II, V-B, contributions | Start now |

### Rules for every test

- Freeze first: record the git commit, both TensorRT engine hashes and every constant from section 1. Nothing is retuned after new data is seen.
- Log raw camera frames and raw dToF packets with hardware timestamps, so any number can be recomputed later.
- Confidence intervals use 1000 bootstrap resamples and a 95% percentile interval. Resample frames for logs and ZJU-L5, points for the tape test, and never pixels.
- A difference between two methods on the same data gets a paired bootstrap on the per-frame difference.
- Timing uses 100 warm-up and 500 timed iterations, reported as median, 5th and 95th percentile.

### T1. DEPTHOR at matched runtime and precision

1. Set MAXN (`sudo nvpmodel -m 0`), run `sudo jetson_clocks`, and log `tegrastats` throughout.
2. Export DEPTHOR-Small and DEPTHOR-Large to ONNX at 480×640, batch 1, and build FP16 engines with `trtexec --fp16`.
3. Check the engines: AbsRel on 50 ZJU-L5 frames must stay within 0.002 of the FP32 PyTorch model.
4. Time them with our harness, and re-time our two rows in the same session.
5. If DEPTHOR will not convert, time both DEPTHOR and our core stages in PyTorch FP16 instead.
6. Re-time the original setup behind 79.4 ms as well, so any change can be explained.

Output: runtime, precision, median, p5, p95 and rate for every row.

### T2. Offline core timing and resolution

1. Record the tensor size at every stage (item 16 in section 1).
2. Time the full perception core, uncertainty terms included, on 500 recorded frames at deployed resolution without ROS.
3. Report the ratio of the in-node frame period (from T4) to this offline median.

### T3. Tape reference

The tape reference is the only independent depth check on our own hardware, so its geometry must be recorded exactly. The pipeline outputs camera-frame axial depth Z, the distance along the camera's forward axis, so every tape point is converted to Z before it is compared.

**Before the session**

1. Reprocess the raw logs of sessions 1 and 2 if they still exist. First confirm whether their tape distances were straight-line range or axial depth (section 2, question 3).
2. Freeze the uncertainty constants and tag the commit. Nothing is refitted on session 3.
3. Collect the geometry from section 1: camera offset (item 27), optical center depth in the lens (item 28), camera pitch and yaw (item 29) and dToF center offset (item 30).

**Setting up each point**

1. Mark the robot center on the floor, and mark the forward axis with a second floor point at least 1 m ahead of it.
2. Place the marker, a printed cross or AprilTag, so its center is unambiguous in the image. For the thin pole, the target must sit on the pole itself.
3. With a steel tape with 1 mm graduations, measure three legs from the robot center: forward along the axis (x\_m), sideways and square to the axis with right positive (y\_m), and the height of the marker center above the floor (z\_m).
4. Keep the tape flat on the floor, use a square for the sideways leg, and use a level or plumb line for the height.
5. Do not touch the robot between measuring and capturing.

Session 3 needs at least 12 new held-out points: 3 in the cone at 1.5 to 4 m, 2 in the cone beyond 4.2 m, 3 outside the cone, 2 straddling the cone edge, and 2 on depth discontinuities including a thin pole.

**Capture**

1. Warm up the Jetson and the dToF for 10 minutes, and keep the lighting constant.
2. Take 5 captures per point, at least 2 minutes apart.
3. Log per capture: raw frames and dToF packets, the marker pixel (u, v), the published depth and variance at that pixel, each variance term separately, the arbitration weight, angle to the nearest anchor, anchor count, condition number, fallback flag, and the raw reading of the dToF zone covering the marker.

**Converting tape legs to depth**

With the camera offset (x\_c, y\_c, z\_c) and pitch theta:

```latex
\begin{aligned}
d &= x_m - x_c, \quad y = y_m - y_c, \quad h = z_c - z_m \\
Z &= d\cos\theta + h\sin\theta \\
r &= \sqrt{d^2 + y^2 + h^2}
\end{aligned}
```

Compare the pipeline against Z. For a level camera, Z = d. The slant range r is only a cross-check and never the reference. Camera yaw also enters Z for sideways points, since 1 degree of yaw shifts Z by about 17 mm at 1 m sideways, so use the calibrated yaw (item 29) for off-axis and out-of-cone points.

Example: the optical center sits 0.120 m ahead of the robot center and 0.250 m above the floor, and the marker is taped at 2.120 m forward, 0 m sideways and 0.100 m high. Then d = 2.000 m, h = 0.150 m, Z = 2.000 m and r = 2.006 m, a gap of 5.6 mm. Moving the marker 0.8 m sideways leaves Z at 2.000 m but raises r to 2.159 m.

**Reference accuracy**

Report the reference as an absolute uncertainty in millimeters (1 sigma), and optionally as a percentage of range. The 1 mm graduation is the resolution; setup errors set the accuracy. Typical budget for one point at about 2 m:

| Source | Typical 1 sigma (mm) |
| --- | --- |
| Tape reading, 1 mm graduations | 0.3 |
| Locating the robot center and forward axis | 1 |
| CAD offset to the optical center | 1 |
| Optical center position inside the lens | 2 |
| Locating the marker center | 1 |
| Combined, root sum of squares | 2.7 |

Expect about 3 to 4 mm at 4 m. Pipeline errors are around 40 mm, so the reference is more than ten times smaller than what it measures, which is the usual requirement.

**Raw dToF check**

For each in-cone point, compare the raw dToF reading with the tape range measured from the dToF center (item 30) along that zone's ray. This measures the error of the sensor reference behind Table II.

**Suggested log, one row per capture (lengths in mm)**

```csv
point_id,session,capture,set,cone_status,x_m,y_m,z_m,x_c,y_c,z_c,pitch_deg,yaw_deg,u_px,v_px,Z_ref,depth_pred,sigma_pred,dtof_raw,V_fit,tau2,term_arb,term_angle,term_nu,floor_applied,omega,anchor_count,cond_number,fallback
```

**Output**

- Median and MAE inside and outside the cone
- Table IV for the calibration, held-out and all points
- The uncertainty ablation: V\_fit alone, then adding tau^2, the three terms, and the floor
- Raw dToF error at the in-cone points
- Per-point data for the tape figure: reference Z, pipeline error, spread over the 5 captures, and predicted sigma

The paper reports Z and the pipeline error per point, plus one sentence on the convention and the reference uncertainty. The release publishes the raw legs, offsets, pitch and yaw, so anyone can rebuild Z.

### T4. In-node latency and data age

1. Timestamp every stage (CUDA events on the GPU, monotonic clock on the CPU) without adding synchronization.
2. For each published frame, log the capture stamp, the dToF map completion stamp and map ID, each stage start and end, and the publish time.
3. Run 10 minutes stationary and 10 minutes driving, with `tegrastats` in parallel.

Output: per-stage median, p5 and p95; publish-rate p5 and p95; data age and anchor age, median and p95; fraction of frames that reuse a dToF map; queue drops.

### T5. Table II ablation

1. On the 1234 pairs and the same hold-out masks, run uniform weights, w ∝ z², w ∝ z, the robust pass, the scattered refiner, the island refiner and arbitration.
2. Score both protocols and every angle band, with bootstrap intervals.
3. Compute paired intervals for robust pass vs w ∝ z, island refiner vs analytic fit, arbitration vs island refiner, and nearest zone vs arbitration.
4. Record the fraction of anchors the robust pass down-weights or rejects.

### T6. Fallback frequency

1. Over all logs, record per frame the anchor count, weighted variance of the predicted disparity, condition number, fitted scale and shift, and whether the shift was held.
2. Report the fire rate per trigger, and count frames with zero anchors, a stale map, or a fitted scale at or below zero.

### T7. Driving comparison per session

1. Split the 600 driving frames by recording session.
2. For each session, compute MAE with and without arbitration and the 15 to 30 degree median, with a paired bootstrap on the difference.

### T8. ZJU-L5 footprint and intervals

1. Project the 8×8 zone grid into the image and report the footprint as a percentage of valid pixels.
2. Score DEPTHOR-Small, ours with ViT-S (affine and ridge) and ours as deployed, inside the footprint, outside it and over all valid pixels.
3. List how our valid-pixel mask differs from the whole-frame protocol, which explains the 6% vs 15% gap.
4. Compute bootstrap intervals for Tables III and V, and paired intervals for ridge vs affine and ours vs DEPTHOR-Small.
5. Report the largest difference from DEPTHOR's published numbers, which is the reproduction tolerance.

### T9. Refiner seeds

1. Train the scattered and island refiners (and mixed, if we keep it) with at least 3 seeds each, using the same data and hyperparameters.
2. Evaluate on the Table II protocols, and report mean ± std and the island gain over the analytic 0.055 m for each seed.
3. Fill item 25 in section 1 while doing this.

## 5. Release checklist

The reviewer asked us to provide code, calibration files, splits and runtime settings, and the paper's footnote points to [github.com/leoxie080808/RingFusion](https://github.com/leoxie080808/RingFusion). The repository must be public and contain the following before submission.

- [ ] Pipeline source and the ROS 2 node
- [ ] Training code for backbone distillation and the refiner
- [ ] TensorRT build settings and both FP16 engines
- [ ] Calibration: Kannala-Brandt intrinsics, rectification lookup table, camera-to-dToF extrinsic, dToF ray table
- [ ] Hold-out masks for the random and center protocols
- [ ] Distillation split (1800 training and 200 held-out frame IDs) and the refiner training split
- [ ] ZJU-L5 split indices and valid-pixel masks (indices and masks only, if the dataset licence restricts images)
- [ ] Tape points, each labelled calibration or held out
- [ ] Timing harness, Jetson power settings, ROS 2 QoS and queue depth
- [ ] A tagged release matching the camera-ready paper, ideally archived on Zenodo for a permanent DOI

Anything we cannot release goes in the response letter, with the reason.
