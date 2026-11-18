# ICRAE 2026: capture session and remaining tasks

**Written 2026-11-17 for the capture session that evening. Presentation only.**

Everything recorded with these tools is for the ICRAE 2026 talk (Nagoya, Nov 20 to 22). None of it
changes the paper's numbers or the pipeline. Every slide that uses it carries a "Not in the paper" tag.

## What this commit adds

| File | Shot | Runs when | What it writes |
|---|---|---|---|
| `tools/diagnostics/record_clip_sigma.py` | C1 | stack running | `record_clip.py` frames plus a σ panel from `/depth_var` on a log scale (0.01 to 3 m), with one shared colour scale for the dToF and fused panels |
| `tools/diagnostics/record_map_frames.py` | M1 | stack running | for every `/depth`: the rectified camera frame it came from, depth and variance (float16), stamps and intrinsics, for a map built offline |
| `tools/diagnostics/heldout_qual.py` | H1 | node stopped | one held-out frame: camera with kept and withheld zones, nearest-zone fill, ours, and the error at the withheld zones |
| `tools/diagnostics/zjul5_qual.py` | Z1 | node stopped | ZJU-L5 images with RGB, ground truth, ours, DEPTHOR-Small and both error maps on one colour scale |
| `docs/ICRAE2026_CAPTURE.md` | | | this file |

Already in the repo and also used: `record_clip.py` (C2, C3), `record_cloudseq.py` (C4), `rate_live.py`.

## Remaining tasks

### Before the session, on the Orin

- [ ] Check `git status` on the Orin. If it shows no changed tracked files, `git pull` this commit.
      If it does, don't pull. Copy the four scripts into `tools/diagnostics/` instead.
- [ ] Source ROS 2 and the workspace (two of the scripts import `rclpy` and `ringfusion_msgs` at startup).
- [ ] Cache the Depth Anything V2 Small weights for Z1. This needs internet once.
- [ ] Run the import check and `--help` for each script, then a 5 s dry run of the two live recorders.
- [ ] Check that ZJU-L5 (the folder with `data.json`) and DEPTHOR (`~/external/Depthor`) are on the Orin.
- [ ] Check that the logged pairs (`ros2_ws/data/real/`) and both engines are there. Git ignores all of them.
- [ ] Free at least 30 GB of disk for bag backups (`df -h`).
- [ ] Charge the robot batteries.
- [ ] Set up the scene: clutter at 0.5 to 1 m, a thin pole, and a far wall 3 to 4 m away. A corridor works well.
- [ ] Bring a tape measure, floor tape for marks, and a phone for photos. Keep the lighting steady.

```bash
source /opt/ros/humble/setup.bash && source ~/RingFusion/ros2_ws/install/setup.bash
export R=~/RingFusion OUT=~/icrae_rerun
export CAL=$R/ros2_ws/src/ringfusion_bringup/config/calibration.yaml

# cache the weights zjul5_qual.py loads
python3 -c "from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M; \
m='depth-anything/Depth-Anything-V2-Small-hf'; P.from_pretrained(m); M.from_pretrained(m); print('cached')"

# imports and options
python3 -c "import cv2, numpy, scipy, yaml, transformers; print('ok')"
for f in record_clip_sigma record_map_frames heldout_qual zjul5_qual; do
  python3 $R/tools/diagnostics/$f.py --help > /dev/null && echo "$f ok"; done

# with the stack running (step 1 below): 5 s dry runs, then look at the output
python3 $R/tools/diagnostics/record_clip_sigma.py --dir /tmp/dry_c1 --secs 5 --calib $CAL
python3 $R/tools/diagnostics/record_map_frames.py --dir /tmp/dry_m1 --secs 5 --calib $CAL
```

`record_map_frames.py` should report 0 dropped and 0 unpaired. If `unpaired` climbs, rerun it with
`--max-skew 0.12`.

### Tonight: shot list

The three Must items are the reason for the session. Do them first, then the Nice items as time allows.

| ID | Capture | Slide | Priority | Where | Status |
|---|---|---|---|---|---|
| C1 | Live drive with an uncertainty (σ) panel, 3 takes of 30 s | 20 Uncertainty, live | Must | Robot | [ ] |
| C2 | Far-field ceiling: mixed-depth scene, dToF and fused depth on the same colour scale | 24 Ceiling, live | Must | Robot | [ ] |
| Z1 | ZJU-L5 side by side on 3 test images | 19 ZJU-L5 side by side | Must | Orin, offline | [ ] |
| C3 | Demo drive re-recorded with matched colour scales | 8 Live demo | Nice | Robot | [ ] |
| C4 | Point-cloud sequence, 30 s | 9 Point cloud in 3D | Nice | Robot | [ ] |
| H1 | One held-out frame | 13 Held-out frame | Nice | Orin, offline | [ ] |
| P1 | Photos: robot three-quarter view, sensor close-up, tape setup | 1, 7 and 16 | Nice | Phone | [ ] |
| F1 | Flat-wall test at 0.5, 1 and 2 m | Backup | Optional | Robot | [ ] |
| M1 | Map frames: one slow 60 s loop for a map built offline | 25 Future work | Optional | Robot | [ ] |

### Wednesday Nov 18

- [ ] **Leo:** zip `clips`, `zjul5`, `heldout`, `photos`, `logs`, plus `map1` if recorded (leave the bags
      out unless asked), and send it over.
- [ ] **Claude:** cut 15 s clips, swap the captures in for the mock-ups (C1 on slide 20, C2 on 24, Z1 on 19,
      H1 on 13, C4 on 9), build the M1 map clip if it was recorded, update the speaker notes, and recheck
      that the talk stays under 12 minutes.
- [ ] **Leo:** install Barlow Condensed, open the deck in PowerPoint, check that every video plays, turn on
      Embed fonts, and save the .pptx plus a PDF backup to the USB.

### Later

- [ ] Update the main README to match the conference paper and tidy up the results sections.

## Session procedure

About 3 hours, Must items first.

### 1. Setup and freeze (20 min)

```bash
export R=~/RingFusion OUT=~/icrae_rerun
export CAL=$R/ros2_ws/src/ringfusion_bringup/config/calibration.yaml
mkdir -p $OUT/{clips,bags,zjul5,heldout,logs,photos}
sudo nvpmodel -m 0 && sudo jetson_clocks && nvpmodel -q

# the pipeline must be the paper's build: the diff (committed and uncommitted) must be empty
git -C $R log --oneline -1 > $OUT/logs/freeze.txt
git -C $R diff --stat 3e592d8 -- ros2_ws training >> $OUT/logs/freeze.txt
sha256sum $R/student_v4_heldout_fp16.engine $R/residual_v7_fov73_fp16.engine >> $OUT/logs/freeze.txt

ros2 launch ringfusion_bringup single_module.launch.py port:=/dev/ttyACM1 \
  backbone_engine:=$R/student_v4_heldout_fp16.engine \
  residual_engine:=$R/residual_v7_fov73_fp16.engine
```

Deployed settings: arbitration on, ridge 0. Change nothing. Let it warm up for 10 minutes, then check
the rate. Expect about 9.4 Hz. Below 8.5 Hz, stop and check the clocks before recording anything.

```bash
python3 $R/tools/diagnostics/rate_live.py --secs 60 --label "rerun 2026-11-17" --out $OUT/logs/rate_check.json
```

### 2. C1, uncertainty clip (30 min, Must)

Start 1 m from the clutter, pass the thin pole, then turn toward the far wall so the image edges and
the far field light up in the σ panel. Record 3 takes of 30 s, each with a bag backup.

```bash
ros2 bag record -o $OUT/bags/c1_take1 /image /tof /depth /depth_var /cloud &
python3 $R/tools/diagnostics/record_clip_sigma.py --dir $OUT/clips/c1_take1 --secs 30 --fps 15 --calib $CAL
kill %1
```

### 3. C2, far-field ceiling (20 min, Must)

Park the robot so a wide far wall fills the image from edge to edge, with near objects at 0.5 to 1 m.
Tape the distance to the wall and write it down. Record 10 s with the wall at 2, 3 and 4 m, then one
slow 15 s approach from 4 m to 1 m. Both depth panels use the same 0.15 to 5 m scale.

```bash
python3 $R/tools/diagnostics/record_clip.py --dir $OUT/clips/c2_wall_3m --secs 10 --fps 15 \
  --tof-lo 0.15 --tof-hi 5.0 --d-lo 0.15 --d-hi 5.0 --calib $CAL
```

What to look for: inside the dToF view the wall reads close to its tape distance, because arbitration
uses the sensor there. Outside the view, the same wall should read no deeper than about 1.4 to 1.5 m.
That contrast is the slide.

### 4. C3 and C4, demo clip and point cloud (25 min, Nice)

C3 is the same route as the paper's drive, 2 takes of 30 s, with matched scales.

```bash
python3 $R/tools/diagnostics/record_clip.py --dir $OUT/clips/c3_take1 --secs 30 --fps 15 \
  --tof-lo 0.15 --tof-hi 5.0 --d-lo 0.15 --d-hi 5.0 --calib $CAL
python3 $R/tools/diagnostics/record_cloudseq.py --calib $CAL --dir $OUT/clips/c4_cloud --secs 30 --fps 12
```

### 5. P1, photos (10 min, Nice)

Robot three-quarter view at robot height, the sensor module close up, and the tape setup with a marker
in place. Landscape, plain background.

### 6. F1, flat wall (20 min, Optional)

Face a plain wall square-on at 0.5, 1 and 2 m by tape. Record a 10 s bag at each and note whether depth
is published and looks sane.

```bash
ros2 bag record -o $OUT/bags/f1_wall_1m /image /tof /depth /depth_var
```

### 7. M1, map frames (15 min, Optional)

Only after C1 and C2 are saved, with the stack still running. Drive one slow loop around the room at
walking pace or slower, turn smoothly, and finish where you started so the map can close the loop. Keep
textured things in view and avoid long stretches facing a blank wall. 60 s is enough for a 7 to 10 s
clip played at 6 to 8 times speed.

```bash
python3 $R/tools/diagnostics/record_map_frames.py --dir $OUT/map1 --secs 60 --calib $CAL
```

It saves about 0.55 MB per frame (around 300 MB a minute) and prints the dropped and unpaired counts
every 30 frames. Both should stay near zero. Do a second take into `map2` if there is time. The map is
built offline afterwards, keeping only points with a low σ so the far-field ceiling doesn't smear the walls.

### 8. Z1 and H1, offline on the Orin (45 min)

Stop the node first. Z1 runs our closed-form fit on the same Depth Anything V2 ViT-S weights as DEPTHOR
(the paper's 0.094 AbsRel row) and DEPTHOR-Small, on three test images picked automatically: one where
we match DEPTHOR, one typical, and one with a far background. It prints the pooled AbsRel, which should
reproduce the paper's 0.094.

```bash
python3 $R/tools/diagnostics/zjul5_qual.py --zjul5-root <ZJU-L5 dir> \
  --depthor-root ~/external/Depthor --pick match,median,far --out $OUT/zjul5
python3 $R/tools/diagnostics/heldout_qual.py --stems-file $R/docs/demo/benchmarks/val_stems.txt \
  --protocol center --calib $CAL --out $OUT/heldout
```

## Checks before packing up

Ten minutes here saves a second session.

- [ ] Watch every clip once, end to end: no frozen panels, no black dToF panel, labels readable.
- [ ] Encode each clip folder: `ffmpeg -framerate 15 -i f%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 18 take.mp4`
- [ ] Pick the best take of C1 and C2 and note the 15 s to use.
- [ ] σ panel: areas outside the dToF view and the far field look higher than near, in-view areas.
      If not, keep the bag before re-recording.
- [ ] C2: tape distance written down for every wall position.
- [ ] Z1: three image sets saved, with the per-image AbsRel for both methods in the log.
- [ ] `rate_check.json` shows about 9.4 Hz, and `freeze.txt` shows no diff under `ros2_ws` or `training`.
- [ ] Copy `~/icrae_rerun` off the robot before shutting down.

If a Must capture fails, the current slides still work as they are, so nothing here blocks the talk.
