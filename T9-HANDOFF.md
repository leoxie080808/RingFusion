# T9 (refiner seeds) — handoff for the second AGX Orin

Goal: train the **scattered** and **island** refiners with **3 seeds each** (6 runs), so the paper
can report mean ± std over seeds instead of a single run. Everything here happens on machine B.
**Do not evaluate on B** — ship the checkpoints back and score them on machine A. Reason in §6.

---

## 0. Before you start — what machine A must hand you

| | What | Why |
|---|---|---|
| 1 | A pushed commit containing `--seed` in `training/train_residual.py` | Without it every seed trains identically — the flag did not exist before this round |
| 2 | `ros2_ws/data/real/` (**3.0 GB**) | gitignored; the refiner's training set (1234 rgb + 1234 tof) |
| 3 | `training/runs/student_v4_heldout/student_best.pth` (**15 MB**) | gitignored; the frozen backbone the refiner trains against |

Items 2 and 3 come on a USB stick. **Nothing else is needed** — no TensorRT engines (T9 trains in
PyTorch), no ZJU-L5, no `data/rect`, no `data/teacher`. `calibration.yaml` is tracked and arrives
with the clone.

---

## 1. Clone

```bash
git clone --depth 1 <repo-url> ~/RingFusion
cd ~/RingFusion
git log --oneline -1          # confirm this commit contains --seed:
grep -n "add_argument('--seed'" training/train_residual.py    # must print a line
```

`--depth 1` is deliberate: the full history is ~7 GB.

## 2. Copy the two gitignored payloads

Destination paths must match exactly.

```bash
mkdir -p ~/RingFusion/ros2_ws/data ~/RingFusion/training/runs
cp -r /media/<stick>/real/                 ~/RingFusion/ros2_ws/data/real/
cp -r /media/<stick>/student_v4_heldout/   ~/RingFusion/training/runs/student_v4_heldout/
```

Verify:

```bash
ls ~/RingFusion/ros2_ws/data/real/rgb | wc -l     # expect 1234
ls ~/RingFusion/ros2_ws/data/real/tof | wc -l     # expect 1234
ls -la ~/RingFusion/training/runs/student_v4_heldout/student_best.pth   # expect ~15 MB
```

## 3. Gate — confirm the software stack matches machine A

```bash
cd ~/RingFusion
python3 tools/diagnostics/envinfo.py --compare docs/demo/benchmarks/timing_pipeline_r31.json
```

Exit **0** = match, proceed. Exit **1** = it prints which versions differ.

A mismatch does **not** block T9 — training is PyTorch and is far less version-sensitive than
TensorRT inference. But **report the diff to machine A before starting**, and record it, because
it determines whether anything else can ever be offloaded to this board. The reference stack is
L4T R36.5.0 · TensorRT 10.3.0.30 · CUDA 12.6.68 · cuDNN 9.3.0.75.

## 4. Put the board in the same state as machine A

```bash
sudo nvpmodel -m 0        # MAXN. Persists across reboots.
sudo jetson_clocks        # DOES NOT persist -- re-run after every boot.
python3 tools/diagnostics/envinfo.py | grep -A3 jetson_clocks_applied   # want: true
```

Also stop anything else using the GPU — this board has one GPU and training will be the only
thing on it for ~12 hours.

## 5. Run the six trainings

Two configs × three seeds. **`scattered` is `--holdout random`; `island` is `--holdout island`.**

```bash
cd ~/RingFusion
for cfg in random island; do
  for seed in 0 1 2; do
    name="t9_${cfg}_seed${seed}"
    echo "=== $name ==="
    python3 training/train_residual.py --real \
      --rgb   ros2_ws/data/real/rgb \
      --tof   ros2_ws/data/real/tof \
      --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
      --student-ckpt training/runs/student_v4_heldout/student_best.pth \
      --out   training/runs/$name \
      --holdout $cfg --island 16 --holdout-frac 0.25 \
      --epochs 111 --patience 0 \
      --seed $seed 2>&1 | tee training/runs/$name.log
  done
done
```

### Why `--epochs 111 --patience 0` and not the README's `--epochs 200 --patience 8`

The shipped `residual_v7` ran **111 epochs** (best validation −0.2328 at epoch 108) and its
patience-8 early stop **never fired** — it was stopped by hand. So `--epochs 200` would run the
full 200 epochs, ≈ 3.3 h per run and ≈ **20 h** for six.

Pinning 111 matches the shipped model's actual budget and bounds the job at ≈ **11.5 h**. Setting
`--patience 0` disables early stopping so every seed runs an **identical** schedule — otherwise
seeds could stop at different epochs and the reported ± std would mix initialisation variance
with stopping-point variance, which is not the quantity the paper wants.

**Do not change any other hyperparameter.** Optimizer AdamW, lr 1e-3, wd 1e-4, batch 8, input
288 × 384, cosine schedule, `--holdout-frac 0.25`, `--island 16` are all deployed values and must
stay fixed for the comparison to mean anything.

### What `--seed` does and does not touch

It seeds weight initialisation, batch ordering and augmentation noise. It **deliberately does not
reseed the train/val split**, which stays pinned at 0 — every seed trains and validates on the
same frames, so the reported spread is initialisation variance alone.

### Expected timing and what a bad run looks like

| | Expect | If you see |
|---|---|---|
| Per epoch | ~1 min | > 2 min → something else is on the GPU |
| Per run | ~114 min | — |
| All six | **~11.5 h** | — |
| First log line | `residual parameters: 463,971 \| device cuda \| seed N` | `device cpu` → CUDA not visible, **stop** |
| `island` best val | around −0.23 | worse than −0.15 → report before continuing |
| Coverage | trending to ~0.68 | stuck far from 0.68 → report |

Each run writes `residual_best.pth` and `residual_last.pth` into its `--out` directory.

## 6. Ship back — checkpoints only, do not score them here

```bash
cd ~/RingFusion/training/runs
tar czf t9_results.tar.gz t9_*_seed*/residual_best.pth t9_*_seed*/residual_last.pth t9_*.log
```

Also include the output of:

```bash
python3 tools/diagnostics/envinfo.py --out t9_env_machineB.json
```

**Why evaluation stays on machine A:** scoring means building TensorRT FP16 engines and running
them over the 1234-frame set. Engines are built for a specific GPU + TensorRT version, and a
rebuilt FP16 engine can shift numerics enough that the rows would not be comparable with the
existing r31 record. Training has no such exposure, which is exactly why T9 is the right job to
offload and the accuracy tests are not.

## 7. Report back

1. The `--compare` result from §3 (match, or the exact diff).
2. `t9_env_machineB.json` — it now carries a `machine` block (hostname, serial, model, RAM) so the
   two boards are distinguishable in the paper's provenance record.
3. The six best validation losses, and any run that deviated from the table in §5.
4. `t9_results.tar.gz`.
