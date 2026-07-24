"""Train the residual correction net g_phi (§4.4).

The residual predicts per-pixel corrections (da, db) to the closed-form affine fit
plus an extra variance tau^2. Its inputs are built with the SAME geometry and
anchoring the robot runs at inference (anchoring_bridge), so it corrects the exact
fit it will see. ToF is simulated from the GT depth, so only (rgb, depth) pairs are
needed to start.

Loss = log-depth L1 + 0.2 * Gaussian NLL. The NLL is what makes the variance
*calibrated*; coverage (fraction within 1 sigma) should approach 0.68.

Usage:
    python train_residual.py --rgb data/syn/rgb --depth data/syn/depth \
        --student-ckpt runs/student/student_best.pth --out runs/residual
"""
import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from anchoring_bridge import (default_calib, simulate_tof, build_residual_inputs,
                              calib_from_yaml, build_real_supervision)
from losses import residual_loss, coverage
from models.residual import ResidualRefinerNet, apply_residual, count_parameters
from models.student import DepthStudent
from residual_data import ResidualDepthDataset
from residual_real_data import ResidualRealDataset


def assemble_batch(disp, rgb_norm, gt_depth, valid, calib, rng, args):
    """Turn a raw batch into the residual net's inputs. Runs the numpy anchoring
    per sample (cheap) and drops samples whose fit is underdetermined.

    disp: (B,1,H,W) student output. Returns tensors on CPU, or None if all dropped.
    """
    disp_np = disp[:, 0].detach().cpu().numpy()
    xs, keep, aa, bb, vars_, gts, valids = [], [], [], [], [], [], []
    for i in range(disp_np.shape[0]):
        gt = gt_depth[i, 0].numpy()
        tof_d, tof_v, conf = simulate_tof(gt, calib, noise_frac=args.noise_frac,
                                          dropout=args.dropout, rng=rng)
        info = build_residual_inputs(disp_np[i], tof_d, tof_v, calib,
                                     confidence=conf, min_confidence=args.min_confidence)
        if info is None or info['var_analytic'] is None:
            continue
        D0 = info['D0']
        med = np.median(D0[D0 > 0]) if np.any(D0 > 0) else 1.0
        logD0 = np.log(np.clip(D0, 1e-3, None) / max(med, 1e-3)).astype(np.float32)
        x6 = torch.cat([
            rgb_norm[i],
            torch.from_numpy(logD0)[None],
            torch.from_numpy(info['anchor_depth'])[None],
            torch.from_numpy(info['anchor_mask'])[None],
        ], dim=0)
        xs.append(x6)
        keep.append(i)
        aa.append(info['a']); bb.append(info['b'])
        vars_.append(torch.from_numpy(info['var_analytic'])[None])
        gts.append(gt_depth[i])
        valids.append(valid[i])
    if not xs:
        return None
    return {
        'x': torch.stack(xs), 'keep': keep,
        'a': torch.tensor(aa, dtype=torch.float32).view(-1, 1, 1, 1),
        'b': torch.tensor(bb, dtype=torch.float32).view(-1, 1, 1, 1),
        'var': torch.stack(vars_), 'gt': torch.stack(gts),
        'valid': torch.stack(valids).float(),
    }


def assemble_batch_real(disp, rgb_norm, tof_dist, tof_conf, calib, rng, args):
    """Real-data counterpart of assemble_batch: no dense GT -- each frame's ToF is
    split into anchors (drive the fit + net input) and hold-outs (sparse target),
    via anchoring_bridge.build_real_supervision. Same output contract as
    assemble_batch, but 'gt'/'valid' are sparse (nonzero only at held-out pixels)."""
    disp_np = disp[:, 0].detach().cpu().numpy()
    xs, keep, aa, bb, vars_, gts, valids = [], [], [], [], [], [], []
    for i in range(disp_np.shape[0]):
        td = tof_dist[i].numpy()                         # (rows,cols), NaN invalid
        tv = np.isfinite(td)
        conf = tof_conf[i].numpy()
        info = build_real_supervision(disp_np[i], td, tv, calib,
                                      holdout_frac=args.holdout_frac, rng=rng,
                                      confidence=conf, min_confidence=args.min_confidence)
        if info is None:
            continue
        D0 = info['D0']
        med = np.median(D0[D0 > 0]) if np.any(D0 > 0) else 1.0
        logD0 = np.log(np.clip(D0, 1e-3, None) / max(med, 1e-3)).astype(np.float32)
        x6 = torch.cat([
            rgb_norm[i],
            torch.from_numpy(logD0)[None],
            torch.from_numpy(info['anchor_depth'])[None],
            torch.from_numpy(info['anchor_mask'])[None],
        ], dim=0)
        xs.append(x6); keep.append(i)
        aa.append(info['a']); bb.append(info['b'])
        vars_.append(torch.from_numpy(info['var_analytic'])[None])
        gts.append(torch.from_numpy(info['D_gt'])[None])
        valids.append(torch.from_numpy(info['valid_gt'])[None])
    if not xs:
        return None
    return {
        'x': torch.stack(xs), 'keep': keep,
        'a': torch.tensor(aa, dtype=torch.float32).view(-1, 1, 1, 1),
        'b': torch.tensor(bb, dtype=torch.float32).view(-1, 1, 1, 1),
        'var': torch.stack(vars_), 'gt': torch.stack(gts),
        'valid': torch.stack(valids).float(),
    }


def _step(residual, packed, disp, args, device):
    """Shared forward + loss for both the synthetic and real paths."""
    x = packed['x'].to(device)
    disp_keep = disp[packed['keep']]
    a = packed['a'].to(device); b = packed['b'].to(device)
    var = packed['var'].to(device); gt = packed['gt'].to(device)
    valid_k = packed['valid'].to(device)
    out = residual(x)
    D_pred, _ = apply_residual(disp_keep, a, b, out)
    log_tau2 = out[:, 2:3]
    loss = residual_loss(D_pred, gt, var, log_tau2, valid_k, nll_weight=args.nll_weight)
    cov = coverage(D_pred, gt, var, log_tau2, valid_k)
    return loss, cov


def run_batch(student, residual, batch, calib, rng, args, device):
    rgb_norm, gt_depth, valid = batch
    with torch.no_grad():
        disp = student(rgb_norm.to(device))
    packed = assemble_batch(disp, rgb_norm, gt_depth, valid, calib, rng, args)
    if packed is None:
        return None
    return _step(residual, packed, disp, args, device)


def run_batch_real(student, residual, batch, calib, rng, args, device):
    rgb_norm, tof_dist, tof_conf = batch
    with torch.no_grad():
        disp = student(rgb_norm.to(device))
    packed = assemble_batch_real(disp, rgb_norm, tof_dist, tof_conf, calib, rng, args)
    if packed is None:
        return None
    return _step(residual, packed, disp, args, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb', required=True)
    ap.add_argument('--depth', help='dense GT depth dir (synthetic path; required unless --real)')
    ap.add_argument('--real', action='store_true',
                    help='train on REAL paired (rgb, 32x32 ToF) logs via held-out anchors '
                         '(needs --tof and --calib; no dense GT)')
    ap.add_argument('--tof', help='real ToF .npz dir (with --real)')
    ap.add_argument('--calib', help='calibration.yaml for the real calib (with --real)')
    ap.add_argument('--holdout-frac', type=float, default=0.25,
                    help='fraction of each frame\'s ToF zones held out as supervision (real path)')
    ap.add_argument('--student-ckpt', required=True)
    ap.add_argument('--out', default='runs/residual')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--size', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--depth-scale', type=float, default=1.0,
                    help='multiply loaded depth by this (0.001 for 16-bit-mm PNGs)')
    ap.add_argument('--hfov', type=float, default=90.0, help='camera h-FOV for synthetic calib')
    ap.add_argument('--noise-frac', type=float, default=0.02)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--min-confidence', type=int, default=-1)
    ap.add_argument('--nll-weight', type=float, default=0.2)
    ap.add_argument('--grad-clip', type=float, default=1.0,
                    help='max global grad norm; 0 disables. Tames NLL gradient blowups.')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.real:
        if not (args.tof and args.calib):
            ap.error('--real requires --tof and --calib')
    elif not args.depth:
        ap.error('--depth is required unless --real is set')

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    H, W = args.size
    calib = calib_from_yaml(args.calib, (H, W)) if args.real else default_calib(H, W, hfov_deg=args.hfov)
    run_batch_fn = run_batch_real if args.real else run_batch
    rng = np.random.default_rng(0)

    student = DepthStudent(pretrained=False).to(device).eval()
    student.load_state_dict(torch.load(args.student_ckpt, map_location=device))
    for p in student.parameters():
        p.requires_grad_(False)

    residual = ResidualRefinerNet().to(device)
    print(f"residual parameters: {count_parameters(residual):,} | device {device}")

    if args.real:
        full = ResidualRealDataset(args.rgb, args.tof, size=tuple(args.size))
    else:
        full = ResidualDepthDataset(args.rgb, args.depth, size=tuple(args.size),
                                    depth_scale=args.depth_scale)
    n_val = max(1, int(len(full) * args.val_frac))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers)

    opt = torch.optim.AdamW(residual.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, total))))

    best_val = float('inf')
    for epoch in range(args.epochs):
        residual.train()
        run = cov_run = n = 0.0
        for batch in train_loader:
            res = run_batch_fn(student, residual, batch, calib, rng, args, device)
            if res is None:
                continue
            loss, cov = res
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: the Gaussian NLL can explode when predicted variance
            # goes tiny on a large-error (far/degenerate) pixel -> huge gradients that
            # knock the net into a bad basin (observed as loss diverging to 1000s).
            # Clipping the global norm keeps a single bad batch from wrecking training.
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(residual.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            run += loss.item(); cov_run += cov.item(); n += 1
        tr = run / max(1, n)
        tr_cov = cov_run / max(1, n)

        residual.eval()
        vrun = vcov = vn = 0.0
        with torch.no_grad():
            for batch in val_loader:
                res = run_batch_fn(student, residual, batch, calib, rng, args, device)
                if res is None:
                    continue
                loss, cov = res
                vrun += loss.item(); vcov += cov.item(); vn += 1
        vl = vrun / max(1, vn)
        vcoverage = vcov / max(1, vn)
        print(f"[epoch {epoch}] train {tr:.4f} (cov {tr_cov:.2f})  "
              f"val {vl:.4f} (cov {vcoverage:.2f} -> target 0.68)")

        torch.save(residual.state_dict(), os.path.join(args.out, 'residual_last.pth'))
        if vl < best_val:
            best_val = vl
            torch.save(residual.state_dict(), os.path.join(args.out, 'residual_best.pth'))
            print(f"  new best (val {vl:.4f}) -> residual_best.pth")

    print(f"done. best val {best_val:.4f}")


if __name__ == '__main__':
    main()
