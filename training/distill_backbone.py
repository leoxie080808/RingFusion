"""Distill Depth Anything V2 into the small student backbone (§3.4, §3.5).

    photos -> cached teacher disparity -> train student

Loss is scale-and-shift-invariant (the student, like the teacher, is only correct
up to a global affine): trimmed SSI MAE + 0.5 * multi-scale gradient matching.

Usage:
    python distill_backbone.py --images data/rect --cache data/teacher \
        --out runs/student --epochs 60 --batch 16
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, random_split

from data import DistillDataset
from losses import distill_loss, ssi_loss
from models.student import DepthStudent, count_parameters


def cosine_warmup(optimizer, warmup_steps, total_steps, min_factor):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', default='runs/student')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--size', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--resume', default='')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    amp = device.startswith('cuda')

    full = DistillDataset(args.images, args.cache, size=tuple(args.size), augment=True)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))
    # val split should not be augmented; share the underlying arrays but disable aug
    val_view = DistillDataset(args.images, args.cache, size=tuple(args.size), augment=False)
    val_set.dataset = val_view

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=amp)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=amp)

    model = DepthStudent(pretrained=not args.no_pretrained).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
    print(f"student parameters: {count_parameters(model):,}  "
          f"| train {n_train} val {n_val} | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    sched = cosine_warmup(opt, int(0.05 * total_steps), total_steps, 1e-6 / args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for it, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=amp):
                pred = model(img)
                loss = distill_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if (it + 1) % 50 == 0:
                print(f"epoch {epoch} it {it + 1}/{len(train_loader)} "
                      f"loss {run / (it + 1):.4f} lr {sched.get_last_lr()[0]:.2e}")

        model.eval()
        val = 0.0
        with torch.no_grad():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                val += ssi_loss(model(img), target).item()
        val /= max(1, len(val_loader))
        print(f"[epoch {epoch}] train {run / len(train_loader):.4f}  val_ssi {val:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out, 'student_last.pth'))
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), os.path.join(args.out, 'student_best.pth'))
            print(f"  new best (val_ssi {val:.4f}) -> student_best.pth")

    print(f"done. best val_ssi {best_val:.4f}")


if __name__ == '__main__':
    main()
