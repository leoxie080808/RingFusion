"""Training losses.

Backbone distillation (§3.4): both teacher and student are affine-invariant, so
the loss must be invariant to scale and shift. We align the student to the teacher
with the *same* closed-form least-squares fit used at inference (Stage 5), then
compare -- trimmed MAE plus multi-scale gradient matching.

Residual (§4.4): accuracy in log-depth plus a Gaussian NLL. The NLL is the entire
mechanism behind the calibration claim: err^2/var punishes over-confidence and
log(var) punishes under-confidence, so the minimum sits where predicted variance
matches realized squared error. Train on accuracy alone and the variance is
meaningless.
"""
import torch


# --- backbone distillation --------------------------------------------------

def align_ssi(pred, target):
    """Least-squares scale+shift aligning pred to target. Same math as Stage 5."""
    p = pred.flatten(1)
    t = target.flatten(1)
    ones = torch.ones_like(p)
    s_pp = (p * p).sum(1); s_p = p.sum(1); s_1 = ones.sum(1)
    s_pt = (p * t).sum(1); s_t = t.sum(1)
    det = s_pp * s_1 - s_p * s_p
    a = torch.where(det.abs() > 1e-8, (s_pt * s_1 - s_p * s_t) / det, torch.ones_like(det))
    b = torch.where(det.abs() > 1e-8, (s_pp * s_t - s_p * s_pt) / det, torch.zeros_like(det))
    return a.view(-1, 1, 1, 1) * pred + b.view(-1, 1, 1, 1)


def ssi_loss(pred, target, trim=0.2):
    """Trimmed MAE on the aligned pair. Dropping the worst `trim` fraction stops
    the student from faithfully learning the teacher's failures on glass/sky."""
    aligned = align_ssi(pred, target)
    err = (aligned - target).abs().flatten(1)
    k = int(err.shape[1] * (1.0 - trim))
    kept, _ = err.sort(dim=1)
    return kept[:, :k].mean()


def gradient_loss(pred, target, scales=4):
    """Multi-scale gradient matching preserves depth discontinuities (edges)."""
    total = 0.0
    p, t = align_ssi(pred, target), target
    for _ in range(scales):
        d = p - t
        total = total + d.diff(dim=-1).abs().mean() + d.diff(dim=-2).abs().mean()
        p, t = p[:, :, ::2, ::2], t[:, :, ::2, ::2]
    return total


def distill_loss(student_out, teacher_out, grad_weight=0.5, trim=0.2):
    return ssi_loss(student_out, teacher_out, trim) + \
        grad_weight * gradient_loss(student_out, teacher_out)


# --- residual ---------------------------------------------------------------

# log_tau2 is an unbounded network output; exp() of a large value overflows to inf,
# which turns the NLL (var.log(), err2/var) into inf -> NaN gradients -> dead training.
# Clamp it to a sane band (exp(-8)..exp(8) = 3e-4..3e3) and floor the total variance.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0
VAR_FLOOR = 1e-6
# A degenerate residual can drive D_pred to the ~10 km depth clamp, giving err2 ~1e8
# that dominates the NLL and thrashes training. Clamp D_pred to a physically-plausible
# ceiling in the loss so a single bad pixel can't blow up the objective (the ToF GT is
# already range-gated well below this in build_real_supervision).
MAX_DEPTH_LOSS = 15.0


def residual_loss(D_pred, D_gt, var_analytic, log_tau2, valid, nll_weight=0.2):
    """Accuracy (log-depth L1) + Gaussian NLL over valid pixels."""
    tau2 = log_tau2.clamp(LOGVAR_MIN, LOGVAR_MAX).exp()
    var = (var_analytic + tau2).clamp(min=VAR_FLOOR)  # bounded, strictly positive
    D_pred = D_pred.clamp(max=MAX_DEPTH_LOSS)
    err2 = (D_pred - D_gt) ** 2

    denom = valid.sum().clamp(min=1.0)
    l_depth = ((D_pred.clamp(min=1e-3).log() - D_gt.clamp(min=1e-3).log()).abs()
               * valid).sum() / denom
    l_nll = (0.5 * (err2 / var + var.log()) * valid).sum() / denom
    return l_depth + nll_weight * l_nll


@torch.no_grad()
def coverage(D_pred, D_gt, var_analytic, log_tau2, valid):
    """Fraction of valid pixels whose error is within 1 sigma. Should approach
    0.68 for a calibrated Gaussian (~0.95 = too timid, ~0.30 = over-confident)."""
    tau2 = log_tau2.clamp(LOGVAR_MIN, LOGVAR_MAX).exp()
    sigma = (var_analytic + tau2).clamp(min=VAR_FLOOR).sqrt()
    hit = ((D_pred - D_gt).abs() <= sigma) & (valid > 0)
    return hit.sum().float() / (valid > 0).sum().clamp(min=1).float()
