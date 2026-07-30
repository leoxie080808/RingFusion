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

# Lower bound on the SSI alignment scale, as a fraction of the std-matching scale. Must be
# > 0: at exactly 0 the aligned output stops depending on `pred` and the gradient dies (see
# align_ssi). 0.01 is small enough not to distort a healthy fit and large enough to keep a
# usable gradient when the student starts anti-correlated.
A_FLOOR_FRAC = 0.01


def align_ssi(pred, target, lock_positive=True):
    """Least-squares scale+shift aligning pred to target. Same math as Stage 5.

    lock_positive FLOORS the alignment scale at a small positive value (it used to clamp to
    exactly 0, which killed the gradient -- see the note in the body). Teacher and student emit
    NON-NEGATIVE disparity (the student head ends in ReLU), so the physically correct
    scale is positive. Left free, a<0 gives a mirror-image minimum: a SIGN-FLIPPED
    student (large where the teacher is small) aligns just as well and scores an
    equally low SSI loss, so distillation can silently converge inverted -- which is
    exactly what a from-scratch re-distill did (rho -0.998). Clamping a>=0 removes that
    basin so the loss only rewards the correct sign. b is then the LS-optimal shift for
    the floored a: b = mean(t - a*p)  (identical to the free-fit b whenever a is unfloored)."""
    p = pred.flatten(1)
    t = target.flatten(1)
    ones = torch.ones_like(p)
    s_pp = (p * p).sum(1); s_p = p.sum(1); s_1 = ones.sum(1)
    s_pt = (p * t).sum(1); s_t = t.sum(1)
    det = s_pp * s_1 - s_p * s_p
    a = torch.where(det.abs() > 1e-8, (s_pt * s_1 - s_p * s_t) / det, torch.ones_like(det))
    if lock_positive:
        # DO NOT clamp to exactly 0. That was a dead-gradient trap: at a>=0 boundary the
        # aligned output becomes 0*pred + mean(target), a CONSTANT, so d(ssi)/d(pred) is
        # identically zero and the model can never recover. Measured 2026-07-30 on a
        # from-scratch re-distill: a random init is mildly anti-correlated with the teacher
        # (corr -0.12, chance), least squares asks for a = -48, the clamp sets a = 0, and
        # ssi_loss then reports 189.95 with EXACTLY zero gradient for every subsequent step.
        # Training froze at val_ssi 167.24 for 39 epochs against v3's 3.51, identically for
        # amp on/off and three learning rates -- the giveaway that the loss no longer
        # depended on the prediction at all.
        #
        # Floor `a` at a small POSITIVE fraction of the scale that would match the two
        # standard deviations instead. The aligned output still depends on pred, so the
        # gradient survives and pushes the student toward positive correlation, which is
        # what the clamp was for. The original sign-inversion basin (rho -0.998) stays
        # excluded because a can never go negative.
        scale_ref = (t.std(1) / p.std(1).clamp(min=1e-6)) * A_FLOOR_FRAC
        a = torch.maximum(a, scale_ref)
    b = (s_t - a * s_p) / s_1
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


def structure_loss(D_pred, D_base, valid, scales=3):
    """Tie the residual to Network A's RELATIVE geometry where there is no ToF target.

    The hold-out scheme can only supervise inside the ToF's footprint (measured: a
    447x340 px box = 7.5% of the frame) and only inside its range gate (the logged
    ToF never returns past 6.1 m). Outside that the net is unconstrained, and on-robot
    it shows: it invents a 13 px lattice Network A does not have (anchor-pitch power
    A ~3x vs B 15-23x over broadband) and runs the far field to the 20 m clamp in a
    room measuring 2.8 m.

    But A's depth is only missing SCALE -- its relative geometry is valid over the whole
    frame. So off-target we match GRADIENTS of log-depth, not values: the net stays free
    to rescale (its entire job, since log-scale differences vanish under d/dx) while
    being penalised for inventing structure A never saw. Multi-scale so it constrains
    broad drift as well as per-pixel speckle.

    Same construction as gradient_loss() above, applied to (B, A) instead of
    (student, teacher). valid is the supervised mask -- this acts on 1 - valid.
    """
    free = (1.0 - valid)
    d = (D_pred.clamp(min=1e-3).log() - D_base.clamp(min=1e-3).log()) * free
    total = 0.0
    for _ in range(scales):
        if d.shape[-1] < 2 or d.shape[-2] < 2:
            break
        total = total + d.diff(dim=-1).abs().mean() + d.diff(dim=-2).abs().mean()
        d = d[:, :, ::2, ::2]
    return total


def residual_loss(D_pred, D_gt, var_analytic, log_tau2, valid, nll_weight=0.2,
                  D_base=None, struct_weight=0.0):
    """Accuracy (log-depth L1) + Gaussian NLL over valid pixels, plus an optional
    structure term tying the net to Network A off-target (see structure_loss).

    struct_weight=0 reproduces the original objective exactly."""
    tau2 = log_tau2.clamp(LOGVAR_MIN, LOGVAR_MAX).exp()
    var = (var_analytic + tau2).clamp(min=VAR_FLOOR)  # bounded, strictly positive
    D_pred = D_pred.clamp(max=MAX_DEPTH_LOSS)
    err2 = (D_pred - D_gt) ** 2

    denom = valid.sum().clamp(min=1.0)
    l_depth = ((D_pred.clamp(min=1e-3).log() - D_gt.clamp(min=1e-3).log()).abs()
               * valid).sum() / denom
    l_nll = (0.5 * (err2 / var + var.log()) * valid).sum() / denom
    loss = l_depth + nll_weight * l_nll
    if struct_weight > 0.0 and D_base is not None:
        loss = loss + struct_weight * structure_loss(D_pred, D_base, valid)
    return loss


@torch.no_grad()
def coverage(D_pred, D_gt, var_analytic, log_tau2, valid):
    """Fraction of valid pixels whose error is within 1 sigma. Should approach
    0.68 for a calibrated Gaussian (~0.95 = too timid, ~0.30 = over-confident)."""
    tau2 = log_tau2.clamp(LOGVAR_MIN, LOGVAR_MAX).exp()
    sigma = (var_analytic + tau2).clamp(min=VAR_FLOOR).sqrt()
    hit = ((D_pred - D_gt).abs() <= sigma) & (valid > 0)
    return hit.sum().float() / (valid > 0).sum().clamp(min=1).float()
