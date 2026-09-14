"""Resampling intervals for the r31 revision.

The reviewer asked for per-run variation or confidence intervals on the statistics.
Two things matter for getting that right on depth data.

FRAME-LEVEL, NOT PIXEL-LEVEL. Errors within a frame are strongly correlated -- one bad
fit moves every pixel in that frame together, and a depth discontinuity in the wrong
place moves a whole region. Resampling pixels treats them as independent and returns an
interval that is far too tight, sometimes by an order of magnitude. The resampling unit
has to be whatever the errors are correlated within, which here is the frame. Every
function below takes `groups` -- a list of per-frame arrays -- and resamples the LIST.

PAIRED, WHERE THE CLAIM IS A DIFFERENCE. "MAE falls from 0.264 to 0.247" is a claim about
a difference measured on the same frames. Bootstrapping the two arms separately and
comparing their intervals throws away that pairing and will call a real effect
insignificant. paired_diff() resamples the frame list once and applies it to both arms.

B is reported in every result so the paper can state it. Cost scales with the total
sample count: B=1000 over 1228 frames x ~700 held-out zones is ~24 s per statistic, so
the big ablation wants B=1000 while the 15-point tape set can afford B=10000.
"""
import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -> (lo, hi).

    Used for every coverage fraction in the paper. At n = 11 and n = 15 the normal
    approximation is not usable (it can return bounds outside [0, 1] and is badly
    anti-conservative near p = 1), and the paper's coverage columns are exactly that
    regime: 10/11 and 15/15. Wilson stays inside [0, 1] and does not collapse at the
    endpoints, which is the whole point when the claim is 'coverage reached 1.000'.
    """
    n = int(n)
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _as_groups(groups):
    return [np.asarray(g, float).ravel() for g in groups]


def _flatten(groups):
    """-> (flat, starts, sizes). Lets a draw be gathered with one fancy-index.

    The obvious implementation, np.concatenate([groups[j] for j in draw]) per iteration,
    copies every element through a Python list on all B iterations. Over 1228 frames x
    ~750 held-out zones x B=1000 that is ~10^9 element copies per statistic per method,
    which puts the ablation into hours. Flattening once and building the gather index
    arithmetically keeps the whole thing in numpy.
    """
    sizes = np.array([g.size for g in groups], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return np.concatenate(groups) if groups else np.empty(0), starts, sizes


def _draw(flat, starts, sizes, idx):
    """Gather the concatenation of the groups named by `idx`, without a Python loop."""
    counts = sizes[idx]
    total = int(counts.sum())
    if total == 0:
        return np.empty(0)
    # offset[k] maps position-within-output -> position-within-flat for group idx[k]
    out_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    base = np.repeat(starts[idx] - out_starts, counts)
    return flat[base + np.arange(total)]


def frame_bootstrap(groups, stat=np.median, B=10000, seed=0, alpha=0.05):
    """Percentile bootstrap over GROUPS (one array per frame) -> dict.

    stat is applied to the concatenation of the resampled groups, so a frame
    contributing more pixels carries proportionally more weight -- the same weighting
    the point estimate uses. Returns the point estimate on the real data alongside the
    interval so the two can never drift apart in the write-up.
    """
    gs = _as_groups(groups)
    gs = [g for g in gs if g.size]
    if not gs:
        return {'point': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'B': int(B), 'n_groups': 0, 'n_samples': 0}
    flat, starts, sizes = _flatten(gs)
    point = float(stat(flat))
    rng = np.random.default_rng(seed)
    n = len(gs)
    draws = np.empty(B, float)
    for i in range(B):
        idx = rng.integers(0, n, n)
        draws[i] = stat(_draw(flat, starts, sizes, idx))
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {'point': point, 'lo': float(lo), 'hi': float(hi), 'B': int(B),
            'n_groups': n, 'n_samples': int(sum(g.size for g in gs))}


def paired_diff(groups_a, groups_b, stat=np.median, B=10000, seed=0, alpha=0.05):
    """Bootstrap the DIFFERENCE stat(b) - stat(a) with the frame draw shared -> dict.

    groups_a[i] and groups_b[i] must be the same frame scored two ways. Adds `p_two_sided`,
    the fraction of resamples whose difference has the opposite sign to the point estimate,
    doubled -- the usual bootstrap two-sided test. It answers the question the A/B claim
    actually poses: given frame-to-frame variation, would this difference survive a
    different 600 frames?
    """
    ga, gb = _as_groups(groups_a), _as_groups(groups_b)
    if len(ga) != len(gb):
        raise ValueError(f'paired arms differ in frame count: {len(ga)} vs {len(gb)}')
    keep = [i for i in range(len(ga)) if ga[i].size and gb[i].size]
    ga, gb = [ga[i] for i in keep], [gb[i] for i in keep]
    if not ga:
        return {'point': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'B': int(B), 'n_groups': 0}
    fa, sa, za = _flatten(ga)
    fb, sb, zb = _flatten(gb)
    a0, b0 = float(stat(fa)), float(stat(fb))
    point = b0 - a0
    rng = np.random.default_rng(seed)
    n = len(ga)
    draws = np.empty(B, float)
    for i in range(B):
        idx = rng.integers(0, n, n)          # one frame draw, applied to BOTH arms
        draws[i] = (stat(_draw(fb, sb, zb, idx)) - stat(_draw(fa, sa, za, idx)))
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    opp = float(np.mean(draws >= 0.0) if point < 0 else np.mean(draws <= 0.0))
    return {'stat_a': a0, 'stat_b': b0, 'point': point, 'lo': float(lo), 'hi': float(hi),
            'p_two_sided': min(1.0, 2.0 * opp), 'B': int(B), 'n_groups': n}


def summarise(groups, B=10000, seed=0):
    """The four numbers the paper reports per cell, each with an interval."""
    stats = {'median': np.median, 'mae': np.mean,
             'rmse': lambda x: float(np.sqrt(np.mean(x ** 2))),
             'p95': lambda x: float(np.quantile(x, 0.95))}
    return {k: frame_bootstrap(groups, f, B=B, seed=seed) for k, f in stats.items()}


def fmt(d, places=3):
    """'0.055 [0.051, 0.059]' -- the form these go into the tables in."""
    if not np.isfinite(d.get('point', np.nan)):
        return 'n/a'
    return f"{d['point']:.{places}f} [{d['lo']:.{places}f}, {d['hi']:.{places}f}]"
