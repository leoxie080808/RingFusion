"""Re-analysis of the LIVE-4 tape captures for revision 31. Reads only; runs anywhere.

Three questions the reviewer's comments put to the tape reference, all answerable from
the saved captures without going back to the robot:

  1. Which of the 15 points are the 11 the sigma constants were tuned on, and which are
     genuinely held out? The paper says "tuned on eleven of the points and scored here on
     all fifteen", which reads as fifteen independent points.
  2. What do the coverage fractions actually support? 10/11 and 11/15 are quoted to three
     decimals against targets 0.683 and 0.954.
  3. Can the out-of-cone claim of Section V-E be sourced from held-out points instead of
     from the calibration set?

TWO DIFFERENT INTERVALS, AND THE PAPER NEEDS BOTH. live4_sigma_n15 already carries lo/hi
for coverage -- but those are the spread across 5 REPEAT CAPTURES of the same 15 points,
so they measure pipeline noise on a stationary scene. They say nothing about how well 15
points pin down the coverage of the population of points we could have placed. That is
the binomial interval, and it is much wider. A claim that coverage "moved toward its
target" is a claim about the population, so it needs the binomial one.

Point matching is by pixel distance. Both sessions used the same camera on the same rig
in the same room, so a marker re-measured in the second session lands within a few px of
where it was in the first. Ground truth is compared as a tiebreak: two targets can sit
close in pixel space and far apart in depth, which is a depth discontinuity rather than a
re-measurement, and those are reported separately rather than silently merged.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import frame_bootstrap, wilson                      # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(HERE, 'docs', 'demo', 'benchmarks')

MATCH_PX = 45.0       # tape_eval's own neighbourhood radius; also the re-click tolerance
MATCH_GT_M = 0.10     # beyond this the two are different targets, not a re-measurement


def load(name):
    with open(os.path.join(BENCH, name)) as f:
        return json.load(f)


def match_points(cal_pts, test_pts):
    """-> (rows, n_reused, n_new). One row per test point, nearest calibration point."""
    rows = []
    for q in test_pts:
        best, bd = None, float('inf')
        for p in cal_pts:
            d = math.hypot(q['u'] - p['u'], q['v'] - p['v'])
            if d < bd:
                best, bd = p, d
        same_pixel = bd <= MATCH_PX
        dgt = abs(q['gt'] - best['gt']) if best else float('nan')
        if same_pixel and dgt <= MATCH_GT_M:
            role = 'calibration'          # same marker, re-measured
        elif same_pixel:
            role = 'ambiguous'            # close in pixels, far in depth: separate targets
        else:
            role = 'heldout'              # no calibration point anywhere near
        rows.append({'id': q['id'], 'u': q['u'], 'v': q['v'], 'gt': q['gt'],
                     'cone': q.get('cone'), 'what': q.get('what', ''),
                     'role': role, 'nearest_cal_px': round(bd, 1),
                     'nearest_cal_gt': best['gt'] if best else None,
                     'gt_delta_m': round(dgt, 3)})
    n_reused = sum(r['role'] == 'calibration' for r in rows)
    n_new = sum(r['role'] == 'heldout' for r in rows)
    return rows, n_reused, n_new


def _nsig(p):
    """|error| in sigmas. The n=11 captures predate the nsig field and store only err
    and sigma, so derive it rather than dropping those points from the coverage counts."""
    if p.get('nsig') is not None:
        return abs(p['nsig'])
    s = p.get('sigma')
    if not s:
        return None
    return abs(p['err']) / s


def coverage_block(pts, label):
    """Coverage at 1 and 2 sigma over a point set, with the BINOMIAL interval."""
    nsig = [x for x in (_nsig(p) for p in pts) if x is not None]
    n = len(nsig)
    k1 = sum(x <= 1.0 for x in nsig)
    k2 = sum(x <= 2.0 for x in nsig)
    lo1, hi1 = wilson(k1, n)
    lo2, hi2 = wilson(k2, n)
    return {'label': label, 'n': n,
            'cov1': {'k': k1, 'frac': k1 / n if n else float('nan'),
                     'wilson_lo': round(lo1, 3), 'wilson_hi': round(hi1, 3),
                     'target': 0.683},
            'cov2': {'k': k2, 'frac': k2 / n if n else float('nan'),
                     'wilson_lo': round(lo2, 3), 'wilson_hi': round(hi2, 3),
                     'target': 0.954},
            'worst_nsig': max(nsig) if nsig else float('nan')}


def error_block(pts, label, B=10000):
    """Median / MAE / RMSE over a point set, bootstrapped.

    Each point is its own resampling group: the tape reference is point-wise, so the
    unit of independence is the point, not a frame. With n this small the interval is
    the headline result -- it is what says whether the number can be quoted at all.
    """
    err = [[abs(p['err'])] for p in pts]
    out = {'label': label, 'n': len(pts)}
    if not pts:
        return out
    out['median'] = frame_bootstrap(err, np.median, B=B)
    out['mae'] = frame_bootstrap(err, np.mean, B=B)
    out['rmse'] = frame_bootstrap(err, lambda x: float(np.sqrt(np.mean(x ** 2))), B=B)
    return out


def main():
    cal = load('live4_sigma_2026-08-04.json')              # the 11 the constants were fit on
    cal_after = load('live4_sigma_after_2026-08-04.json')  # same 11, after the sigma terms
    test = load('live4_sigma_n15_2026-08-05.json')         # the 15-point re-test

    rows, n_reused, n_new = match_points(cal['points'], test['points'])
    by_role = {r['role'] for r in rows}
    heldout = [p for p, r in zip(test['points'], rows) if r['role'] in ('heldout', 'ambiguous')]
    strict_heldout = [p for p, r in zip(test['points'], rows) if r['role'] == 'heldout']

    pts = test['points']
    in_cone = [p for p in pts if p.get('cone') == 'in']
    out_cone = [p for p in pts if p.get('cone') == 'out']
    edge = [p for p in pts if p.get('cone') == 'edge']
    ho_out_cone = [p for p in heldout if p.get('cone') == 'out']

    report = {
        'generated_for': 'ringfusion_r31',
        'inputs': ['live4_sigma_2026-08-04.json', 'live4_sigma_after_2026-08-04.json',
                   'live4_sigma_n15_2026-08-05.json'],
        'label': 'r31 desk re-analysis of the saved LIVE-4 tape captures. No new capture.',

        'provenance': {
            'question': 'Are the 15 points independent of the 11 the constants were tuned on?',
            'match_px': MATCH_PX, 'match_gt_m': MATCH_GT_M,
            'n_calibration_points': len(cal['points']),
            'n_test_points': len(pts),
            'n_reused_from_calibration': n_reused,
            'n_ambiguous': sum(r['role'] == 'ambiguous' for r in rows),
            'n_strictly_heldout': n_new,
            'rows': rows,
            'finding': (f'{n_reused} of {len(pts)} points are calibration markers '
                        f're-measured; {n_new} are strictly held out. The independent '
                        f'sample is {n_new}, not {len(pts)}.'),
        },

        'coverage': {
            'note': ('Wilson intervals are the SAMPLING uncertainty from having this many '
                     'points. The lo/hi already in live4_sigma_n15 are the spread across 5 '
                     'repeat captures of the SAME points, i.e. pipeline noise. They are not '
                     'interchangeable and the paper needs the Wilson one for any claim '
                     'about where coverage sits relative to its target.'),
            'n11_before_sigma_terms': coverage_block(cal['points'], 'n=11, before'),
            'n11_after_sigma_terms': coverage_block(cal_after['points'], 'n=11, after'),
            'n15_all': coverage_block(pts, 'n=15, all points'),
            'n15_heldout_only': coverage_block(heldout, 'n=15, held out + ambiguous'),
            'repeat_spread_from_file': test.get('all_15'),
        },

        'accuracy': {
            'note': ('Depth error against tape. Bootstrapped over points, B=10000. These '
                     'intervals are what decides whether the tape reference can carry an '
                     'accuracy claim or only a validation one.'),
            'all_15': error_block(pts, 'all 15 points'),
            'in_cone': error_block(in_cone, 'inside the dToF cone'),
            'cone_edge': error_block(edge, 'on the cone edge'),
            'out_of_cone': error_block(out_cone, 'outside the dToF cone'),
            'heldout_only': error_block(heldout, 'held out + ambiguous'),
            'strictly_heldout': error_block(strict_heldout, 'strictly held out'),
        },

        'out_of_cone_claim': {
            'question': ('Section V-E quotes errors 1.07 and 1.52 m against sigma 2.10 and '
                         '1.84. Those are calibration-set points. Can the claim be re-sourced '
                         'to held-out points?'),
            'paper_source': {
                'file': 'live4_sigma_2026-08-04.json',
                'points': [{'id': p['id'], 'u': p['u'], 'v': p['v'], 'gt': p['gt'],
                            'pred': round(p['pred'], 3), 'err': round(p['err'], 3),
                            'sigma': round(p['sigma'], 3)}
                           for p in cal['points']
                           if abs(abs(p['err']) - 1.071) < 0.02 or abs(abs(p['err']) - 1.519) < 0.02],
                'role': 'CALIBRATION SET -- the constants were fitted on these points',
            },
            'heldout_replacement': [
                {'id': p['id'], 'what': p.get('what'), 'gt': p['gt'],
                 'pred': round(p['pred'], 3), 'err': round(p['err'], 3),
                 'sigma': round(p['sigma'], 3), 'nsig': round(p['nsig'], 3),
                 'pointing_sensitivity_m': round(p.get('best_within_45px', float('nan')), 3)}
                for p in ho_out_cone],
            'all_out_of_cone_n15': [
                {'id': p['id'], 'role': next(r['role'] for r in rows if r['id'] == p['id']),
                 'gt': p['gt'], 'err': round(p['err'], 3), 'sigma': round(p['sigma'], 3),
                 'nsig': round(p['nsig'], 3)}
                for p in out_cone],
        },

        'pointing_sensitivity': {
            'note': ('best_within_45px is how far off the BEST pixel within 45 px is. Where '
                     'that is much smaller than the reported error, most of the error is '
                     'where the click landed, not what the pipeline predicted.'),
            'points': [
                {'id': p['id'], 'cone': p.get('cone'), 'abs_err': round(abs(p['err']), 3),
                 'best_within_45px': round(p['best_within_45px'], 3),
                 'attributable_to_pointing': round(abs(p['err']) - p['best_within_45px'], 3),
                 'flagged_suspect': p.get('suspect_misclick', False)}
                for p in pts if p.get('best_within_45px', 0) > 0.05],
        },
    }

    out_path = os.path.join(BENCH, 'tape_stats_r31.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=1)

    # --- console summary -------------------------------------------------------------
    P = report['provenance']
    print(f"\n=== provenance ===\n  {P['finding']}")
    print(f"  {'id':<4}{'role':<13}{'cone':<6}{'gt':>7}  {'px to nearest cal':>18}  what")
    for r in rows:
        print(f"  {r['id']:<4}{r['role']:<13}{str(r['cone']):<6}{r['gt']:>7.3f}  "
              f"{r['nearest_cal_px']:>18.1f}  {r['what'][:34]}")

    print('\n=== coverage, with binomial (Wilson) intervals ===')
    for key in ('n11_before_sigma_terms', 'n11_after_sigma_terms', 'n15_all', 'n15_heldout_only'):
        c = report['coverage'][key]
        for s in ('cov1', 'cov2'):
            b = c[s]
            print(f"  {c['label']:<28} {s} {b['k']}/{c['n']} = {b['frac']:.3f}  "
                  f"95% [{b['wilson_lo']:.3f}, {b['wilson_hi']:.3f}]   target {b['target']}")

    print('\n=== depth error against tape, bootstrapped over points ===')
    for key in ('all_15', 'in_cone', 'cone_edge', 'out_of_cone', 'strictly_heldout'):
        e = report['accuracy'][key]
        if 'median' not in e:
            continue
        m, a = e['median'], e['mae']
        print(f"  {e['label']:<26} n={e['n']:<3} median {m['point']:.3f} "
              f"[{m['lo']:.3f}, {m['hi']:.3f}]   MAE {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")

    print('\n=== out-of-cone claim ===')
    for p in report['out_of_cone_claim']['paper_source']['points']:
        print(f"  paper quotes: err {abs(p['err']):.2f} m vs sigma {p['sigma']:.2f}  "
              f"(id {p['id']}, CALIBRATION set)")
    print('  held-out out-of-cone points available instead:')
    for p in report['out_of_cone_claim']['heldout_replacement']:
        print(f"    id{p['id']:<3} gt {p['gt']:.2f}  err {p['err']:+.3f}  sigma {p['sigma']:.3f}  "
              f"{p['nsig']:.2f} sigma   {str(p['what'])[:30]}")

    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
