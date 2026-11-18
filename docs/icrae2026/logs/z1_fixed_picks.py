"""Run tools/diagnostics/zjul5_qual.py with its pick bug worked around, without editing it.

zjul5_qual.pick_indices returns positions into `stats` (scored images only), but main() uses
them as dataset indices (keep[i], by_index[i]). Once any test image is skipped, the two
diverge and the rendered image is not the one picked. Here pick_indices is wrapped to return
stats[pos]['index'] instead. Also dumps every image's per-image metrics, so the picks can
be checked.
"""
import json, os, sys

sys.path.insert(0, os.path.expanduser('~/RingFusion/tools/diagnostics'))
import zjul5_qual as Z  # noqa: E402

_orig = Z.pick_indices
OUT = sys.argv[sys.argv.index('--out') + 1]   # captured now: model loading later rewrites sys.argv


def fixed(stats, picks, far_m):
    chosen = [(p, stats[pos]['index']) for p, pos in _orig(stats, picks, far_m)]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'per_image.json'), 'w') as f:
        json.dump([{'index': s['index'], 'name': s['name'], 'far_frac': s['far_frac'],
                    'absrel_ours': s['all_ours']['absrel'],
                    'absrel_depthor': s['all_depthor']['absrel']} for s in stats], f, indent=0)
    print('picks (position -> dataset index):',
          [(p, pos, stats[pos]['index']) for (p, pos) in _orig(stats, picks, far_m)], flush=True)
    return chosen


Z.pick_indices = fixed
Z.main()
