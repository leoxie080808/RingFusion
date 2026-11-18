#!/usr/bin/env python3
"""Record the frozen uncertainty constants and the commit they came from.

T3 requires that nothing is refitted on session 3: the sigma model must be fixed BEFORE the
tape points are captured, or the tape reference stops being independent and becomes a fit.
Saying so in prose is not evidence. This writes the actual values, read out of the modules
that the node imports, next to the commit hash and the dirty/clean state of the tree, so
"these constants, that commit" is checkable rather than asserted.

Run it before the session, commit the JSON, and tag that commit. Run it again after and the
two files must be identical.
"""
import importlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))

# module -> the constants that participate in sigma. Deliberately explicit: a wildcard sweep
# of module-level uppercase names would silently start or stop tracking things as the code
# moves, which is the opposite of a freeze.
WANTED = {
    'ringfusion_perception.blend': [
        'NEAR_DEG', 'FAR_DEG', 'BLEND_SCALE',          # arbitration ramp
        'SCENE_CAP_K', 'SCENE_CAP_FLOOR_M',            # far-field ceiling
        'DISAGREE_K', 'SUPPORT_FRAC', 'SPREAD_K', 'SPREAD_WIN',   # the three sigma terms
    ],
    'ringfusion_perception.pipeline': ['ROI_OUTSIDE_SIGMA_FRAC', 'ROI_MASK_STRIDE'],
    'ringfusion_perception.anchoring': ['RANGE_WEIGHT_P'],
}


def git(*args):
    try:
        return subprocess.check_output(('git',) + args, cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        return None


def main():
    out = os.path.join(_REPO, 'docs', 'demo', 'benchmarks', 't1',
                       'sigma_constants_frozen.json')
    if len(sys.argv) > 1:
        out = sys.argv[1]

    consts, missing = {}, []
    for mod, names in WANTED.items():
        m = importlib.import_module(mod)
        for n in names:
            if hasattr(m, n):
                consts[f'{mod.split(".")[-1]}.{n}'] = getattr(m, n)
            else:
                missing.append(f'{mod}.{n}')
    if missing:
        sys.exit('these constants no longer exist, so the freeze is not describing the '
                 'shipped model -- fix WANTED before relying on this file:\n  '
                 + '\n  '.join(missing))

    dirty = git('status', '--porcelain')
    doc = {'commit': git('rev-parse', 'HEAD'),
           'commit_subject': git('log', '-1', '--pretty=%s'),
           'tree_clean': dirty == '',
           # Named so it cannot be mistaken for a clean freeze in a later audit.
           'uncommitted_files': [] if not dirty else dirty.splitlines(),
           'tag': git('describe', '--tags', '--exact-match') or None,
           'constants': consts}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(doc, open(out, 'w'), indent=1, sort_keys=True)

    print(f'commit {doc["commit"][:8] if doc["commit"] else "??"} '
          f'({"clean" if doc["tree_clean"] else "DIRTY"})'
          + (f'  tag {doc["tag"]}' if doc['tag'] else '  tag: none yet'))
    for k, v in consts.items():
        print(f'  {k:42s} {v}')
    print(f'wrote {out}')
    if not doc['tree_clean']:
        print('\n!! tree is DIRTY, so this is not yet a freeze: the constants recorded here '
              'are not the ones any commit contains. Commit first, then re-run and tag.')


if __name__ == '__main__':
    main()
