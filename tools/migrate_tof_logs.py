"""One-time migration: mirror paired-log ToF maps to match tof_source.MIRROR_COLUMNS.

Paired logs recorded before 2026-07-28 hold the RAW TMF8829 map, whose column order runs
opposite to the camera's +x. `tof_source.MIRROR_COLUMNS` now flips that at the driver, so
every NEW log is already correct -- but training on a mix of old and new would feed the
net two contradictory conventions. This rewrites the old ones in place (after a backup).

Only `dist_m` and `confidence` are touched; the paired RGB is unaffected (it was always
stored rectified and in camera convention).

    python tools/migrate_tof_logs.py --tof-dir ros2_ws/data/real/tof --dry-run
    python tools/migrate_tof_logs.py --tof-dir ros2_ws/data/real/tof

Idempotent guard: a migrated file carries `mirrored=True`, and is skipped on re-runs.
"""
import argparse
import glob
import os
import shutil

import numpy as np


def migrate(path, dry=False):
    d = dict(np.load(path))
    if bool(d.get('mirrored', np.array(False))):
        return 'skip'
    if 'dist_m' not in d:
        return 'nodist'
    if dry:
        return 'would'
    d['dist_m'] = np.ascontiguousarray(np.fliplr(np.asarray(d['dist_m'], np.float32)))
    if 'confidence' in d:
        d['confidence'] = np.ascontiguousarray(np.fliplr(d['confidence']))
    d['mirrored'] = np.array(True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **d)
    os.replace(tmp, path)
    return 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-backup', action='store_true')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.tof_dir, '**', '*.npz'), recursive=True))
    if not files:
        print(f'no .npz under {a.tof_dir}')
        return
    print(f'{len(files)} files under {a.tof_dir}')

    if not a.dry_run and not a.no_backup:
        bak = a.tof_dir.rstrip('/') + '_prefix_backup'
        if os.path.exists(bak):
            print(f'backup already exists at {bak} -- refusing to overwrite it')
            return
        print(f'backing up -> {bak}')
        shutil.copytree(a.tof_dir, bak)

    n = {}
    for f in files:
        r = migrate(f, a.dry_run)
        n[r] = n.get(r, 0) + 1
    print('  ' + ', '.join(f'{k}: {v}' for k, v in sorted(n.items())))
    if a.dry_run:
        print('dry run -- nothing written')


if __name__ == '__main__':
    main()
