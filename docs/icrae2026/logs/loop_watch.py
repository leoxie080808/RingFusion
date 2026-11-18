"""Watch a record_map_frames.py output dir; SIGINT the recorder when the view returns to frame 0.

Usage: loop_watch.py DIR PID [min_secs] [thresh] [hold_frames]
Closed = mean abs diff of small blurred greyscale frames vs frame 0 below `thresh`
for `hold_frames` consecutive frames, after at least `min_secs` of recording.
"""
import glob, os, signal, sys, time
import cv2, numpy as np

d, pid = sys.argv[1], int(sys.argv[2])
min_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 14.0
hold = int(sys.argv[5]) if len(sys.argv) > 5 else 8


def small(p):
    im = cv2.imread(p, 0)
    if im is None:
        return None
    return cv2.GaussianBlur(cv2.resize(im, (80, 60), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)


def alive():
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


ref = None
while alive() and ref is None:
    fs = sorted(glob.glob(os.path.join(d, 'rgb', '*.jpg')))
    if len(fs) >= 2:
        ref = small(fs[0])
    time.sleep(0.3)
t0 = time.time()
print(f'watch: reference frame captured; closing allowed after {min_secs:.0f} s', flush=True)

seen, run, best, last_print = 0, 0, 1e9, 0
while alive():
    fs = sorted(glob.glob(os.path.join(d, 'rgb', '*.jpg')))
    for p in fs[seen:-1]:                       # skip the newest; it may be half-written
        cur = small(p)
        if cur is None:
            continue
        diff = float(np.abs(cur - ref).mean())
        el = time.time() - t0
        if el > min_secs:
            best = min(best, diff)
            run = run + 1 if diff < thresh else 0
        if el - last_print > 5:
            print(f'watch: {el:5.1f} s  frame {os.path.basename(p)}  diff vs start {diff:5.1f}  '
                  f'(best after {min_secs:.0f}s: {best if best < 1e9 else float("nan"):.1f})', flush=True)
            last_print = el
        if run >= hold:
            print(f'watch: LOOP CLOSED at {el:.1f} s, frame {os.path.basename(p)}, diff {diff:.1f} -> stopping recorder', flush=True)
            os.kill(pid, signal.SIGINT)
            sys.exit(0)
    seen = max(seen, len(fs) - 1)
    time.sleep(0.2)
print(f'watch: recorder exited on its own (cap reached); best diff after {min_secs:.0f}s was {best:.1f}', flush=True)
