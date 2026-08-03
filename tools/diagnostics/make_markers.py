#!/usr/bin/env python3
"""Print-ready fiducial markers for the tape ground-truth session.

A marker's job is to answer "which pixel does this measurement belong to?". So it needs an
UNAMBIGUOUS CENTRE -- an X, whose crossing you both measure to and click on -- and enough
pixels ON THE FULL-RES IMAGE to click accurately.

WHAT DOES NOT SET THE SIZE. It is tempting to size a marker so it covers several cells of
the depth map. That is unreachable: the backbone works at 384x288, so the depth grid has
fx ~= 89 px and even a 12 cm marker spans only 2.7 depth cells at 4 m. You would need a
half-metre board.

It does not matter, because the marker is not the object being measured. It is a label
saying "this pixel", lying flat against a wall or box whose depth is what we actually want.
The depth behind the paper IS the depth of the paper. (The one place this breaks: a marker
straddling a depth discontinuity, where the grid blurs foreground into background. Hence
the placement rule -- keep markers a hand's width clear of any edge or corner.)

So the only real constraint is clicking, which happens on the rectified 1640x1232 image
where fx = 379.6 px. Pixels spanned = fx * size / distance:

              0.4 m    1 m    2 m    3 m    4 m   4.5 m
      9 cm     85px   34px   17px   11px  8.5px  7.6px
     12 cm    114px   46px   23px   15px   11px   10px

6 px is the floor for placing a click; the capture tool's x8 zoom inset makes 8 px
comfortable. 9 cm clears the floor across the entire range a robot on a 3.66 m field can
produce, and -- the reason it is the default -- FOUR fit on one portrait A4 instead of two,
so twenty markers is five sheets rather than ten.

ONE SIZE, DELIBERATELY. An earlier version graded the size by distance (5/9/15 cm), which
holds the pixel span constant and is optically ideal. It is also unusable in practice:
markers get stuck to the field BEFORE the robot is parked, so the distance is not known
when the size has to be chosen.

Use a smaller marker only where 9 cm physically will not sit flat (a narrow box edge, a
chair rail) -- that is a decision about the SURFACE, which you can make while placing, not
about distance, which you cannot.

Every page carries a 100 mm scale bar. CHECK IT WITH A RULER after printing -- "fit to page"
is the default in many print dialogs and silently shrinks everything by a few percent, which
would put that same percent of bias into every measurement of the session and, worse, look
completely normal.
"""
import argparse
import glob
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

REPO = Path(__file__).resolve().parents[2]   # tools/diagnostics/ -> repo root

A4_W, A4_H = 210.0, 297.0        # mm, PORTRAIT (a 2x2 grid of 9 cm markers)
MARGIN = 10.0
TITLE_H = 10.0                   # reserved at the top
BAR_H = 14.0                     # reserved at the bottom for the scale bar
CAP_H = 6.0                      # caption strip under each marker, cut out WITH it
CUT_GAP = 10.0                   # mm of white between neighbours, for scissors
# Magenta: rare in indoor scenes, so it will not be confused with clutter, and it stays
# distinct from the grey/beige a VEX field and a lab wall are made of.
INK = '#C2185B'
CUT = '#B0B0B0'                  # cut guides: visible on paper, ignorable in the image


def draw_marker(ax, x, y, size, mid):
    """One marker with its lower-left at (x, y) mm, `size` mm across.

    The X's crossing is the centre: it is defined by four long edges rather than one small
    printed dot, so it survives being photographed at 8 px across.
    """
    ax.add_patch(Rectangle((x, y), size, size, fill=False, ec='black', lw=0.7))
    ax.plot([x, x + size], [y, y + size], color=INK, lw=size * 0.045, solid_capstyle='butt')
    ax.plot([x, x + size], [y + size, y], color=INK, lw=size * 0.045, solid_capstyle='butt')
    # A small white disc at the crossing keeps the exact centre visible instead of buried
    # under two overlapping strokes -- that point is what gets measured and clicked.
    ax.add_patch(plt.Circle((x + size / 2, y + size / 2), size * 0.035,
                            fc='white', ec=INK, lw=0.6, zorder=3))
    ax.text(x + size / 2, y - 1.5, f'#{mid}   {size:.0f} mm',
            ha='center', va='top', fontsize=7, color='black')


def scale_bar(ax, y):
    x0 = MARGIN
    ax.plot([x0, x0 + 100], [y, y], color='black', lw=1.2)
    for xx in (x0, x0 + 50, x0 + 100):
        ax.plot([xx, xx], [y - 2, y + 2], color='black', lw=1.2)
    ax.text(x0 + 50, y + 3.5,
            'THIS LINE MUST MEASURE EXACTLY 100 mm — print at 100 %, NOT "fit to page"',
            ha='center', va='bottom', fontsize=6.5, color='black')


def grid_positions(size, ncol, nrow):
    """Lower-left corners of an ncol x nrow grid, plus the cut lines between cells.

    Returns ([], []) rather than overlapping markers if the grid does not fit -- markers
    printed on top of each other would be discovered only after they were on the wall.
    Every neighbour is separated by at least CUT_GAP of white so scissors have somewhere
    to go without clipping the black border (which is what the corner-click mode needs).
    """
    uw = A4_W - 2 * MARGIN
    uh = A4_H - 2 * MARGIN - TITLE_H - BAR_H
    cell_h = size + CAP_H
    gap_x = (uw - ncol * size) / (ncol - 1) if ncol > 1 else 0.0
    gap_y = (uh - nrow * cell_h) / (nrow - 1) if nrow > 1 else 0.0
    if gap_x < (CUT_GAP if ncol > 1 else 0) or gap_y < (CUT_GAP if nrow > 1 else 0):
        return [], []
    gap_x = min(gap_x, 16.0)                      # keep columns from drifting to the edges
    gap_y = min(gap_y, 20.0)
    span_x = ncol * size + (ncol - 1) * gap_x
    span_y = nrow * cell_h + (nrow - 1) * gap_y
    x0 = MARGIN + (uw - span_x) / 2
    y_top = A4_H - MARGIN - TITLE_H - (uh - span_y) / 2

    out = []
    for r in range(nrow):
        for c in range(ncol):
            out.append((x0 + c * (size + gap_x),
                        y_top - (r + 1) * cell_h - r * gap_y + CAP_H))
    # Cut guides down the middle of each gutter, so "cut on the dashed line" always leaves
    # a clear border on both neighbours.
    vs = [x0 + c * (size + gap_x) + size + gap_x / 2 for c in range(ncol - 1)]
    hs = [y_top - (r + 1) * (cell_h + gap_y) + gap_y / 2 for r in range(nrow - 1)]
    return out, (vs, hs, x0, span_x, y_top, span_y)


def draw_cuts(ax, guides, pad=6.0):
    vs, hs, x0, span_x, y_top, span_y = guides
    for x in vs:
        ax.plot([x, x], [y_top - span_y - pad, y_top + pad], color=CUT, lw=0.5,
                ls=(0, (3, 3)), zorder=0)
    for y in hs:
        ax.plot([x0 - pad, x0 + span_x + pad], [y, y], color=CUT, lw=0.5,
                ls=(0, (3, 3)), zorder=0)


def page(pdf, markers, guides, title):
    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W); ax.set_ylim(0, A4_H); ax.axis('off'); ax.set_aspect('equal')
    ax.text(MARGIN, A4_H - 6, title, fontsize=9, va='top', color='black')
    draw_cuts(ax, guides)
    for x, y, s, mid in markers:
        draw_marker(ax, x, y, s, mid)
    scale_bar(ax, MARGIN + 4)
    pdf.savefig(fig); plt.close(fig)


def emit(pdf, count, size, ncol, nrow, title, first_id):
    """Lay `count` markers of `size` across as many pages as the grid needs."""
    slots, guides = grid_positions(size, ncol, nrow)
    if not slots:
        raise SystemExit(f'{size:.0f} mm will not fit a {ncol}x{nrow} grid on A4 portrait '
                         f'with a {CUT_GAP:.0f} mm cutting gap')
    mid = first_id
    while mid < first_id + count:
        ms = []
        for x, y in slots:
            if mid >= first_id + count:
                break
            ms.append((x, y, size, mid))
            mid += 1
        page(pdf, ms, guides, title)
    return mid


def main():
    ap = argparse.ArgumentParser()
    # Anchored to the repo, not the shell's cwd: the natural place to run this from is
    # ros2_ws (where everything else lives), and a cwd-relative default fails there with
    # a matplotlib traceback that says nothing about the actual problem.
    ap.add_argument('--out', default=str(REPO / 'docs/demo/tape_markers.pdf'))
    ap.add_argument('--size', type=float, default=90.0, help='marker side in mm')
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--cols', type=int, default=2)
    ap.add_argument('--rows', type=int, default=2)
    ap.add_argument('--small', type=int, default=4,
                    help='extra 5 cm markers for surfaces too narrow for the main size')
    # PNGs are written by default, not on request: they are the copy that gets committed
    # and looked at, so a run that refreshed only the PDF would leave stale sheets in the
    # repo that still LOOK like the current ones.
    ap.add_argument('--no-png', action='store_true',
                    help='skip the per-page PNGs (PDF only)')
    ap.add_argument('--dpi', type=int, default=300)
    a = ap.parse_args()

    per_page = a.cols * a.rows
    with PdfPages(a.out) as pdf:
        nxt = emit(pdf, a.count, a.size, a.cols, a.rows,
                   f'{a.size/10:.0f} cm — use at ANY distance 0.4–4.5 m '
                   f'— cut on the dashed lines', 1)
        if a.small:
            # Grid sized to the count so the sheet does not print half-empty with cut
            # guides running through blank paper.
            sc = min(3, a.small)
            sr = max(1, -(-a.small // sc))
            emit(pdf, a.small, 50.0, sc, sr,
                 '5 cm — only where 9 cm will not sit flat (narrow edges, rails)', nxt)

    pages = -(-a.count // per_page) + (1 if a.small else 0)
    print(f'wrote {a.out}')
    print(f'{a.count} x {a.size/10:.0f} cm ({per_page} per portrait A4) '
          f'+ {a.small} x 5 cm  =  {pages} pages')
    print(f'{CUT_GAP:.0f} mm cutting gap between neighbours, dashed guides down each gutter')

    if not a.no_png:
        stem = a.out[:-4] if a.out.endswith('.pdf') else a.out
        for old in sorted(glob.glob(f'{stem}_p*.png')):
            os.remove(old)                     # or a shrunk run leaves orphaned pages behind
        subprocess.run(['pdftoppm', '-png', '-r', str(a.dpi), '-aa', 'yes',
                        '-aaVector', 'yes', a.out, f'{stem}_p'], check=True)
        for f in sorted(glob.glob(f'{stem}_p-*.png')):   # pdftoppm writes "_p-1.png"
            os.rename(f, f.replace('_p-', '_p'))
        made = sorted(glob.glob(f'{stem}_p*.png'))
        print(f'wrote {len(made)} PNG at {a.dpi} dpi: {made[0]} .. {made[-1]}')
        if len(made) != pages:
            raise SystemExit(f'PNG count {len(made)} != {pages} pages -- stale files?')

    print('Print at 100% scale, PORTRAIT. The 100 mm bar is a sanity check on how big the '
          'markers came out; it does NOT affect accuracy (nothing measures the marker).')


if __name__ == '__main__':
    main()
