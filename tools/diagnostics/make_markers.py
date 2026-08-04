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

INK WEIGHT: POINTS ARE NOT MILLIMETRES (fixed 2026-08-04)
--------------------------------------------------------
The first version sized the X with `lw=size * 0.045`, which reads like "4.5 % of the marker"
-- 4 mm on a 90 mm marker. It is not. Matplotlib line widths are in POINTS, and 1 pt =
0.353 mm, so that expression printed a **1.43 mm** stroke: barely a third of the intended
width. The border was worse, `lw=0.7` = **0.25 mm**.

The consequence only shows up in the field. Marker legibility is not set by the outline but
by the thinnest printed feature, and at 1640x1232 one pixel subtends distance/379.6:

                    0.5 m    1 m     2 m     3 m    4.5 m
   1 px covers      1.3mm   2.6mm   5.3mm   7.9mm   11.9mm
   old X, 1.43 mm   1.1px   0.5px   0.3px   0.2px    0.1px   <- sub-pixel past ~0.5 m
   new X, 8.1 mm    6.2px   3.1px   1.5px   1.0px    0.7px

Sub-pixel ink does not render faint, it renders as CONTRAST COLLAPSE -- the stroke averages
into the white paper around it and the marker becomes a pale square. That is why the tape
session could not read markers past ~2 m and why automatic detection failed every way it was
tried: the ink was never actually in the image to detect.

So every dimension in this file is now declared in mm and converted with pt() at the point of
use. Nothing here is expressed in points.
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

# Ink weights, as a fraction of the marker side. Held as fractions rather than absolute mm
# so the 5 cm markers get proportionally scaled ink instead of the 9 cm weights, which would
# swallow them. Values are in MILLIMETRES once multiplied -- see pt() before using them.
STROKE_FRAC = 0.090              # X limbs      -> 8.1 mm on a 90 mm marker
DOT_FRAC    = 0.065              # centre disc RADIUS -> 5.85 mm (11.7 mm across)
BORDER_FRAC = 0.013              # square outline -> 1.2 mm

MM_PER_PT = 25.4 / 72.0


def pt(mm):
    """mm -> matplotlib points.

    Every dimension in this file is authored in mm because that is what comes out of the
    printer and what a ruler checks. Matplotlib line widths are in points, and conflating
    the two is exactly the bug described in the module docstring, so the conversion is
    never done by eye.
    """
    return mm / MM_PER_PT


def draw_marker(ax, x, y, size, mid):
    """One marker with its lower-left at (x, y) mm, `size` mm across.

    The X's crossing is the centre: it is defined by four long edges rather than one small
    printed dot, so it survives being photographed at 8 px across.
    """
    stroke = size * STROKE_FRAC
    border = Rectangle((x, y), size, size, fill=False, ec='black',
                       lw=pt(size * BORDER_FRAC))
    ax.add_patch(border)
    # The limbs are CLIPPED to the square, not merely butt-capped. A butt cap is square to
    # the line's own direction, so on a 45-degree diagonal it still projects a triangle of
    # ink past each corner -- at 8 mm that is several millimetres outside the border, and the
    # border is exactly the edge corner-clicking looks for. Clipping is the only thing that
    # actually keeps the ink inside the cut line.
    for xs, ys in (([x, x + size], [y, y + size]),
                   ([x, x + size], [y + size, y])):
        line, = ax.plot(xs, ys, color=INK, lw=pt(stroke), solid_capstyle='butt')
        line.set_clip_path(border)
    # A white disc at the crossing keeps the exact centre visible instead of buried under two
    # overlapping strokes -- that point is what gets measured and clicked. It must stay
    # comfortably wider than the limbs or the bold X simply covers it.
    ax.add_patch(plt.Circle((x + size / 2, y + size / 2), size * DOT_FRAC,
                            fc='white', ec=INK, lw=pt(size * BORDER_FRAC), zorder=3))
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
    # Declared up front: Python rejects a `global` that appears after the name has already
    # been read in the same scope, and the argparse defaults below read both of these.
    global STROKE_FRAC, DOT_FRAC

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
    ap.add_argument('--stroke-frac', type=float, default=STROKE_FRAC,
                    help='X limb width as a fraction of the marker side '
                         f'(default {STROKE_FRAC} = {90*STROKE_FRAC:.1f} mm at 90 mm)')
    ap.add_argument('--dot-frac', type=float, default=DOT_FRAC,
                    help='centre disc RADIUS as a fraction of the marker side '
                         f'(default {DOT_FRAC} = {2*90*DOT_FRAC:.1f} mm across at 90 mm)')
    a = ap.parse_args()

    STROKE_FRAC, DOT_FRAC = a.stroke_frac, a.dot_frac
    if DOT_FRAC * 2 <= STROKE_FRAC * 1.3:
        raise SystemExit(f'centre disc ({2*a.size*DOT_FRAC:.1f} mm across) is not clearly '
                         f'wider than the X limbs ({a.size*STROKE_FRAC:.1f} mm) -- the bold '
                         f'X would swallow it. Raise --dot-frac or lower --stroke-frac.')

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
    stroke = a.size * STROKE_FRAC
    print(f'ink: X limbs {stroke:.1f} mm, centre disc {2*a.size*DOT_FRAC:.1f} mm across, '
          f'border {a.size*BORDER_FRAC:.1f} mm')
    print('  X limb width in camera pixels (1640x1232, fx=379.6):')
    print('   ' + ''.join(f'{d:>8}' for d in ['0.5 m', '1 m', '2 m', '3 m', '4.5 m']))
    print('   ' + ''.join(f'{379.6 * (stroke/1000.0) / d:>8.1f}'
                          for d in [0.5, 1.0, 2.0, 3.0, 4.5]))
    print('  (below ~1 px the stroke averages into the paper and the marker goes pale)')

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
