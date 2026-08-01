#!/usr/bin/env python3
"""Print-ready fiducial markers for the tape ground-truth session.

A marker's job is to answer "which pixel does this measurement belong to?". So it needs an
UNAMBIGUOUS CENTRE -- an X, whose crossing you both measure to and click on -- and enough
pixels on the sensor to click accurately.

ONE SIZE, DELIBERATELY. An earlier version graded the size by distance (5/9/15 cm), which
holds the pixel span roughly constant and is optically ideal. It is also unusable in practice:
markers get stuck to the field BEFORE the robot is parked, so the distance is not known when
the size has to be chosen.

12 cm is the size that removes the decision. Pixels spanned = fx * size / distance with the
rectified fx ~= 380 px:

      0.4 m   1 m    2 m    3 m    4 m   4.5 m
      114px  46px   23px   15px   11px   10px

Clickable across the entire range a robot on a 3.66 m VEX field can produce -- 10 px is
comfortable with the capture tool's x8 zoom inset, and 6 px is the floor. Two fit per
landscape A4.

Use a smaller marker only where 12 cm physically will not sit flat (a narrow box edge, a
chair rail) -- that is a decision about the SURFACE, which you can make while placing, not
about distance, which you cannot.

Every page carries a 100 mm scale bar. CHECK IT WITH A RULER after printing -- "fit to page"
is the default in many print dialogs and silently shrinks everything by a few percent, which
would make every marker the wrong size and, worse, look fine.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

A4_W, A4_H = 297.0, 210.0        # mm, LANDSCAPE (two 12 cm markers side by side)
MARGIN = 12.0
# Magenta: rare in indoor scenes, so it will not be confused with clutter, and it stays
# distinct from the grey/beige a VEX field and a lab wall are made of.
INK = '#C2185B'


def draw_marker(ax, x, y, size, mid):
    """One marker at (x, y) mm, `size` mm across, with an X whose crossing is the centre."""
    ax.add_patch(Rectangle((x, y), size, size, fill=False, ec='black', lw=0.7))
    ax.plot([x, x + size], [y, y + size], color=INK, lw=size * 0.045, solid_capstyle='butt')
    ax.plot([x, x + size], [y + size, y], color=INK, lw=size * 0.045, solid_capstyle='butt')
    # A small white disc at the crossing keeps the exact centre visible instead of buried
    # under two overlapping strokes -- that point is what gets measured and clicked.
    ax.add_patch(plt.Circle((x + size / 2, y + size / 2), size * 0.035,
                            fc='white', ec=INK, lw=0.6, zorder=3))
    ax.text(x + size / 2, y - 3.5, f'#{mid}   {size:.0f} mm',
            ha='center', va='top', fontsize=7, color='black')


def scale_bar(ax, y):
    x0 = MARGIN
    ax.plot([x0, x0 + 100], [y, y], color='black', lw=1.2)
    for xx in (x0, x0 + 50, x0 + 100):
        ax.plot([xx, xx], [y - 2, y + 2], color='black', lw=1.2)
    ax.text(x0 + 50, y + 4,
            'THIS LINE MUST MEASURE EXACTLY 100 mm — print at 100 %, NOT "fit to page"',
            ha='center', va='bottom', fontsize=7, color='black')


def page(pdf, markers, title):
    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W); ax.set_ylim(0, A4_H); ax.axis('off'); ax.set_aspect('equal')
    ax.text(MARGIN, A4_H - 8, title, fontsize=9, va='top', color='black')
    for x, y, s, mid in markers:
        draw_marker(ax, x, y, s, mid)
    scale_bar(ax, 12)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/demo/tape_markers.pdf')
    ap.add_argument('--size', type=float, default=120.0, help='marker side in mm')
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--small', type=int, default=4,
                    help='extra 6 cm markers for surfaces too narrow for the main size')
    a = ap.parse_args()

    S = a.size
    per_page = max(1, int((A4_W - 2 * MARGIN + 8) // (S + 8)))
    with PdfPages(a.out) as pdf:
        mid = 1
        while mid <= a.count:
            ms = []
            for c in range(per_page):
                if mid > a.count:
                    break
                x = MARGIN + c * (S + 8)
                ms.append((x, A4_H - MARGIN - 12 - S, S, mid))
                mid += 1
            page(pdf, ms, f'{S/10:.0f} cm — use at ANY distance 0.4–4.5 m')
        # a few small ones for surfaces the main size cannot sit flat on
        if a.small:
            small = 60.0
            ppp = max(1, int((A4_W - 2 * MARGIN + 8) // (small + 8)))
            done = 0
            while done < a.small:
                ms = []
                for c in range(ppp):
                    if done >= a.small:
                        break
                    ms.append((MARGIN + c * (small + 8),
                               A4_H - MARGIN - 12 - small, small, mid))
                    mid += 1; done += 1
                page(pdf, ms, '6 cm — only where 12 cm will not sit flat (narrow edges)')

    print(f'wrote {a.out}')
    print(f'{a.count} x {S/10:.0f} cm ({per_page} per landscape A4) + {a.small} x 6 cm')
    print('Print at 100% scale, LANDSCAPE, then CHECK the 100 mm bar with a ruler.')


if __name__ == '__main__':
    main()
