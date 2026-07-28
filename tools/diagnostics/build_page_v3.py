#!/usr/bin/env python3
"""Build the current-state RingFusion analytics page (geometry fix + Network B v3)."""
import base64
import json

DEMO = "/home/leroi-ultio/RingFusion/docs/demo"
OUT = f"{DEMO}/network_b_v1_vs_v2.html"          # same file -> same artifact URL

import sys
sys.path.insert(0, "/tmp/claude-1000/-home-leroi-ultio-RingFusion-ros2-ws/"
                   "5313505d-56c6-4b05-9a49-5a77f3868308/scratchpad")
import viewer_snippet

AB = json.load(open(f"{DEMO}/moving_ab_v3.json"))["summary"]
CLIP = ("data:video/mp4;base64," +
        base64.b64encode(open(f"{DEMO}/pipeline_clip_v3.mp4", "rb").read()).decode())
CSEQ = ("data:video/mp4;base64," +
        base64.b64encode(open(f"{DEMO}/clips/drive_cloudseq_colordepth.mp4", "rb").read()).decode())
CMETA = json.load(open(f"{DEMO}/clips/drive_cloudseq_meta.json"))
VIEWER = viewer_snippet.build(CSEQ, CMETA)

# sigma_cal.py, 8 frames, ~940 anchors/frame
SIG = [
    ("v1",      0.179, 0.159, 0.612, 0.958, 0.490),
    ("v2",      0.239, 0.091, 0.264, 0.895, 0.913),
    ("v3 best", 0.198, 0.078, 0.578, 0.976, 0.943),
    ("v3 last", 0.198, 0.079, 0.590, 0.973, 0.942),
]

HTML = f"""<style>
:root {{
  --bg:#0e1217; --panel:#161d24; --panel-2:#1d262e; --line:#2b3841; --hair:#3c4c5722;
  --ink:#e8eff5; --muted:#93a4b3; --faint:#6b7d8b;
  --accent:#5ad1c8; --good:#4cc76b; --warn:#e3aa2e; --crit:#f2685e;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme: light) {{
  :root {{ --bg:#f3f6f8; --panel:#fff; --panel-2:#e9eff4; --line:#d5dee6; --hair:#8aa0b422;
    --ink:#101820; --muted:#54626e; --faint:#7b8a97;
    --accent:#0c8a80; --good:#1e8a3f; --warn:#966806; --crit:#c13b32; }}
}}
:root[data-theme="dark"] {{
  --bg:#0e1217; --panel:#161d24; --panel-2:#1d262e; --line:#2b3841; --hair:#3c4c5722;
  --ink:#e8eff5; --muted:#93a4b3; --faint:#6b7d8b;
  --accent:#5ad1c8; --good:#4cc76b; --warn:#e3aa2e; --crit:#f2685e;
}}
:root[data-theme="light"] {{
  --bg:#f3f6f8; --panel:#fff; --panel-2:#e9eff4; --line:#d5dee6; --hair:#8aa0b422;
  --ink:#101820; --muted:#54626e; --faint:#7b8a97;
  --accent:#0c8a80; --good:#1e8a3f; --warn:#966806; --crit:#c13b32;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:40px 22px 76px;
  display:flex; flex-direction:column; gap:38px; }}
header {{ display:flex; flex-direction:column; gap:9px;
  border-bottom:1px solid var(--line); padding-bottom:22px; }}
.eyebrow {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); }}
h1 {{ font-size:28px; font-weight:640; margin:0; letter-spacing:-.015em; text-wrap:balance; }}
.lede {{ color:var(--muted); font-size:15px; margin:0; max-width:68ch; }}
.lede b {{ color:var(--ink); font-weight:600; }}
h2 {{ font-size:12px; font-family:var(--mono); letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin:0 0 15px; font-weight:600; }}
h3 {{ font-size:15.5px; margin:0 0 4px; font-weight:620; }}
p {{ margin:0 0 10px; }}
.vid {{ background:#0a0d10; border:1px solid var(--line); border-radius:10px;
  overflow:hidden; line-height:0; }}
.vid video {{ display:block; width:100%; height:auto; }}
.vnote {{ display:grid; grid-template-columns:repeat(4,1fr); gap:11px; margin-top:13px; }}
@media (max-width:860px) {{ .vnote {{ grid-template-columns:repeat(2,1fr); }} }}
.vnote div {{ background:var(--panel); border:1px solid var(--line); border-radius:9px;
  padding:11px 13px; font-size:13px; color:var(--muted); }}
.vnote b {{ color:var(--ink); font-weight:620; }}
.tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
@media (max-width:800px) {{ .tiles {{ grid-template-columns:repeat(2,1fr); }} }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 15px; display:flex; flex-direction:column; gap:3px; }}
.tile .k {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--faint); }}
.tile .v {{ font-family:var(--mono); font-size:23px; font-weight:640;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
.tile .n {{ font-size:12.5px; color:var(--muted); }}
.v.good {{ color:var(--good); }} .v.crit {{ color:var(--crit); }}
.note {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:15px 17px; }}
.note.warn {{ border-left:3px solid var(--warn); }}
.note.good {{ border-left:3px solid var(--good); }}
.note p {{ margin:0; color:var(--muted); font-size:14px; max-width:74ch; }}
.note b {{ color:var(--ink); }}
.tblwrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ text-align:left; padding:10px 15px; border-bottom:1px solid var(--hair); }}
thead th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--faint); background:var(--panel-2);
  border-bottom:1px solid var(--line); font-weight:600; }}
tbody tr:last-child td {{ border-bottom:none; }}
td.n, th.n {{ font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.d {{ color:var(--muted); font-size:13px; }}
tr.win td {{ background:color-mix(in srgb, var(--good) 9%, transparent); }}
.win b {{ color:var(--good); }}
code {{ font-family:var(--mono); font-size:12.5px; background:var(--panel-2);
  padding:1px 5px; border-radius:4px; }}
.gloss {{ display:flex; flex-direction:column; gap:12px; }}
.cloudwrap {{ background:#0b0e11; border:1px solid var(--line); border-radius:10px;
  overflow:hidden; }}
.cloudwrap canvas {{ display:block; width:100%; height:clamp(340px, 52vh, 560px);
  cursor:grab; touch-action:none; }}
.cloudwrap canvas:active {{ cursor:grabbing; }}
.pcbar {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:10px 13px; border-top:1px solid var(--line); background:var(--panel-2); }}
.pcbar button {{ font-family:var(--mono); font-size:11.5px; color:var(--ink);
  background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:5px 10px; cursor:pointer; }}
.pcbar button:hover {{ border-color:var(--accent); color:var(--accent); }}
.pcbar button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.pcbar .sep {{ width:1px; height:18px; background:var(--line); margin:0 4px; }}
.pcbar .pchint {{ font-size:12px; color:var(--faint); margin-left:auto; }}
.pctransport {{ gap:10px; }}
.pctransport button {{ min-width:34px; }}
#pcplay {{ min-width:58px; }}
#pcseek {{ flex:1; min-width:140px; -webkit-appearance:none; appearance:none;
  height:5px; border-radius:3px; background:var(--line); outline:none; cursor:pointer; }}
#pcseek::-webkit-slider-thumb {{ -webkit-appearance:none; appearance:none;
  width:14px; height:14px; border-radius:50%; background:var(--accent);
  border:2px solid var(--panel); cursor:grab; }}
#pcseek::-moz-range-thumb {{ width:12px; height:12px; border-radius:50%;
  background:var(--accent); border:2px solid var(--panel); cursor:grab; }}
#pcseek:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}
.pctime {{ font-family:var(--mono); font-size:11.5px; color:var(--muted);
  font-variant-numeric:tabular-nums; white-space:nowrap; min-width:92px;
  text-align:right; }}
@media (prefers-reduced-motion: reduce) {{ .cloudwrap canvas {{ scroll-behavior:auto; }} }}
footer {{ border-top:1px solid var(--line); padding-top:16px; color:var(--faint);
  font-size:12.5px; font-family:var(--mono); }}
</style>

<div class="wrap">

<header>
  <div class="eyebrow">RingFusion · on-robot analytics · 2026-07-28</div>
  <h1>Calibration fix, and Network B v3</h1>
  <p class="lede">Two changes since the last run. The ToF↔camera projection turned out to be
    <b>mirrored and axis-swapped</b> — a real calibration bug that had been corrupting every
    measurement. And Network B was <b>retrained on the corrected geometry</b>. Together they fix
    the far-field blowup, the depth bias, and the uncertainty channel.</p>
</header>

<section>
  <h2>The pipeline running — 4 panels, live</h2>
  <div class="vid">
    <video src="{CLIP}" controls loop muted playsinline preload="metadata"
           aria-label="Camera, ToF heatmap, fused depth and top-down obstacle map side by side"></video>
  </div>
  <div class="vnote">
    <div><b>Camera</b> — raw view, fisheye corrected.</div>
    <div><b>ToF 32×32</b> — the sensor's own readings, now un-mirrored. Black squares got no
      return. The only real distance measurement in the system.</div>
    <div><b>Fused depth</b> — camera detail carrying the ToF's scale, at full resolution.</div>
    <div><b>Top-down</b> — the depth map unprojected to a bird's-eye map. Robot at bottom-centre,
      rings at 1/2/3 m, ground faint grey, obstacles bright and coloured by distance.</div>
  </div>
  <p style="color:var(--muted);font-size:14px;margin-top:12px;max-width:74ch">Colour is fixed
    across the clip (blue near, red far) rather than auto-scaled per frame, so a colour means the
    same distance at second 1 and second 34. Motion verified rather than assumed: consecutive
    camera frames differ by 10.75 grey levels against 1.76 for a stationary capture, and the view
    diverges from its start and stays diverged.</p>
</section>

<section>
  <h2>Fly through the depth cloud</h2>
  <p style="color:var(--muted);font-size:14px;margin:0 0 14px;max-width:74ch">The same drive as
    3D geometry. Every point is one pixel of <code>/depth</code> unprojected through the camera
    intrinsics, coloured from the camera. It plays as the drive plays — <b>drag to orbit, scroll
    to zoom, at any moment</b>. Pause to hold a frame and inspect it.</p>
  {VIEWER}
  <div class="note" style="margin-top:14px">
    <p>Each frame is an <b>independent cloud from the robot's current viewpoint</b>, not a
      stitched map of the room — this stack has no odometry, so there is nothing to register
      successive frames against. You are flying around what the robot can see right now, and it
      is a floating shell rather than a solid: only surfaces facing the camera are ever measured.</p>
    <p style="margin-top:9px">Depth is 8-bit over 0.15–4.0 m (~15 mm per level), and video
      compression adds ~26 mm on top — both well under the pipeline's own ~200 mm error. Points
      beyond 4 m are dropped, so the far field is cut off rather than wrong.</p>
  </div>
</section>

<section>
  <h2>The calibration bug</h2>
  <div class="tiles">
    <div class="tile"><span class="k">disparity ↔ truth</span>
      <span class="v good">0.737→0.917</span><span class="n">ρ against real ToF</span></div>
    <div class="tile"><span class="k">best-case MAE</span>
      <span class="v good">0.319→0.251</span><span class="n">metres, −21%</span></div>
    <div class="tile"><span class="k">centre-depth bias</span>
      <span class="v good">−14%→+6.5%</span><span class="n">matched region vs ToF</span></div>
    <div class="tile"><span class="k">/depth rate</span>
      <span class="v">13.7<span style="font-size:13px;color:var(--muted)"> Hz</span></span>
      <span class="n">unchanged, v3 is free</span></div>
  </div>
  <div class="note good" style="margin-top:14px">
    <p>The ToF grid was <b>horizontally mirrored</b> relative to the camera — pan the robot right
      and the heatmap slid left — and <code>fov_h</code>/<code>fov_v</code> were <b>swapped</b>
      (the module sees taller than it is wide, configured as the reverse). Two independent errors
      that compounded. Found by sweeping every grid orientation × FOV assignment and scoring each
      by how well the backbone's disparity tracked true inverse depth; confirmed by a visual
      overlay and by eye on the previous clip.</p>
    <p style="margin-top:9px"><b>It also overturned an earlier conclusion.</b> Measured under the
      broken projection, the Depth Anything V2 teacher scored ρ 0.750 against the student's 0.737,
      which read as "monocular depth is intrinsically hard here, the student is at its teacher's
      ceiling". Wrong — both were being scored against misprojected anchors. Corrected, the
      student alone reaches <b>0.917</b>. The backbone was never the bottleneck.</p>
  </div>
</section>

<section>
  <h2>Uncertainty — the channel that was broken</h2>
  <div class="tblwrap">
    <table>
      <thead><tr><th>Engine</th><th class="n">MAE</th><th class="n">median σ</th>
        <th class="n">coverage ±1σ</th><th class="n">coverage ±2σ</th>
        <th class="n">corr(σ, |error|)</th></tr></thead>
      <tbody>
        {''.join(
          f'<tr class="{"win" if nm.startswith("v3 last") else ""}"><td>{"<b>" if nm.startswith("v3 last") else ""}{nm}{"</b>" if nm.startswith("v3 last") else ""}</td>'
          f'<td class="n">{mae:.3f} m</td><td class="n">{ms:.3f} m</td>'
          f'<td class="n">{c1:.3f}</td><td class="n">{c2:.3f}</td>'
          f'<td class="n">{cr:.3f}</td></tr>'
          for nm, mae, ms, c1, c2, cr in SIG)}
        <tr><td class="d">ideal</td><td class="d">—</td><td class="d">—</td>
          <td class="n d">0.683</td><td class="n d">0.955</td><td class="d">higher</td></tr>
      </tbody>
    </table>
  </div>
  <div class="note" style="margin-top:14px">
    <p><b>corr(σ, |error|) is the one that matters.</b> It asks whether the model's stated
      confidence actually tracks where it is wrong. v1 scored 0.490 — barely better than a
      constant, and it was <i>most confident exactly where it was most wrong</i>. v3 reaches
      <b>0.943</b>. Combined with σ at the anchors being 0.078 m while the whole-frame median
      stays near 0.84 m, the shape is now correct: confident where the ToF constrains it,
      uncertain where it is extrapolating.</p>
    <p style="margin-top:9px">Coverage is the fraction of true depths inside the ±1σ band; 0.683
      is the calibrated ideal. v3's 0.590 is mildly overconfident but far better than v2's 0.264,
      which was severely so — a failure only visible once the geometry was fixed.</p>
  </div>
</section>

<section>
  <h2>Under motion — {AB['frames']} frames over {AB['duration_s']:.0f} s of driving</h2>
  <div class="tblwrap">
    <table>
      <thead><tr><th>Metric</th><th class="n">Closed-form</th><th class="n">v2</th>
        <th class="n">v3</th><th>Want</th></tr></thead>
      <tbody>
        <tr class="win"><td><b>anchor MAE</b> (vs real ToF)</td>
          <td class="n">{AB['A']['anchor_mae_mean']:.3f} m</td>
          <td class="n">{AB['v1']['anchor_mae_mean']:.3f} m</td>
          <td class="n"><b>{AB['v2']['anchor_mae_mean']:.3f} m</b></td><td class="d">lower</td></tr>
        <tr><td>max depth (mean)</td><td class="n">{AB['A']['max_mean']:.2f} m</td>
          <td class="n">{AB['v1']['max_mean']:.2f} m</td>
          <td class="n">{AB['v2']['max_mean']:.2f} m</td><td class="d">≤ ToF's {AB['tof_max_over_run']:.1f} m</td></tr>
        <tr><td>frames at 20 m clamp</td><td class="n">0</td><td class="n">0</td>
          <td class="n">0</td><td class="d">zero</td></tr>
        <tr><td>pixels &gt; 5 m</td><td class="n">{AB['A']['frac_gt5_mean_pct']:.3f}%</td>
          <td class="n">{AB['v1']['frac_gt5_mean_pct']:.3f}%</td>
          <td class="n">{AB['v2']['frac_gt5_mean_pct']:.3f}%</td><td class="d">zero</td></tr>
        <tr><td>max depth (worst frame)</td><td class="n">{AB['A']['max_worst']:.2f} m</td>
          <td class="n">{AB['v1']['max_worst']:.2f} m</td>
          <td class="n">{AB['v2']['max_worst']:.2f} m</td><td class="d">≤ ToF's {AB['tof_max_over_run']:.1f} m</td></tr>
        <tr><td>frame-to-frame jump</td><td class="n">{AB['A']['jump_mean_m']:.4f} m</td>
          <td class="n">{AB['v1']['jump_mean_m']:.4f} m</td>
          <td class="n">{AB['v2']['jump_mean_m']:.4f} m</td><td class="d">near baseline</td></tr>
        <tr><td>median depth</td><td class="n">{AB['A']['median_depth_mean']:.3f} m</td>
          <td class="n">{AB['v1']['median_depth_mean']:.3f} m</td>
          <td class="n">{AB['v2']['median_depth_mean']:.3f} m</td><td class="d">plausible</td></tr>
      </tbody>
    </table>
  </div>
  <p style="color:var(--muted);font-size:14px;margin-top:12px;max-width:74ch">v3 is the most
    accurate of the three — <b>32% better than the closed-form path</b> and 19% better than v2 —
    which is the key difference from v2, which used to be conservative <i>and</i> less accurate
    than doing nothing. It costs almost nothing in stability: frame-to-frame jump is 0.0290 m
    against the raw backbone's 0.0271 m, about 7% extra. For contrast, on the pre-fix stack v1
    <b>doubled</b> the jitter and slammed into the 20 m clamp on 47 of 102 moving frames.</p>
  <div class="note good" style="margin-top:14px">
    <p><b>Nothing hit the clamp this time — not even v2.</b> Zero frames above 5 m, zero above
      3 m, for all three engines, with the ToF seeing out to {AB['tof_max_over_run']:.1f} m. Fed
      correct geometry, v2 behaves. That says the old 20 m catastrophe was substantially a
      <i>calibration</i> artefact rather than purely a training failure — the residual was
      extrapolating wildly because its anchors were landing on the wrong pixels.</p>
  </div>
</section>

<section>
  <h2>Still open</h2>
  <div class="gloss">
    <div class="note warn"><h3>The v2-vs-v3 comparison confounds two changes</h3>
      <p>v3 was retrained on corrected geometry <i>and</i> at a lower structure-loss weight
        (0.15 vs 0.3). Both moved together, so how much of its win comes from which is not
        separable from this run. A control at weight 0.3 on corrected geometry would settle it.</p></div>
    <div class="note warn"><h3>v3 still under-corrects</h3>
      <p>It deviates from the closed-form path by only 0.019 m on average. The structure-loss
        weight was lowered from 0.3 to 0.15 and accuracy improved, so there is likely more to gain
        going lower still.</p></div>
    <div class="note warn"><h3>ToF covers 7.5% of the frame</h3>
      <p>Unchanged by any of this. The sensor's readings land in a small central box; everything
        outside is monocular extrapolation carrying the affine fit.</p></div>
    <div class="note warn"><h3>Checkpoint selection is noisy</h3>
      <p>The validation split is ~62 samples against a heavy-tailed NLL loss, so val loss swung
        wildly (1.30 → −0.34) across the run. <code>best</code> and <code>last</code> landed
        statistically identical, so it did not bite this time — but the split should be widened
        before it does.</p></div>
  </div>
</section>

<footer>student_v3_fp16 backbone · residual_v3_last_fp16 · AGX Orin, jetson_clocks on ·
{AB["frames"]} moving frames, {AB['anchors_mean']:.0f} anchors/frame · 2026-07-28</footer>

</div>
"""

open(OUT, "w").write(HTML)
print(f"wrote {OUT} ({len(HTML)/1e6:.2f} MB)")
