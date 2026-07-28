"""Animated point-cloud viewer: a colour+depth video unprojected to points on the GPU.

Kept out of the page's f-string so the GLSL/JS braces don't need doubling.
build(video_b64, meta) -> HTML fragment.

The video is one frame per timestep, LEFT half colour and RIGHT half depth-as-luma. A
vertex shader samples both halves per point, so playback drives the cloud and the camera
stays free -- orbit/zoom while it plays.
"""
import json

_TMPL = r"""
<div class="cloudwrap">
  <canvas id="pcv" aria-label="Interactive 3D point cloud, animated over the drive"></canvas>
  <video id="pcsrc" style="display:none" loop muted playsinline preload="auto" src="__DATA__"></video>
  <div class="pcbar">
    <button type="button" data-view="follow">Behind robot</button>
    <button type="button" data-view="top">Top-down</button>
    <button type="button" data-view="side">Side</button>
    <button type="button" data-view="iso">Iso</button>
    <span class="pchint">drag to orbit · scroll to zoom</span>
  </div>
  <div class="pcbar pctransport">
    <button type="button" id="pcplay" aria-label="Play or pause the cloud">Pause</button>
    <button type="button" id="pcback" aria-label="Step back one frame">&#9664;|</button>
    <button type="button" id="pcfwd" aria-label="Step forward one frame">|&#9654;</button>
    <input type="range" id="pcseek" min="0" max="1000" value="0" step="1"
           aria-label="Scrub through the drive">
    <span class="pctime" id="pctime">0.0 / 0.0 s</span>
  </div>
</div>
<script>
(function () {
  var M = __META__;
  var cv = document.getElementById('pcv');
  var vid = document.getElementById('pcsrc');
  var gl = cv.getContext('webgl2', {antialias: true, alpha: false});
  if (!gl) {
    cv.outerHTML = '<p style="padding:18px;color:#93a4b3">This view needs WebGL2.</p>';
    return;
  }

  var VS = `#version 300 es
  precision highp float;
  uniform sampler2D uTex;
  uniform mat4 uMVP;
  uniform vec2 uGrid;      // points across, down
  uniform vec4 uK;         // fx, fy, cx, cy  (in stored pixels)
  uniform vec2 uZ;         // zmin, zmax
  uniform vec2 uQ;         // qmin, qspan  (valid luma levels; 0 is the no-data sentinel)
  uniform float uSize;
  out vec3 vCol;
  out float vDrop;
  void main() {
    float i = float(gl_VertexID);
    float gx = mod(i, uGrid.x);
    float gy = floor(i / uGrid.x);
    vec2 uv = (vec2(gx, gy) + 0.5) / uGrid;          // 0..1 within one half
    // depth is the RIGHT half, colour the LEFT half
    float q = texture(uTex, vec2(0.5 + uv.x * 0.5, uv.y)).r * 255.0;
    // valid levels start at uQ.x; the gap below it absorbs codec ringing so compression
    // cannot push a near-range point onto the no-data sentinel
    vDrop = q < (uQ.x * 0.5) ? 1.0 : 0.0;
    float z = uZ.x + clamp((q - uQ.x) / uQ.y, 0.0, 1.0) * (uZ.y - uZ.x);
    vCol = texture(uTex, vec2(uv.x * 0.5, uv.y)).rgb;
    // pixel coords in the STORED frame -> camera ray -> metric point
    float px = uv.x * uGrid.x;
    float py = uv.y * uGrid.y;
    float X = (px - uK.z) / uK.x * z;
    float Y = (py - uK.w) / uK.y * z;
    // camera frame is x right, y DOWN, z forward -> flip for a y-up view
    gl_Position = uMVP * vec4(X, -Y, -z, 1.0);
    gl_PointSize = uSize / max(gl_Position.w, 0.05);
  }`;
  var FS = `#version 300 es
  precision mediump float;
  in vec3 vCol; in float vDrop;
  out vec4 o;
  void main() {
    if (vDrop > 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    o = vec4(vCol, 1.0);
  }`;

  function sh(t, s) {
    var o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(o));
    return o;
  }
  var prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog); gl.useProgram(prog);
  var vao = gl.createVertexArray(); gl.bindVertexArray(vao);

  var tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  var U = {
    mvp: gl.getUniformLocation(prog, 'uMVP'),
    grid: gl.getUniformLocation(prog, 'uGrid'),
    k: gl.getUniformLocation(prog, 'uK'),
    z: gl.getUniformLocation(prog, 'uZ'),
    q: gl.getUniformLocation(prog, 'uQ'),
    size: gl.getUniformLocation(prog, 'uSize'),
    tex: gl.getUniformLocation(prog, 'uTex')
  };
  gl.enable(gl.DEPTH_TEST);

  var GX = M.w, GY = M.h, NPTS = GX * GY;
  var az = 0.0, el = 0.22, dist = 3.2, tgt = [0, 0, -1.6];

  function mul(a, b) {
    var o = new Float32Array(16);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
      var s = 0; for (var k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  }
  function persp(f, asp, zn, zf) {
    var t = 1 / Math.tan(f / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (zf + zn) / (zn - zf); o[11] = -1;
    o[14] = 2 * zf * zn / (zn - zf); return o;
  }
  function look(e, c, u) {
    var z = [e[0]-c[0], e[1]-c[1], e[2]-c[2]];
    var zl = Math.hypot(z[0],z[1],z[2]) || 1; z = [z[0]/zl, z[1]/zl, z[2]/zl];
    var x = [u[1]*z[2]-u[2]*z[1], u[2]*z[0]-u[0]*z[2], u[0]*z[1]-u[1]*z[0]];
    var xl = Math.hypot(x[0],x[1],x[2]) || 1; x = [x[0]/xl, x[1]/xl, x[2]/xl];
    var y = [z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]];
    var o = new Float32Array(16);
    o[0]=x[0]; o[4]=x[1]; o[8]=x[2];
    o[1]=y[0]; o[5]=y[1]; o[9]=y[2];
    o[2]=z[0]; o[6]=z[1]; o[10]=z[2];
    o[12]=-(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]);
    o[13]=-(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]);
    o[14]=-(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]);
    o[15]=1; return o;
  }

  function draw() {
    var w = cv.clientWidth, h = cv.clientHeight;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== (w * dpr | 0) || cv.height !== (h * dpr | 0)) {
      cv.width = w * dpr; cv.height = h * dpr;
    }
    if (vid.readyState >= 2) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, vid);
    }
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(0.043, 0.055, 0.067, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    var ce = Math.cos(el), se = Math.sin(el);
    var eye = [tgt[0] + dist * ce * Math.sin(az),
               tgt[1] + dist * se,
               tgt[2] + dist * ce * Math.cos(az)];
    var V = look(eye, tgt, [0, 1, 0]);
    var P = persp(1.05, w / h, 0.05, 60.0);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(U.mvp, false, mul(P, V));
    gl.uniform2f(U.grid, GX, GY);
    gl.uniform4f(U.k, M.fx, M.fy, M.cx, M.cy);
    gl.uniform2f(U.z, M.zmin, M.zmax);
    gl.uniform2f(U.q, M.qmin, M.qspan);
    gl.uniform1f(U.size, 3.0 * dpr);
    gl.uniform1i(U.tex, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, NPTS);
    requestAnimationFrame(draw);
  }

  var drag = false, lx = 0, ly = 0;
  cv.addEventListener('pointerdown', function (e) {
    drag = true; lx = e.clientX; ly = e.clientY;
    try { cv.setPointerCapture(e.pointerId); } catch (x) {}
  });
  cv.addEventListener('pointerup', function (e) {
    drag = false; try { cv.releasePointerCapture(e.pointerId); } catch (x) {}
  });
  cv.addEventListener('pointermove', function (e) {
    if (!drag) return;
    az -= (e.clientX - lx) * 0.008;
    el = Math.max(-1.45, Math.min(1.45, el + (e.clientY - ly) * 0.008));
    lx = e.clientX; ly = e.clientY;
  });
  cv.addEventListener('wheel', function (e) {
    e.preventDefault();
    dist = Math.max(0.4, Math.min(20, dist * (e.deltaY > 0 ? 1.11 : 0.9)));
  }, {passive: false});

  var views = {
    follow: [0.0, 0.22, 3.2], top: [0.0, 1.40, 4.2],
    side: [1.5708, 0.15, 3.6], iso: [0.8, 0.55, 3.8]
  };
  Array.prototype.forEach.call(document.querySelectorAll('.pcbar button[data-view]'), function (b) {
    b.addEventListener('click', function () {
      var v = views[b.getAttribute('data-view')];
      az = v[0]; el = v[1]; dist = v[2];
    });
  });
  // ---- transport: play/pause, frame step, scrub ----
  var pb = document.getElementById('pcplay');
  var seek = document.getElementById('pcseek');
  var tlab = document.getElementById('pctime');
  var DT = 1.0 / (M.fps || 7.5);
  var scrubbing = false;

  function fmt(t) { return (isFinite(t) ? t : 0).toFixed(1); }
  function syncLabel() {
    tlab.textContent = fmt(vid.currentTime) + ' / ' + fmt(vid.duration) + ' s';
  }
  function setPlay(p) { pb.textContent = p ? 'Pause' : 'Play'; }

  pb.addEventListener('click', function () {
    if (vid.paused) { vid.play().catch(function () {}); } else { vid.pause(); }
  });
  vid.addEventListener('play', function () { setPlay(true); });
  vid.addEventListener('pause', function () { setPlay(false); });

  function step(dir) {
    vid.pause();
    var d = vid.duration || 0;
    var t = vid.currentTime + dir * DT;
    vid.currentTime = Math.max(0, Math.min(d - 1e-3, t));
  }
  document.getElementById('pcback').addEventListener('click', function () { step(-1); });
  document.getElementById('pcfwd').addEventListener('click', function () { step(1); });

  seek.addEventListener('pointerdown', function () { scrubbing = true; });
  window.addEventListener('pointerup', function () { scrubbing = false; });
  seek.addEventListener('input', function () {
    var d = vid.duration;
    if (isFinite(d) && d > 0) { vid.currentTime = (seek.value / 1000) * d; syncLabel(); }
  });
  vid.addEventListener('timeupdate', function () {
    if (!scrubbing && isFinite(vid.duration) && vid.duration > 0) {
      seek.value = String((vid.currentTime / vid.duration) * 1000);
    }
    syncLabel();
  });
  vid.addEventListener('loadedmetadata', syncLabel);

  // arrow keys step, space toggles -- only once the viewer has focus
  cv.setAttribute('tabindex', '0');
  cv.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    else if (e.key === ' ') { pb.click(); e.preventDefault(); }
  });

  vid.play().catch(function () { setPlay(false); });
  syncLabel();
  requestAnimationFrame(draw);
})();
</script>
"""


def build(b64, meta):
    return (_TMPL
            .replace('__META__', json.dumps(meta))
            .replace('__DATA__', b64))
